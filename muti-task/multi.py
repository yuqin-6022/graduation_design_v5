#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time      : 2020/4/15 17:51
# @Author    : Shawn Li
# @FileName  : MMoE_hp.py
# @IDE       : PyCharm
# @Blog      : 暂无

import tensorflow as tf
import numpy as np
import pandas as pd
import kerastuner
from kerastuner import HyperModel
from kerastuner.tuners.bayesian import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state, compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score
from datetime import datetime
import time
import os
import json


# MultiMetrics---------------------------------------------------------------------------------------------------------
# 默认顺序：dloc、ed、overload-----------------------------------------------------------------------------------------
class MultiMetrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super().__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        val_predicts = self.model.predict(self.validation_data[0])
        task_num = len(val_predicts)

        _val_f1_all = 0

        for i in range(task_num):
            val_predict = np.argmax(val_predicts[i], -1)
            val_targ = self.validation_data[1][i]

            if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
                val_targ = np.argmax(val_targ, -1)

            _val_f1 = f1_score(val_targ, val_predict, average='weighted')
            _val_recall = recall_score(val_targ, val_predict, average='weighted')
            _val_precision = precision_score(val_targ, val_predict, average='weighted')

            _val_f1_all = _val_f1_all + _val_f1

            logs['task_%d_val_f1' % i] = _val_f1
            logs['task_%d_val_recall' % i] = _val_recall
            logs['task_%d_val_precision' % i] = _val_precision
            print(" — task_%d — val_f1: %f — val_precision: %f — val_recall: %f" % (
            i, _val_f1, _val_precision, _val_recall))

        _val_f1_mean = _val_f1_all / task_num
        logs['val_f1_mean'] = _val_f1_mean
        print(" — val_f1_mean: %f" % _val_f1_mean)

        return


# MyHyperModel---------------------------------------------------------------------------------------------------------
class MyHyperModel(HyperModel):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def build(self, hp):

        inputs = tf.keras.Input(shape=self.input_shape)

        experts_num = hp.Int('experts_num', min_value=2, max_value=6, step=1)

        # gates--------------------------------------------------------------------------------------------------------
        dloc_gate_dense = tf.keras.layers.Dense(units=experts_num, name='dloc_dense')(inputs)
        dloc_gate_out = tf.keras.layers.Softmax(name='gate_weight_for_dloc')(dloc_gate_dense)
        ed_gate_dense = tf.keras.layers.Dense(units=experts_num, name='ed_dense')(inputs)
        ed_gate_out = tf.keras.layers.Softmax(name='gate_weight_for_ed')(ed_gate_dense)
        overload_gate_dense = tf.keras.layers.Dense(units=experts_num, name='overload_dense')(inputs)
        overload_gate_out = tf.keras.layers.Softmax(name='gate_weight_for_overload')(overload_gate_dense)
        gates_out = [dloc_gate_out, ed_gate_out, overload_gate_out]
        gates_out = tf.concat(gates_out, 1)
        gates_out = tf.reshape(gates_out, (-1, 3, experts_num))

        # experts------------------------------------------------------------------------------------------------------
        experts_out = []
        expert_out_units = hp.Int('expert_out_unit', min_value=16, max_value=64, step=16)
        for i in range(experts_num):
            expert_fc_out = inputs
            for j in range(hp.Int('experts%d_fc' % i, min_value=1, max_value=2, step=1)):
                expert_fc_out = tf.keras.layers.Dense(units=hp.Int('units%d' % j, min_value=64, max_value=256, step=64),
                                                      activation='relu', name=('expert%d_dense%d' % (i, j)))(expert_fc_out)
                expert_fc_out = tf.keras.layers.BatchNormalization(name=('expert%d_dense%d_bn' % (i, j)))(expert_fc_out)  # bn
            expert_fc_out = tf.keras.layers.Dense(units=expert_out_units, activation='relu', name=('expert%d_same_dense' % i))(expert_fc_out)  # 统一输出units数
            expert_fc_out = tf.keras.layers.BatchNormalization(name=('expert%d_same_dense_bn' % i))(expert_fc_out)  # bn
            experts_out.append(expert_fc_out)  # 将多个expert的输出拼接
        experts_out = tf.concat(experts_out, 1)
        experts_out = tf.reshape(experts_out, (-1, experts_num, expert_out_units))

        # towers-------------------------------------------------------------------------------------------------------
        tower_input = tf.matmul(gates_out, experts_out)
        # dloc_towers--------------------------------------------------------------------------------------------------
        dloc_tower_out = tower_input[:, 0]
        for i in range(hp.Int('dloc_tower_fc', min_value=1, max_value=2, step=1)):
            dloc_tower_out = tf.keras.layers.Dense(units=hp.Int('units%d' % i, min_value=64, max_value=256, step=64),
                                                   activation='relu', name=('dloc_tower_dense%d' % i))(dloc_tower_out)
            dloc_tower_out = tf.keras.layers.BatchNormalization(name=('dloc_tower_dense%d_bn' % i))(dloc_tower_out)  # bn
            dloc_tower_out = tf.keras.layers.Dropout(
                rate=hp.Float('dloc_dropout_rate%d' % i, min_value=0, max_value=0.5, step=0.05), name=('dloc_tower_dense%d_dropout' % i))(
                dloc_tower_out)  # dropout
        dloc_tower_out = tf.keras.layers.Dense(units=100, name='dloc_tower_output_dense')(dloc_tower_out)
        dloc_tower_out = tf.keras.layers.Softmax(name='dloc_softmax')(dloc_tower_out)
        # ed_towers----------------------------------------------------------------------------------------------------
        ed_tower_out = tower_input[:, 1]
        for i in range(hp.Int('ed_tower_fc', min_value=1, max_value=2, step=1)):
            ed_tower_out = tf.keras.layers.Dense(units=hp.Int('units%d' % i, min_value=64, max_value=256, step=64),
                                                 activation='relu', name=('ed_tower_dense%d' % i))(ed_tower_out)
            ed_tower_out = tf.keras.layers.BatchNormalization(name=('ed_tower_dense%d_bn' % i))(ed_tower_out)  # bn
            ed_tower_out = tf.keras.layers.Dropout(
                rate=hp.Float('ed_dropout_rate%d' % i, min_value=0, max_value=0.5, step=0.05), name=('ed_tower_dense%d_dropout' % i))(ed_tower_out)  # dropout
        ed_tower_out = tf.keras.layers.Dense(units=10, name='ed_tower_output_dense')(ed_tower_out)
        ed_tower_out = tf.keras.layers.Softmax(name='ed_softmax')(ed_tower_out)
        # overload_towers----------------------------------------------------------------------------------------------
        overload_tower_out = tower_input[:, 2]
        for i in range(hp.Int('ed_tower_fc', min_value=1, max_value=2, step=1)):
            overload_tower_out = tf.keras.layers.Dense(units=hp.Int('units%d' % i, min_value=64, max_value=256, step=64),
                                                       activation='relu', name=('overload_tower_dense%d' % i))(overload_tower_out)
            overload_tower_out = tf.keras.layers.BatchNormalization(name=('overload_tower_dense%d_bn' % i))(overload_tower_out)  # bn
            overload_tower_out = tf.keras.layers.Dropout(
                rate=hp.Float('overload_dropout_rate%d' % i, min_value=0, max_value=0.5, step=0.05), name=('overload_tower_dense%d_dropout' % i))(
                overload_tower_out)  # dropout
        overload_tower_out = tf.keras.layers.Dense(units=4, name='overload_tower_output_dense')(overload_tower_out)
        overload_tower_out = tf.keras.layers.Softmax(name='overload_softmax')(overload_tower_out)

        model = tf.keras.Model(inputs=inputs, outputs=[dloc_tower_out, ed_tower_out, overload_tower_out])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss={
                'dloc_softmax': 'sparse_categorical_crossentropy',
                'ed_softmax': 'sparse_categorical_crossentropy',
                'overload_softmax': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'dloc_softmax': hp.Int('dloc_loss_weight', min_value=1, max_value=10, step=1),
                'ed_softmax': hp.Int('ed_loss_weight', min_value=1, max_value=10, step=1),
                'overload_softmax': 1
            },
            metrics=['accuracy']
        )

        model.summary()

        return model


# 终端运行-------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Starting...')
    start_time = time.time()

    model_type = 'bn_after'

    snr_list = list(range(1, 11))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # snr_list = list(range(1, 6))  # [1, 2, 3, 4, 5]
    # snr_list = list(range(6, 11))  # [6, 7, 8, 9, 10]
    # snr_list = list(range(1, 6, 2))  # [1, 3, 5]
    # snr_list = list(range(6, 11, 2))  # [6, 8, 10]

    # 设置gpu---------------------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    print(gpus, cpus)

    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]
    )

    # for gpu in gpus:
    #     tf.config.experimental.set_virtual_device_configuration(
    #         gpu,
    #         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
    #     )

    OBJECTIVE = kerastuner.Objective("val_f1_mean", direction="max")
    # OBJECTIVE = 'val_accuracy'
    MAX_TRIALS = 25
    EPOCHS = 2500
    FIT_EPOCHS = 50000
    BATCH_SIZE = 1024
    CUR_PATH = os.getcwd()
    DATETIME = datetime.now().strftime('%Y%m%d%H%M%S')

    KERAS_TUNER_DIR = os.path.join(CUR_PATH, 'keras_tuner_dir')
    if not os.path.exists(KERAS_TUNER_DIR):
        os.makedirs(KERAS_TUNER_DIR)

    BEST_F1_MODEL_DIR = os.path.join(CUR_PATH, 'models')
    if not os.path.exists(BEST_F1_MODEL_DIR):
        os.makedirs(BEST_F1_MODEL_DIR)

    # 数据集-----------------------------------------------------------------------------------------------------------
    train_df = pd.read_csv('../dataset/train.csv')
    test_df = pd.read_csv('../dataset/test.csv')

    TEST_SIZE = 2700

    x_list = ['Sn%d' % i for i in snr_list]
    x_list.append('Tt')
    x_train_origin = train_df.loc[:, x_list].copy().values
    y_train_origin = train_df[['dloc', 'ED', 'overload_loc']].copy().values

    x_test = test_df.iloc[:, x_list].copy().values
    y_test = test_df[['dloc', 'ED', 'overload_loc']].copy().values

    x_train, x_valid, y_train, y_valid = train_test_split(x_train_origin, y_train_origin, test_size=TEST_SIZE)

    y_dloc_train = y_train[:, 0]
    y_dloc_valid = y_valid[:, 0]
    y_dloc_test = y_test[:, 0]

    y_ED_train = y_train[:, 1]
    y_ED_valid = y_valid[:, 1]
    y_ED_test = y_test[:, 1]

    y_overload_train = y_train[:, 2]
    y_overload_valid = y_valid[:, 2]
    y_overload_test = y_test[:, 2]

    # 标准化处理-------------------------------------------------------------------------------------------------------
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)

    # 处理标准化后大于三倍标准差的值，此处标准差为1
    for i in range(x_train.shape[1]):
        x_train[:, i][x_train[:, i] > 3] = 3
        x_train[:, i][x_train[:, i] < -3] = -3

        x_valid[:, i][x_valid[:, i] > 3] = 3
        x_valid[:, i][x_valid[:, i] < -3] = -3

        x_test[:, i][x_test[:, i] > 3] = 3
        x_test[:, i][x_test[:, i] < -3] = -3

    # 超参搜索开始-----------------------------------------------------------------------------------------------------
    # 考虑样本权重-----------------------------------------------------------------------------------------------------
    # my_class_weight = compute_class_weight('balanced', np.unique(y_train), y_train).tolist()
    # cw = dict(zip(np.unique(y_train), my_class_weight))
    # print(cw)

    # keras-tuner部分设置----------------------------------------------------------------------------------------------
    # CALLBACKS = [tf.keras.callbacks.EarlyStopping(patience=3)]
    CALLBACKS = [
        MultiMetrics(valid_data=(x_valid, [y_dloc_valid, y_ED_valid, y_overload_valid]))
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.5, mode='auto')
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1', patience=10, factor=0.5, mode='max')
    ]
    best_f1_model_path = os.path.join(BEST_F1_MODEL_DIR, '%s.hdf5' % model_type)
    FIT_CALLBACKS = [
        MultiMetrics(valid_data=(x_valid, [y_dloc_valid, y_ED_valid, y_overload_valid])),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.5, mode='auto'),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1', patience=10, factor=0.5, mode='max'),
        tf.keras.callbacks.ModelCheckpoint(best_f1_model_path, monitor='val_f1_mean', verbose=2, save_best_only=True, mode='max')
    ]
    PROJECT_NAME = '%s_keras_tuner_dir' % model_type

    # 实例化贝叶斯优化器
    hypermodel = MyHyperModel((x_train.shape[1],))
    tuner = BayesianOptimization(hypermodel, objective=OBJECTIVE, max_trials=MAX_TRIALS, directory=KERAS_TUNER_DIR,
                                 project_name=PROJECT_NAME)
    # 开始计时超参数搜索
    tuner_start_time = datetime.now()
    tuner_start = time.time()
    # 开始超参数搜索
    tuner.search(x_train, [y_dloc_train, y_ED_train, y_overload_train], batch_size=BATCH_SIZE, epochs=EPOCHS,
                 validation_data=(x_valid, [y_dloc_valid, y_ED_valid, y_overload_valid]), callbacks=CALLBACKS)
    # tuner.search(x_train, y_train, batch_size=TUNER_BATCH_SIZE, epochs=TUNER_EPOCHS, validation_data=(x_valid, y_valid))
    # 结束计时超参数搜索
    tuner_end_time = datetime.now()
    tuner_end = time.time()
    # 统计超参数搜索用时
    tuner_duration = tuner_end - tuner_start

    # 获取前BEST_NUM个最优超参数--------------------------------------------------------------
    best_models = tuner.get_best_models()
    best_model = best_models[0]

    history = best_model.fit(x_train, [y_dloc_train, y_ED_train, y_overload_train], batch_size=BATCH_SIZE, epochs=FIT_EPOCHS,
                             validation_data=(x_valid, [y_dloc_valid, y_ED_valid, y_overload_valid]), callbacks=FIT_CALLBACKS, verbose=2)
    print(best_model.evaluate(x_test, [y_dloc_test, y_ED_test, y_overload_test]))

    # 恢复到最佳权重
    model = tf.keras.models.load_model(best_f1_model_path)
    print(model.evaluate(x_test, [y_dloc_test, y_ED_test, y_overload_test]))
    y_pred = model.predict(x_test)
    y_dloc_pred = np.argmax(y_pred[0], axis=1)
    y_ed_pred = np.argmax(y_pred[1], axis=1)
    y_overload_pred = np.argmax(y_pred[2], axis=1)

    pred_csv = 'multi_pred.csv'
    if not os.path.exists(pred_csv):
        pred_df = test_df[['dloc', 'ED', 'overload_loc']].copy()
    else:
        pred_df = pd.read_csv(pred_csv)
    pred_df['%s_dloc' % model_type] = y_dloc_pred
    pred_df['%s_ED' % model_type] = y_ed_pred
    pred_df['%s_overload' % model_type] = y_overload_pred

    pred_df.to_csv(pred_csv, index=False)

    end_time = time.time()
    time_consuming = end_time - start_time
    print('Time_consuming: %d' % int(time_consuming))

    print('Finish!')

