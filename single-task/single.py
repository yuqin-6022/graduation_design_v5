#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time      : 2020/4/15 18:13
# @Author    : Shawn Li
# @FileName  : single_task_hp.py
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
import argparse


# Metrics--------------------------------------------------------------------------------------------------------------
class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='weighted')
        _val_recall = recall_score(val_targ, val_predict, average='weighted')
        _val_precision = precision_score(val_targ, val_predict, average='weighted')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return


# MyHyperModel---------------------------------------------------------------------------------------------------------
class MyHyperModel(HyperModel):
    def __init__(self, input_shape, output_num):
        super().__init__()
        self.input_shape = input_shape
        self.output_num = output_num

    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape))
        for i in range(hp.Int('num_layers', min_value=1, max_value=5, step=1)):
            model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                         min_value=16,
                                                         max_value=256,
                                                         step=16),
                                            activation='relu'))
            model.add(tf.keras.layers.BatchNormalization())  # bn
        # 输出层
        # model.add(tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(hp.Choice('l2_rate', [1e-2, 1e-3, 1e-4]))))
        model.add(tf.keras.layers.Dense(units=self.output_num, activation='softmax'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy']
        )

        return model


# 终端运行-------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Starting...')
    start_time = time.time()

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--y-type', type=str, default=None)
    # parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    y_type = args.y_type

    model_type = 'bn_after-nodropout'
    # y_type = 'dloc'
    # y_type = 'ED'
    # y_type = 'overload_loc'

    snr_list = list(range(1, 11))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # snr_list = [1, 3, 5, 6, 8, 10]
    # snr_list = [1, 5, 6, 10]
    # snr_list = [2, 4, 7, 9]
    # snr_list = list(range(1, 6))  # [1, 2, 3, 4, 5]
    # snr_list = list(range(6, 11))  # [6, 7, 8, 9, 10]
    # snr_list = list(range(1, 6, 2))  # [1, 3, 5]
    # snr_list = list(range(6, 11, 2))  # [6, 8, 10]

    # 设置gpu---------------------------------------------------------------------------------
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    #
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    # print(gpus, cpus)
    #
    # tf.config.experimental.set_virtual_device_configuration(
    #     gpus[0],
    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
    # )


    # for gpu in gpus:
    #     tf.config.experimental.set_virtual_device_configuration(
    #         gpu,
    #         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
    #     )

    OBJECTIVE = kerastuner.Objective("val_f1", direction="max")
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
    KERAS_TUNER_DIR = os.path.join(KERAS_TUNER_DIR, '%s_keras_tuner_dir' % model_type)
    if not os.path.exists(KERAS_TUNER_DIR):
        os.makedirs(KERAS_TUNER_DIR)

    BEST_F1_MODEL_DIR = os.path.join(CUR_PATH, 'models')
    if not os.path.exists(BEST_F1_MODEL_DIR):
        os.makedirs(BEST_F1_MODEL_DIR)
    BEST_F1_MODEL_DIR = os.path.join(BEST_F1_MODEL_DIR, '%s' % model_type)
    if not os.path.exists(BEST_F1_MODEL_DIR):
        os.makedirs(BEST_F1_MODEL_DIR)

    # 数据集-----------------------------------------------------------------------------------------------------------
    train_df = pd.read_csv('../dataset/train.csv')
    test_df = pd.read_csv('../dataset/test.csv')

    print('--------------------------------------------------------------------------------------------------------')
    print(y_type)
    print('--------------------------------------------------------------------------------------------------------')

    TEST_SIZE = 2700

    x_list = ['Sn%d' % i for i in snr_list]
    x_list.append('Tt')
    x_train_origin = train_df.loc[:, x_list].copy().values
    y_train_origin = train_df[y_type].copy().values

    x_test = test_df.loc[:, x_list].copy().values
    y_test = test_df[y_type].copy().values

    x_train, x_valid, y_train, y_valid = train_test_split(x_train_origin, y_train_origin, test_size=TEST_SIZE)

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
    my_class_weight = compute_class_weight('balanced', np.unique(y_train), y_train).tolist()
    cw = dict(zip(np.unique(y_train), my_class_weight))
    print(cw)

    # keras-tuner部分设置----------------------------------------------------------------------------------------------
    # CALLBACKS = [tf.keras.callbacks.EarlyStopping(patience=3)]
    CALLBACKS = [
        Metrics(valid_data=(x_valid, y_valid))
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.5, mode='auto')
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1', patience=10, factor=0.5, mode='max')
    ]
    best_f1_model_path = os.path.join(BEST_F1_MODEL_DIR, '%s.hdf5' % y_type)
    FIT_CALLBACKS = [
        Metrics(valid_data=(x_valid, y_valid)),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.5, mode='auto'),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1', patience=10, factor=0.5, mode='max'),
        tf.keras.callbacks.ModelCheckpoint(best_f1_model_path, monitor='val_f1', verbose=2, save_best_only=True, mode='max')
    ]
    PROJECT_NAME = os.path.join(KERAS_TUNER_DIR, '%s_keras_tuner_dir' % y_type)

    # 实例化贝叶斯优化器
    y_num = len(train_df[y_type].unique()) + (y_type == 'ED')  # ED从1~9开始，所以补一个输出
    hypermodel = MyHyperModel((x_train.shape[1],), y_num)
    tuner = BayesianOptimization(hypermodel, objective=OBJECTIVE, max_trials=MAX_TRIALS, directory=KERAS_TUNER_DIR, project_name=PROJECT_NAME)
    # 开始计时超参数搜索
    tuner_start_time = datetime.now()
    tuner_start = time.time()
    # 开始超参数搜索
    tuner.search(x_train, y_train, class_weight=cw, batch_size=BATCH_SIZE, epochs=EPOCHS,
                 validation_data=(x_valid, y_valid), callbacks=CALLBACKS)
    # tuner.search(x_train, y_train, batch_size=TUNER_BATCH_SIZE, epochs=TUNER_EPOCHS, validation_data=(x_valid, y_valid))
    # 结束计时超参数搜索
    tuner_end_time = datetime.now()
    tuner_end = time.time()
    # 统计超参数搜索用时
    tuner_duration = tuner_end - tuner_start

    # 获取前BEST_NUM个最优超参数--------------------------------------------------------------
    best_models = tuner.get_best_models()
    best_model = best_models[0]

    history = best_model.fit(x_train, y_train, class_weight=cw, batch_size=BATCH_SIZE, epochs=FIT_EPOCHS, validation_data=(x_valid, y_valid), callbacks=FIT_CALLBACKS, verbose=2)
    print(best_model.evaluate(x_test, y_test))

    # 恢复到最佳权重
    model = tf.keras.models.load_model(best_f1_model_path)
    print(model.evaluate(x_test, y_test))
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    pred_csv = 'single_%s_pred.csv' % y_type
    if not os.path.exists(pred_csv):
        pred_df = test_df.loc[:, [y_type]].copy()
    else:
        pred_df = pd.read_csv(pred_csv)
    pred_df[model_type] = y_pred

    pred_df.to_csv(pred_csv, index=False)

    end_time = time.time()
    time_consuming = end_time - start_time
    print('Time_consuming: %d' % int(time_consuming))

    print('Finish!')

