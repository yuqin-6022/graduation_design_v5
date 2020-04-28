#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time      : 2020/4/15 17:45
# @Author    : Shawn Li
# @FileName  : single_lr.py
# @IDE       : PyCharm
# @Blog      : 暂无

import tensorflow as tf
import numpy as np
import pandas as pd
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


# 终端运行-------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Starting...')
    start_time = time.time()

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--y-type', type=str, default=None)
    # parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    y_type = args.y_type

    # 设置gpu---------------------------------------------------------------------------------
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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

    CUR_PATH = os.getcwd()
    DATETIME = datetime.now().strftime('%Y%m%d%H%M%S')

    EPOCHS = 50000
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.01

    MODEL_DIR = os.path.join(CUR_PATH, 'models')
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

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

    my_class_weight = compute_class_weight('balanced', np.unique(y_train), y_train).tolist()
    cw = dict(zip(np.unique(y_train), my_class_weight))
    print(cw)

    softmax_num = len(train_df[y_type].unique()) + (y_type == 'ED')  # ED从1~9开始，所以补一个输出
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x_train.shape[1], )),
        tf.keras.layers.Dense(units=softmax_num, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=['accuracy']
    )

    model_path = os.path.join(MODEL_DIR, 'lr_%s.hdf5' % y_type)
    CALLBACKS = [
        Metrics(valid_data=(x_valid, y_valid)),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.9, min_lr=0.001, mode='auto'),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_f1', verbose=2, save_best_only=True, mode='max')
    ]

    history = model.fit(x_train, y_train, class_weight=cw, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_valid, y_valid), callbacks=CALLBACKS, verbose=2)
    print(model.evaluate(x_test, y_test))

    # 恢复到最佳权重
    model = tf.keras.models.load_model(model_path)
    print(model.evaluate(x_test, y_test))
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    pred_csv = 'baseline_pred.csv'
    if not os.path.exists(pred_csv):
        pred_df = test_df.loc[:, ['dloc', 'ED', 'overload_loc']].copy()
    else:
        pred_df = pd.read_csv(pred_csv)
    pred_df['baseline_%s' % y_type] = y_pred

    pred_df.to_csv(pred_csv, index=False)

    end_time = time.time()
    time_consuming = end_time - start_time
    print('Time_consuming: %d' % int(time_consuming))

    print('Finish!')

