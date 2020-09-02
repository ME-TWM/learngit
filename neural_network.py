# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import csv
from matplotlib import pyplot as plt
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from math import *
import time


class neural_network(QThread):

    update_msg = pyqtSignal(dict)  # QThread线程信号(括号内指定信号类型)
    update_info = pyqtSignal(str)
    update_acc = pyqtSignal(str)
    update_top_acc = pyqtSignal(str)
    stop_time = pyqtSignal()
    start_pso = pyqtSignal()
    input_dir = None
    output_dir = None
    input_dimension = None
    learning_rate = None
    num_steps = None
    batch_size = None
    target_acc = None
    network_struct = None
    deepth = None
    acc = 0
    update_plot_acc = pyqtSignal(float)
    shut_down = False
    start_nn = False
    suspend = False

    # 载入CSV文件并以np.array数组返回
    def loaddata(self, input_data, output_data):
        # data_set = np.zeros(shape=[4000, 28], dtype=np.float32)  # 数据类型默认为float64
        data_set = []
        line_num = 0  # 行号
        with open(input_data, 'r') as file:
            # csvReader = csv.reader(file)
            for line in file:
                line_num += 1
                if line_num == 1:
                    continue
                # 以，作为分隔符并转为float格式，将行append进data_set
                data_set.append([float(x) for x in line.split(',')])
        line_num = 0
        with open(output_data, 'r') as file:
            for line in file:
                line_num += 1
                if line_num == 1:
                    continue
                result_data = [float(x) for x in line.split(',')]
                data_set[int(result_data[0])] = data_set[int(result_data[0])] + result_data[1:]
        new = []
        for item in data_set:
            if len(item) == 106:
                new.append(item)
        new = np.array(new)
        np.random.shuffle(new)
        print('check point:\nshape of data:', new.shape)
        return new

    # 构造网络时只用一个样本来考虑即可（多个样本都是重复一个样本的计算过程），比如一个样本28个特征（即一行28列），w1=[28,30]就是
    # 指下一层有30个神经元，每个各28个权重（因为tf.matmul为矩阵相乘，tf.multiply为对应元素相乘）
    # 一个样本输出为一行，所以bias形状应为[1,30]

    def network(self, inputs, reuse, num=3, activation_function=tf.nn.relu):
        print('any question?')
        with tf.variable_scope('cqc', reuse=reuse):
            # 由于network_struct末尾被多加了个0，所以mag大一个数量级
            # mag = 10 ** len(self.network_struct)
            # neural_num = int(self.network_struct) // mag
            print('it is impossible????')
            weights1 = tf.get_variable('w1', shape=[self.input_dimension, self.network_struct[0]], initializer=tf.random_normal_initializer())
            bias1 = tf.get_variable('b1', shape=[1, self.network_struct[0]], initializer=tf.zeros_initializer())
            print('it is possible????')
            h1 = tf.matmul(inputs, weights1) + bias1
            activate1 = activation_function(h1)
            # num = num - 1
            if self.deepth == 1:
                weights_out = tf.get_variable('w4', shape=[self.network_struct[0], 6], initializer=tf.random_normal_initializer())
                bias_out = tf.get_variable('b4', shape=[1, 6], initializer=tf.zeros_initializer())
                outputs = tf.matmul(activate1, weights_out) + bias_out
                return outputs
            print('possible??????')
            # mag = 10 ** len(self.network_struct)
            # neural_num = int(self.network_struct) // mag
            weights2 = tf.get_variable('w2', shape=[self.network_struct[0], self.network_struct[1]], initializer=tf.random_normal_initializer())
            bias2 = tf.get_variable('b2', shape=[1, self.network_struct[1]], initializer=tf.zeros_initializer())
            h2 = tf.matmul(activate1, weights2) + bias2
            activate2 = activation_function(h2)
            # num = num - 1
            print('hahaha\n')
            if self.deepth == 2:
                weights_out = tf.get_variable('w4', shape=[self.network_struct[1], 6], initializer=tf.random_normal_initializer())
                bias_out = tf.get_variable('b4', shape=[1, 6], initializer=tf.zeros_initializer())
                outputs = tf.matmul(activate2, weights_out) + bias_out
                return outputs

            print('i am here\n')
            # mag = 10 ** len(self.network_struct)
            # neural_num = int(self.network_struct) // mag

            weights3 = tf.get_variable('w3', shape=[self.network_struct[1], self.network_struct[2]], initializer=tf.random_normal_initializer())
            bias3 = tf.get_variable('b3', shape=[1, self.network_struct[2]], initializer=tf.zeros_initializer())
            h3 = tf.matmul(activate2, weights3) + bias3
            activate3 = activation_function(h3)
            # self.network_struct = self.network_struct[1:]
            # old_neural_num = neural_num
            # num = num - 1
            if self.deepth == 3:
                weights_out = tf.get_variable('w4', shape=[self.network_struct[2], 6], initializer=tf.random_normal_initializer())
                bias_out = tf.get_variable('b4', shape=[1, 6], initializer=tf.zeros_initializer())
                outputs = tf.matmul(activate3, weights_out) + bias_out
                return outputs

            # 回归问题输出不用经过激活函数（？？？）



    # 返回一个batch的数据
    def next_batch(self, subdata, batch_size, train=True):
        if train == True:
            np.random.shuffle(subdata)   # 多维数组时只对第一维（行）打乱
            #print(subdata.shape)
            train_set = subdata[0:batch_size, 0:100]
            train_label = subdata[0:batch_size, 100:]/1e4  # 将输出规范化
            #test_set = data[4001:, 1:29]
            #test_label = data[4001:, 29:]
            return train_set, train_label
        test_set = subdata[:, 0:100]
        test_label = subdata[:, 100:]/1e4
        return test_set, test_label

    def run(self):
        while True:
            if self.start_nn == False:
                time.sleep(1)  # 设置检查时间
            else:
                # self.update_info.emit('working....please wait.\n')
                self.network_struct = [int(i) for i in self.network_struct.split('x')]
                self.deepth = len(self.network_struct)
                learning_rate = float(self.learning_rate)  # 学习率
                num_steps = int(self.num_steps)  # 迭代次数
                batch_size = int(self.batch_size)  # 每批样本数(这个数越大，每一次训练的loss也越大)
                target_acc = float(self.target_acc)  # target_loss, 定义目标函数合格值(注意这个loss是一个batch的loss，而不是1个样本的)

                x_place = tf.placeholder(tf.float32, [None, self.input_dimension], name='x_place')
                y_place = tf.placeholder(tf.float32, [None, 6], name='y_place')

                data = self.loaddata(self.input_dir, self.output_dir)
                print(data)
                # 前2500个数据作为训练集
                train_data = data[0:2000, :]
                test_data = data[2000:, :]

                train_result = self.network(x_place, reuse=False, activation_function=tf.nn.relu)
                print('this is it')
                b = tf.constant(value=1, dtype=tf.float32)
                logits_eval = tf.multiply(train_result, b, name='logits_eval')

                loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_place - logits_eval)))
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
                mean_accuracy = tf.reduce_mean(1 - tf.abs((logits_eval - y_place) / y_place))
                print('whatever')
                test = self.network(x_place, reuse=True, activation_function=tf.nn.relu)
                # test_error =

                loss_data = []
                train_round = []
                test_sample = []
                mode_1_dev = []
                mode_2_dev = []
                mode_3_dev = []
                mode_4_dev = []
                torsion_dev = []
                bend_dev = []
                max_acc = 0
                msg = {}

                init = tf.global_variables_initializer()
                saver = tf.train.Saver()
                with tf.Session() as sess:
                    sess.run(init)
                    i = 0
                    while True:
                        train_set, train_label = self.next_batch(train_data, batch_size=batch_size)
                        # 每次迭代将一个batch的数据喂入训练优化器
                        sess.run(train_step, feed_dict={x_place: train_set, y_place: train_label})
                        losses, self.acc = sess.run([loss, mean_accuracy], feed_dict={x_place: train_set, y_place: train_label})
                        # 每隔200次迭代显示一次loss
                        self.update_acc.emit(str(self.acc))
                        self.update_plot_acc.emit(self.acc)
                        msg['loss'] = str(losses)
                        self.update_msg.emit(msg)
                        if i % 200 == 0:
                            train_msg = '\nround ' + str(i) + '  train loss: ' + str(losses)
                            print(train_msg)
                            loss_data.append(losses)
                            print('aa')
                            train_round.append(i)
                        i += 1
                        if self.acc > max_acc:
                            print('bb')
                            max_acc = self.acc
                            saver.save(sess, '.\suprised_model.ckpt', global_step=i)
                            msg['top_acc'] = str(max_acc)
                            self.update_top_acc.emit(str(max_acc))
                        print('cc')
                        while self.suspend:
                            time.sleep(1)
                        # 结束循环
                        print('dd')
                        if max_acc >= target_acc or i > num_steps or self.shut_down == True:
                            self.stop_time.emit()
                            break
                        print('ee')
                    # self.update_info.emit('training completed.')
                    test_set, test_label = self.next_batch(test_data, batch_size=batch_size, train=False)
                    # self.update_info.emit('check point:\nshape of test data:', test_set.shape)

                    for i in range(100):
                        # 每次迭代喂入1个数据
                        predict_val = sess.run(test, feed_dict={x_place: test_set[i:i+1, :]})
                        test_sample.append(i)
                        deviation = np.fabs(predict_val - test_label[i:i+1, :]) / test_label[i:i+1, :]
                        # print(deviation)
                        mode_1_dev.append(deviation[0][0])
                        mode_2_dev.append(deviation[0][1])
                        mode_3_dev.append(deviation[0][2])
                        mode_4_dev.append(deviation[0][3])
                        torsion_dev.append(deviation[0][4])
                        bend_dev.append(deviation[0][5])
                        # 每隔10个数据显示一次实际值和预测值
                        if i%10 == 0:
                            print('test label:', test_label[i:i+1, :]*1e4, 'pridict value:', predict_val*1e4)

                #    print('predict result:')
                #    print(sess.run(test, feed_dict={xp: x_test}))
                    # 打印loss图表
                    plt.figure(1)
                    #plt.subplot(121)
                    plt.figure(figsize=(15, 5))   # 指定输出像素为1500*500
                    plt.plot(train_round, loss_data, 'black')
                    plt.ylabel('train loss')

                    # 打印预测误差图表
                    plt.figure(2)
                    #plt.subplot(122)
                    plt.figure(figsize=(15, 5))
                    mode_1, = plt.plot(test_sample, mode_1_dev,  'x')
                    mode_2, = plt.plot(test_sample, mode_2_dev, 'x')
                    mode_3, = plt.plot(test_sample, mode_3_dev, 'x')
                    mode_4 = plt.plot(test_sample, mode_4_dev, 'x')
                    torsion = plt.plot(test_sample, torsion_dev, 'x')
                    bend = plt.plot(test_sample, bend_dev, 'x')

                    plt.legend([mode_1, mode_2, mode_3, torsion, bend], ['dev of mode1', 'dev of mode2', 'dev of mode3', 'dev of torsion',
                                                                         'dev of bend'])
                    #plt.ylabel('deviation')
                    plt.show()
                while self.start_nn == True:
                    time.sleep(1)
                continue


