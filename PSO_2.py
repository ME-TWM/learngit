# -*- coding: utf-8 -*-

import numpy as np
import random
from math import *
import tensorflow as tf
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import time
import copy
from network_model import *
from matplotlib import pyplot as plt


class PSO_model(QThread):
    update_result = pyqtSignal(str)
    # update_result = pyqtSignal(list)
    signal = False
    update_boundary = pyqtSignal()
    transmit_to_up = pyqtSignal(list)
    transmit_to_low = pyqtSignal(list)
    init_global_best = None
    init_local_best = None
    # def __init__(self, size_pop=None, v_limit=None, pop_max=None, pop_min=None, idimension=None, odimension=None,
    #              fitness_fun=None, maxg=None, c1=2.0, c2=2.0, target_value=None, error=None):  # 注意target_value为一个元组
    size_pop = None  # 种群规模
    v_limit = 1  # 速度限制(如何定义？)
    # self.pop_limit = pop_limit  # 飞行空间限制
    pop_max = None  # ！
    pop_min = None  # ！
    input_dimension = None  # 种群维度 #！
    output_dimension = 6
    # fitness_model = fitness_fun  # fitness_model应该返回一个含6个元素的元组
    maxg = 500  # 迭代次数
    c1 = 2
    c2 = 2
    target_value = None
    error = None
    weight = None
    span = None
    best_distance = None
    boundary_matrix = None
    # 适应度历史纪录，大小为(maxg, size_pop, odimension)
    # self.history_of_fitness = np.zeros(shape=(self.maxg, self.size_pop, self.output_dimension))
    global_best = []  # 历史全局最优，为一个含3个元素的列表,分别表示fitness,pos,vel
    single_best_fitness = []  # 大小为size_pop的列表，记录每个粒子的历史最佳适应值，与pos,vel同步更新
    # single_best_pos = np.zeros(shape=(size_pop, 3))  # 每个粒子自身的历史最优适应值的位置，大小为(size_pop, 3)
    # single_best_vel = np.zeros(shape=(size_pop, 3))  # 每个粒子自身的历史最优适应值的速度
    qualified_index = None
    fitness_log = []  # 记录每次迭代粒子的最佳适应值
    predict_log = []  # 记录每次迭代粒子的最佳预测值
    suspend = False

    def update_vel(self, v_current, pos_current, gbest, zbest):
        v_current = v_current + self.c1 * np.random.random([self.size_pop, self.input_dimension]) * (zbest - pos_current) + \
                    self.c2 * np.random.random([self.size_pop, self.input_dimension]) * (gbest - pos_current)
        v_current = v_current.clip(-self.v_limit, self.v_limit)
        return v_current

    def update_pos(self, pos_current, v_after_update):
        pos_current = pos_current + v_after_update * 1  # 速度*时间=距离
        for i in range(self.size_pop):
            for j in range(self.input_dimension):
                if pos_current[i, j] > self.boundary_matrix[0][j]:
                    pos_current[i, j] = self.boundary_matrix[0][j]
                elif pos_current[i, j] < self.boundary_matrix[1][j]:
                    pos_current[i, j] = self.boundary_matrix[1][j]
        pos_current = self.self_adaptive_mutation(pos_current)

        return pos_current

    # 自适应变异(防止陷入局部最值)
    def self_adaptive_mutation(self, pos):
        if random.random() > 0.8:
            ran_list = []
            while True:
                num = random.randint(0, self.input_dimension - 1)
                if num in ran_list:
                    continue
                if len(ran_list) == self.input_dimension / 2:  # 一半的值赋予随机值
                    break
                ran_list.append(num)
            for j in range(len(pos)):
                for i in ran_list:
                    pos[j][i] = round(random.uniform(self.pop_min, self.pop_max), 1)
        return pos

    # 判断当前粒子群是否存在满足所有目标条件的粒子，若有则返回其索引
    def _judge_function(self, fitness_array):
        current_error = abs(fitness_array) / self.target_value  # 相对误差
        judge_array = np.array(current_error < self.error)  # 布尔型数组，判断是否在容许误差内
        index = 0
        qualified_index = []
        for val in judge_array:
            if val.all() == True:  # 记录所有预测值都在误差范围内的粒子的索引,即合格的粒子
                qualified_index.append(index)
            index += 1
        if len(qualified_index):
            return qualified_index
        else:
            return None

    # pos_array大小为(size_pop, idimension)
    def fitness_value_function(self, pos_array):
        # fitness_model(pos_array)大小为(num_pop, odimension)
        print('the input of nn jiushi:')
        print(pos_array)
        print(pos_array.shape)
        predict_array = self.fitness_model.predict_value(pos_array)
        print('the predict val:')
        print(predict_array)
        # fitness_array相当于每个粒子odimension个目标函数与理想值之间的差值
        fitness_array = self.target_value - predict_array
        # fitness_array大小为(size_pop, 6),更新后直接拿去判断是否有合格粒子
        self.qualified_index = self._judge_function(fitness_array)
        print(fitness_array.shape)
        print('差值：', fitness_array)
        # fitness大小为(size_pop, )
        print('span:', self.span)
        fitness = np.sqrt(np.sum(((abs(fitness_array) / self.span) ** 2) * self.weight, axis=1))
        # 参照 distance = np.sqrt(np.sum(((abs(fitness_array) / (self.span)) ** 2) * self.weight, axis=1))
        self.predict_log.append(predict_array[fitness.argmin()])
        # 更新按照fitness来更新，判断是否结束迭代用judge_function来判断
        return fitness

    def update_best_value(self, fitness, pos_array, vel_array):
        for i in range(self.size_pop):
            if fitness[i] < self.single_best_fitness[i]:
                self.single_best_fitness[i] = fitness[i]
                self.single_best_pos[i] = pos_array[i]
                self.single_best_vel[i] = vel_array[i]
            if fitness[i] < self.global_best[0]:
                self.global_best[0] = fitness[i]
                self.global_best[1] = pos_array[i]
                # self.global_best[2] = vel_array[i]
        self.update_boundary.emit()  # 发射更新边界信号
        # 判断是否超过上下界
        print('single best pos:')
        print(self.single_best_pos)
        print('top matrix:')
        print(self.boundary_matrix[0])
        print('low matrix:')
        print(self.boundary_matrix[1])
        # upper_judge: 20个100维的粒子的矩阵高于boundary的上限的为true
        upper_judge = np.array(self.single_best_pos > self.boundary_matrix[0])
        print('upper judge:', upper_judge)
        # lower_judge：20个100维的粒子的矩阵低于boundary的下限的为true
        lower_judge = np.array(self.single_best_pos < self.boundary_matrix[1])
        print('lower judge:', lower_judge)
        reach_upper = []
        reach_lower = []
        num_of_upper = np.array([0*i for i in range(self.input_dimension)])
        num_of_lower = np.array([0*i for i in range(self.input_dimension)])
        for j in range(len(upper_judge[0])):  # j in rage 100
            for i in range(len(upper_judge)):  # i in rage size_pop
                if upper_judge[i, j]:
                    num_of_upper[j] += 1
        print('各维度打到顶部的离粒子数', num_of_upper)
        for i in range(len(num_of_upper)):  # i in rage 100
            if num_of_upper[i] > (self.size_pop // 5):  # 超过1/5的粒子的历史最佳位置的某一维度超出上界
                reach_upper.append(i)
        print(reach_upper)
        for j in range(len(lower_judge[0])):
            for i in range(len(lower_judge)):
                if lower_judge[i, j]:
                    num_of_lower[j] += 1
        print('各位度达到底部的离子束：', num_of_lower)
        for i in range(len(num_of_lower)):
            if num_of_lower[i] > (self.size_pop // 5):  # 超过1/5的粒子的历史最佳位置的某一维度低于下界
                reach_lower.append(i)

        print(reach_lower)

        self.transmit_to_up.emit(reach_upper)
        self.transmit_to_low.emit(reach_lower)
        self.fitness_log.append(self.global_best[0])

    def run(self):
        while True:
            if self.signal:
                self.update_result.emit('solving....please wait.')
                tar = copy.deepcopy(self.target_value)
                err = copy.deepcopy(self.error)
                self.target_value = np.array([self.target_value], dtype=float)  # 一个理想的目标值,大小为(size_pop, odimension),形参为元组
                self.error = np.array(self.error, dtype=float)  # 允许误差(停止迭代条件)，是一个odimension元素的向量，每个元素表示每个目标函数的容许误差
                for i in range(self.size_pop - 1):
                    self.target_value = np.concatenate((self.target_value, np.array([tar], dtype=float)))
                    self.error = np.concatenate((self.error, np.array(err, dtype=float)))
                self.error = self.error.reshape(self.size_pop, self.output_dimension)
                self.fitness_model = nn_model()
                # 初始化
                velocity = 2 * self.v_limit * (np.random.random((self.size_pop, self.input_dimension)) - 0.5)
                # 注意下面与常规PSO的区别
                position = np.random.random((self.size_pop, self.input_dimension))
                print('over here')

                # 求出最优值，将相似度最高的case接入
                self.global_best = [self.best_distance, self.init_global_best]
                # 将选出的case来初始化部分历史最优，pop_max + 2即将历史最优初始化为比初始边界更大的边界
                self.single_best_pos = np.concatenate((self.init_local_best,
                                                       (self.pop_max + 2) * (np.random.random((self.size_pop - len(self.init_local_best), self.input_dimension)))))
                self.single_best_vel = velocity
                # 历史最佳适应值
                print('global best:1')
                self.single_best_fitness = self.fitness_value_function(self.single_best_pos)
                print('global best:2')
                self.update_result.emit('global best:')
                self.update_result.emit(str(self.global_best))
                self.update_result.emit('\n\n')
                # print(self.global_best)
                self.update_result.emit('single best fitness:')
                # print(self.single_best_fitness)
                self.update_result.emit(str(self.single_best_fitness.tolist()))
                self.update_result.emit('\n\nsingle best pop:')
                # print(self.single_best_pos)
                self.update_result.emit(str(self.single_best_pos.tolist()))
                # 开始迭代
                for i in range(self.maxg):
                    # 进化
                    self.update_result.emit('\n\n*********************** iteration %d ***********************' % (i + 1))
                    velocity = self.update_vel(velocity, position, self.global_best[1], self.single_best_pos)
                    position = self.update_pos(position, velocity)

                    # self.update_result.emit('position:')
                    # self.update_result.emit(str(position.tolist()))
                    # self.update_result.emit('\n\n')
                    position = self.self_adaptive_mutation(position)  # 自适应变异
                    fitness = self.fitness_value_function(position)

                    # self.update_result.emit('fitness is :')
                    # self.update_result.emit(fitness)
                    self.update_best_value(fitness, position, velocity)
                    self.update_result.emit('\n\n')
                    self.update_result.emit('global best:')
                    #print(self.global_best)
                    self.update_result.emit(str(self.global_best))
                    self.update_result.emit('\n\n')
                    self.update_result.emit('single best fitness:')
                    self.update_result.emit(str(self.single_best_fitness))
                    self.update_result.emit('\n\n')
                    self.update_result.emit('single best pop:')
                    self.update_result.emit(str(self.single_best_pos))
                    self.update_result.emit('\n\n')

                    if self.qualified_index:
                        self.update_result.emit('qualified solution was found!\n')
                        self.update_result.emit('total iteration : %d ' % i)
                        self.update_result.emit('qualified solution:\n')
                        for index in self.qualified_index:
                            self.update_result.emit(str(position[index]))
                            self.update_result.emit('predict result:')
                            self.update_result.emit(str(self.fitness_model.predict_value([position[index]])))
                            self.update_result.emit('\n')
                        break
                    while self.suspend:
                        time.sleep(1)
                iteration = [i for i in range(self.maxg + 1)]
                plt.figure(1)
                plt.figure(figsize=(15, 5))
                plt.plot(iteration[:-1], self.fitness_log)
                plt.show()

                break
            time.sleep(1)


def ackley(array):
    result = np.zeros(shape=array.shape[0])
    for i in range(len(array)):
        result[i] = -20 * exp(-0.2 * sqrt((array[i][0] ** 2 + array[i][1]**2) / 2)) - \
                    exp((cos(2 * pi * array[i][0]) + cos(2 * pi * array[i][1])) / 2) + 20 + 2.71289
    # 每一个result[i]都应该是一个向量，哪怕只有一个值，而不是标量
    result = result[:, np.newaxis]
    return result

'''
def nn_model(x_array):
    g1 = tf.Graph()  # 调用CNN网络
    sess = tf.Session(graph=g1)
    with sess.as_default():
        with sess.graph.as_default():
            saver = tf.train.import_meta_graph('.\suprised_model.ckpt-13881.meta')  # 载入图结构，保存在.meta文件中？
            saver.restore(sess, ".\suprised_model.ckpt-13881")  # 恢复模型？
            graph = tf.get_default_graph()
            input_shape = graph.get_operation_by_name('x_place').outputs[0]
            output_shape = graph.get_tensor_by_name("logits_eval:0")
            result = sess.run(output_shape, feed_dict={input_shape: x_array})
            # result = result[:, np.newaxis]
            result = result * 1e4  # 别忘了反归一化！！
            return result
'''

'''
if __name__ == '__main__':
    
    array = [1.8, 1.8, 1.6, 1.5, 1.6, 2.0, 1.1, 1.9, 1.4, 1.9, 1.0, 1.5, 2.6, 1.8, 1.5, 1.7, 0.6, 0.9, 2.1, 2.8, 2.9,
             1.4, 0.9, 2.1, 3.2, 2.0, 2.9, 1.4, 2.0, 3.2, 3.8, 0.3, 1.6, 2.3, 2.4, 4.4, 1.2, 3.0, 2.1, 1.2, 2.3, 1.4,
             0.4, 0.5, 3.3, 2.8, 2.0, 0.4, 2.5, 2.2, 2.5, 3.2, 1.4, 2.1, 3.4, 2.1, 1.7, 2.1, 3.0, 4.0, 2.7, 3.0, 6.0,
             1.4, 0.9, 1.7, 2.7, 3.2, 3.2, 2.2, 2.4, 2.1, 3.5, 2.5, 2.8, 2.6, 1.9, 1.3, 2.7, 2.5, 2.5, 3.6, 3.1, 2.2,
             2.7, 2.5, 2.6, 3.9, 2.8, 2.4, 2.8, 3.5, 1.2, 1.9, 4.0, 2.3, 2.6, 3.5, 2.4, 3.4]
    array = np.array(array).reshape(1, 100)
    print(nn_model(array))
    exit()
    
    model = PSO_model(size_pop=20, v_limit=1, pop_min=-3, pop_max=3, idimension=2, odimension=1, fitness_fun=ackley,
                      c1=2, c2=2, maxg=500, target_value=(0.01,), error=(0.01,))
    # print(ackley(np.array([[0, 0]])))
    # exit()
    model.run()
    
    in_tensor = model.global_best[1].reshape(1, 100)
    result = nn_model(in_tensor)
    print(result)
    mode_11 = []
    mode_22 = []
    mode_33 = []
    mode_44 = []
    tor = []
    ben = []
    fit = []
    plt.figure(1)
    plt.figure(figsize=(15, 5))
    for i in range(len(model.predict_log)):
        mode_11.append(model.predict_log[i][0])
        mode_22.append(model.predict_log[i][1])
        mode_33.append(model.predict_log[i][2])
        mode_44.append(model.predict_log[i][3])
        tor.append(model.predict_log[i][4])
        ben.append(model.predict_log[i][5])
    # for i in range(len(model.fitness_log)):
    #     fit.append(model.fitness_log)
    iteration = [i for i in range(model.maxg + 1)]
    # plt.legend([mode_11, mode_22, mode_33, mode_44, tor, ben], ['mode_1', 'mode_2', 'mode_3', 'mode_4', 'torsion', 'bend'])
    plt.figure(1)
    plt.figure(figsize=(15, 5))
    plt.plot(iteration[:-1], model.fitness_log)

    plt.figure(2)
    plt.figure(figsize=(15, 5))
    plt.plot(iteration, mode_11)

    plt.figure(3)
    plt.figure(figsize=(15, 5))
    plt.plot(iteration, mode_22)

    plt.figure(4)
    plt.figure(figsize=(15, 5))
    plt.plot(iteration, mode_33)

    plt.figure(5)
    plt.figure(figsize=(15, 5))
    plt.plot(iteration, mode_44)

    plt.figure(6)
    plt.figure(figsize=(15, 5))
    plt.plot(iteration, tor)

    plt.figure(3)
    plt.figure(figsize=(15, 5))
    plt.plot(iteration, ben)

    plt.show()
    '''

'''有个缺点，就是目标值不能是0'''
