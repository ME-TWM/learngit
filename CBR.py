# -*- coding: utf-8 -*-

from PyQt5.QtCore import QObject, pyqtSignal, QThread
import numpy as np
import time


class CBR(QThread):
    transmit_data = pyqtSignal(list)
    best_case = pyqtSignal(list)
    transmit_span = pyqtSignal(list)
    best_distance = pyqtSignal(float)
    signal = False

    target_value = []
    weight = []
    threshold_sim = None
    input_data = None
    output_data = None

    def run(self):
        while True:
            if self.signal:
                # data_set = np.zeros(shape=[4000, 28], dtype=np.float32)  # 数据类型默认为float64
                data_set = []
                line_num = 0  # 行号
                with open(self.input_data, 'r') as file:
                    # csvReader = csv.reader(file)
                    for line in file:
                        line_num += 1
                        if line_num == 1:
                            continue
                        # 以，作为分隔符并转为float格式，将行append进data_set
                        data_set.append([float(x) for x in line.split(',')])
                line_num = 0
                with open(self.output_data, 'r') as file:
                    for line in file:
                        line_num += 1
                        if line_num == 1:
                            continue
                        result_data = [float(x) for x in line.split(',')]
                        data_set[int(result_data[0])] = data_set[int(result_data[0])] + result_data[1:]
                new = []
                for item in data_set:
                    if len(item) == 106:
                        item = item[-6:] + item[:-6]  # 把result放在前面(不要用extend)
                        new.append(item)
                new = np.array(new)
                # np.random.shuffle(new)
                print('check point:\nshape of data:', new.shape)
                result = new[:, :6]
                print(result.shape)
                max_val = result.max(axis=0)
                min_val = result.min(axis=0)
                print('max val:', max_val)
                print('min_val:', min_val)
                # normali = (result - min_val) / (max_val - min_val)
                span = [max_val - min_val]
                self.transmit_span.emit(span)
                target_value = np.array(list(map(float, self.target_value)))
                print('target: ', target_value)
                weight = np.array(list(map(float, self.weight)))
                threshold_sim = float(self.threshold_sim)
                # qualify_list = []
                distance = np.sqrt(np.sum(((abs(target_value - result) / (span[0])) ** 2) * weight, axis=1))
                similarity = 1 - distance
                self.best_distance.emit(distance.min())
                print('shape of sim:', similarity)
                print('sim最大和最小', similarity.max(), similarity.min())
                # print('距离最大和最小的索引', distance.argmax(), distance.argmin())
                # print(result[distance.argmax()])
                best_solution = []
                best_solution.append(new[similarity.argmax()][6:])
                print(best_solution)
                # 发送最佳案例
                self.best_case.emit(best_solution)
                qualified_case = []
                for i in range(len(new)):
                    if similarity[i] >= threshold_sim:
                        qualified_case.append(new[i])
                self.transmit_data.emit(qualified_case)

                print('the nearest case:', result[distance.argmin()])
                print('case solution:', new[distance.argmin()][0:100])
                self.signal = False

            time.sleep(1)  # 设置检查间隔


