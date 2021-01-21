from PyQt5 import QtWidgets

from PyQt5.QtCore import pyqtSignal

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure

import numpy as np

from array import array

import time

import random

import threading

from datetime import datetime

from matplotlib.dates import date2num, MinuteLocator, SecondLocator, DateFormatter


X_MINUTES = 1

Y_MAX = 1

Y_MIN = 0

INTERVAL = 1

MAXCOUNTER = int(X_MINUTES * 60 / INTERVAL)


class MplCanvas(FigureCanvas):

    def __init__(self):

        self.fig = Figure()

        self.ax = self.fig.add_subplot(111, facecolor='#000000')  # facecolor设置图表背景颜色(or #FFDAB9)

        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        FigureCanvas.updateGeometry(self)

        # self.ax.set_xlabel("time of data generator")

        self.ax.grid(True)  # 显示网格

        self.ax.set_ylabel('Accuracy')

        self.ax.legend()

        self.ax.set_ylim(Y_MIN, Y_MAX)

        self.ax.xaxis.set_major_locator(MinuteLocator())  # every minute is a major locator

        self.ax.xaxis.set_minor_locator(SecondLocator([10, 20, 30, 40, 50]))  # every 10 second is a minor locator

        self.ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))  # tick label formatter

        self.curveObj = None  # draw object

    def plot(self, datax, datay):

        if self.curveObj is None:

            # create draw object once

            self.curveObj, = self.ax.plot_date(np.array(datax), np.array(datay), 'g-')

        else:

            # update data of draw object

            self.curveObj.set_data(np.array(datax), np.array(datay))

            # update limit of X axis,to make sure it can move

            self.ax.set_xlim(datax[0], datax[-1])

        ticklabels = self.ax.xaxis.get_ticklabels()

        for tick in ticklabels:
            tick.set_rotation(25)

        self.draw()


class MplCanvasWrapper(QtWidgets.QWidget):

    def __init__(self, parent=None):

        QtWidgets.QWidget.__init__(self, parent)

        self.canvas = MplCanvas()

        self.vbl = QtWidgets.QVBoxLayout()

        self.ntb = NavigationToolbar(self.canvas, parent)

        self.vbl.addWidget(self.ntb)

        self.vbl.addWidget(self.canvas)

        self.setLayout(self.vbl)

        self.dataX = []

        self.dataY = []

        self.initDataGenerator()

        self.acc = 0.0

    def startPlot(self):

        self.__generating = True

    def pausePlot(self):

        self.__generating = False

        pass

    def initDataGenerator(self):

        self.__generating = False

        self.__exit = False

        self.tData = threading.Thread(name="dataGenerator", target=self.generateData)

        self.tData.start()

    def releasePlot(self):

        self.__exit = True

        self.tData.join()

    def generateData(self):

        counter = 0

        while (True):

            if self.__exit:
                break

            if self.__generating:

                # newData = random.randint(Y_MIN, Y_MAX)
                newData = self.acc

                newTime = date2num(datetime.now())

                self.dataX.append(newTime)

                self.dataY.append(newData)  # 将新的数据增加至x,y列表中

                self.canvas.plot(self.dataX, self.dataY)  # 将新的datax, datay画入图中

                if counter >= MAXCOUNTER:  # 当x,y列表存储的数据量大于max_counter时，把旧的数据弹出

                    self.dataX.pop(0)

                    self.dataY.pop(0)

                else:

                    counter += 1

            time.sleep(INTERVAL)