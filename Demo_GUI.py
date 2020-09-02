# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QFileDialog, QTableWidgetItem, QHeaderView
from PyQt5 import QtCore
from PyQt5.QtGui import QColor
from DemoGUI import Ui_MainWindow
from PSO_2 import *
from neural_network import *
from time_module import *
# from PyQt5.QtCore import *  注意看看把这个注释掉后有没有影响
from CBR import *
import sys


class my_gui(Ui_MainWindow, QMainWindow, QThread):  # 注意这里继承的是QMainWindow而不是QWidget，因为创建的是MainWindow，不是Widget或者dialog
    send_to_pso = pyqtSignal(str)
    qualified_case = None       # 多个线程共用的变量放在这里
    best_qualified_case = None
    selected_case = []
    target = None
    weight = None
    span = None
    best_distance = None
    boundary_table_row = None
    boundary_table_col = None

    def __init__(self):
        super(my_gui, self).__init__()
        self.setupUi(self)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '确认', '确认退出吗？', QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def run_cnn(self):
        if self.lineEdit.text():
            nn.input_dir = self.lineEdit.text()
        else:
            QMessageBox.about(self, '通知', '请输入正确的输入样本路径！')
            self.lineEdit.clear()
            self.lineEdit.setFocus()
        if self.lineEdit_17.text():
            nn.output_dir = self.lineEdit_17.text()
        else:
            QMessageBox.about(self, '通知', '请输入正确的输出样本路径！')
            self.lineEdit_17.clear()
            self.lineEdit_17.setFocus()
        if self.lineEdit_18.text():
            nn.input_dimension = self.lineEdit_18.text()

        if self.lineEdit_19.text():
            nn.network_struct = self.lineEdit_19.text()

        if self.lineEdit_20.text():
            nn.learning_rate = self.lineEdit_20.text()

        if self.lineEdit_21.text():
            nn.num_steps = self.lineEdit_21.text()

        if self.lineEdit_22.text():
            nn.batch_size = self.lineEdit_22.text()

        if self.lineEdit_23.text():
            nn.target_acc = self.lineEdit_23.text()

        nn.start_nn = True
        self.mplCanvas.startPlot()
        pass


    def start_time(self):
        if self.lineEdit.text() and self.lineEdit_17.text():
            t.signal = True

    def open_file(self):
        input_dir, _ = QFileDialog.getOpenFileName(self,
                                                    "选取文件",
                                                    "./",  # 默认打开的文件夹
                                                    "All Files (*);;Text Files (*.csv)")  # 设置文件扩展名过滤,注意用双分号间隔
        self.lineEdit.clear()
        self.lineEdit.setText(input_dir)  # 给lineEdit赋值

    def open_file_2(self):
        output_dir, _ = QFileDialog.getOpenFileName(self,
                                                   "选取文件",
                                                   "./",  # 默认打开的文件夹
                                                   "All Files (*);;Text Files (*.csv)")  # 设置文件扩展名过滤,注意用双分号间隔
        self.lineEdit_17.clear()
        self.lineEdit_17.setText(output_dir)  # 给lineEdit赋值

    def handle_display(self, data):
        if type(data) == dict:
            # self.textEdit.append(data['train_msg'])
            # self.lineEdit_3.setText(data['cur_acc'])
            # self.lineEdit_2.setText(data['top_acc'])
            self.lineEdit_24.setText(data['loss'])
        else:
            # self.textEdit.append(data)
            pass

    def handle_display_acc(self, data):
        self.lineEdit_3.setText(data)

    def handle_display_top_acc(self, data):
        self.lineEdit_2.setText(data)

    def handle_display_time(self, data):
        self.lineEdit_4.setText(data)

    def handle_display_plot(self, data):
        self.mplCanvas.acc = data

    def shut_down_plot(self):
        nn.shut_down = True
        # self.releasePlot()
        self.mplCanvas.pausePlot()
        pass

    def pause_plot(self):
        self.mplCanvas.pausePlot()
        pass

    def go_on_nn(self):
        nn.suspend = False
        t.suspend = False
        self.mplCanvas.startPlot()
        pass

    def suspend_nn(self):
        nn.suspend = True
        t.suspend = True

    def auto_fill(self):
        self.lineEdit_18.setText('100')
        self.lineEdit_19.setText('100x50x20')
        self.lineEdit_20.setText('0.001')
        self.lineEdit_21.setText('40000')
        self.lineEdit_22.setText('100')
        self.lineEdit_23.setText('0.99')

    def reset(self):
        pass

    # 不把best_qualified_case的设置放在show_table里，因为qualified_case的设置已经用了一次show_table，使用两次的话可能会两次重复画表格
    def set_init_global_best(self, data):
        self.best_qualified_case = data

    def set_span(self, data):
        self.span = data[0]

    def set_best_distance(self, data):
        self.best_distance = data

    def show_table(self, data):
        self.qualified_case = data
        self.tableWidget.clear()
        row = len(data)
        col = int(self.lineEdit_18.text()) + 6
        self.tableWidget.setRowCount(row)  # 设置行数
        self.tableWidget.setColumnCount(col + 1)  # 设置列数
        self.y_header_1 = ['使用', '一阶扭转', '尾门框扭转', '一阶弯曲', '前舱横摆', '扭转刚度', '弯曲刚度']
        self.y_header_2 = ['con_'+str(i+1) for i in range(col - 6)]
        self.y_header = self.y_header_1 + self.y_header_2
        self.tableWidget.setHorizontalHeaderLabels(self.y_header)  # 设置表头内容
        self.tableWidget.verticalHeader().setVisible(True)  # 显示垂直表头
        self.tableWidget.horizontalHeader().setVisible(True)  # 显示水平表头
        self.tableWidget.setColumnWidth(0, 30)  # 设置第0列宽度30
        # 添加复选框
        for k in range(row):
            item_checked = QTableWidgetItem()  # 复选框
            item_checked.setCheckState(QtCore.Qt.Unchecked)  # 如果使用这句就表示复选框是未选的
            self.tableWidget.setItem(k, 0, item_checked)
        # 添加数据
        i = 0  # 第几行（从0开始）
        j = 1  # 第几列（从0开始）
        for case in data:
            for item in case:
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(item)))  # 设置j行i列的内容为item
                j += 1
            i += 1
            j = 1
        # self.tableWidget.setColumnWidth(j, 80)  # 设置j列的宽度(要设置所有的宽度的话只能用一个for循环来设置所有列宽)
        # self.tableWidget.setRowHeight(i, 50)  # 设置i行的高度

        # 根据widget的大小自适应调整表格（我加的， 注意import QHeaderView）
        # self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def retrieve(self):
        '''
        self.lineEdit_5.setText('48')
        self.lineEdit_6.setText('54')
        self.lineEdit_7.setText('84')
        self.lineEdit_8.setText('87')
        self.lineEdit_9.setText('3471')
        self.lineEdit_10.setText('-42852')
        '''
        self.lineEdit_5.setText('55')
        self.lineEdit_6.setText('60')
        self.lineEdit_7.setText('89')
        self.lineEdit_8.setText('95')
        self.lineEdit_9.setText('3800')
        self.lineEdit_10.setText('-50000')
        self.lineEdit_26.setText('0.2')
        self.lineEdit_27.setText('0.2')
        self.lineEdit_28.setText('0.10')
        self.lineEdit_29.setText('0.1')
        self.lineEdit_30.setText('0.2')
        self.lineEdit_31.setText('0.2')
        self.target = [float(self.lineEdit_5.text()), float(self.lineEdit_6.text()), float(self.lineEdit_7.text()),
                       float(self.lineEdit_8.text()), float(self.lineEdit_9.text()), float(self.lineEdit_10.text())]
        self.weight = [float(self.lineEdit_26.text()), float(self.lineEdit_27.text()), float(self.lineEdit_28.text()),
                       float(self.lineEdit_29.text()), float(self.lineEdit_30.text()), float(self.lineEdit_31.text())]
        # 下面的cbr_system也可以替换为CBR，但如果替换的话修改的就是CBR这个类的参数，而不是cbr_system这个实例的参数！！存在风险
        cbr_system.target_value.extend(self.target)
        cbr_system.weight.extend(self.weight)
        cbr_system.threshold_sim = self.doubleSpinBox.value()
        cbr_system.input_data = self.lineEdit.text()
        cbr_system.output_data = self.lineEdit_17.text()
        cbr_system.signal = True

    def run_pso(self):
        print('zhekeba')
        if self.lineEdit_11.text() and self.lineEdit_12.text() and self.lineEdit_13.text() and self.lineEdit_14.text() \
                and self.lineEdit_15.text() and self.lineEdit_16.text():
            pso.target_value = self.target
            error = [float(self.lineEdit_11.text()), float(self.lineEdit_12.text()), float(self.lineEdit_13.text()),
                     float(self.lineEdit_14.text()), float(self.lineEdit_15.text()), float(self.lineEdit_16.text())]
            pso.error = error
            pso.size_pop = int(self.lineEdit_25.text())
            pso.pop_max = self.doubleSpinBox_2.value()
            pso.pop_min = self.doubleSpinBox_3.value()
            pso.input_dimension = int(self.lineEdit_18.text())
            # 注意init_global_best, init_local_best只有变量！！没有6个结果
            print('any body home?')
            pso.init_global_best = self.best_qualified_case[0]
            print('nibaba', pso.init_global_best)
            for i in range(len(self.qualified_case)):
                if self.tableWidget.item(i, 0).checkState():
                    print('halo')
                    self.selected_case.append(self.qualified_case[i])
            print('zaici', self.selected_case)
            pso.init_local_best = np.array(self.selected_case)[:, 6:]
            print('nimama', pso.init_local_best)
            pso.weight = self.weight
            pso.span = self.span
            pso.best_distance = self.best_distance
            print('nimab')

            # 显示修正边界表格
            self.tableWidget_2.clear()
            self.boundary_table_row = 2
            self.boundary_table_col = int(self.lineEdit_18.text())
            self.tableWidget_2.setRowCount(self.boundary_table_row)  # 设置行数
            self.tableWidget_2.setColumnCount(self.boundary_table_col)  # 设置列数
            x_header = ['上边界', '下边界']
            self.tableWidget_2.setHorizontalHeaderLabels(self.y_header_2)  # 设置表头内容
            self.tableWidget_2.setVerticalHeaderLabels(x_header)
            self.tableWidget_2.verticalHeader().setVisible(True)  # 显示垂直表头
            self.tableWidget_2.horizontalHeader().setVisible(True)  # 显示水平表头
            # 添加数据
            for i in range(self.boundary_table_row):
                for j in range(self.boundary_table_col):
                    if i == 0:
                        self.tableWidget_2.setItem(i, j, QTableWidgetItem(str(self.doubleSpinBox_2.value())))
                    else:
                        self.tableWidget_2.setItem(i, j, QTableWidgetItem(str(self.doubleSpinBox_3.value())))
            # 根据widget的大小自适应调整表格（我加的， 注意import QHeaderView）
            self.tableWidget_2.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
            pso.boundary_matrix = np.zeros((self.boundary_table_row, self.boundary_table_col))
            for i in range(self.boundary_table_row):
                for j in range(self.boundary_table_col):
                    pso.boundary_matrix[i, j] = float(self.tableWidget_2.item(i, j).text())

            pso.signal = True
        else:
            QMessageBox.about(self, '通知', '请输入正确的目标值和允许误差！')
        '''
        con = self.textEdit_2.toPlainText()
        # print(con)
        pso.condition = con
        pso.signal = True
        '''

    def update_up(self, data):
        up_data = ['con_'+str(i+1) for i in data]
        self.textEdit.setText(str(up_data))

    def update_low(self, data):
        low_data = ['con_' + str(i + 1) for i in data]
        self.textEdit_2.setText(str(low_data))

    def rectified_boundary(self):
        pso.boundary_matrix = np.zeros((self.boundary_table_row, self.boundary_table_col))
        for i in range(self.boundary_table_row):
            for j in range(self.boundary_table_col):
                pso.boundary_matrix[i, j] = float(self.tableWidget_2.item(i, j).text())

    def handle_display_pso(self, data):
        self.textEdit_3.append(data)

    def suspend_pso(self):
        pso.suspend = True

    def go_on_pso(self):
        pso.suspend = False

    def reset_nn(self):
        self.lineEdit_2.clear()
        self.lineEdit_3.clear()
        self.lineEdit_4.clear()
        self.lineEdit_24.clear()
        self.lineEdit_18.clear()
        self.lineEdit_19.clear()
        self.lineEdit_20.clear()
        self.lineEdit_21.clear()
        self.lineEdit_22.clear()
        self.lineEdit_23.clear()
        nn.start_nn = False
        t.stop_signal = False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = my_gui()
    window.pushButton_2.clicked.connect(window.run_cnn)  # 注意函数后面不要加括号
    window.pushButton_2.clicked.connect(window.start_time)
    window.pushButton.clicked.connect(window.open_file)  # 函数执行完后才打印出QT界面
    window.pushButton_4.clicked.connect(window.open_file_2)
    window.pushButton_3.clicked.connect(window.run_pso)
    window.pushButton_5.clicked.connect(window.auto_fill)
    window.pushButton_7.clicked.connect(window.shut_down_plot)
    window.pushButton_8.clicked.connect(window.go_on_nn)
    window.pushButton_10.clicked.connect(window.retrieve)
    window.pushButton_11.clicked.connect(window.suspend_pso)
    window.pushButton_12.clicked.connect(window.go_on_pso)
    window.pushButton_9.clicked.connect(window.reset_nn)
    window.pushButton_13.clicked.connect(window.suspend_nn)
    window.pushButton_13.clicked.connect(window.pause_plot)

    t = time_e()  # 相当于初始化__init__函数
    t.update_time.connect(window.handle_display_time)
    t.start()  # 相当于执行run()函数

    nn = neural_network()
    nn.update_msg.connect(window.handle_display)
    nn.update_info.connect(window.handle_display)
    nn.update_acc.connect(window.handle_display_acc)
    nn.update_top_acc.connect(window.handle_display_top_acc)
    nn.stop_time.connect(t.handle_stop_signal)
    nn.update_plot_acc.connect(window.handle_display_plot)
    nn.start()  # 单独在这里运行线程，不能放在主进程里，否则会被视为主进程的一部分（？）

    cbr_system = CBR()
    cbr_system.transmit_data.connect(window.show_table)
    cbr_system.best_case.connect(window.set_init_global_best)
    cbr_system.transmit_span.connect(window.set_span)
    cbr_system.best_distance.connect(window.set_best_distance)
    cbr_system.start()

    pso = PSO_model()
    pso.update_result.connect(window.handle_display_pso)
    pso.update_boundary.connect(window.rectified_boundary)
    pso.transmit_to_up.connect(window.update_up)
    pso.transmit_to_low.connect(window.update_low)
    pso.start()

    window.show()

    sys.exit(app.exec_())

'''
将新生成的py文件开头修改如下：
from PyQt5 import QtCore, QtGui, QtWidgets

from mplCanvasWrapper import MplCanvasWrapper
# import background_rc

try:

    _fromUtf8 = QtCore.QString.fromUtf8

except AttributeError:

    def _fromUtf8(s):

        return s

try:

    _encoding = QtWidgets.QApplication.UnicodeUTF8


    def _translate(context, text, disambig):

        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)

except AttributeError:

    def _translate(context, text, disambig):

        return QtWidgets.QApplication.translate(context, text, disambig)


class Ui_MainWindow(QtWidgets.QMainWindow):
'''