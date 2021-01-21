# -*- coding: utf-8 -*-

from PyQt5.QtCore import QObject, pyqtSignal, QThread
import time


# 线程2
class time_e(QThread):
    update_time = pyqtSignal(str)
    signal = False
    stop_signal = False
    suspend = False
    time_continue = 0
    total_suspend = 0

    def run(self):
        while True:
            if self.signal:
                time_start = time.time()
                while not self.stop_signal:
                    time_elapse = time.time() - time_start - self.total_suspend
                    m, s = divmod(time_elapse, 60)
                    h, m = divmod(m, 60)
                    time_elapse = str(int(h)) + 'h(s) ' + str(int(m)) + 'm(s) ' + str(int(s)) + 's'
                    self.update_time.emit(time_elapse)
                    time.sleep(1)  # 设置更新间隔
                    time_suspend = time.time()  # 记录暂停时间点
                    while self.suspend:
                        time.sleep(1)
                        self.time_continue = time.time() - time_suspend  # 记录暂停时间
                    self.total_suspend += self.time_continue
                    self.time_continue = 0
                while self.stop_signal:
                    time.sleep(1)
                self.signal = False

            time.sleep(1)  # 设置检查间隔

    def handle_stop_signal(self):
        self.stop_signal = True

