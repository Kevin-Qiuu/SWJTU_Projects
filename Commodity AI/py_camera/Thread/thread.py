# -*- coding: utf-8 -*-
import time
from PyQt5.QtCore import QThread, pyqtSignal

#定义一个线程类
class New_Thread(QThread):
    #自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    finishSignal = pyqtSignal(str)

    # 带一个参数t
    def __init__(self, t,parent=None):
        super(New_Thread, self).__init__(parent)

        self.t = t
    #run函数是子线程中的操作，线程启动后开始执行
    def run(self):
        for i in range(self.t):
            time.sleep(1)
            #发射自定义信号
            #通过emit函数将参数i传递给主线程，触发自定义信号
            self.finishSignal.emit('this')  # 注意这里与_signal = pyqtSignal(str)中的类型相同
