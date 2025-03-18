import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from Thread.Thread_gui import *
from Thread.thread import New_Thread


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # 绑定按钮点击事件
        self.Button_Start.clicked.connect(self.Start)
        self.Button_Stop.clicked.connect(self.Stop)

    def Stop(self):
        print('End')
        self.thread.terminate()  # 终止线程

    def Start(self):
        print('Start clicked.')
        self.thread = New_Thread(t=100)  # 实例化一个线程，参数t设置为100
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.Change)
        # 启动线程，执行线程类中run函数
        self.thread.start()

    # 接受通过emit传来的信息，执行相应操作
    def Change(self, msg):
        print(msg)
        self.label.setText(str(msg))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
