# Ui_Mainwindow.py
from PyQt5 import QtCore, QtGui, QtWidgets

# main.py
# 使用Device manager进行多相机管理

# 导入程序运行必须模块
import platform
import sys
# PyQt5中使用的基本控件都在PyQt5.QtWidgets模块中
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
# GraphicsView相关模块
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
# PyQt5核心模块
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from cv2 import imread
# 导入designer工具生成的Ui_MainWindow模块
from Ui_Multicamera_photograph import Ui_MainWindow
# 格式转换模块
from PyQt5.QtGui import QImage, QPixmap
# Pyrealsense相关模块
import pyrealsense2 as rs
import cv2
# from skimage import io
from realsense_device_manager import DeviceManager  # 多相机管理
# 其他重要模块
import os
import time
import datetime as dt
import numpy as np
import threading as th
# raise exception相关模块
import ctypes
import inspect



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1338, 625)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Btn_takePhotos = QtWidgets.QPushButton(self.centralwidget)
        self.Btn_takePhotos.setGeometry(QtCore.QRect(30, 540, 75, 31))
        self.Btn_takePhotos.setObjectName("Btn_takePhotos")
        self.label_top = QtWidgets.QLabel(self.centralwidget)
        self.label_top.setGeometry(QtCore.QRect(40, 10, 701, 16))
        self.label_top.setObjectName("label_top")
        self.label_PhotoLeft = QtWidgets.QLabel(self.centralwidget)
        self.label_PhotoLeft.setGeometry(QtCore.QRect(40, 40, 640, 480))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_PhotoLeft.sizePolicy().hasHeightForWidth())
        self.label_PhotoLeft.setSizePolicy(sizePolicy)
        self.label_PhotoLeft.setObjectName("label_PhotoLeft")
        self.label_PhotoRight = QtWidgets.QLabel(self.centralwidget)
        self.label_PhotoRight.setGeometry(QtCore.QRect(680, 40, 640, 480))
        self.label_PhotoRight.setObjectName("label_PhotoRight")
        self.label_Directory = QtWidgets.QLabel(self.centralwidget)
        self.label_Directory.setGeometry(QtCore.QRect(230, 540, 771, 31))
        self.label_Directory.setObjectName("label_Directory")
        self.Btn_chooseDirectory = QtWidgets.QPushButton(self.centralwidget)
        self.Btn_chooseDirectory.setGeometry(QtCore.QRect(110, 540, 98, 31))
        self.Btn_chooseDirectory.setObjectName("Btn_chooseDirectory")
        self.label_PhotoLeft.raise_()
        self.Btn_takePhotos.raise_()
        self.label_top.raise_()
        self.label_PhotoRight.raise_()
        self.label_Directory.raise_()
        self.Btn_chooseDirectory.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1338, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Btn_takePhotos.setText(_translate("MainWindow", "拍照"))
        self.label_top.setText(_translate("MainWindow", "以640*480分辨率显示图片"))
        self.label_PhotoLeft.setText(_translate("MainWindow", "TextLabel"))
        self.label_PhotoRight.setText(_translate("MainWindow", "TextLabel"))
        self.label_Directory.setText(_translate("MainWindow", "当前无路径"))
        self.Btn_chooseDirectory.setText(_translate("MainWindow", "选择储存路径"))


class MyMainForm(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        # 创建父对象
        super(MyMainForm, self).__init__(parent)
        # 设置ui
        self.setupUi(self)
        # 初始化控件
        self.label_Directory.setText('默认路径为当前文件夹')
        # 初始化添加信号和槽
        # 当dis_update作为信号相应的时候触发camera_view
        self.dis_update_l.connect(self.camera_view_l)
        self.dis_update_r.connect(self.camera_view_r)
        self.Btn_takePhotos.clicked.connect(self.Btn_takePhotos_clicked)
        self.Btn_chooseDirectory.clicked.connect(self.choose_saveDirectory)
        self.thread_camera = None
        self.takePhotos = False
        self.saveDirectory = "./"

    # 在对应的页面类的内部，与def定义的函数同级
    dis_update_l = pyqtSignal(QPixmap)
    dis_update_r = pyqtSignal(QPixmap)

    def Btn_takePhotos_clicked(self):  # 控制是否进行拍照
        self.takePhotos = True

    def choose_saveDirectory(self):
        filepath = QFileDialog.getExistingDirectory(self, "选择文件保存路径", "/")
        self.saveDirectory = filepath
        self.label_Directory.setText(self.saveDirectory)

    # 添加一个退出的提示事件
    def closeEvent(self, event):
        """我们创建了一个消息框，上面有俩按钮：Yes和No.第一个字符串显示在消息框的标题栏，第二个字符串显示在对话框，
              第三个参数是消息框的俩按钮，最后一个参数是默认按钮，这个按钮是默认选中的。返回值在变量reply里。"""

        reply = QMessageBox.question(self, 'Message', "Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # 判断返回值，如果点击的是Yes按钮，我们就关闭组件和应用，否则就忽略关闭事件
        if reply == QMessageBox.Yes:
            self.stop_thread(self.thread_camera)
            event.accept()
        else:
            event.ignore()

    def open_camera(self):
        # target选择开启摄像头的函数
        # self.thread_camera = th.Thread(target=self.multiCamera_photograph_npImage)
        self.thread_camera = th.Thread(
            target=self.multiCamera_photograph_npImage)
        self.thread_camera.start()
        print('Open Camera')

    def camera_view_l(self, Pixmap):  # 左图像
        # 调用setPixmap函数设置显示Pixmap
        self.label_PhotoLeft.setPixmap(Pixmap)
        # 调用setScaledContents将图像比例化显示在QLabel上
        self.label_PhotoLeft.setScaledContents(True)

    def camera_view_r(self, Pixmap):  # 右图像
        # 调用setPixmap函数设置显示Pixmap
        self.label_PhotoRight.setPixmap(Pixmap)
        # 调用setScaledContents将图像比例化显示在QLabel上
        self.label_PhotoRight.setScaledContents(True)

    def _async_raise(self, tid, exctype):
        """raises the exception, performs cleanup if needed"""
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def stop_thread(self, thread):
        self._async_raise(thread.ident, SystemExit)

    def multiCamera_photograph_npImage(self):
        resolution_width = 640  # pixels #不用1280而用640是为了与深度图像保持分辨率一致
        resolution_height = 480  # pixels
        frame_rate = 30  # fps

        # frames the auto-exposure controller to stablise
        dispose_frames_for_stablisation = 30
        try:
            # Enable the streams from all the intel realsense devices
            rs_config = rs.config()
            rs_config.enable_stream(rs.stream.depth, resolution_width,
                                    resolution_height, rs.format.z16, frame_rate)
            rs_config.enable_stream(rs.stream.infrared, 1, resolution_width,
                                    resolution_height, rs.format.y8, frame_rate)
            rs_config.enable_stream(rs.stream.color, resolution_width,
                                    resolution_height, rs.format.bgr8, frame_rate)

            # Use the device manager class to enable the devices and get the frames
            device_manager = DeviceManager(rs.context(), rs_config)
            device_manager.enable_all_devices()

            # Allow some frames for the auto-exposure controller to stablise
            for frame in range(dispose_frames_for_stablisation):
                frames = device_manager.poll_frames()

            assert (len(device_manager._available_devices) > 0)

            # Continue acquisition until terminated with Ctrl+C by the user
            images_color = []  # 存放多个相机的图像
            images_depth = []  # 存放多个相机的图像
            serials = []  # 存放多个相机的编号
            numOfPhotos = 0  # 拍摄组编号
            while 1:
                # _enabled_devices[device_serial] = (Device(pipeline, pipeline_profile, product_line))
                for (serial, device) in device_manager._enabled_devices.items():
                    # 此处可以通过device.pipeline达到与pipeline = rs.pipeline()
                    # serial: 相机编号
                    # device: Device(pipeline, pipeline_profile, product_line) #见realSense_device_manager
                    # TODO：对每个相机流进行操作
                    streams = device.pipeline_profile.get_streams()
                    # frameset will be a pyrealsense2.composite_frame object
                    frames = device.pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_frame = frames.get_depth_frame()
                    depth_image = np.asanyarray(depth_frame.get_data())
                    # 存储多图片
                    images_color.append(color_image)
                    images_depth.append(depth_image)
                    serials.append(serial)
                    ##################################
                # Save the results
                if (self.takePhotos == True):
                    now_date = dt.datetime.now().strftime('%F')
                    now_time = dt.datetime.now().strftime('%F_%H%M%S')

                    # 新建一个以当天日期为名字的文件夹
                    # path_ok = os.path.exists(now_date) # os.path.exists默认检测当前文件夹
                    nowDateSaveDirectory = os.path.join(self.saveDirectory, now_date)
                    path_ok = os.path.exists(nowDateSaveDirectory)
                    if (path_ok == False):
                        os.mkdir(nowDateSaveDirectory)

                    if (os.path.isdir(nowDateSaveDirectory)):
                        # id = self.lineEdit_id.text()
                        id = str(numOfPhotos)
                        # if (re.match('^[a-zA-Z0-9_]*$', id) and (id != '')):
                        depth_full_path_l = os.path.join(
                            './', now_date, id + '_l_depth.png')
                        depth_full_path_r = os.path.join(
                            './', now_date, id + '_r_depth.png')
                        color_full_path_l = os.path.join(
                            './', now_date, id + '_l_color.png')
                        color_full_path_r = os.path.join(
                            './', now_date, id + '_r_color.png')
                        self.label_Directory.setText(nowDateSaveDirectory)
                        # 保存color_image
                        cv2.imencode('.png', images_color[0])[1].tofile(color_full_path_l)
                        cv2.imencode('.png', images_color[1])[1].tofile(color_full_path_r)
                        # 保存depth_image
                        cv2.imencode('.png', images_depth[0])[1].tofile(depth_full_path_l)
                        cv2.imencode('.png', images_depth[1])[1].tofile(depth_full_path_r)
                        # print('ok')
                    self.takePhotos = False
                    numOfPhotos += 1
                # Visualise the results
                flag = False  # 是否是第一张图片
                for image in images_color:
                    if flag == False:
                        # #将第一张图片放到左边
                        MyQImage = QImage(
                            image, resolution_width, resolution_height, QImage.Format_BGR888)
                        pixmap = QPixmap.fromImage(MyQImage)
                        self.dis_update_l.emit(pixmap)  # 左
                        # print('test image l done')
                        time.sleep(DELAY)
                        flag = True
                    elif flag == True:
                        # 将第二张图片放到右边
                        MyQImage = QImage(
                            image, resolution_width, resolution_height, QImage.Format_BGR888)
                        pixmap = QPixmap.fromImage(MyQImage)
                        self.dis_update_r.emit(pixmap)  # 右
                        # print('test image r done')
                        time.sleep(DELAY)
                multi_serial = ""
                flag = False
                for serial in serials:
                    if flag == False:
                        multi_serial = multi_serial + serial
                        flag = True
                    elif flag == True:
                        multi_serial = multi_serial + " and " + serial
                self.label_top.setText(
                    'Color image from RealSense Device Nr: ' + multi_serial)
                # 清除当前缓存的数据
                images_color.clear()
                images_depth.clear()
                # time.sleep(DELAY)

        except KeyboardInterrupt:
            print("The program was interupted by the user. Closing the program...")

        finally:
            device_manager.disable_streams()


if __name__ == "__main__":
    # 固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    # 初始化
    myWin = MyMainForm()
    # 将窗口控件显示在屏幕上
    myWin.show()
    myWin.open_camera()

    print("started")
    # 程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
