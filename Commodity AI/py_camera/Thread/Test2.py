import numpy as np
import math
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog
from qt_material import apply_stylesheet
import cv2
import time
import os
import  traceback



class Ui_MainWindow(object):
    def __init__(self):
        super().__init__()

        def __init__(self):
            super().__init__()
            import pyrealsense2 as rs
            self.timer_camera = QtCore.QTimer()  # 初始化定时器
            self.timer_realsense = QtCore.QTimer()
            self.timer_camera_check = QtCore.QTimer()
            self.timer_camera_check.setInterval(10000)
            self.timer_camera_check.start(500)
            self.timer_save = QtCore.QTimer()
            self.timer_save.setInterval(10000)

            ##检测设备连接
            self.connect_device = []
            self.pipeline = {}
            self.depth_frame = {}
            self.color_frame = {}
            self.frames = {}
            self.aligned_frames = {}
            self.img_r = {}
            self.depth_image = {}
            self.shoot_count = {}
            self.time_interval = 10000
            self.count_seted = 0
            self.folder_path = ''
            # self.folder_path = QtWidgets.QFileDialog.getExistingDirectory(None, "选择文件夹")
            # # 初始化保存路径
            # self.folder_path = os.path.join(self.folder_path, "out", time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
            # ##创建多级目录的文件夹用makedirs而不是mkdir
            # os.makedirs(self.folder_path)
            # os.makedirs(os.path.join(self.folder_path, "color"))
            # os.makedirs(os.path.join(self.folder_path, "depth"))
            # os.makedirs(os.path.join(self.folder_path, "points"))

            self.resolution_depth = ['256 × 144',
                                     '424 × 240',
                                     '480 × 270',
                                     '640 × 360',
                                     '640 × 400',
                                     '640 × 480',
                                     '848 × 100',
                                     '848 × 480',
                                     '1280 × 720']

            self.resolution_rgb = ['320 × 180',
                                   '320 × 240',
                                   '424 × 240',
                                   '640 × 360',
                                   '640 × 480',
                                   '848 × 480',
                                   '960 × 540',
                                   '1280 × 720',
                                   '1920 × 1080']

            for d in rs.context().devices:
                print('Found device: ',
                      d.get_info(rs.camera_info.name), ' ',
                      d.get_info(rs.camera_info.serial_number))
                if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                    self.connect_device.append(d.get_info(rs.camera_info.serial_number))
            self.camera_num = len(rs.context().devices)

            # #初始化一个camera_check监视器
            # camera_check = MonitorVariable(self.camera_num)

            print(self.connect_device)

            for i in range(self.camera_num):
                self.pipeline[i] = rs.pipeline()

            # 定义IMU数据流（惯性测量单元）   不需要
            # self.imu_pipeline = rs.pipeline()
            # 定义点云
            # self.imu_config = rs.config()
            # self.imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)  # acceleration
            # self.imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)  # gyroscope

            self.pc = rs.pointcloud()
            self.points = rs.points()

            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            # 设置对其方式为：深度图向RGB图对齐
            self.align_to = rs.stream.color
            self.alignedFs = rs.align(self.align_to)
            # 创建着色器(其实这个可以替代opencv的convertScaleAbs()和applyColorMap()函数了,但是是在多少米范围内map呢?)
            self.colorizer = rs.colorizer()
            # 为每个操作设置tag
            self.tag_hole_filling = 0
            # self.tag_decimation = 0
            # self.tag_spatial = 0
            # self.tag_temporal = 0
            # 定义孔填充过滤器
            self.hole_filling = rs.hole_filling_filter()

            # 创建抽取过滤器，这个相当于池化，用于降低分辨率和消除噪声,且经过实测此操作可以降低CPU占用率
            self.decimation = rs.decimation_filter()
            # 池化步长设置
            self.pooling_stride = 2
            self.decimation.set_option(rs.option.filter_magnitude, self.pooling_stride)
            # 空间滤波器是域转换边缘保留平滑的快速实现,实测会明显增加CPU负载，不建议使用
            self.spatial = rs.spatial_filter()
            # 我们可以通过增加smooth_alpha和smooth_delta选项来强调滤镜的效果：
            self.spatial.set_option(rs.option.filter_magnitude, 5)
            self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial.set_option(rs.option.filter_smooth_delta, 20)
            # 该过滤器还提供一些基本的空间孔填充功能：
            self.spatial.set_option(rs.option.holes_fill, 3)
            # 定义时间滤波器，不建议在运动场景下使用，会有伪影出现
            self.temporal = rs.temporal_filter()
            # 设置选择的相机
            self.camera_choosed = 0
            self.tag_save_data = 0
            self.tag_diff_data = 0

            # 初始化保存路径
            self.save_path = os.path.join(os.getcwd(), "out", time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
            ##创建多级目录的文件夹用makedirs而不是mkdir
            os.makedirs(self.save_path)
            os.makedirs(os.path.join(self.save_path, "color"))
            os.makedirs(os.path.join(self.save_path, "depth"))
            os.makedirs(os.path.join(self.save_path, "points"))
            ##保存的信息的初始化
            self.saved_color_image = None  # 保存的临时图片
            self.saved_depth_mapped_image = None

            # 深度图像画面预设
            self.preset = 0
            # 记录点云的数组
            self.vtx = None
            self.height = 720
            self.width = 1280
            # 输入图片大小
            self.img_size = 512
            # 输出特征图尺寸及其步长
            self.feature_size = 32
            self.stride = 16
            # 设置其他按钮字体
            self.font2 = QFont()
            self.font2.setFamily("宋体")
            self.font2.setPixelSize(22)
            self.font2.setBold(True)
            # 深度相机返回的图像和三角函数值
            self.depth_image = {}
            self.img_d = None
            self.pitch = None
            self.roll = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(370, 323)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 20, 61, 16))
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(80, 20, 211, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(300, 20, 51, 23))
        self.pushButton.setObjectName("pushButton")

        self.pushButton2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton2.setGeometry(QtCore.QRect(100, 20, 51, 23))
        self.pushButton2.setObjectName("pushButton")

        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(20, 50, 331, 261))
        self.listWidget.setObjectName("listWidget")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.bindList)
        self.pushButton2.clicked.connect(self.bindList2)

    def bindList(self):
        import pyrealsense2 as rs
        from PyQt5.QtWidgets import QFileDialog
        import os  # 导入os模块
        # 创建选择路径对话框
        print('hello1')
        dir = QFileDialog.getExistingDirectory(None, "选择文件夹路径", os.getcwd())
        self.lineEdit.setText(dir)  # 在文本框中显示选择的路径
        list = os.listdir(dir)  # 遍历选择的文件夹
        self.listWidget.addItems(list)  # 将文件夹中的所有文件显示在列表中

    def bindList2(self):
        import pyrealsense2 as rs
        from PyQt5.QtWidgets import QFileDialog
        import os  # 导入os模块
        # 创建选择路径对话框
        try:
            print('hello2')
            dir = QFileDialog.getExistingDirectory(None, "选择文件夹路径", os.getcwd())
            print(3)
        except Exception as e:
            traceback.print_exc()
        # self.lineEdit.setText(dir)  # 在文本框中显示选择的路径
        # list = os.listdir(dir)  # 遍历选择的文件夹
        # self.listWidget.addItems(list)  # 将文件夹中的所有文件显示在列表中

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "选择路径："))
        self.pushButton.setText(_translate("MainWindow", "选择"))
        self.pushButton2.setText(_translate("MainWindow", "选择2"))


# 主方法
if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()  # 创建窗体对象
    ui = Ui_MainWindow()  # 创建PyQt5设计的窗体对象
    ui.setupUi(MainWindow)  # 调用PyQt5窗体的方法对窗体对象进行初始化设置
    MainWindow.show()  # 显示窗体
    sys.exit(app.exec_())  # 程序关闭时退出进程