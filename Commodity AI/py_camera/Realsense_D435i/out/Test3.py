import numpy as np
import math
from PyQt5 import QtCore, QtGui, QtWidgets
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
        self.folder_path = os.getcwd()
        self.pathChoosed = 0

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

        for i in range(self.camera_num):
            print(self.connect_device[i])

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
        MainWindow.resize(1920, 1080)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.button_hole_filling = QtWidgets.QPushButton(self.centralwidget)
        self.button_hole_filling.setGeometry(QtCore.QRect(230, 360, 171, 61))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.button_hole_filling.setFont(font)
        self.button_hole_filling.setObjectName("button_hole_filling")
        self.button_open_camera = QtWidgets.QPushButton(self.centralwidget)
        self.button_open_camera.setGeometry(QtCore.QRect(230, 260, 171, 61))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.button_open_camera.setFont(font)
        self.button_open_camera.setObjectName("button_open_camera")
        self.button_open_camera_2 = QtWidgets.QPushButton(self.centralwidget)
        self.button_open_camera_2.setGeometry(QtCore.QRect(230, 550, 171, 61))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.button_open_camera_2.setFont(font)
        self.button_open_camera_2.setObjectName("button_open_camera_2")
        self.button_save_data = QtWidgets.QPushButton(self.centralwidget)
        self.button_save_data.setGeometry(QtCore.QRect(230, 460, 171, 61))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.button_save_data.setFont(font)
        self.button_save_data.setObjectName("button_save_data")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(230, 640, 171, 61))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        for i in range(self.camera_num):
            self.comboBox.addItem("相机"+str(i + 1))
        self.label_show_camera1 = QtWidgets.QLabel(self.centralwidget)
        self.label_show_camera1.setGeometry(QtCore.QRect(1230, 130, 640, 480))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_show_camera1.setFont(font)
        self.label_show_camera1.setScaledContents(False)
        self.label_show_camera1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_show_camera1.setObjectName("label_show_camera1")
        self.label_show_camera = QtWidgets.QLabel(self.centralwidget)
        self.label_show_camera.setGeometry(QtCore.QRect(490, 130, 640, 480))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_show_camera.setFont(font)
        self.label_show_camera.setScaledContents(False)
        self.label_show_camera.setAlignment(QtCore.Qt.AlignCenter)
        self.label_show_camera.setObjectName("label_show_camera")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(745, 60, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(1420, 60, 261, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.comboBox_rgb = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_rgb.setGeometry(QtCore.QRect(230, 50, 171, 61))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.comboBox_rgb.setFont(font)
        self.comboBox_rgb.setObjectName("comboBox_rgb")
        for d in self.resolution_rgb:
            self.comboBox_rgb.addItem(d)
        self.comboBox_rgb.setCurrentIndex(4)
        self.button_save_data_2 = QtWidgets.QPushButton(self.centralwidget)
        self.button_save_data_2.setGeometry(QtCore.QRect(230, 820, 171, 61))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.button_save_data_2.setFont(font)
        self.button_save_data_2.setObjectName("button_save_data_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(60, 730, 151, 61))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(80, 50, 131, 61))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.doubleSpinBox_savetime = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_savetime.setGeometry(QtCore.QRect(230, 730, 171, 61))
        self.doubleSpinBox_savetime.setDecimals(2)
        self.doubleSpinBox_savetime.setMinimum(0.5)
        self.doubleSpinBox_savetime.setSingleStep(0.1)
        self.doubleSpinBox_savetime.setObjectName("doubleSpinBox_savetime")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(60, 160, 151, 61))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.comboBox_depth = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_depth.setGeometry(QtCore.QRect(230, 160, 171, 61))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.comboBox_depth.setFont(font)
        self.comboBox_depth.setObjectName("comboBox_depth")
        for d in self.resolution_depth:
            self.comboBox_depth.addItem(d)
        self.comboBox_depth.setCurrentIndex(5)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # self.menu.menuAction().triggered.connect(self.menu_trigger)
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.button_open_camera_2.clicked.connect(self.button_close_camera_click)
        self.button_hole_filling.clicked.connect(self.button_hole_filling_click)
        # self.button_decimation.clicked.connect(self.button_decimation_click)
        # self.button_spatial.clicked.connect(self.button_spatial_click)
        # self.button_temporal.clicked.connect(self.button_temporal_click)
        self.timer_realsense.timeout.connect(self.get_photo)
        self.timer_camera_check.timeout.connect(self.check_camera)
        self.doubleSpinBox_savetime.valueChanged.connect(self.spin_box_change)
        self.button_save_data_2.clicked.connect(self.button_auto_save_click)
        # self.photo_thread.photoSignal.connect(self.button_save_data_click)
        self.timer_save.timeout.connect(self.button_save_data_click)
        self.button_save_data.clicked.connect(self.button_save_data_click)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MyCamera"))
        self.button_hole_filling.setText(_translate("MainWindow", "孔洞填充"))
        self.button_open_camera.setText(_translate("MainWindow", "打开相机"))
        self.button_open_camera_2.setText(_translate("MainWindow", "关闭相机"))
        self.button_save_data.setText(_translate("MainWindow", "保存信息"))
        self.label_show_camera1.setText(_translate("MainWindow", "Camera Depth"))
        self.label_show_camera.setText(_translate("MainWindow", "Camera RGB"))
        self.label.setText(_translate("MainWindow", "CAMERA RGB"))
        self.label_2.setText(_translate("MainWindow", "CAMERA DEPTH ALIGNED"))
        self.button_save_data_2.setText(_translate("MainWindow", "自动保存"))
        self.label_3.setText(_translate("MainWindow", "保存间隔时间"))
        self.label_4.setText(_translate("MainWindow", "RGB分辨率"))
        self.label_5.setText(_translate("MainWindow", "Depth分辨率"))

    # def slot_init(self):  # 建立通信连接
    #     # self.menu.menuAction().triggered.connect(self.menu_trigger)
    #     self.button_open_camera.clicked.connect(self.button_open_camera_click)
    #     self.button_open_camera_2.clicked.connect(self.button_close_camera_click)
    #     self.button_hole_filling.clicked.connect(self.button_hole_filling_click)
    #     # self.button_decimation.clicked.connect(self.button_decimation_click)
    #     # self.button_spatial.clicked.connect(self.button_spatial_click)
    #     # self.button_temporal.clicked.connect(self.button_temporal_click)
    #     self.timer_realsense.timeout.connect(self.get_photo)
    #     self.timer_camera_check.timeout.connect(self.check_camera)
    #     self.doubleSpinBox_savetime.valueChanged.connect(self.spin_box_change)
    #     self.button_save_data_2.clicked.connect(self.button_auto_save_click)
    #     # self.photo_thread.photoSignal.connect(self.button_save_data_click)
    #     self.timer_save.timeout.connect(self.button_save_data_click)
    #     self.button_save_data.clicked.connect(self.button_save_data_click)

    def check_camera(self):
        import pyrealsense2 as rs
        if self.camera_num > len(rs.context().devices):
            self.connect_device.clear()
            self.pipeline.clear()
            for d in rs.context().devices:
                print('Found device: ',
                      d.get_info(rs.camera_info.name), ' ',
                      d.get_info(rs.camera_info.serial_number))
                if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                    self.connect_device.append(d.get_info(rs.camera_info.serial_number))
            self.camera_num = len(rs.context().devices)
            for i in range(self.camera_num):
                self.pipeline[i] = rs.pipeline()
            self.camera_num = len(rs.context().devices)
            self.comboBox.clear()
            for i in range(self.camera_num):
                self.comboBox.addItem("相机" + str(i + 1))
            if self.camera_num > 0:
                self.comboBox.setCurrentIndex(0)
            QMessageBox.about(self.centralwidget, '警告', '           一个相机已被移除                 ')

        if self.camera_num < len(rs.context().devices):
            self.connect_device.clear()
            self.pipeline.clear()
            for d in rs.context().devices:
                print('Found device: ',
                      d.get_info(rs.camera_info.name), ' ',
                      d.get_info(rs.camera_info.serial_number))
                if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                    self.connect_device.append(d.get_info(rs.camera_info.serial_number))
            self.camera_num = len(rs.context().devices)
            for i in range(self.camera_num):
                self.pipeline[i] = rs.pipeline()
            self.camera_num = len(rs.context().devices)
            self.comboBox.clear()
            for i in range(self.camera_num):
                self.comboBox.addItem("相机" + str(i + 1))
            if self.camera_num > 0:
                self.comboBox.setCurrentIndex(0)
            QMessageBox.about(self.centralwidget, '警告', '           一个相机已被添加                 ')

    # def button_message_click(self):
    #     ###超级大坑！！！！self一定要和button对应
    #     QMessageBox.about(self.centralwidget, '关于', '这是一个关于对话框')
    #     return

    def button_auto_save_click(self):
        if self.camera_num == 0:
            QMessageBox.about(self.centralwidget, '警告', '请连接相机')
            return
        if self.tag_save_data == 0:
            self.tag_save_data = 1
            self.button_save_data_2.setText(u'关闭自动')
            self.timer_save.start(self.doubleSpinBox_savetime.value() * 1000)
        else:
            self.tag_save_data = 0
            self.button_save_data_2.setText(u'开启自动')
            self.timer_save.stop()

    def spin_box_change(self):
        self.time_interval = int(self.doubleSpinBox_savetime.value() * 1000)
        self.timer_save.setInterval(self.time_interval)

    def button_save_data_click(self):
        if self.camera_num == 0:
            QMessageBox.about(self.centralwidget, '警告', '请连接相机')
            return
        if self.tag_save_data == 0:
            print('hello')
            try:
                print('hello2')
                folder_path = QFileDialog.getExistingDirectory(None, "选择文件夹","")
                return
                print(3)
            except Exception as e:
                traceback.print_exc()
            self.folder_path = folder_path
            print(2)
            #初始化保存路径
            self.folder_path = os.path.join(self.folder_path, "out", time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
            ##创建多级目录的文件夹用makedirs而不是mkdir
            os.makedirs(self.folder_path)
            os.makedirs(os.path.join(self.folder_path, "color"))
            os.makedirs(os.path.join(self.folder_path, "depth"))
            os.makedirs(os.path.join(self.folder_path, "points"))
        ply = {}
        for i in range(self.camera_num):
            ply[i] = rs.save_to_ply(os.path.join((self.folder_path), "points", "camera"+ str(i + 1) +"_{}.ply".format(self.shoot_count[i])))
            ply[i].set_option(rs.save_to_ply.option_ply_binary, True)
            ply[i].set_option(rs.save_to_ply.option_ply_normals, True)
        colorized = {}
        for i in range(self.camera_num):
            colorized[i] = self.colorizer.process(self.frames[i])
            ply[i].process(colorized[i])
        depth_data = {}
        for i in range(self.camera_num):
            depth_data[i] = np.asanyarray(self.depth_frame[i].get_data(), dtype="float16")

        depth_image = {}
        for i in range(self.camera_num):
            depth_image[i] = self.depth_image[i]

        color_image = {}
        for i in range(self.camera_num):
            ##按理说只有读才会出现 rgb -> bgr的情况才对
            color_image[i] = cv2.cvtColor(self.img_r[i], cv2.COLOR_BGR2RGB)

        depth_mapped_image = {}
        for i in range(self.camera_num):
            depth_mapped_image[i] = cv2.applyColorMap(cv2.convertScaleAbs(depth_image[i], alpha=0.03), cv2.COLORMAP_JET)

        saved_color_image = color_image
        ##将深度数据生成图片要用！
        saved_depth_mapped_image = depth_mapped_image

        # 彩色图片保存为png格式(时间间隔大于1s)
        for i in range(self.camera_num):
            cv2.imwrite(os.path.join((self.folder_path), "color", "camera"+ str(i + 1) +"_{}.png".format(self.shoot_count[i])), saved_color_image[i])
        # 深度信息由采集到的float16直接保存为npy格式

        for i in range(self.camera_num):
            np.save(os.path.join((self.folder_path), "depth", "camera"+ str(i + 1) +"_{}".format(self.shoot_count[i])), depth_data[i])

        for i in range(self.camera_num):
            self.shoot_count[i] += 1

        # qmessage = QMessageBox()
        # qmessage.setMinimumSize(100,100)
        # qmessage.about(self.centralwidget, '保存数据', '                  保存成功                     ')
        # cv2.imshow("save", np.hstack((saved_color_image, saved_depth_mapped_image)))

    def button_close_camera_click(self):
        ##关闭之前的帧很重要！！！！！！！
        if self.camera_num == 0:
            QMessageBox.about(self.centralwidget, '警告', '请连接相机')
            return
        self.timer_camera.stop()
        self.timer_realsense.stop()
        for i in range(self.camera_num):
            self.pipeline[i].stop()
        # self.imu_pipeline.stop()
        _translate = QtCore.QCoreApplication.translate
        self.label_show_camera1.setText(_translate("MainWindow", "Camera Depth"))
        self.label_show_camera.setText(_translate("MainWindow", "Camera RGB"))
        ##最后一定要打开检索
        self.timer_camera_check.start(500)

    def button_open_camera_click(self):
        import pyrealsense2 as rs
        if self.camera_num == 0:
            QMessageBox.about(self.centralwidget, '警告', '请连接相机')
            return
        current_index = self.comboBox_rgb.currentIndex()
        if current_index == 0:
            self.config.enable_stream(rs.stream.color, 320, 180, rs.format.rgb8, 30)
        if current_index == 1:
            self.config.enable_stream(rs.stream.color, 320, 240, rs.format.rgb8, 30)
        if current_index == 2:
            self.config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 30)
        if current_index == 3:
            self.config.enable_stream(rs.stream.color, 640, 360, rs.format.rgb8, 30)
        if current_index == 4:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        if current_index == 5:
            self.config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
        if current_index == 6:
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.rgb8, 30)
        if current_index == 7:
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
        if current_index == 8:
            self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)

        current_index = self.comboBox_depth.currentIndex()
        if current_index == 0:
            self.config.enable_stream(rs.stream.depth, 256, 144, rs.format.z16, 30)
        if current_index == 1:
            self.config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
        if current_index == 2:
            self.config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 30)
        if current_index == 3:
            self.config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
        if current_index == 4:
            self.config.enable_stream(rs.stream.depth, 640, 400, rs.format.z16, 30)
        if current_index == 5:
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        if current_index == 6:
            self.config.enable_stream(rs.stream.depth, 848, 100, rs.format.z16, 30)
        if current_index == 7:
            self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        if current_index == 8:
            self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        if self.count_seted == 0:
            self.count_seted = 1
            self.shoot_count = {}
            for i in range(self.camera_num):
                self.shoot_count.setdefault(i,0)
        self.timer_camera_check.stop()
        if not self.timer_camera.isActive():
            # 开启深度相机数据流
            for i in range(self.camera_num):
                self.config.enable_device(self.connect_device[i])
                self.pipeline[i].start(self.config)
            # depth_sensor = profile.get_device().first_depth_sensor()
            # 开启IMU数据流
            # imu_profile = self.imu_pipeline.start(self.imu_config)
            # sensor = self.pipeline.get_active_profile().get_device().query_sensors()[0]
            # sensor.set_option(rs.option.visual_preset, self.preset)
            ##开启定时器
            self.timer_realsense.start(50)
            self.timer_camera.start(50)

    def button_hole_filling_click(self):
        if self.camera_num == 0:
            QMessageBox.about(self.centralwidget, '警告', '请连接相机')
            return
        if self.tag_hole_filling == 0:
            self.tag_hole_filling = 1
            self.button_hole_filling.setText(u'关闭孔填充滤波')
        else:
            self.tag_hole_filling = 0
            self.button_hole_filling.setText(u'开启孔填充滤波')

    def get_photo(self):

        # 从数据流中读取一帧并进行对齐
        try:
            for i in range(self.camera_num):
                self.frames[i] = self.pipeline[i].wait_for_frames()

            for i in range(self.camera_num):
                self.aligned_frames[i] = self.alignedFs.process(self.frames[i])

            # 分别获得深度帧和RGB帧
            ##下面两个信息必须共享
            for i in range(self.camera_num):
                self.depth_frame[i] = self.aligned_frames[i].get_depth_frame()
                self.color_frame[i] = self.aligned_frames[i].get_color_frame()

            # 获取帧的宽高(一次获取就好)
            self.width = self.depth_frame[0].get_width()
            self.height = self.depth_frame[0].get_height()

            # 对深度图像进行后处理
            if self.tag_hole_filling == 1:
                for i in range(self.camera_num):
                    self.depth_frame[i] = self.hole_filling.process(self.depth_frame[i])

            # if self.tag_decimation == 1:
            #     depth_frame = self.decimation.process(depth_frame)
            # if self.tag_spatial == 1:
            #     depth_frame = self.spatial.process(depth_frame)
            # if self.tag_temporal == 1:
            #     depth_frame = self.temporal.process(depth_frame)
            # 获取点云
            # self.pc.map_to(self.color_frame)
            # points = self.pc.calculate(self.depth_frame)
            # self.vtx = np.asanyarray(points.get_vertices())
            # self.vtx = np.reshape(self.vtx, (self.height, self.width, -1))
            # 将深度帧和RBG帧转换为数组

            for i in range(self.camera_num):
                self.img_r[i] = np.asanyarray(self.color_frame[i].get_data())

            for i in range(self.camera_num):
                self.depth_image[i] = np.asanyarray(self.colorizer.colorize(self.depth_frame[i]).get_data())

            # 读取图片到label_show_camera控件中(这个只用渲染一组就好了)
            showImage = QtGui.QImage(self.img_r[self.comboBox.currentIndex()
    ].data, self.img_r[self.comboBox.currentIndex()
    ].shape[1], self.img_r[self.comboBox.currentIndex()
    ].shape[0], QtGui.QImage.Format_RGB888)

            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

            showImage1 = QtGui.QImage(self.depth_image[self.comboBox.currentIndex()
    ].data, self.depth_image[self.comboBox.currentIndex()
    ].shape[1], self.depth_image[self.comboBox.currentIndex()
    ].shape[0],
                                      QtGui.QImage.Format_RGB888)

            self.label_show_camera1.setPixmap(QtGui.QPixmap.fromImage(showImage1))
        except:
            QMessageBox.about(self.centralwidget, '警告', '        摄像头已弹出      ')
            self.check_camera()
            self.timer_camera.stop()
            self.timer_realsense.stop()
            for i in range(self.camera_num):
                self.pipeline[i].stop()
            if self.camera_num == 0:
                QMessageBox.about(self.centralwidget, '警告', '请连接相机')
            _translate = QtCore.QCoreApplication.translate
            self.label_show_camera1.setText(_translate("MainWindow", "Camera Depth"))
            self.label_show_camera.setText(_translate("MainWindow", "Camera RGB"))
            ##最后一定要打开检索
            self.timer_camera_check.start(500)
            return


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()  # 创建窗体对象
    ui = Ui_MainWindow()  # 创建PyQt5设计的窗体对象
    ui.setupUi(MainWindow)  # 调用PyQt5窗体的方法对窗体对象进行初始化设置
    MainWindow.show()  # 显示窗体
    sys.exit(app.exec_())  # 程序关闭时退出进程