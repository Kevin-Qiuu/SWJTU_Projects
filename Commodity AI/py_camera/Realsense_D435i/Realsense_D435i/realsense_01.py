import cv2
import numpy as np
import math
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyrealsense2 as rs



class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.timer_camera = QtCore.QTimer()  # 初始化定时器
        self.timer_realsense = QtCore.QTimer()
        # 定义realsense视频数据流
        self.pipeline = rs.pipeline()
        # 定义IMU数据流
        self.imu_pipeline = rs.pipeline()
        # 定义点云
        self.imu_config = rs.config()
        self.imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)  # acceleration
        self.imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)  # gyroscope
        self.pc = rs.pointcloud()
        self.config = rs.config()
        # 设置对其方式为：深度图向RGB图对齐
        self.align_to = rs.stream.color
        self.alignedFs = rs.align(self.align_to)
        # 创建着色器(其实这个可以替代opencv的convertScaleAbs()和applyColorMap()函数)
        self.colorizer = rs.colorizer()
        # 深度图像画面预设
        self.preset = 0
        # 记录点云的数组
        self.vtx = None
        self.height = 720
        self.width = 1280
        # 设置字体
        self.font2 = QFont()
        self.font2.setFamily("宋体")
        self.font2.setPixelSize(22)
        self.font2.setBold(True)
        # 深度相机返回的图像和三角函数值
        self.depth_image = None
        self.img_r = None
        self.pitch = None
        self.roll = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        MainWindow.setAcceptDrops(True)

        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("background-color: rgb(240, 248, 255);")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # 所有控件的定义要从此处才能定义
        # 显示RGB图
        self.label_show_camera = QtWidgets.QLabel(self.centralwidget)
        self.label_show_camera.setGeometry(QtCore.QRect(0, 0, 960, 540))
        self.label_show_camera.setStyleSheet("background-color: rgb(255,255,255);")
        self.label_show_camera.setObjectName("label_show_camera")
        self.label_show_camera.setScaledContents(True)
        # 显示深度图
        self.label_show_camera1 = QtWidgets.QLabel(self.centralwidget)
        self.label_show_camera1.setGeometry(QtCore.QRect(960, 0, 960, 540))
        self.label_show_camera1.setStyleSheet("background-color: rgb(255,255,255);")
        self.label_show_camera1.setObjectName("label_show_camera1")
        self.label_show_camera1.setScaledContents(True)
        # 显示相机位姿
        self.label_show_camera2 = QtWidgets.QLabel(self.centralwidget)
        self.label_show_camera2.setGeometry(QtCore.QRect(960, 560, 512, 512))
        self.label_show_camera2.setStyleSheet("background-color: rgb(255,255,255);")
        self.label_show_camera2.setObjectName("label_show_camera2")
        self.label_show_camera2.setScaledContents(True)

        # 显示目标坐标
        self.label_show_performance = QtWidgets.QLabel(self.centralwidget)
        self.label_show_performance.setGeometry(QtCore.QRect(300, 560, 600, 30))
        self.label_show_performance.setObjectName("label_show_performance")
        self.label_show_performance.setScaledContents(True)
        self.label_show_performance.setStyleSheet("color:blue")
        self.label_show_performance.setFont(self.font2)

        # 显示加速度
        self.label_show_accel = QtWidgets.QLabel(self.centralwidget)
        self.label_show_accel.setGeometry(QtCore.QRect(1480, 660, 600, 30))
        self.label_show_accel.setObjectName("label_show_accel")
        self.label_show_accel.setScaledContents(True)
        self.label_show_accel.setStyleSheet("color:blue")
        self.label_show_accel.setFont(self.font2)

        # 显示欧拉角
        self.label_show_pose = QtWidgets.QLabel(self.centralwidget)
        self.label_show_pose.setGeometry(QtCore.QRect(1480, 760, 600, 30))
        self.label_show_pose.setObjectName("label_show_pose")
        self.label_show_pose.setScaledContents(True)
        self.label_show_pose.setStyleSheet("color:blue")
        self.label_show_pose.setFont(self.font2)

        # 开启相机的按钮
        self.button_open_camera = QtWidgets.QPushButton(self.centralwidget)
        self.button_open_camera.setGeometry(QtCore.QRect(20, 560, 200, 50))
        self.button_open_camera.setStyleSheet("color: rgb(255, 255, 255);\n"
                                              "border-color: rgb(0, 0, 0);\n"
                                              "background-color: rgb(50,50,237);")
        self.button_open_camera.setFlat(False)
        self.button_open_camera.setObjectName("button_open_camera")
        self.button_open_camera.setFont(QFont("宋体", 12, QFont.Bold))


        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Realsense D435i"))
        self.button_open_camera.setText(_translate("MainWindow", "Start Detection"))
        self.label_show_accel.setText(_translate("MainWindow", "Accel"))
        self.label_show_pose.setText(_translate("MainWindow", "Pose"))

    def slot_init(self):  # 建立通信连接
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_realsense.timeout.connect(self.get_photo)


    def button_open_camera_click(self):
        if not self.timer_camera.isActive():
            # 开启深度相机数据流
            profile = self.pipeline.start(self.config)
            depth_sensor = profile.get_device().first_depth_sensor()
            # 开启IMU数据流
            imu_profile = self.imu_pipeline.start(self.imu_config)
            # 设置画面预设
            sensor = self.pipeline.get_active_profile().get_device().query_sensors()[0]
            sensor.set_option(rs.option.visual_preset, self.preset)
            self.timer_realsense.start(50)
            self.timer_camera.start(50)


    def get_photo(self):
        # 从数据流中读取一帧并进行对齐
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.alignedFs.process(frames)
        # 分别获得深度帧和RGB帧
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # 获取IMU数据
        imu_frames = self.imu_pipeline.wait_for_frames()
        accel_frame = imu_frames.first_or_default(rs.stream.accel, rs.format.motion_xyz32f)
        #gyro_frame = imu_frames.first_or_default(rs.stream.gyro, rs.format.motion_xyz32f)
        accel_info = accel_frame.as_motion_frame().get_motion_data()
        gx, gy, gz = accel_info.x, accel_info.y, accel_info.z
        self.label_show_accel.setText('Accel  gx:' + format(gx, '.3f') + ' gy:' + format(gy, '.3f') + ' gz:' + format(gz, '.3f'))
        # 计算欧拉角
        # 俯仰角
        pitch = int(180/math.pi * math.atan(abs(gz)/math.sqrt(gx * gx + gy * gy)))
        # 滚转角
        roll = int(180/math.pi * math.atan(abs(gx)/math.sqrt(gy * gy + gz * gz)))
        # 计算出三角函数值方便坐标转换
        vector = math.sqrt(gx * gx + gy * gy + gz * gz)
        sin_pitch = abs(gz) / vector
        cos_pitch = math.sqrt(gx * gx + gy * gy) / vector
        sin_roll = abs(gx) / vector
        cos_roll = abs(gy * gy + gz * gz) / vector
        self.label_show_pose.setText('Pose  pitch:' + str(pitch) + '° roll:' + str(roll) + '°')
        # 获取帧的宽高
        self.width = depth_frame.get_width()
        self.height = depth_frame.get_height()
        # 获取点云
        self.pc.map_to(color_frame)
        points = self.pc.calculate(depth_frame)
        self.vtx = np.asanyarray(points.get_vertices())
        self.vtx = np.reshape(self.vtx, (self.height, self.width, -1))
        # 将深度帧和RBG帧转换为数组
        img_r = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

        self.img_r = img_r
        self.depth_image = depth_image
        self.pitch = (sin_pitch, cos_pitch)
        self.roll = (sin_roll, cos_roll)
        showImage = QtGui.QImage(img_r.data, img_r.shape[1], img_r.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 读取图片到label_show_camera控件中
        showImage1 = QtGui.QImage(depth_image.data, depth_image.shape[1], depth_image.shape[0],
                                  QtGui.QImage.Format_RGB888)
        self.label_show_camera1.setPixmap(QtGui.QPixmap.fromImage(showImage1))


    # 将相机当前姿态绘制到界面上
    def IMU_visdom(self, gx, gy, gz):
        canvas_size = 512
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype="uint8")
        # 绘制坐标轴
        cv2.line(canvas, (256, 256), (256, 450), (0, 255, 0), 2)
        cv2.line(canvas, (256, 450), (256 - 5, 450 - 10), (0, 255, 0), 2)
        cv2.line(canvas, (256, 450), (256 + 5, 450 - 10), (0, 255, 0), 2)
        cv2.putText(canvas, 'Y', (266, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.line(canvas, (256, 256), (88, 353), (0, 0, 255), 2)
        cv2.line(canvas, (88, 353), (88 + 10, 353), (0, 0, 255), 2)
        cv2.line(canvas, (88, 353), (88 + 5, 353 - 10), (0, 0, 255), 2)
        cv2.putText(canvas, 'Z', (76, 359), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.line(canvas, (256, 256), (88, 159), (255, 0, 0), 2)
        cv2.line(canvas, (88, 159), (88 + 10, 159), (255, 0, 0), 2)
        cv2.line(canvas, (88, 159), (88 + 5, 159 + 10), (255, 0, 0), 2)
        cv2.putText(canvas, 'X', (83, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        # 我们认为194个像素点为坐标轴长度，对应重力加速度大小，这里取10, 首先计算各个分量在对应坐标轴上的距离
        dis_y = (256, int(gy / 10 * 194))
        dis_x = (int(gx / 10 * 194 / 2 * 1.732), int(gx / 10 * 194 / 2))
        dis_z = (int(gz / 10 * 194 / 2 * 1.732), int(gz / 10 * 194 / 2))
        cv2.circle(canvas, (256, 256 - dis_y[1]), 3, color=(255, 255, 255))
        cv2.circle(canvas, (256 - dis_x[0], 256 - dis_x[1]), 3, color=(255, 255, 255))
        cv2.circle(canvas, (256 - dis_z[0], 256 + dis_z[1]), 3, color=(255, 255, 255))
        point_xyz = (256 + dis_x[0] + dis_z[0], 256 + dis_x[1] - dis_z[1] + dis_y[1])
        cv2.line(canvas, (256, 256), point_xyz, (255, 255, 255), 2)
        showImage2 = QtGui.QImage(canvas.data, canvas.shape[1], canvas.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera2.setPixmap(QtGui.QPixmap.fromImage(showImage2))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    ui.slot_init()
    mainWindow.show()
    sys.exit(app.exec_())
