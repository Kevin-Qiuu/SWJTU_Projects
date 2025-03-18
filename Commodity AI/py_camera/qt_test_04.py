import cv2
import numpy as np
import math
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyrealsense2 as rs
from PyQt5 import uic

if __name__ == '__main__':
    app = QApplication(sys.argv)

    ui = uic.loadUi("./camera.ui")
    # 展示窗口
    ui.show()

    app.exec()
