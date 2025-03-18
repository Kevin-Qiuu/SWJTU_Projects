import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # self.initUI()

    def initUI(self,MainWindow):
        self.setWindowTitle("文件夹选择")
        self.setGeometry(100, 100, 300, 200)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        button = QPushButton("选择文件夹", self.centralwidget)
        button.clicked.connect(self.openFolderDialog)
        button.setGeometry(100, 80, 100, 30)

    def openFolderDialog(self):
        folder_path = QFileDialog.getExistingDirectory(None, "选择文件夹")
        print("选择的文件夹路径：", folder_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = MainWindow()
    ui.initUI(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())