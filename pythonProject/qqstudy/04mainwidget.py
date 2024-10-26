import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QDesktopWidget, QPushButton, QHBoxLayout
from PyQt5.QtGui import QIcon

class MainWindow(QMainWindow):
    def __init__(self,parent=None):
        super(MainWindow,self).__init__(parent)

        self.resize(400,200)

        self.status = self.statusBar()
        self.status.showMessage("这是状态栏",5000)

        self.setWindowTitle("mainwindow example")

        self.button1 = QPushButton("关闭窗口")
        self.button1.clicked.connect(self.onbuttonClick)


        layout = QHBoxLayout()
        layout.addWidget(self.button1)


        main_frame = QWidget(self)
        main_frame.setLayout(layout)
        self.setCentralWidget(main_frame)

    def onbuttonClick(self):
        # sender是发信号的对象
        sender = self.sender()
        print(sender.text()+"被按下了")
        qApp = QApplication.instance()
        qApp.quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())