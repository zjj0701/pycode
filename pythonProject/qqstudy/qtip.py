import sys

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QToolTip, QApplication


class winform(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        QToolTip.setFont(QFont('SansSerif', 10))
        self.setToolTip('this is the tooltip')
        self.setGeometry(200,300,400,300)
        self.setWindowTitle("tip")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = winform()
    win.show()
    sys.exit(app.exec_())