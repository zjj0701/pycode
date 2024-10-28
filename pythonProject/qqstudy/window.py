# -*- coding = UTF-8 -*-
import sys
from PyQt5.QtWidgets import QApplication,QWidget

app = QApplication(sys.argv)
window = QWidget()
window.resize(300,200)
window.move(500,500)
window.setWindowTitle("test qwidget")
window.show()
app.exec_()