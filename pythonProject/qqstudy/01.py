import sys
from PyQt5 import QtWidgets,QtCore

app = QtWidgets.QApplication(sys.argv)

# __表示私有

widget = QtWidgets.QWidget()
widget.resize(360,360)
widget.setWindowTitle("hello")
widget.show()

sys.exit(app.exec_())