from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QCheckBox
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("QCheckBox 示例")

        # 创建复选框
        self.checkbox = QCheckBox("同意条款", self)

        # 设置复选框状态改变事件
        self.checkbox.stateChanged.connect(self.checkbox_changed)

        # 将复选框设置为中央控件
        self.setCentralWidget(self.checkbox)

    def checkbox_changed(self, state):
        if state == Qt.Checked:
            print("复选框被勾选")
        else:
            print("复选框未被勾选")

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
