from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("QComboBox 示例")

        # 创建下拉列表
        self.combobox = QComboBox(self)

        # 添加选项
        self.combobox.addItems(["选项 1", "选项 2", "选项 3"])

        # 连接下拉列表选项改变事件
        self.combobox.currentIndexChanged.connect(self.combobox_changed)

        # 将下拉列表设置为中央控件
        self.setCentralWidget(self.combobox)

    def combobox_changed(self, index):
        # 获取当前选中的文本
        text = self.combobox.currentText()
        print(f"当前选中: {text}")

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
