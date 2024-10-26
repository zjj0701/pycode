from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("QLineEdit 示例")

        # 创建文本框
        self.line_edit = QLineEdit(self)

        # 设置默认提示文本
        self.line_edit.setPlaceholderText("请输入文本")

        # 将文本框设置为中央控件
        self.setCentralWidget(self.line_edit)

        # 连接文本输入结束的信号到槽函数
        self.line_edit.returnPressed.connect(self.return_pressed)

    def return_pressed(self):
        # 获取用户输入的文本
        text = self.line_edit.text()
        print(f"用户输入: {text}")

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
