import sys  # 导入 sys 模块，用于与 Python 解释器交互
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton  # 从 PyQt5 中导入所需的类
# QLabel 显示文本，QMainWindow主窗口类，
# 创建一个主窗口类，继承自 QMainWindow
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()  # 调用父类 QMainWindow 的初始化方法
        self.setWindowTitle("PyQt5 第一个窗口")  # 设置窗口标题

        #创建按钮
        button = QPushButton("选择",self)
        # 设置按钮点击事件
        button.clicked.connect(self.button_clicked)
        # 将按钮设置为窗口的中央控件
        self.setCentralWidget(button)


    def button_clicked(self):
        print("ok")
# 创建一个 PyQt5 应用程序对象
app = QApplication(sys.argv)

# 创建主窗口实例
window = MainWindow()
window.show()  # 显示窗口

# 进入应用程序的事件循环，保持应用程序运行，直到关闭窗口
sys.exit(app.exec_())
