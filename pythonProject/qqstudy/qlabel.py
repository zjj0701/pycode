import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication


class Label(QWidget):
    def __init__(self):
        super().__init__()

        label1 = QLabel(self)
        label2 = QLabel(self)
        label3 = QLabel(self)
        label4 = QLabel(self)

        label1.setText("文本标签")
        label1.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.black)
        label1.setPalette(palette)
        label1.setAlignment(Qt.AlignCenter)

        label2.setText("label2:")

        label3.setAlignment(Qt.AlignCenter)
        label3.setToolTip("label3")
        label3.setPixmap(QPixmap("./emoji-smile.svg"))

        label4.setText("label4:")
        label4.setAlignment(Qt.AlignCenter)
        label4.setToolTip("label4")

        vbox = QVBoxLayout()
        vbox.addWidget(label1)
        vbox.addStretch()
        vbox.addWidget(label2)
        vbox.addStretch()
        vbox.addWidget(label3)
        vbox.addStretch()
        vbox.addWidget(label4)
        vbox.addStretch()

        label1.setOpenExternalLinks(True)

        label4.setOpenExternalLinks(False)
        label4.linkActivated.connect(self.linkHovered)



    def linkHovered(self):
        print("hhhhhhhhhhhhhh")

    def linkClicked(self):
        print("cccccccccccccccc")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    label = Label()
    label.show()
    sys.exit(app.exec_())