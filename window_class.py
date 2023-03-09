import os
import sys

import cv2
import numpy
import numpy as np
import torch
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
from PIL import ImageDraw
from PIL import ImageChops
import PIL
import recognizer


class ClickLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal()
    width = 600
    height = 400
    center = height // 2
    white = (255, 255, 255)
    black = (0, 0, 0)
    green = (0, 128, 0)
    image1 = PIL.Image.new("RGB", (width, height), black)
    draw = ImageDraw.Draw(image1)
    old_x = -1
    old_y = -1

    def mouseMoveEvent(self, event):
        painter = QtGui.QPainter(self.pixmap())
        pen = QtGui.QPen()
        pen.setWidth(15)
        painter.setPen(pen)

        if self.old_x != -1:
            painter.drawLine(self.old_x, self.old_y, event.x(), event.y())

        painter.end()
        self.parent().update()
        # do the PIL image/draw (in memory) drawings
        if self.old_x != -1:
            self.draw.line([self.old_x, self.old_y, event.x(), event.y()], self.white, width=15)

        self.old_x = event.x()
        self.old_y = event.y()
        # PIL image can be saved as .png .jpg .gif or .bmp file (among others)
        filename = "my_drawing.jpg"
        self.image1.save(filename)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.old_x = -1

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(600, 600)
        self.textEdit = QtWidgets.QTextEdit()
        self.label = ClickLabel()
        self.label.resize(600, 400)
        canvas = QtGui.QPixmap(600, 400)
        canvas.fill(Qt.white)
        self.label.setPixmap(canvas)
        self.label.clicked.connect(self.draw_something)
        self.btnPress1 = QtWidgets.QPushButton("recognize")
        self.btnPress1.clicked.connect(self.button1_clicked)
        self.setCentralWidget(self.label)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.textEdit)
        layout.addWidget(self.label)
        layout.addWidget(self.btnPress1)
        self.main_widget = QtWidgets.QWidget(self)
        self.main_widget.setLayout(layout)
        self.setCentralWidget(self.main_widget)

    def button1_clicked(self):
        image_data_temp = cv2.imread("my_drawing.jpg")  # Read Image as numbers
        image_temp_resize = cv2.resize(image_data_temp, (recognizer.IMAGE_SIZE, recognizer.IMAGE_SIZE))
        result_arr = []
        X_Data = np.asarray(image_temp_resize) / (255.0)
        X_Data = X_Data.reshape(-1, recognizer.IMAGE_SIZE, recognizer.IMAGE_SIZE, 3)

        X = torch.tensor(X_Data, dtype=torch.float32)
        print("sample dimension one is {0}".format(len(X)))
        print("sample dimension two is {0}".format(len(X[0])))
        print("sample dimension three is {0}".format(len(X[0][0])))
        print("sample dimension four is {0}".format(len(X[0][0][0])))

        y_pred = recognizer.model(X)

        print("result dimension one is {0}".format(len(y_pred)))
        print("result dimension two is {0}".format(len(y_pred[0])))
        print("result dimension three is {0}".format(len(y_pred[0][0])))
        print("result dimension four is {0}".format(len(y_pred[0][0][0])))
        #print(y_pred[0][0])

        is_max = y_pred[0][0][0][0]
        selected_index = 0

        print("is_max value is {0}".format(is_max))
        print("and selected index is  is {0}".format(selected_index))
        for idx, x in enumerate(y_pred[0][0][0]):
            if x > is_max:
                is_max = x
                selected_index = idx
        print("after the loop is_max value is {0}".format(is_max))
        print("after the loop selected index is  is {0}".format(selected_index))
        print("categories are")
        print(recognizer.CATEGORIES)
        categories = recognizer.CATEGORIES
        self.textEdit.append(categories[selected_index - 1])

        canvas = QtGui.QPixmap(600, 400)
        canvas.fill(Qt.white)
        self.label.setPixmap(canvas)
        os.remove("my_drawing.jpg")
        ClickLabel.image1 = PIL.Image.new("RGB", (ClickLabel.width, ClickLabel.height), ClickLabel.white)
        ClickLabel.draw = ImageDraw.Draw(ClickLabel.image1)

