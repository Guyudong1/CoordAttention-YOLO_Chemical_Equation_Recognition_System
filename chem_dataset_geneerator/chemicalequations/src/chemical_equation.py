import cv2
import numpy as np
import os
import random
from typing import List, Tuple, Dict

class ChemicalEquation:
    def __init__(self):
        self.elements = []  # 存储方程式中所有元素的信息
        self.width = 0
        self.height = 0
        self.image = None
        self.annotations = []  # 存储YOLO标注信息

    def add_element(self, symbol: str, bbox: Tuple[int, int, int, int], class_id: int):
        """添加一个元素到方程式中"""
        # bbox: (x_center, y_center, width, height) - 相对于整个图像的比例坐标
        self.elements.append({
            'symbol': symbol,
            'bbox': bbox,
            'class_id': class_id
        })