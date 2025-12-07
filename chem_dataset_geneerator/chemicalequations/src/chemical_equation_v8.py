import cv2
import numpy as np
import os
import random
from typing import List, Tuple, Dict


class ChemicalEquation:
    def __init__(self):
        self.elements = []  # 每个元素保存完整信息
        self.image = None
        self.width = 0
        self.height = 0

    def add_element(self, original_class, new_class, bbox, class_id, color):
        self.elements.append({
            'original_class': original_class,
            'new_class': new_class,
            'bbox': bbox,
            'class_id': class_id,
            'color': color
        })