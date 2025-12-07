import os
import cv2
import numpy as np
import random
from pathlib import Path

# 配置路径
txt_path = 'elements.txt'        # 元素列表
letters_base = Path('raw_letters')  # 原始字母数据集
output_dir = Path('generated_elements_V1')  # 输出文件夹
output_dir.mkdir(exist_ok=True)

# 水平拼接函数
def concatenate_images(images, gap=5):
    """水平拼接多张图像，中间加gap像素间距"""
    if not images:
        return None
    max_h = max(img.shape[0] for img in images)
    # 对齐高度
    aligned_imgs = []
    for img in images:
        h, w = img.shape[:2]
        new_img = cv2.copyMakeBorder(img, 0, max_h - h, 0, 0, cv2.BORDER_CONSTANT, value=255)
        aligned_imgs.append(new_img)
    # 拼接
    combined = aligned_imgs[0]
    for img in aligned_imgs[1:]:
        combined = np.hstack([combined, 255*np.ones((max_h, gap), dtype=np.uint8), img])
    return combined

# 读取元素列表
with open(txt_path, 'r') as f:
    elements = [line.strip() for line in f if line.strip()]

# 遍历每个元素
for el in elements:
    char_images = []
    for ch in el:
        if 'A' <= ch <= 'Z':
            folder = letters_base / f"{ch}_cap"
        elif 'a' <= ch <= 'z':
            folder = letters_base / ch
        else:
            continue  # 忽略非字母字符

        if not folder.exists() or not any(folder.iterdir()):
            print(f"警告：文件夹为空或不存在 {folder}")
            continue

        img_file = random.choice(list(folder.glob('*.png')))
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        char_images.append(img)

    if char_images:
        element_img = concatenate_images(char_images)
        if element_img is not None:
            save_path = output_dir / f"{el}.png"
            cv2.imwrite(str(save_path), element_img)
            print(f"已生成 {save_path}")
