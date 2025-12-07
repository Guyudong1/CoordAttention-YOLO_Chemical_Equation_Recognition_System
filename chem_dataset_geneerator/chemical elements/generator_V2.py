import cv2
import numpy as np
import random
from pathlib import Path

# ========== 参数配置 ==========
txt_path = 'elements.txt'           # 元素列表
letters_base = Path('raw_letters')  # 原始字母数据集
save_dir = Path('generated_elements_V2')  # 输出保存路径
target_size = (45, 45)              # 输出图像尺寸
gap = 2                             # 字母间距（像素）
scale_small = 1                 # 第二字母缩放比例（越小越矮）
show_time = 100                     # 每张图显示毫秒数
# =================================

save_dir.mkdir(exist_ok=True)

def trim_white(img, thresh=250):
    """裁掉图像四周白边，只保留字母区域"""
    coords = cv2.findNonZero(255 - img)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]

def pad_to_height(img, h):
    """上下填充以对齐高度"""
    pad_top = (h - img.shape[0]) // 2
    pad_bottom = h - img.shape[0] - pad_top
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=255)

def combine_and_resize(img1, img2, gap=2, target_size=(45, 45), scale_small=0.7):
    """将两张图像紧密拼接，第二张更小，再缩放回目标尺寸"""
    img1 = trim_white(img1)
    img2 = trim_white(img2)

    # 缩小第二个字母
    h2, w2 = img2.shape
    img2 = cv2.resize(img2, (int(w2 * scale_small), int(h2 * scale_small)), interpolation=cv2.INTER_AREA)

    max_h = max(img1.shape[0], img2.shape[0])
    img1 = pad_to_height(img1, max_h)
    img2 = pad_to_height(img2, max_h)

    combined = np.hstack([img1, 255 * np.ones((max_h, gap), dtype=np.uint8), img2])
    resized = cv2.resize(combined, target_size, interpolation=cv2.INTER_AREA)
    return resized

# 读取元素列表
with open(txt_path, 'r') as f:
    elements = [line.strip() for line in f if line.strip()]

# 遍历生成
for el in elements:
    if len(el) == 1:
        folder = letters_base / (f"{el}_cap" if el.isupper() else el)
        if not folder.exists():
            print(f"缺少 {folder}")
            continue
        img_file = random.choice(list(folder.glob('*.png')))
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        result = cv2.resize(img, target_size)
    else:
        imgs = []
        for ch in el:
            folder = letters_base / (f"{ch}_cap" if ch.isupper() else ch)
            if not folder.exists():
                print(f"缺少 {folder}")
                break
            img_file = random.choice(list(folder.glob('*.png')))
            imgs.append(cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE))
        if len(imgs) == 2:
            result = combine_and_resize(imgs[0], imgs[1], gap=gap, target_size=target_size, scale_small=scale_small)
        else:
            continue

    # 保存
    save_path = save_dir / f"{el}.png"
    cv2.imwrite(str(save_path), result)
    print(f"已生成 {el}.png")

    # 显示效果
    cv2.imshow(el, result)
    cv2.waitKey(show_time)
    cv2.destroyWindow(el)

print("全部元素已生成完毕！")
