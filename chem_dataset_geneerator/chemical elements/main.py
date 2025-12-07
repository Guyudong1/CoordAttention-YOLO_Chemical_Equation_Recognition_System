import cv2
import numpy as np
import random
from pathlib import Path

# ========== 参数配置 ==========
txt_path = 'elements.txt'
letters_base = Path('raw_letters')
save_root = Path('generated_dataset')
target_size = (45, 45)
samples_per_class = 500
gap_min, gap_max = 2, 3        # 字母间距随机范围
base_scale_small = 0.7
dy = 6.5
# =================================

save_root.mkdir(exist_ok=True)

# 特殊字母调整表
LETTER_ADJUST = {
    'g': {'scale': 1.1, 'dy': 10},
    'y': {'scale': 1.1, 'dy': 10},
    'p': {'scale': 1.0, 'dy': 10},
    'q': {'scale': 1.0, 'dy': 10},
    'd': {'scale': 1.1, 'dy': 0},
    'b': {'scale': 1.1, 'dy': 0},
    'h': {'scale': 1.1, 'dy': 0},
    'l': {'scale': 1.1, 'dy': 0},
    'f': {'scale': 1.1, 'dy': 0},
}

# ========== 工具函数 ==========
def load_gray_image(path):
    """兼容 RGBA 的安全加载函数（把透明背景变白）"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:  # RGBA -> 灰度
        alpha = img[:, :, 3] / 255.0
        img_rgb = img[:, :, :3]
        white_bg = np.ones_like(img_rgb, dtype=np.uint8) * 255
        img_rgb = (img_rgb * alpha[..., None] + white_bg * (1 - alpha[..., None])).astype(np.uint8)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray

def trim_white(img, thresh=250):
    coords = cv2.findNonZero(255 - img)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]

def pad_to_height(img, h, dy):
    pad_top = (h - img.shape[0]) // 2 + dy
    pad_bottom = h - img.shape[0] - pad_top
    pad_top = max(0, pad_top)
    pad_bottom = max(0, pad_bottom)
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=255)

def combine_on_canvas(img1, img2, gap=2, canvas_size=(60, 60), target_size=(45, 45), letter2=None):
    """在较大画布上拼接后整体缩小"""
    img1 = trim_white(img1)
    img2 = trim_white(img2)

    # 个性化缩放与偏移
    if letter2 and letter2 in LETTER_ADJUST:
        s = base_scale_small * LETTER_ADJUST[letter2]['scale']
        dy = LETTER_ADJUST[letter2]['dy']
    else:
        s = base_scale_small
        dy = 0

    h2, w2 = img2.shape
    img2 = cv2.resize(img2, (max(1, int(w2 * s)), max(1, int(h2 * s))), interpolation=cv2.INTER_AREA)

    max_h = max(img1.shape[0], img2.shape[0])
    img1 = pad_to_height(img1, max_h, 0)
    img2 = pad_to_height(img2, max_h, dy)

    canvas = np.ones(canvas_size, dtype=np.uint8) * 255

    gap = random.randint(gap_min, gap_max)
    total_w = img1.shape[1] + img2.shape[1] + gap
    if total_w > canvas_size[1]:
        scale = (canvas_size[1] - gap) / total_w
        img1 = cv2.resize(img1, (max(1, int(img1.shape[1]*scale)), max(1, int(img1.shape[0]*scale))))
        img2 = cv2.resize(img2, (max(1, int(img2.shape[1]*scale)), max(1, int(img2.shape[0]*scale))))
        max_h = max(img1.shape[0], img2.shape[0])

    y1 = (canvas_size[0] - max_h) // 2
    x1 = (canvas_size[1] - total_w) // 2
    if x1 < 0: x1 = 0
    x2 = x1 + img1.shape[1] + gap

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    try:
        canvas[y1:y1+h1, x1:x1+w1] = np.minimum(canvas[y1:y1+h1, x1:x1+w1], img1)
        canvas[y1:y1+h2, x2:x2+w2] = np.minimum(canvas[y1:y1+h2, x2:x2+w2], img2)
    except ValueError:
        print(f"⚠️ 拼接失败：{letter2}, 可能尺寸越界")
        return None

    return cv2.resize(canvas, target_size, interpolation=cv2.INTER_AREA)

# ========== 主循环 ==========
with open(txt_path, 'r', encoding='utf-8') as f:
    elements = [line.strip() for line in f if line.strip()]

for el in elements:
    save_dir = save_root / el
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在生成 {el} 的样本...")
    for i in range(samples_per_class):
        if len(el) == 1:
            folder = letters_base / (f"{el}_cap" if el.isupper() else el)
            if not folder.exists():
                print(f"缺少 {folder}")
                continue
            img_file = random.choice(list(folder.glob('*.png')))
            img = load_gray_image(img_file)
            if img is None: continue
            result = cv2.resize(img, target_size)
        else:
            imgs = []
            for ch in el:
                folder = letters_base / (f"{ch}_cap" if ch.isupper() else ch)
                if not folder.exists():
                    print(f"缺少 {folder}")
                    break
                img_file = random.choice(list(folder.glob('*.png')))
                img = load_gray_image(img_file)
                if img is None or img.size == 0:
                    print(f"⚠️ 空图: {img_file}")
                    break
                imgs.append(img)
            if len(imgs) == 2:
                result = combine_on_canvas(
                    imgs[0], imgs[1],
                    gap=random.randint(gap_min, gap_max),
                    canvas_size=(60, 60),
                    target_size=target_size,
                    letter2=el[1]
                )
                if result is None:
                    continue
            else:
                continue

        save_path = save_dir / f"{el}_{i:03d}.png"
        cv2.imwrite(str(save_path), result)

    print(f"{el} 共生成 {samples_per_class} 张样本完成。")

print("全部元素数据集已生成完毕！")
