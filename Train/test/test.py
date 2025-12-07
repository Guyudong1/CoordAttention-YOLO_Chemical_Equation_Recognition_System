from ultralytics import YOLO
import cv2
import os
import numpy as np

CELL_MODEL_PATH = "../../Model/chemical_equations_training/cell_only_training/weights/best.pt"

INPUT_DIR = "../test_img_1"
OUTPUT_DIR = "../test_res_1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载 Cell 模型
cell_model = YOLO(CELL_MODEL_PATH)

# 支持的图片类型
valid_ext = [".jpg", ".jpeg", ".png", ".bmp"]

# 遍历输入文件夹
for filename in os.listdir(INPUT_DIR):
    if not any(filename.lower().endswith(ext) for ext in valid_ext):
        continue

    img_path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(img_path)

    if img is None:
        print("加载失败：", img_path)
        continue

    # ----------------------------
    # 1. cell 检测
    # ----------------------------
    results = cell_model(img)[0]
    boxes = results.boxes.xyxy.cpu().numpy()

    # 创建掩码图像（纯白背景）
    mask_img = np.ones_like(img) * 255  # 纯白背景

    # ----------------------------
    # 2. 绘制红色 cell 框（原图）
    # ----------------------------
    img_with_boxes = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # ----------------------------
    # 3. 绘制掩码图（灰色半透明区域）
    # ----------------------------
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # 创建灰色半透明掩码
        gray_color = (128, 128, 128)  # 灰色
        alpha = 0.5  # 透明度

        # 提取当前box区域
        roi = mask_img[y1:y2, x1:x2]

        # 创建与ROI相同大小的颜色层
        color_layer = np.full_like(roi, gray_color, dtype=np.uint8)

        # 混合原区域与灰色（alpha混合）
        blended_roi = cv2.addWeighted(roi, 1 - alpha, color_layer, alpha, 0)

        # 将混合后的区域放回掩码图
        mask_img[y1:y2, x1:x2] = blended_roi

        # 可选：在掩码图上添加边界框以便更清晰
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), (0, 0, 0), 1)

    # ----------------------------
    # 4. 保存带框图片和掩码图片
    # ----------------------------
    # 保存带红色框的原图
    box_save_path = os.path.join(OUTPUT_DIR, f"cell_{filename}")
    cv2.imwrite(box_save_path, img_with_boxes)

    # 保存掩码图
    mask_save_path = os.path.join(OUTPUT_DIR, f"mask_{filename}")
    cv2.imwrite(mask_save_path, mask_img)

    print(f"已处理并保存: {box_save_path}")
    print(f"已处理并保存: {mask_save_path}")

print("\n全部完成！")