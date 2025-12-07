from ultralytics import YOLO
import cv2
import os
import numpy as np

CELL_MODEL_PATH = "../../Model/chemical_equations_training/cell_only_training_5struct/weights/best.pt"

INPUT_DIR = "../test_img_1"
OUTPUT_DIR = "../test_res_2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# YOLO 类别映射（根据你的模型）
id_to_label = {
    0: 'sub',   # 下标
    1: '+',     # 加号
    2: 'aR',    # arrowR
    3: 'a2',    # arrow2
    4: 'ot'     # 其他
}

# 每类颜色（BGR）
color_mapping = {
    'sub': (0, 0, 255),      # 红色
    '+':   (0, 255, 0),      # 绿色
    'aR':  (255, 0, 0),      # 蓝色
    'a2':  (255, 0, 0),      # 蓝色
    'ot':  (128, 128, 128)   # 灰色
}

# 加载模型
cell_model = YOLO(CELL_MODEL_PATH)

valid_ext = [".jpg", ".jpeg", ".png", ".bmp"]

for filename in os.listdir(INPUT_DIR):
    if not any(filename.lower().endswith(ext) for ext in valid_ext):
        continue

    img_path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(img_path)

    if img is None:
        print("加载失败：", img_path)
        continue

    # ----------------------------
    # 1. YOLO 预测
    # ----------------------------
    results = cell_model(img)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    # ----------------------------
    # 2. 创建白色背景掩码
    # ----------------------------
    mask_img = np.ones_like(img) * 255

    # ----------------------------
    # 3. 在原图画框（不同类别不同颜色）
    # ----------------------------
    img_with_boxes = img.copy()

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)

        label_name = id_to_label.get(cls, 'ot')
        color = color_mapping[label_name]

        # 画框
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_with_boxes, label_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ----------------------------
    # 4. 掩码图绘制
    # ----------------------------
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)

        label_name = id_to_label.get(cls, 'ot')
        color = color_mapping[label_name]

        # 半透明填充
        alpha = 0.5
        roi = mask_img[y1:y2, x1:x2]
        color_layer = np.full_like(roi, color, dtype=np.uint8)
        blended_roi = cv2.addWeighted(roi, 1 - alpha, color_layer, alpha, 0)
        mask_img[y1:y2, x1:x2] = blended_roi

        # 边框
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), (0, 0, 0), 1)

    # ----------------------------
    # 5. 保存
    # ----------------------------
    box_save_path = os.path.join(OUTPUT_DIR, f"cell_{filename}")
    mask_save_path = os.path.join(OUTPUT_DIR, f"mask_{filename}")

    cv2.imwrite(box_save_path, img_with_boxes)
    cv2.imwrite(mask_save_path, mask_img)

    print(f"已处理并保存: {box_save_path}")
    print(f"已处理并保存: {mask_save_path}")

print("\n全部完成！")
