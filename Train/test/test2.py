import cv2
from ultralytics import YOLO
import numpy as np
import os

# ===============================================
# 配置区域
# ===============================================
CELL_MODEL_PATH = "../../Model/chemical_equations_training/cell_only_training_5struct/weights/best.pt"
ELEMENT_MODEL_PATH = "../../Model/chemical_elements_training/phase2_elements_augmented_3/weights/best.pt"

INPUT_FOLDER = "../test_img_1"
OUTPUT_FOLDER = "../test_res_11"

BIG_IOU_THRESHOLD = 0.75
SMALL_IOU_THRESHOLD = 0.4


# ===============================================
# 类别映射
# ===============================================
id_to_label = {
    0: "sub",
    1: "+",
    2: "aR",
    3: "a2",
    4: "ot",
}

color_mapping = {
    "sub": (0, 0, 255),
    "+": (0, 255, 0),
    "aR": (255, 0, 0),
    "a2": (255, 0, 0),
    "ot": (128, 128, 128)
}


# ===============================================
# IoU（修复后）
# ===============================================
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)   # ← 修复了你的错误（你之前写成 y3）

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


# ===============================================
# 大框去重（保留大框）
# ===============================================
def filter_big_boxes(boxes, threshold=BIG_IOU_THRESHOLD):
    kept = []
    for box in boxes:
        keep = True
        x1, y1, x2, y2 = box
        area1 = (x2 - x1) * (y2 - y1)

        for k in kept:
            x3, y3, x4, y4 = k
            area2 = (x4 - x3) * (y4 - y3)
            iou = compute_iou(box, k)

            if iou > threshold:
                if area1 <= area2:
                    keep = False
                else:
                    kept.remove(k)
                break

        if keep:
            kept.append(box)

    return kept


# ===============================================
# 小框去重（真正有效的 NMS）
# ===============================================
def filter_small_boxes(boxes, scores, threshold=SMALL_IOU_THRESHOLD):
    order = np.argsort(scores)[::-1]  # 按置信度从高到低排序
    kept = []
    used = [False] * len(boxes)

    for i in order:
        if used[i]:
            continue

        kept.append(i)
        for j in order:
            if used[j] or i == j:
                continue
            if compute_iou(boxes[i], boxes[j]) > threshold:
                used[j] = True

    return kept


# ===============================================
# 大框绘制
# ===============================================
def draw_big_box(img, box, color):
    x1, y1, x2, y2 = map(int, box)
    overlay = img.copy()

    alpha = 0.35
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)

    img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


# ===============================================
# 小框绘制
# ===============================================
def draw_small_box(img, box, cls_name, ox, oy):
    x1, y1, x2, y2 = map(int, box)
    x1 += ox
    x2 += ox
    y1 += oy
    y2 += oy

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(img, cls_name, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


# ===============================================
# 单图完整流程
# ===============================================
def process_single_image(img_path, cell_model, element_model):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Error] Cannot read: {img_path}")
        return None

    H, W = img.shape[:2]

    # ======================
    # 大框检测
    # ======================
    cell_results = cell_model(img)[0]
    boxes = cell_results.boxes.xyxy.cpu().numpy().astype(int)
    classes = cell_results.boxes.cls.cpu().numpy().astype(int)

    if len(boxes) == 0:
        print(f"[Warning] No big box in {img_path}")
        return img

    big_boxes = filter_big_boxes(boxes)

    # 绘制大框
    for i, box in enumerate(big_boxes):
        x1, y1, x2, y2 = box

        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))

        cls_name = id_to_label.get(classes[i], "ot")
        color = color_mapping[cls_name]

        draw_big_box(img, (x1, y1, x2, y2), color)

    # ======================
    # 小框检测
    # ======================
    for box in big_boxes:
        x1, y1, x2, y2 = box

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        result = element_model(crop, conf=0.10)[0]

        elem_boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        elem_ids = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        names = element_model.names

        if len(elem_boxes) == 0:
            continue

        # 🔥真正有效的小框去重
        indices = filter_small_boxes(elem_boxes, scores)

        for idx in indices:
            draw_small_box(img, elem_boxes[idx], names[elem_ids[idx]], x1, y1)

    return img


# ===============================================
# 批量处理文件夹
# ===============================================
if __name__ == "__main__":
    print("Loading models...")
    cell_model = YOLO(CELL_MODEL_PATH)
    element_model = YOLO(ELEMENT_MODEL_PATH)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("Processing folder...")

    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(INPUT_FOLDER, filename)
        print(f"Processing: {filename}")

        try:
            out_img = process_single_image(img_path, cell_model, element_model)
            if out_img is not None:
                save_path = os.path.join(OUTPUT_FOLDER, filename)
                cv2.imwrite(save_path, out_img)
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

    print("\n全部处理完成！")
