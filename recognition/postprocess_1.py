import os
import cv2
import numpy as np
from ultralytics import YOLO

# ===============================================
# 配置区域
# ===============================================
CELL_MODEL_PATH = "../Model/chemical_equations_training/cell_only_training_5struct/weights/best.pt"
ELEMENT_MODEL_PATH = "../Model/chemical_elements_training/phase2_elements_augmented_3/weights/best.pt"

BIG_IOU_THRESHOLD = 0.75
SMALL_IOU_THRESHOLD = 0.5

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

special_no_merge = {"+", "aR", "a2"}   # 这些结构标签不与元素合并


# ===============================================
# IoU 计算
# ===============================================
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


# ===============================================
# 大框去重
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
# 小框去重：NMS
# ===============================================
def filter_small_boxes(boxes, scores, threshold=SMALL_IOU_THRESHOLD):
    order = np.argsort(scores)[::-1]
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
# 绘制函数
# ===============================================
def draw_big_box(img, box, color):
    x1, y1, x2, y2 = map(int, box)
    overlay = img.copy()

    alpha = 0.35
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)

    img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_small_box(img, box, cls_name, ox, oy):
    x1, y1, x2, y2 = map(int, box)
    x1 += ox
    x2 += ox
    y1 += oy
    y2 += oy

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(img, cls_name, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)


# ===============================================
# 单图处理：返回 tokens + 保存图像
# ===============================================
def process_single_image(img_path, output_folder, cell_model, element_model):

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read: {img_path}")

    H, W = img.shape[:2]

    # -------- 大框检测 --------
    cell_results = cell_model(img)[0]
    boxes = cell_results.boxes.xyxy.cpu().numpy().astype(int)
    classes = cell_results.boxes.cls.cpu().numpy().astype(int)

    if len(boxes) == 0:
        print("[Warning] No big box detected")
        return img, []

    big_boxes = filter_big_boxes(boxes)

    # 大框对应结构标签
    big_box_classes = []
    for bb in big_boxes:
        for i, ori in enumerate(boxes):
            if np.array_equal(bb, ori):
                big_box_classes.append(id_to_label.get(classes[i], "ot"))
                break

    # 绘制大框
    for i, box in enumerate(big_boxes):
        draw_big_box(img, box, color_mapping[big_box_classes[i]])

    # -------- 小框检测 + token 收集 --------
    collected_tokens = []

    for bi, box in enumerate(big_boxes):
        x1, y1, x2, y2 = box
        struct_label = big_box_classes[bi]

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        result = element_model(crop, conf=0.15)[0]
        elem_boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        elem_ids = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        names = element_model.names

        if len(elem_boxes) == 0:
            continue

        indices = filter_small_boxes(elem_boxes, scores)

        for idx in indices:
            element_label = names[elem_ids[idx]]

            if struct_label in special_no_merge:
                final_label = struct_label
            else:
                final_label = f"{struct_label}({element_label})"

            # ------ token 收集 ------
            ex1, ey1, ex2, ey2 = elem_boxes[idx]
            cx = x1 + (ex1 + ex2) / 2
            cy = y1 + (ey1 + ey2) / 2
            collected_tokens.append((final_label, cx, cy))

            draw_small_box(img, elem_boxes[idx], final_label, x1, y1)

    # 最终排序
    collected_tokens.sort(key=lambda t: t[1])

    # 保存图像
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(save_path, img)

    return img, collected_tokens


# ===============================================
# 主入口：处理 1 张图片
# ===============================================
if __name__ == "__main__":
    INPUT_IMAGE = "../Train/test_img_1/equation_7.jpg"
    OUTPUT_FOLDER = "post_res2"

    print("Loading models...")
    cell_model = YOLO(CELL_MODEL_PATH)
    element_model = YOLO(ELEMENT_MODEL_PATH)

    print("\nProcessing one image...\n")
    image, tokens = process_single_image(INPUT_IMAGE, OUTPUT_FOLDER,
                                         cell_model, element_model)

    print("\n========== TOKEN LIST ==========")
    for t in tokens:
        print(t)

    print("\n处理完成！图片已保存。")
