from ultralytics import YOLO
import cv2
import numpy as np
# from preprocess import preprocess

# ------------------------
# 绘制小元素标签
# ------------------------
def draw_element_box_on_big_image(big_img, box, class_id, class_names, offset_x, offset_y):
    (x1, y1, x2, y2) = map(int, box)
    x1 += offset_x
    x2 += offset_x
    y1 += offset_y
    y2 += offset_y

    cv2.rectangle(big_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    label = f"{class_names[class_id]}"
    scale = 0.4
    thickness = 1

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    label_y = max(y1 - 5, 10)

    overlay = big_img.copy()
    cv2.rectangle(
        overlay,
        (x1, label_y - t_size[1] - 4),
        (x1 + t_size[0] + 4, label_y),
        (0, 255, 0),
        -1
    )
    big_img[:] = cv2.addWeighted(overlay, 0.3, big_img, 0.7, 0)
    cv2.putText(big_img, label, (x1 + 2, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness)

# ------------------------
# 判断一个 box 是否被其他 box 覆盖比例超过阈值
# ------------------------
def discard_covered_boxes(boxes, cover_thr=0.7):
    boxes = np.array(boxes)
    keep = []

    for i, box_a in enumerate(boxes):
        x1a, y1a, x2a, y2a = box_a
        area_a = (x2a - x1a) * (y2a - y1a)
        discard_flag = False

        for j, box_b in enumerate(boxes):
            if i == j:
                continue
            x1b, y1b, x2b, y2b = box_b
            # 计算交集
            inter_x1 = max(x1a, x1b)
            inter_y1 = max(y1a, y1b)
            inter_x2 = min(x2a, x2b)
            inter_y2 = min(y2a, y2b)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            if inter_area / area_a > cover_thr:
                discard_flag = True
                break

        if not discard_flag:
            keep.append(box_a)

    return np.array(keep)

# ------------------------
# 主流程
# ------------------------
def run_once(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"❌ 无法加载图像：{img_path}")

    processed_img = img.copy()
    # processed_img = preprocess(img)
    if len(processed_img.shape) == 2:
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

    h, w = processed_img.shape[:2]

    cell_model = YOLO('../Model/chemical_equations_training/cell_only_training/weights/best.pt')
    element_model = YOLO("../Model/chemical_elements_training/phase2_elements_augmented_60x60_transformer/weights/best.pt")

    draw_img = processed_img.copy()

    # ---------- Step1: cell 检测 ----------
    cell_results = cell_model(processed_img)[0]

    cell_boxes = cell_results.boxes.xyxy.cpu().numpy()
    cell_boxes = sorted(cell_boxes, key=lambda b: (b[1], b[0]))

    # ---------- Step2: 去掉被覆盖红框 ----------
    cell_boxes = discard_covered_boxes(cell_boxes, cover_thr=0.7)

    # ---------- Step3: 绘制红框 ----------
    for box in cell_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # ---------- Step4: element 检测 ----------
    for box in cell_boxes:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = processed_img[y1:y2, x1:x2]
        elem_results = element_model.predict(crop)[0]

        elem_boxes = elem_results.boxes.xyxy.cpu().numpy()
        elem_scores = elem_results.boxes.conf.cpu().numpy()
        elem_ids = elem_results.boxes.cls.cpu().numpy().astype(int)
        class_names = element_model.names

        if len(elem_scores) > 0:
            best_idx = np.argmax(elem_scores)
            best_box = elem_boxes[best_idx]
            best_id = elem_ids[best_idx]
            draw_element_box_on_big_image(draw_img, best_box, best_id, class_names, x1, y1)

    return draw_img

# ------------------------
# main
# ------------------------
def main():
    img_path = r"test_res2/processed_tt(1).jpg"
    result = run_once(img_path)
    cv2.imshow("YOLO Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
