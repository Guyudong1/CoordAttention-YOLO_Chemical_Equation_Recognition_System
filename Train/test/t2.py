from ultralytics import YOLO
import cv2
import os

# ------------------------
# 配置
# ------------------------
CELL_MODEL_PATH = '../../Model/chemical_equations_training/cell_only_training/weights/best.pt'
IMAGE_PATH = '../test_img/equation_10.jpg'
SAVE_DIR = '../cropped_cells'
IOU_THRESHOLD = 0.75   # IOU 超过这个值就视为重复，只保留大框

# ------------------------
# IoU 计算函数
# ------------------------
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

    if union_area == 0:
        return 0

    return inter_area / union_area


# ------------------------
# 根据 IOU 去除重复框（只保留大的）
# ------------------------
def filter_overlapping_boxes(boxes):
    kept = []

    for box in boxes:
        keep_flag = True
        x1, y1, x2, y2 = box
        area1 = (x2 - x1) * (y2 - y1)

        for k in kept:
            x3, y3, x4, y4 = k
            area2 = (x4 - x3) * (y4 - y3)

            iou = compute_iou(box, k)
            if iou > IOU_THRESHOLD:
                # 如果两个框高度重叠，则保留面积大的
                if area1 <= area2:
                    keep_flag = False
                else:
                    kept.remove(k)
                break

        if keep_flag:
            kept.append(box)

    return kept


# ------------------------
# 创建保存目录
# ------------------------
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------
# 加载模型
# ------------------------
cell_model = YOLO(CELL_MODEL_PATH)

# ------------------------
# 预测格子
# ------------------------
cell_results = cell_model(
    IMAGE_PATH,
    conf=0.5,   # 置信度阈值，低于这个的框不会输出
    iou=0.3,    # NMS IOU 阈值，重复框会被抑制
)
img = cv2.imread(IMAGE_PATH)

boxes = [list(map(int, box.xyxy[0])) for box in cell_results[0].boxes]

# 排序
boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

# 去除大小框重叠重复
clean_boxes = filter_overlapping_boxes(boxes)

# ------------------------
# 裁剪并保存
# ------------------------
for i, (x1, y1, x2, y2) in enumerate(clean_boxes):
    cropped = img[y1:y2, x1:x2]
    save_path = os.path.join(SAVE_DIR, f"cell_{i+1:03d}.jpg")
    cv2.imwrite(save_path, cropped)
    print(f"已保存：{save_path}")

print("\n所有格子已裁剪并保存完成！")
