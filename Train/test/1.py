import os
from ultralytics import YOLO
import cv2
import numpy as np

# ------------------------
# 配置参数
# ------------------------
CELL_MODEL_PATH = '../../Model/chemical_equations_training/cell_only_training/weights/best.pt'  # 红格检测模型
ELEMENT_MODEL_PATH = '../../Model/chemical_elements_training/phase2_elements_augmented_60x60/weights/best.pt'  # 绿格检测模型
IMAGE_PATH = '../../recognition/test_res2/processed_tt(1).jpg'  # 输入图片路径
SAVE_DIR = '../cropped_cells'  # 临时保存裁剪格子的目录
IOU_THRESHOLD = 0.75  # IOU 超过这个值就视为重复，只保留大框


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
# 小标签、不遮挡的绘制函数（只显示标签，不显示置信度）
# ------------------------
def draw_results_small_labels(image, boxes, scores, class_ids, class_names, color=(0, 255, 0)):
    img = image.copy()

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)

        # 绘制边框（厚度为1）
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        # 只显示标签，不显示置信度
        label = f"{class_names[class_id]}"

        # 小字体
        text_scale = 0.5
        text_thickness = 1

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0]

        # 标签放在上方，不遮住内容
        label_x = x1
        label_y = max(y1 - 5, 15)

        # 半透明背景
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (label_x, label_y - text_size[1] - 5),
            (label_x + text_size[0] + 5, label_y),
            color,
            -1
        )

        # 透明度处理
        alpha = 0.6
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # 写文字
        cv2.putText(
            img, label,
            (label_x + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale, (255, 255, 255), text_thickness
        )

    return img


# ------------------------
# 主程序
# ------------------------
def main():
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("步骤1: 加载模型...")
    # 加载红格检测模型（单元格检测）
    cell_model = YOLO(CELL_MODEL_PATH)
    # 加载绿格检测模型（元素检测）
    element_model = YOLO(ELEMENT_MODEL_PATH)

    print("步骤2: 读取输入图片...")
    # 读取原始图片
    original_img = cv2.imread(IMAGE_PATH)
    if original_img is None:
        print(f"错误: 无法读取图片 {IMAGE_PATH}")
        return

    img_height, img_width = original_img.shape[:2]
    print(f"图片尺寸: {img_width} x {img_height}")

    print("步骤3: 检测红格（单元格）...")
    # 预测格子
    cell_results = cell_model(
        IMAGE_PATH,
        conf=0.5,  # 置信度阈值
        iou=0.3,  # NMS IOU 阈值
    )

    # 提取红格框
    red_boxes = [list(map(int, box.xyxy[0])) for box in cell_results[0].boxes]
    print(f"检测到 {len(red_boxes)} 个红格")

    # 排序（从上到下，从左到右）
    red_boxes = sorted(red_boxes, key=lambda b: (b[1], b[0]))

    # 去除重叠重复框
    clean_red_boxes = filter_overlapping_boxes(red_boxes)
    print(f"去重后剩余 {len(clean_red_boxes)} 个红格")

    # 在原始图片上绘制红格检测结果
    red_detection_img = original_img.copy()
    for i, (x1, y1, x2, y2) in enumerate(clean_red_boxes):
        cv2.rectangle(red_detection_img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 红色框，厚度为1
        cv2.putText(red_detection_img, f'Cell {i + 1}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)  # 文字厚度也为1

    print("步骤4: 裁剪红格并检测绿格（元素）...")
    # 存储所有绿格检测结果
    all_green_results = []

    for i, (x1, y1, x2, y2) in enumerate(clean_red_boxes):
        # 裁剪红格区域
        cropped_cell = original_img[y1:y2, x1:x2]
        if cropped_cell.size == 0:
            continue

        # 保存临时文件用于元素检测
        temp_cell_path = os.path.join(SAVE_DIR, f"cell_{i + 1:03d}.jpg")
        cv2.imwrite(temp_cell_path, cropped_cell)

        print(f"  检测红格 {i + 1} 中的绿格...")

        # 在裁剪的红格中检测绿格（元素）
        element_results = element_model(temp_cell_path)

        if len(element_results[0].boxes) > 0:
            # 提取绿格检测结果并转换坐标到原始图片
            boxes = element_results[0].boxes.xyxy.cpu().numpy()
            scores = element_results[0].boxes.conf.cpu().numpy()
            class_ids = element_results[0].boxes.cls.cpu().numpy().astype(int)

            # 将坐标转换回原始图片坐标系
            for j in range(len(boxes)):
                boxes[j][0] += x1  # x1
                boxes[j][1] += y1  # y1
                boxes[j][2] += x1  # x2
                boxes[j][3] += y1  # y2

            # 存储结果
            for box, score, class_id in zip(boxes, scores, class_ids):
                all_green_results.append({
                    'box': box,
                    'score': score,
                    'class_id': class_id
                })

    print(f"总共检测到 {len(all_green_results)} 个绿格（元素）")

    print("步骤5: 生成最终结果图像...")
    # 创建最终结果图像（红格 + 绿格）
    final_result_img = original_img.copy()

    # 首先绘制红格（厚度为1）
    for x1, y1, x2, y2 in clean_red_boxes:
        cv2.rectangle(final_result_img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 红色框，厚度为1

    # 然后绘制绿格
    if all_green_results:
        green_boxes = [result['box'] for result in all_green_results]
        green_scores = [result['score'] for result in all_green_results]
        green_class_ids = [result['class_id'] for result in all_green_results]
        green_class_names = element_model.names

        # 使用小标签绘制绿格（只显示标签，不显示置信度）
        final_result_img = draw_results_small_labels(
            final_result_img, green_boxes, green_scores, green_class_ids, green_class_names, color=(0, 255, 0)
        )

    print("步骤6: 显示结果...")
    # 显示所有结果
    cv2.imshow("1. 原始图片", original_img)
    cv2.imshow("2. 红格检测结果", red_detection_img)
    cv2.imshow("3. 最终结果（红格+绿格）", final_result_img)

    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 可选：保存结果图片
    cv2.imwrite("../final_detection_result.jpg", final_result_img)
    print("最终结果已保存为: ../final_detection_result.jpg")

    # 清理临时文件
    print("清理临时文件...")
    for fname in os.listdir(SAVE_DIR):
        if fname.startswith("cell_") and fname.endswith(".jpg"):
            os.remove(os.path.join(SAVE_DIR, fname))


if __name__ == '__main__':
    main()