import os
from ultralytics import YOLO
import cv2
import numpy as np

# -----------------------
# 小标签、不遮挡的绘制函数
# -----------------------
def draw_results_small_labels(image, boxes, scores, class_ids, class_names):
    img = image.copy()

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)

        # 绘制边框（细线）
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # 标签内容
        label = f"{class_names[class_id]} {score:.2f}"

        # 小字体
        text_scale = 0.4
        text_thickness = 1

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0]

        # 标签放在上方，不遮住内容
        label_x = x1
        label_y = max(y1 - 5, 10)

        # 半透明背景
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (label_x, label_y - text_size[1] - 4),
            (label_x + text_size[0] + 4, label_y),
            (0, 255, 0),
            -1
        )

        # 透明度处理
        alpha = 0.3
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # 写文字
        cv2.putText(
            img, label,
            (label_x + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale, (0, 0, 0), text_thickness
        )

    return img


# -----------------------
# 主程序：遍历文件夹检测
# -----------------------
def main():
    # 模型路径
    model = YOLO('../Model/chemical_elements_training/phase2_elements_augmented_60x60/weights/best.pt')

    # 要遍历的图片文件夹
    folder_path = "cropped_cells"

    # 支持的图片格式
    exts = ['.jpg', '.jpeg', '.png', '.bmp']

    for fname in os.listdir(folder_path):
        if not any(fname.lower().endswith(ext) for ext in exts):
            continue  # 跳过非图片文件

        img_path = os.path.join(folder_path, fname)
        print(f"\n正在检测：{img_path}")

        # YOLO 预测
        results = model(img_path)[0]

        # 读取图像
        img = cv2.imread(img_path)

        # YOLO 结果提取
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        class_names = model.names

        # 可视化标注（小标签版本）
        vis = draw_results_small_labels(img, boxes, scores, class_ids, class_names)

        # 显示
        cv2.imshow("Detected", vis)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
