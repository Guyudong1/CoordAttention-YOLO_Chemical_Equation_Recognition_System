from ultralytics import YOLO
import os
import pandas as pd

def train_yolov11_elements_augmented():
    # 设置工作目录到模型文件夹
    os.chdir(r"E:\sophomore\study\pytorch\pythonProject\MachineLearning\Chemical equation recognition\Model")

    # 使用 YOLOv11 nano 预训练模型
    model = YOLO('yolo11n.pt')

    # 训练
    results = model.train(
        data=r"..\data\elements.yaml",  # 增强后的60x60数据集配置
        epochs=10,                 # 根据效果可调整
        imgsz=60,                  # 输入图片尺寸为60x60
        batch=8,
        device='0',                # GPU id
        project='chemical_elements_training',
        name='phase2_elements_augmented_60x60',
        exist_ok=True,
        augment=True,              # 数据增强
        optimizer='SGD',           # 小目标训练SGD更稳
        lr0=0.01,
        patience=10
    )

    # 输出训练指标
    print("\n=== 训练指标 ===")
    if hasattr(results, 'metrics') and results.metrics is not None:
        for key, value in results.metrics.items():
            print(f"{key}: {value}")

    # 尝试读取训练 CSV 日志
    try:
        csv_path = os.path.join('chemical_elements_training', 'phase2_elements_augmented_60x60', 'results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print("\n=== 最近5个epoch ===")
            print(df.tail())

            # 最后5个epoch平均指标
            last_5 = df.tail(5)
            avg_map50 = last_5.get('metrics/mAP50(B)', last_5.get('metrics/mAP_0.5', pd.Series([0]))).mean()
            avg_map5095 = last_5.get('metrics/mAP50-95(B)', last_5.get('metrics/mAP_0.5:0.95', pd.Series([0]))).mean()
            avg_precision = last_5.get('metrics/precision(B)', last_5.get('metrics/precision', pd.Series([0]))).mean()
            avg_recall = last_5.get('metrics/recall(B)', last_5.get('metrics/recall', pd.Series([0]))).mean()

            print(f"\n=== 最近5个epoch平均指标 ===")
            print(f"平均 mAP50: {avg_map50:.4f}")
            print(f"平均 mAP50-95: {avg_map5095:.4f}")
            print(f"平均 precision: {avg_precision:.4f}")
            print(f"平均 recall: {avg_recall:.4f}")
    except Exception as e:
        print(f"读取日志文件时出错: {e}")


if __name__ == "__main__":
    train_yolov11_elements_augmented()
