import os
import pandas as pd
from ultralytics import YOLO

def train_yolov11_elements_augmented_transformer():
    # 设置工作目录到模型文件夹
    os.chdir(r"E:\sophomore\study\pytorch\pythonProject\MachineLearning\Chemical equation recognition\Model")

    # 使用自定义 YAML 初始化模型（带 Transformer）
    model = YOLO('yolo11s.pt')

    # 训练
    results = model.train(
        data=r"..\data\elements_2.yaml",  # 增强后的 60x60 数据集配置
        epochs=10,                      # 根据效果可调整
        imgsz=96,                        # 输入图片尺寸为 60x60/96/128
        batch=32,
        device='0',                      # GPU id
        project='chemical_elements_training',
        name='phase2_elements_augmented_12',
        exist_ok=True,
        augment=True,                    # 数据增强
        optimizer='SGD',                 # 小目标训练 SGD 更稳
        lr0=0.003,
        lrf=0.1,
        momentum=0.937,
        warmup_epochs=5,
        patience=10,
        box=0.05,  # 对小目标 box loss 更敏感
        cls=0.5,
        dfl=1.5
    )

    # 输出训练指标
    print("\n=== 训练指标 ===")
    if hasattr(results, 'metrics') and results.metrics is not None:
        for key, value in results.metrics.items():
            print(f"{key}: {value}")

    # 尝试读取训练 CSV 日志
    try:
        csv_path = os.path.join(
            'chemical_elements_training',
            'phase2_elements_augmented_12',
            'results.csv'
        )
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
    train_yolov11_elements_augmented_transformer()
