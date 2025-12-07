from ultralytics import YOLO
import os
import pandas as pd


def train_cell_only_yolo():
    # 切换到数据目录
    os.chdir(r"E:\sophomore\study\pytorch\pythonProject\MachineLearning\Chemical equation recognition\Model")

    # ---------- 初始化模型 ----------
    # 使用 YOLO 官方预训练权重 (yolov8n.pt)，类别数量已在 data.yaml 中设置为 1
    model = YOLO('yolo11n.pt')

    # ---------- 训练 ----------
    results = model.train(
        data=r'../data/equations_2.yaml',  # 你的 data.yaml
        epochs=10,
        imgsz=1024,
        batch=8,
        device='0',
        project='chemical_equations_training',
        name='cell_only_training_5struct_2',
        exist_ok=True,
        plots=True,

        # ---------- 数据增强 ----------
        degrees=2.0,
        translate=0.05,
        scale=0.9,
        shear=1.0,
        perspective=0.0005,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.2,
        hsv_h=0,
        hsv_s=0,        hsv_v=0,
        lr0=0.001,
        optimizer='AdamW',
    )

    # ---------- 输出主要指标 ----------
    print("\n=== 训练完成，汇总主要指标 ===")
    # 输出训练指标
    print("\n=== 训练指标 ===")
    if hasattr(results, 'metrics') and results.metrics is not None:
        for key, value in results.metrics.items():
            print(f"{key}: {value}")

    # 尝试读取训练 CSV 日志
    try:
        csv_path = os.path.join(
            'chemical_equations_training',
            'cell_only_training_5struct_2',
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

    # ---------- 验证 ----------
    print("\n=== 验证模型 ===")
    model.val(
        data='../data/equations.yaml',
        imgsz=1024,
        split='val'
    )


if __name__ == "__main__":
    train_cell_only_yolo()
