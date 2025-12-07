from ultralytics import YOLO
import os
import pandas as pd


def custom_training_with_metrics():
    os.chdir(r"/Model")

    model = YOLO('yolo11n.pt')

    # 训练模型
    results = model.train(
        data='..\data\equations.yaml',
        epochs=10,
        imgsz=(1024, 157),      # 保持方程长宽比例（宽可变，高固定157）
        batch=8,
        device='0',
        project='equation',
        name='phase2_2',
        exist_ok=True,
        plots=True,
        optimizer='Adam',
        lr0=1e-4,               # 小学习率，防止破坏特征
        mosaic=0.0,             # ❌ 禁用 Mosaic
        mixup=0.0,              # ❌ 禁用 MixUp
        copy_paste=0.0,         # ❌ 禁用 Copy-Paste
        translate=0.05,         # ✅ 允许少量平移
        scale=0.2,              # ✅ 允许缩放
        shear=0.0,              # ❌ 禁止剪切
        perspective=0.0,        # ❌ 禁止透视变化
        degrees=2.0,            # ✅ 允许微小旋转
    )

    # 获取训练过程中的指标
    print("\n=== 训练指标 ===")
    if hasattr(results, 'results_dict'):
        for key, value in results.results_dict.items():
            print(f"{key}: {value}")

    # 读取CSV日志文件获取详细历史
    try:
        csv_path = 'chemical_elements_training/phase1_elements_custom/results.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print("\n=== 训练历史 (最后5个epoch) ===")
            print(df.tail())

            # 计算平均指标
            last_10_epochs = df.tail(10)
            avg_map50 = last_10_epochs['metrics/mAP50(B)'].mean()
            avg_map = last_10_epochs['metrics/mAP50-95(B)'].mean()
            avg_precision = last_10_epochs['metrics/precision(B)'].mean()
            avg_recall = last_10_epochs['metrics/recall(B)'].mean()

            print(f"\n=== 最后10个epoch平均指标 ===")
            print(f"平均mAP50: {avg_map50:.4f}")
            print(f"平均mAP50-95: {avg_map:.4f}")
            print(f"平均precision: {avg_precision:.4f}")
            print(f"平均recall: {avg_recall:.4f}")
    except Exception as e:
        print(f"读取日志文件时出错: {e}")


if __name__ == "__main__":
    custom_training_with_metrics()