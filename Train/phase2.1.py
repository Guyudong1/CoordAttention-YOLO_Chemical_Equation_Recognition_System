from ultralytics import YOLO
import os
import pandas as pd


def finetune_equations_yolo11n():
    # 切换到数据目录
    os.chdir(r"/Model")

            # 使用阶段一训练好的模型作为预训练权重
    model = YOLO(r'../Model/chemical_elements_training/phase1_elements_custom/weights/best.pt')

    # ---------- 训练 ----------
    results = model.train(
        data=r'../data/equations.yaml',
        epochs=10,
        imgsz=1024,
        batch=8,
        device='0',
        project='chemical_equations_training',
        name='phase2_equations_finetune',
        exist_ok=True,
        plots=True,

        # ---------- 增强 ----------
        degrees=2.0,
        translate=0.05,
        scale=0.9,
        shear=1.0,
        perspective=0.0005,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        lr0=0.001,
        optimizer='AdamW',
    )

    # ---------- 输出主要指标 ----------
    print("\n=== 训练完成，汇总主要指标 ===")
    try:
        csv_path = os.path.join('chemical_equations_training', 'phase2_equations_finetune', 'results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print("\n=== 最后10轮平均指标 ===")
            last_epochs = df.tail(10)
            avg_map50 = last_epochs['metrics/mAP50(B)'].mean()
            avg_map5095 = last_epochs['metrics/mAP50-95(B)'].mean()
            avg_precision = last_epochs['metrics/precision(B)'].mean()
            avg_recall = last_epochs['metrics/recall(B)'].mean()
            print(f"平均 mAP50: {avg_map50:.4f}")
            print(f"平均 mAP50-95: {avg_map5095:.4f}")
            print(f"平均 Precision: {avg_precision:.4f}")
            print(f"平均 Recall: {avg_recall:.4f}")
    except Exception as e:
        print(f"读取日志文件时出错: {e}")

    # ---------- 验证 ----------
    print("\n=== 验证模型 ===")
    model.val(
        data='equations.yaml',
        imgsz=1024,
        split='val'
    )


if __name__ == "__main__":
    finetune_equations_yolo11n()
