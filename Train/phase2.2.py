from ultralytics import YOLO
import torch
import os
import pandas as pd

def train_equation_structure_phase2():
    """
    第二阶段：结构理解训练（保持元素识别能力）
    - 冻结 backbone + 分类头（detect.cls）
    - 禁用破坏性增强（Mosaic / MixUp）
    - 仅学习方程式结构特征
    """

    # === 路径配置 ===
    os.chdir(r"/Model")
    pretrained_model = r"E:\sophomore\study\pytorch\pythonProject\MachineLearning\Chemical equation recognition\model\chemical_elements_training\phase1_elements_custom\weights\best.pt"
    data_yaml = r"..\data\equations.yaml"

    # === 加载模型 ===
    model = YOLO(pretrained_model)
    print("\n已加载 Phase1 预训练权重。")

    # === 冻结 backbone + 分类层 ===
    for name, param in model.model.named_parameters():
        if "backbone" in name or "detect.cls" in name:
            param.requires_grad = False

    frozen_params = [name for name, p in model.model.named_parameters() if not p.requires_grad]
    print(f"已冻结参数层：{len(frozen_params)} 层")

    # === 训练 ===
    results = model.train(
        data=data_yaml,
        epochs=5,              # 训练 20 轮
        imgsz=(1024, 157),      # 保持方程长宽比例（宽可变，高固定157）
        batch=8,
        device='0',
        project='chemical_equation_training',
        name='phase2_structure_only',
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

    print("\n=== Phase2 结构学习训练完成 ===")

    # === 结果统计 ===
    try:
        csv_path = os.path.join("chemical_equation_training", "phase2_structure_only", "results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print("\n=== 训练历史 (最后5个epoch) ===")
            print(df.tail())

            avg_map50 = df["metrics/mAP50(B)"].tail(5).mean()
            avg_precision = df["metrics/precision(B)"].tail(5).mean()
            avg_recall = df["metrics/recall(B)"].tail(5).mean()

            print(f"\n平均 mAP50: {avg_map50:.4f}")
            print(f"平均 Precision: {avg_precision:.4f}")
            print(f"平均 Recall: {avg_recall:.4f}")
    except Exception as e:
        print(f"读取训练日志出错: {e}")

if __name__ == "__main__":
    train_equation_structure_phase2()
