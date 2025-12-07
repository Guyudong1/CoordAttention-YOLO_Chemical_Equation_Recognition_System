import os
import yaml
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

def create_oversampled_dataset_yaml():
    """创建一个新的YAML配置，通过修改训练集列表来实现过采样"""

    # 读取原始YAML配置
    original_yaml_path = r"..\data\elements.yaml"
    with open(original_yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 获取训练集目录路径
    train_dir = config['train']  # 这是 'elements_new/images/train'

    # 构建完整的训练集目录路径
    full_train_dir = os.path.join(config['path'], train_dir)

    print(f"训练目录: {full_train_dir}")

    # 检查路径是否存在
    if not os.path.exists(full_train_dir):
        print(f"错误: 训练目录不存在: {full_train_dir}")
        return None

    # 获取训练目录下所有的图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(full_train_dir).glob(f'*{ext}'))

    print(f"在目录中找到 {len(image_files)} 个图像文件")

    # 过滤掉无效的文件名
    valid_image_files = []
    for file in image_files:
        filename = file.name
        # 过滤掉包含特殊字符的文件名
        if any(char in filename for char in ['}', '{', '|', '\\', ':', '*', '?', '"', '<', '>']):
            print(f"跳过无效文件名: {filename}")
            continue
        valid_image_files.append(file)

    print(f"过滤后有效图像文件: {len(valid_image_files)} 个")

    # 转换为正确的相对路径（使用正斜杠）
    original_train_list = []
    for file in valid_image_files:
        # 确保使用正斜杠，避免路径问题
        rel_path = str(file.relative_to(config['path'])).replace('\\', '/')
        original_train_list.append(rel_path)

    print(f"处理后得到 {len(original_train_list)} 个训练图像路径")

    common_elements_indices = [
        # 极其常见
        0,  # 'H' - 氢
        5,  # 'C' - 碳
        6,  # 'N' - 氮
        7,  # 'O' - 氧
        # 很常见
        10,  # 'Na' - 钠
        11,  # 'Mg' - 镁
        12,  # 'Al' - 铝
        14,  # 'P' - 磷
        15,  # 'S' - 硫
        16,  # 'Cl' - 氯
        18,  # 'K' - 钾
        19,  # 'Ca' - 钙
        # 常见金属元素
        24,  # 'Mn' - 锰
        25,  # 'Fe' - 铁
        28,  # 'Cu' - 铜
        29,  # 'Zn' - 锌
        47,  # 'Ag' - 银
        78,  # 'Pt' - 铂
        79,  # 'Au' - 金
        80,  # 'Hg' - 汞
        # 其他常见
        13,  # 'Si' - 硅
    ]

    print(f"常见元素类别索引: {common_elements_indices}")
    print(f"常见元素包括: {[config['names'][i] for i in common_elements_indices]}")
    print(f"选择的常见元素数量: {len(common_elements_indices)}个")

    # 创建过采样的训练列表
    oversampled_train_list = []

    print("正在分析训练数据并过采样常见元素...")

    processed_count = 0
    common_count = 0
    rare_count = 0
    valid_count = 0
    invalid_count = 0

    for img_rel_path in original_train_list:
        # 构建完整的图像路径
        full_img_path = os.path.join(config['path'], img_rel_path)

        # 检查图像文件是否存在
        if not os.path.exists(full_img_path):
            print(f"警告: 图像文件不存在: {full_img_path}")
            invalid_count += 1
            continue

        # 对应的标签文件路径
        label_path = full_img_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')

        # 检查标签文件是否存在且有效
        if not os.path.exists(label_path):
            print(f"警告: 标签文件不存在: {label_path}")
            invalid_count += 1
            continue

        # 检查标签文件内容
        contains_common = False
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = f.readlines()

            # 单元素数据集，每个文件只有一行
            if labels and labels[0].strip():
                class_idx = int(labels[0].strip().split()[0])
                if class_idx in common_elements_indices:
                    contains_common = True
                    common_count += 1
                else:
                    rare_count += 1
                valid_count += 1
            else:
                print(f"警告: 标签文件为空: {label_path}")
                invalid_count += 1
                continue

        except (ValueError, IndexError) as e:
            print(f"解析标签文件出错 {label_path}: {e}")
            invalid_count += 1
            continue
        except Exception as e:
            print(f"读取标签文件出错 {label_path}: {e}")
            invalid_count += 1
            continue

        # 如果包含常见元素，在训练列表中重复3次
        if contains_common:
            for _ in range(5):  # 过采样3次
                oversampled_train_list.append(img_rel_path)
        else:
            # 稀有元素，只保留1次
            oversampled_train_list.append(img_rel_path)

        processed_count += 1
        if processed_count % 100 == 0:
            print(f"已处理 {processed_count}/{len(original_train_list)} 个样本")

    print(f"\n=== 统计信息 ===")
    print(f"原始训练样本数: {len(original_train_list)}")
    print(f"有效样本数: {valid_count}")
    print(f"无效样本数: {invalid_count}")
    print(f"常见元素样本数: {common_count}")
    print(f"稀有元素样本数: {rare_count}")
    print(f"过采样后训练样本数: {len(oversampled_train_list)}")
    print(f"过采样比例: {len(oversampled_train_list) / len(original_train_list):.2f}x")

    # 检查是否有有效样本
    if len(oversampled_train_list) == 0:
        print("错误: 没有有效的训练样本!")
        return None

    # 创建新的训练列表文件（在原始数据目录下）
    new_train_txt = 'oversampled_train.txt'
    new_train_txt_path = os.path.join(config['path'], new_train_txt)

    with open(new_train_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(oversampled_train_list))

    print(f"已创建训练列表文件: {new_train_txt_path}")

    # 验证生成的文件
    print("\n=== 验证生成的文件 ===")
    if os.path.exists(new_train_txt_path):
        with open(new_train_txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"训练列表文件包含 {len(lines)} 行")
        if len(lines) > 0:
            print(f"前5个路径示例:")
            for i in range(min(5, len(lines))):
                print(f"  {lines[i].strip()}")

            # 检查是否有无效路径
            invalid_paths = []
            for line in lines:
                img_path = os.path.join(config['path'], line.strip())
                label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
                if not os.path.exists(img_path) or not os.path.exists(label_path):
                    invalid_paths.append(line.strip())

            if invalid_paths:
                print(f"警告: 发现 {len(invalid_paths)} 个无效路径")
                for path in invalid_paths[:5]:  # 只显示前5个
                    print(f"  无效: {path}")

    # 创建新的YAML配置
    new_config = config.copy()
    new_config['train'] = new_train_txt  # 使用新的训练列表文件
    new_config['name'] = 'elements_oversampled'

    new_yaml_path = os.path.join(config['path'], 'oversampled_elements.yaml')
    with open(new_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(new_config, f)

    print(f"已创建过采样数据集配置: {new_yaml_path}")

    return new_yaml_path


def clean_invalid_files():
    """清理训练目录中的无效文件"""
    original_yaml_path = r"..\data\elements.yaml"
    with open(original_yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    train_dir = os.path.join(config['path'], config['train'])

    # 查找并删除无效文件
    invalid_chars = ['}', '{', '|', '\\', ':', '*', '?', '"', '<', '>']

    for file in Path(train_dir).glob('*'):
        if any(char in file.name for char in invalid_chars):
            print(f"删除无效文件: {file.name}")
            try:
                file.unlink()
            except Exception as e:
                print(f"删除失败: {e}")


def train_with_oversampling():
    """使用过采样数据集进行训练"""

    os.chdir(r"E:\sophomore\study\pytorch\pythonProject\MachineLearning\Chemical equation recognition\Model")

    # 可选：清理无效文件
    clean_invalid_files()

    # 第一步：创建过采样数据集配置
    new_yaml_path = create_oversampled_dataset_yaml()

    if new_yaml_path is None:
        print("创建过采样数据集失败，退出训练")
        return

    # 第二步：使用新配置训练模型
    model = YOLO('yolo11s.pt')

    print("开始训练...")
    try:
        results = model.train(
            data=new_yaml_path,  # 使用过采样的数据集配置
            epochs=10,
            imgsz=96,
            batch=32,
            device='0',
            project='chemical_elements_training',
            name='phase2_elements_oversampled_1',
            exist_ok=True,
            augment=True,
            optimizer='SGD',
            lr0=0.003,
            lrf=0.1,
            momentum=0.937,
            warmup_epochs=5,
            patience=10,
            box=0.05,
            cls=0.5,
            dfl=1.5
        )
        print("\n=== 过采样训练完成 ===")
    except Exception as e:
        print(f"训练过程中出错: {e}")
        print("尝试使用原始配置训练...")
        # 如果新配置有问题，回退到原始配置
        results = model.train(
            data=r"..\data\elements.yaml",
            epochs=10,
            imgsz=96,
            batch=32,
            device='0',
            project='chemical_elements_training',
            name='phase2_elements_oversampled_1',
            exist_ok=True,
            augment=True,
            optimizer='SGD',
            lr0=0.003,
            lrf=0.1,
            momentum=0.937,
            warmup_epochs=5,
            patience=10,
            box=0.05,
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
            'phase2_elements_oversampled_1',
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
    train_with_oversampling()