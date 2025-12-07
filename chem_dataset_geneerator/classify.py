import os
import random
import shutil
from pathlib import Path

# ======== 指定 class.txt 文件路径 ========
CLASS_FILE = "../chem_dataset_generator/chemical equations/config/classes.txt"

# ======== 读取 class.txt 映射 ========
with open(CLASS_FILE, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

# name -> id 映射
class_name_to_id = {name: idx for idx, name in enumerate(class_names)}


def create_element_labels_and_split(dataset_path, output_path):
    """
    为元素数据集生成标签并按9:1分割训练集和测试集
    Args:
        dataset_path: 原始数据集路径 (final_dataset/elements)
        output_path: 输出路径
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    # 创建输出目录结构
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    # 获取所有元素类别文件夹
    element_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    element_folders.sort()

    # 创建类别映射：使用 class.txt 映射
    class_id_map = {}
    for element_folder in element_folders:
        class_name = element_folder.name
        if class_name not in class_name_to_id:
            raise ValueError(f"类别 {class_name} 不在 class.txt 中")
        class_id = class_name_to_id[class_name]
        class_id_map[class_name] = class_id
        print(f"类别 {class_id}: {class_name}")

    # 统计信息
    total_images = 0
    train_count = 0
    val_count = 0

    # 处理每个元素类别
    for class_name, class_id in class_id_map.items():
        class_folder = dataset_path / class_name

        # 查找所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(class_folder.glob(ext))

        print(f"处理类别 {class_name}: 找到 {len(image_files)} 张图片")

        # 随机打乱图片顺序
        random.shuffle(image_files)

        # 按9:1分割
        split_index = int(len(image_files) * 0.9)
        train_files = image_files[:split_index]
        val_files = image_files[split_index:]

        # 处理训练集
        for img_path in train_files:
            # 复制图片到训练集
            new_img_path = output_path / 'images' / 'train' / img_path.name
            shutil.copy2(img_path, new_img_path)

            # 创建对应的标签文件
            label_path = output_path / 'labels' / 'train' / f"{img_path.stem}.txt"

            # 生成标签内容 (整个图片)
            with open(label_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

            train_count += 1

        # 处理验证集
        for img_path in val_files:
            # 复制图片到验证集
            new_img_path = output_path / 'images' / 'val' / img_path.name
            shutil.copy2(img_path, new_img_path)

            # 创建对应的标签文件
            label_path = output_path / 'labels' / 'val' / f"{img_path.stem}.txt"

            # 生成标签内容
            with open(label_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

            val_count += 1

        total_images += len(image_files)

    # 创建 YOLO 配置文件
    create_yaml_config(output_path, class_names)

    print(f"\n=== 处理完成 ===")
    print(f"总图片数: {total_images}")
    print(f"训练集: {train_count} 张图片")
    print(f"验证集: {val_count} 张图片")
    print(f"类别数: {len(class_names)}")
    print(f"输出路径: {output_path}")

    return class_names


def create_yaml_config(output_path, class_names):
    """创建 YOLO 配置文件"""
    config_content = f"""# 元素识别数据集配置
path: {output_path.absolute()}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""
    config_path = output_path / 'elements.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

    print(f"配置文件已创建: {config_path}")


def check_dataset_structure(dataset_path):
    """检查数据集结构"""
    dataset_path = Path(dataset_path)

    print("=== 检查数据集结构 ===")

    # 检查是否存在元素文件夹
    element_folders = [f for f in dataset_path.iterdir() if f.is_dir()]

    if not element_folders:
        print("错误: 未找到任何元素文件夹")
        return False

    print(f"找到 {len(element_folders)} 个元素文件夹:")

    for folder in element_folders:
        print(f"  - {folder.name}")

        # 检查文件夹中是否有图片
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_count = 0
        for ext in image_extensions:
            image_count += len(list(folder.glob(ext)))

        print(f"    包含 {image_count} 张图片")

    return True


# ======== 使用示例 ========
if __name__ == "__main__":
    elements_path = "../final_dataset/elements"        # 元素数据集路径
    output_path = "../data/elements_yolo_2"     # 输出路径

    if not check_dataset_structure(elements_path):
        print("请检查数据集结构是否正确")
        exit(1)

    class_names = create_element_labels_and_split(elements_path, output_path)

    print(f"\n=== 数据集结构 ===")
    print(f"{output_path}/")
    print(f"├── images/")
    print(f"│   ├── train/     # 训练图片 ({len(list(Path(output_path).glob('images/train/*.*')))} 文件)")
    print(f"│   └── val/       # 验证图片 ({len(list(Path(output_path).glob('images/val/*.*')))} 文件)")
    print(f"├── labels/")
    print(f"│   ├── train/     # 训练标签 ({len(list(Path(output_path).glob('labels/train/*.txt')))} 文件)")
    print(f"│   └── val/       # 验证标签 ({len(list(Path(output_path).glob('labels/val/*.txt')))} 文件)")
    print(f"└── elements.yaml  # 配置文件")
