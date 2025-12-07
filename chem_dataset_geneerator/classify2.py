import shutil
from pathlib import Path
import random


def split_dataset(base_dir, output_dir, train_ratio=0.9):
    """
    将数据集按比例分割为训练集和验证集

    Args:
        base_dir: 数据集根目录，包含images和labels文件夹
        output_dir: 输出目录
        train_ratio: 训练集比例，默认0.9
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    images_dir = base_path / "images"
    labels_dir = base_path / "labels"

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建训练集和验证集目录
    train_images_dir = output_path / "images" / "train"
    train_labels_dir = output_path / "labels" / "train"
    val_images_dir = output_path / "images" / "val"
    val_labels_dir = output_path / "labels" / "val"

    # 创建目录结构
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件（不包括子目录）
    image_files = [f for f in images_dir.iterdir() if
                   f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']]

    # 随机打乱文件列表
    random.shuffle(image_files)

    # 计算分割点
    split_index = int(len(image_files) * train_ratio)

    # 分割文件
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    print(f"总图像数量: {len(image_files)}")
    print(f"训练集数量: {len(train_files)}")
    print(f"验证集数量: {len(val_files)}")
    print(f"训练集比例: {len(train_files) / len(image_files):.2%}")
    print(f"验证集比例: {len(val_files) / len(image_files):.2%}")

    # 复制训练集文件
    for img_file in train_files:
        # 复制图片
        shutil.copy2(img_file, train_images_dir / img_file.name)

        # 复制对应的标签文件
        label_file = labels_dir / img_file.with_suffix('.txt').name
        if label_file.exists():
            shutil.copy2(label_file, train_labels_dir / label_file.name)
        else:
            print(f"警告: 找不到标签文件 {label_file}")

    # 复制验证集文件
    for img_file in val_files:
        # 复制图片
        shutil.copy2(img_file, val_images_dir / img_file.name)

        # 复制对应的标签文件
        label_file = labels_dir / img_file.with_suffix('.txt').name
        if label_file.exists():
            shutil.copy2(label_file, val_labels_dir / label_file.name)
        else:
            print(f"警告: 找不到标签文件 {label_file}")

    print("数据集分割完成！")
    print(f"输出目录: {output_path}")
    print(f"训练集: {train_images_dir} 和 {train_labels_dir}")
    print(f"验证集: {val_images_dir} 和 {val_labels_dir}")


def split_dataset_move(base_dir, output_dir, train_ratio=0.9):
    """
    将数据集按比例分割为训练集和验证集（移动文件而不是复制）
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    images_dir = base_path / "images"
    labels_dir = base_path / "labels"

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建训练集和验证集目录
    train_images_dir = output_path / "images" / "train"
    train_labels_dir = output_path / "labels" / "train"
    val_images_dir = output_path / "images" / "val"
    val_labels_dir = output_path / "labels" / "val"

    # 创建目录结构
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in images_dir.iterdir() if
                   f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']]

    # 随机打乱文件列表
    random.shuffle(image_files)

    # 计算分割点
    split_index = int(len(image_files) * train_ratio)

    # 分割文件
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    print(f"总图像数量: {len(image_files)}")
    print(f"训练集数量: {len(train_files)}")
    print(f"验证集数量: {len(val_files)}")

    # 移动训练集文件
    for img_file in train_files:
        # 移动图片
        shutil.move(str(img_file), str(train_images_dir / img_file.name))

        # 移动对应的标签文件
        label_file = labels_dir / img_file.with_suffix('.txt').name
        if label_file.exists():
            shutil.move(str(label_file), str(train_labels_dir / label_file.name))
        else:
            print(f"警告: 找不到标签文件 {label_file}")

    # 移动验证集文件
    for img_file in val_files:
        # 移动图片
        shutil.move(str(img_file), str(val_images_dir / img_file.name))

        # 移动对应的标签文件
        label_file = labels_dir / img_file.with_suffix('.txt').name
        if label_file.exists():
            shutil.move(str(label_file), str(val_labels_dir / label_file.name))
        else:
            print(f"警告: 找不到标签文件 {label_file}")

    print("数据集分割完成！")
    print(f"输出目录: {output_path}")


# 使用示例
if __name__ == "__main__":
    dataset_path = "chemical equations/equations_3"
    output_path = "../data/equations_yolo_2"

    # 选择复制版本（推荐，保留原文件）
    split_dataset(dataset_path, output_path, train_ratio=0.9)

    # 如果要用移动版本（删除原文件），取消下面的注释
    # split_dataset_move(dataset_path, output_path, train_ratio=0.9)