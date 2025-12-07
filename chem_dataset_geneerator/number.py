import cv2
import numpy as np
from pathlib import Path
import os


def rename_files_simple(folder_path):
    """简单直接的重命名：先改为纯数字，再加前缀"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    all_images = []

    # 收集所有图片文件
    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            all_images.append(file_path)

    if len(all_images) == 0:
        print(f"文件夹 {folder_path.name} 中没有找到图片文件")
        return 0

    # 按文件名排序
    all_images.sort()

    print(f"文件夹 {folder_path.name}: 找到 {len(all_images)} 个文件")

    # 第一步：重命名为临时名称（避免冲突）
    temp_files = []
    for i, img_path in enumerate(all_images, 1):
        try:
            temp_name = f"temp_{i}{img_path.suffix}"
            temp_path = folder_path / temp_name
            img_path.rename(temp_path)
            temp_files.append(temp_path)
        except Exception as e:
            print(f"第一步重命名失败: {img_path.name} -> temp_{i}, 错误: {e}")

    print(f"第一步完成: 重命名为临时文件")

    # 第二步：重命名为最终名称（文件夹名_数字）
    renamed_count = 0
    for i, temp_path in enumerate(temp_files, 1):
        try:
            final_name = f"{folder_path.name}_{i}{temp_path.suffix}"
            final_path = folder_path / final_name
            temp_path.rename(final_path)
            renamed_count += 1
        except Exception as e:
            print(f"第二步重命名失败: {temp_path.name} -> {final_name}, 错误: {e}")

    print(f"文件夹 {folder_path.name}: 成功重命名 {renamed_count} 个文件")
    return renamed_count


def main():
    # 设置基础目录
    input_base_dir = Path("../final_dataset/elements")

    if not input_base_dir.exists():
        print(f"错误：目录 {input_base_dir} 不存在！")
        return

    folders = [f for f in input_base_dir.iterdir() if f.is_dir()]
    print(f"找到 {len(folders)} 个文件夹")

    total_renamed = 0
    for i, folder in enumerate(folders, 1):
        print(f"\n处理进度: {i}/{len(folders)} - {folder.name}")
        renamed_count = rename_files_simple(folder)
        total_renamed += renamed_count

    print(f"\n所有文件夹处理完成！总共重命名了 {total_renamed} 个文件")


if __name__ == "__main__":
    main()