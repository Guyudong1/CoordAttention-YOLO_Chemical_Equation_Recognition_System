import cv2
import numpy as np
import random
from pathlib import Path
import shutil
from scipy import ndimage

# ================= 参数设置 =================
input_base_dir = Path("../final_dataset/elements")
target_num = 1000
img_size = 45


# ==========================================

def count_images(folder_path):
    """计算文件夹中的图像文件数量"""
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    count = 0
    for ext in image_extensions:
        count += len(list(folder_path.glob(ext)))
        count += len(list(folder_path.glob(ext.upper())))
    return count


def get_max_number_from_filenames(folder_path):
    """读取每个文件名后面的数字，返回最大值"""
    max_num = 0

    # 遍历所有图片文件
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            stem = file_path.stem  # 文件名（不含扩展名）

            # 尝试提取文件名中的数字部分
            parts = stem.split('_')

            # 取最后一部分作为可能的数字
            last_part = parts[-1]

            try:
                num = int(last_part)
                if num > max_num:
                    max_num = num
            except ValueError:
                # 如果不能转换为数字，跳过
                continue

    return max_num


def augment_images(input_dir, target_num=1000):
    """对单个文件夹进行数据增强"""
    # 直接使用最大编号作为当前文件数量
    current_count = get_max_number_from_filenames(input_dir)
    print(f"文件夹 {input_dir.name}: 当前图像数量: {current_count}")

    if current_count >= target_num:
        print(f"文件夹 {input_dir.name}: 已满足数量要求，跳过处理\n")
        return

    next_num = current_count + 1
    needed_count = target_num - current_count
    augmented_images = []

    print(f"文件夹 {input_dir.name}: 需要生成 {needed_count} 张增强图像")
    print(f"文件夹 {input_dir.name}: 将从编号 {next_num} 开始命名新文件")

    # 获取所有现有图片用于增强
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    images = []
    for ext in image_extensions:
        images.extend(list(input_dir.glob(ext)))
        images.extend(list(input_dir.glob(ext.upper())))

    # 使用多种增强技术生成图像
    while len(augmented_images) < needed_count:
        img_path = random.choice(images)
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 应用增强
        augmented_img = augment_single_image(img)
        augmented_images.append(augmented_img)

    print(f"文件夹 {input_dir.name}: 成功生成 {len(augmented_images)} 张增强图像")

    # 保存增强图像，使用新的命名规则
    for i, aug_img in enumerate(augmented_images):
        file_num = next_num + i
        save_path = input_dir / f'{input_dir.name}_{file_num}.png'
        cv2.imwrite(str(save_path), aug_img)

    final_count = current_count + len(augmented_images)
    print(f"文件夹 {input_dir.name}: 增强完成！总数量: {final_count} 张\n")


def random_rotation(img, rotation_range=15):
    """随机旋转"""
    angle = random.uniform(-rotation_range, rotation_range)
    M = cv2.getRotationMatrix2D((img_size / 2, img_size / 2), angle, 1)
    rotated = cv2.warpAffine(img, M, (img_size, img_size), borderValue=(255, 255, 255))
    return rotated


def random_scale(img, scale_range=0.2):
    """随机缩放"""
    scale = random.uniform(1 - scale_range, 1 + scale_range)
    new_size = max(1, int(round(img_size * scale)))
    new_size = min(new_size, img_size)

    resized = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_AREA)

    # 放置到canvas中心
    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    offset = (img_size - new_size) // 2
    canvas[offset:offset + new_size, offset:offset + new_size] = resized
    return canvas


def random_translation(img, translate_range=5):
    """随机平移"""
    tx = random.randint(-translate_range, translate_range)
    ty = random.randint(-translate_range, translate_range)

    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(img, M, (img_size, img_size), borderValue=(255, 255, 255))
    return translated


def random_noise(img, noise_level=0.05):
    """随机添加高斯噪声"""
    if random.random() < 0.1:  # 50%概率添加噪声
        noise = np.random.normal(0, noise_level * 255, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    return img


def random_elastic_transform(img, alpha=10, sigma=5):
    """弹性变换（模拟手绘效果）"""
    if random.random() < 0.45:  # 30%概率应用弹性变换
        random_state = np.random.RandomState(None)

        shape = img.shape[:2]
        dx = random_state.rand(*shape) * 2 - 1
        dy = random_state.rand(*shape) * 2 - 1

        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        transformed = np.zeros_like(img)
        for i in range(3):  # 对每个通道分别处理
            transformed[:, :, i] = ndimage.map_coordinates(img[:, :, i], indices, order=1).reshape(shape)

        return transformed
    return img


def random_occlusion(img, max_occlusion_size=10):
    """随机遮挡（模拟部分遮挡）"""
    if random.random() < 0.1:  # 20%概率添加遮挡
        h, w = img.shape[:2]
        occlusion_size = random.randint(5, max_occlusion_size)

        # 随机位置
        x = random.randint(0, w - occlusion_size)
        y = random.randint(0, h - occlusion_size)

        # 创建遮挡（白色矩形）
        img[y:y + occlusion_size, x:x + occlusion_size] = 255

    return img


def augment_single_image(img):
    """对单张图像应用随机增强组合"""
    # 随机选择几种增强技术（避免同时应用太多）
    augmentation_functions = [
        lambda x: random_rotation(x, rotation_range=15),
        lambda x: random_scale(x, scale_range=0.2),
        lambda x: random_translation(x, translate_range=5),
        lambda x: random_noise(x, noise_level=0.03),
        lambda x: random_occlusion(x, max_occlusion_size=8)
    ]

    # 随机选择2-4种增强技术
    num_augmentations = random.randint(2, 4)
    selected_augmentations = random.sample(augmentation_functions, num_augmentations)

    # 按随机顺序应用增强
    random.shuffle(selected_augmentations)

    augmented_img = img.copy()
    for aug_func in selected_augmentations:
        augmented_img = aug_func(augmented_img)

    return augmented_img


def main():
    if not input_base_dir.exists():
        print(f"错误：目录 {input_base_dir} 不存在！")
        return

    folders = [f for f in input_base_dir.iterdir() if f.is_dir()]
    print(f"找到 {len(folders)} 个文件夹")

    folders_to_process = []
    for folder in folders:
        current_count = count_images(folder)
        print(f"文件夹 {folder.name}: 当前有 {current_count} 张图片")

        if current_count < target_num:
            folders_to_process.append(folder)
            print(f"  -> 需要增强到 {target_num} 张")
        else:
            print(f"  -> 已满足要求，跳过处理")

    print(f"\n需要处理的文件夹数量: {len(folders_to_process)}")

    for i, folder in enumerate(folders_to_process, 1):
        print(f"\n处理进度: {i}/{len(folders_to_process)} - {folder.name}")
        augment_images(folder, target_num)

    print("所有文件夹处理完成！")


if __name__ == "__main__":
    main()