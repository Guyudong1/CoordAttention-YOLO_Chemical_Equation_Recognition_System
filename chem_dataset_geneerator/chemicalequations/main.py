import os
import sys
import cv2
from typing import Dict
import numpy as np
import random
import time

from src.generator_V1 import EquationGeneratorV1
from src.generator_V2 import EquationGeneratorV2
from src.generator_V3 import EquationGeneratorV3
from src.generator_V4 import EquationGeneratorV4
from src.generator_V5 import EquationGeneratorV5
from src.generator_V6 import EquationGeneratorV6
from src.generator_V7 import EquationGeneratorV7
from src.generator_V8 import EquationGeneratorV8
from src.utils import save_yolo_annotation, debug_equation, visualize_sample
from src.utils import save_yolo_annotation_v8, visualize_sample_v8

# 以当前时间为随机种子，保证每次运行结果不同
seed = int(time.time())
np.random.seed(seed)
random.seed(seed)
# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加src目录到Python路径
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)


def load_class_mapping(classes_path: str) -> Dict:
    """从classes.txt加载类别映射"""
    class_mapping = {}
    print(f"正在加载类别文件: {classes_path}")
    
    # 检查文件是否存在
    if not os.path.exists(classes_path):
        print(f"错误：文件不存在 - {classes_path}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"文件绝对路径: {os.path.abspath(classes_path)}")
        return class_mapping
    with open(classes_path, "r", encoding='utf-8') as f:
        for idx, line in enumerate(f):
            class_name = line.strip().split('#')[0].strip()
            if class_name:
                class_mapping[class_name] = idx
                print(f"加载类别: {class_name} -> ID: {idx}")
    return class_mapping


def generate_dataset_v1():
    """生成V1版本"""
    # 使用绝对路径配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    symbols_dir = os.path.join(base_dir, "raw_symbols")
    output_dir = os.path.join(base_dir, "generated_data_v1")
    classes_path = os.path.join(base_dir, "config", "classes.txt")
    print(f"项目根目录: {base_dir}")
    print(f"符号目录: {symbols_dir}")
    print(f"输出目录: {output_dir}")
    print(f"类别文件: {classes_path}")
    
    # 检查必要的目录是否存在
    if not os.path.exists(symbols_dir):
        print(f"错误：符号目录不存在 - {symbols_dir}")
        print("请创建 raw_symbols 文件夹并放入化学符号图片")
        return
    
    # 创建输出目录
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    
    # 加载类别映射
    class_mapping = load_class_mapping(classes_path)
    if not class_mapping:
        print("错误：无法加载类别映射，程序退出")
        return
    print(f"成功加载了 {len(class_mapping)} 个类别")
    
    # 创建生成器
    generator = EquationGeneratorV1(symbols_dir, class_mapping)
    
    # 测试生成一个样本
    print("生成测试样本...")
    test_equation = generator.generate_simple_equation()
    debug_equation(test_equation, class_mapping)
    
    # 保存测试样本
    test_image_path = f"{output_dir}/images/test_sample.jpg"
    test_label_path = f"{output_dir}/labels/test_sample.txt"
    test_image_label_path = f"{output_dir}/images/test_sample_with_label.jpg"
    cv2.imwrite(test_image_path, test_equation.image)
    save_yolo_annotation(test_equation, test_label_path)
    print(f"测试样本已保存: {test_image_path}")
    
    # 可视化测试样本
    visualize_sample(test_image_path, test_label_path, test_image_label_path, class_mapping)
    
        
def generate_dataset_v2():
    """生成V2版本"""
    # 使用绝对路径配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    symbols_dir = os.path.join(base_dir, "raw_symbols")
    output_dir = os.path.join(base_dir, "generated_data_v2")
    classes_path = os.path.join(base_dir, "config", "classes.txt")
    print(f"项目根目录: {base_dir}")
    print(f"符号目录: {symbols_dir}")
    print(f"输出目录: {output_dir}")
    print(f"类别文件: {classes_path}")
    
    # 检查必要的目录是否存在
    if not os.path.exists(symbols_dir):
        print(f"错误：符号目录不存在 - {symbols_dir}")
        print("请创建 raw_symbols 文件夹并放入化学符号图片")
        return
    
    # 创建输出目录
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    
    # 加载类别映射
    class_mapping = load_class_mapping(classes_path)
    if not class_mapping:
        print("错误：无法加载类别映射，程序退出")
        return
    print(f"成功加载了 {len(class_mapping)} 个类别")
    
    # 创建生成器
    generator = EquationGeneratorV2(symbols_dir, class_mapping)
    
    # 先测试生成一个样本
    print("生成测试样本...")
    test_equation = generator.generate_simple_equation()
    debug_equation(test_equation, class_mapping)
    
    # 保存测试样本
    test_image_path = f"{output_dir}/images/test_sample.jpg"
    test_label_path = f"{output_dir}/labels/test_sample.txt"
    test_image_label_path = f"{output_dir}/images/test_sample_with_label.jpg"
    cv2.imwrite(test_image_path, test_equation.image)
    save_yolo_annotation(test_equation, test_label_path)
    print(f"测试样本已保存: {test_image_path}")
    
    # 可视化测试样本
    visualize_sample(test_image_path, test_label_path, test_image_label_path, class_mapping)


def generate_dataset_v3():
    """生成V2版本"""
    # 使用绝对路径配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    symbols_dir = os.path.join(base_dir, "raw_symbols")
    output_dir = os.path.join(base_dir, "generated_data_v3")
    classes_path = os.path.join(base_dir, "config", "classes.txt")
    print(f"项目根目录: {base_dir}")
    print(f"符号目录: {symbols_dir}")
    print(f"输出目录: {output_dir}")
    print(f"类别文件: {classes_path}")
    
    # 检查必要的目录是否存在
    if not os.path.exists(symbols_dir):
        print(f"错误：符号目录不存在 - {symbols_dir}")
        print("请创建 raw_symbols 文件夹并放入化学符号图片")
        return
    
    # 创建输出目录
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    
    # 加载类别映射
    class_mapping = load_class_mapping(classes_path)
    if not class_mapping:
        print("错误：无法加载类别映射，程序退出")
        return
    print(f"成功加载了 {len(class_mapping)} 个类别")
    
    # 创建生成器
    generator = EquationGeneratorV3(symbols_dir, class_mapping)
    
    # 先测试生成一个样本
    print("生成测试样本...")
    test_equation = generator.generate_simple_equation()
    debug_equation(test_equation, class_mapping)
    
    # 保存测试样本
    test_image_path = f"{output_dir}/images/test_sample.jpg"
    test_label_path = f"{output_dir}/labels/test_sample.txt"
    test_image_label_path = f"{output_dir}/images/test_sample_with_label.jpg"
    cv2.imwrite(test_image_path, test_equation.image)
    save_yolo_annotation(test_equation, test_label_path)
    print(f"测试样本已保存: {test_image_path}")
    
    # 可视化测试样本
    visualize_sample(test_image_path, test_label_path, test_image_label_path, class_mapping)


def generate_dataset_v4():
    """生成V2版本"""
    # 使用绝对路径配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    symbols_dir = os.path.join(base_dir, "raw_symbols")
    output_dir = os.path.join(base_dir, "generated_data_v4")
    classes_path = os.path.join(base_dir, "config", "classes.txt")
    print(f"项目根目录: {base_dir}")
    print(f"符号目录: {symbols_dir}")
    print(f"输出目录: {output_dir}")
    print(f"类别文件: {classes_path}")

    # 检查必要的目录是否存在
    if not os.path.exists(symbols_dir):
        print(f"错误：符号目录不存在 - {symbols_dir}")
        print("请创建 raw_symbols 文件夹并放入化学符号图片")
        return

    # 创建输出目录
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    # 加载类别映射
    class_mapping = load_class_mapping(classes_path)
    if not class_mapping:
        print("错误：无法加载类别映射，程序退出")
        return
    print(f"成功加载了 {len(class_mapping)} 个类别")

    # 创建生成器
    generator = EquationGeneratorV4(symbols_dir, class_mapping)

    # 先测试生成一个样本
    print("生成测试样本...")
    test_equation = generator.generate_simple_equation()
    debug_equation(test_equation, class_mapping)

    # 保存测试样本
    test_image_path = f"{output_dir}/images/test_sample3.jpg"
    test_label_path = f"{output_dir}/labels/test_sample3.txt"
    test_image_label_path = f"{output_dir}/images/test_sample_with_label3.jpg"
    cv2.imwrite(test_image_path, test_equation.image)
    save_yolo_annotation(test_equation, test_label_path)
    print(f"测试样本已保存: {test_image_path}")

    # 可视化测试样本
    visualize_sample(test_image_path, test_label_path, test_image_label_path, class_mapping)


def generate_dataset_v5():
    """生成V5版本数据集（每条方程式都生成）"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    symbols_dir = os.path.join(base_dir, "raw_symbols")
    output_dir = os.path.join(base_dir, "generated_data_v5")
    classes_path = os.path.join(base_dir, "config", "classes.txt")
    print(f"项目根目录: {base_dir}")
    print(f"符号目录: {symbols_dir}")
    print(f"输出目录: {output_dir}")
    print(f"类别文件: {classes_path}")

    if not os.path.exists(symbols_dir):
        print(f"错误：符号目录不存在 - {symbols_dir}")
        print("请创建 raw_symbols 文件夹并放入化学符号图片")
        return

    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    class_mapping = load_class_mapping(classes_path)
    if not class_mapping:
        print("错误：无法加载类别映射，程序退出")
        return
    print(f"成功加载了 {len(class_mapping)} 个类别")

    generator = EquationGeneratorV5(symbols_dir, class_mapping)

    # 生成所有方程式
    all_equations = generator.generate_all_equations()

    for idx, eq in enumerate(all_equations):
        image_path = f"{output_dir}/images/equation_{idx}.jpg"
        label_path = f"{output_dir}/labels/equation_{idx}.txt"
        image_label_path = f"{output_dir}/images/equation_{idx}_with_label.jpg"

        cv2.imwrite(image_path, eq.image)
        save_yolo_annotation(eq, label_path)

        # 可选：生成带标注的可视化图
        visualize_sample(image_path, label_path, image_label_path, class_mapping)

        print(f"第 {idx+1} 条方程式已保存: {image_path}")

    print(f"\n=== 数据集生成完成，共 {len(all_equations)} 条方程式 ===")


def generate_dataset_v6():
    """生成V6版本数据集（每条方程式都生成）"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    symbols_dir = os.path.join(base_dir, "raw_symbols")
    output_dir = os.path.join(base_dir, "generated_data_v6")
    classes_path = os.path.join(base_dir, "config", "classes.txt")
    print(f"项目根目录: {base_dir}")
    print(f"符号目录: {symbols_dir}")
    print(f"输出目录: {output_dir}")
    print(f"类别文件: {classes_path}")

    if not os.path.exists(symbols_dir):
        print(f"错误：符号目录不存在 - {symbols_dir}")
        print("请创建 raw_symbols 文件夹并放入化学符号图片")
        return

    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    class_mapping = load_class_mapping(classes_path)
    if not class_mapping:
        print("错误：无法加载类别映射，程序退出")
        return
    print(f"成功加载了 {len(class_mapping)} 个类别")

    generator = EquationGeneratorV6(symbols_dir, class_mapping)

    # 生成所有方程式
    all_equations = generator.generate_all_equations()

    for idx, eq in enumerate(all_equations):
        image_path = f"{output_dir}/images/equation_{idx}.jpg"
        label_path = f"{output_dir}/labels/equation_{idx}.txt"
        image_label_path = f"{output_dir}/images/equation_{idx}_with_label.jpg"

        cv2.imwrite(image_path, eq.image)
        save_yolo_annotation(eq, label_path)

        # 生成带标注的可视化图
        visualize_sample(image_path, label_path, image_label_path, class_mapping)

        print(f"第 {idx+1} 条方程式已保存: {image_path}")

    print(f"\n=== 数据集生成完成，共 {len(all_equations)} 条方程式 ===")


def generate_dataset_v7():
    """生成V7版本数据集（每条方程式都生成）"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    symbols_dir = os.path.join(base_dir, "raw_symbols")
    output_dir = os.path.join(base_dir, "generated_data_v7")
    classes_path = os.path.join(base_dir, "config", "classes.txt")
    print(f"项目根目录: {base_dir}")
    print(f"符号目录: {symbols_dir}")
    print(f"输出目录: {output_dir}")
    print(f"类别文件: {classes_path}")

    if not os.path.exists(symbols_dir):
        print(f"错误：符号目录不存在 - {symbols_dir}")
        print("请创建 raw_symbols 文件夹并放入化学符号图片")
        return

    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    class_mapping = load_class_mapping(classes_path)
    if not class_mapping:
        print("错误：无法加载类别映射，程序退出")
        return
    print(f"成功加载了 {len(class_mapping)} 个类别")

    generator = EquationGeneratorV7(symbols_dir, class_mapping)

    # 生成所有方程式
    all_equations = generator.generate_all_equations()

    for idx, eq in enumerate(all_equations):
        image_path = f"{output_dir}/images/equation_{idx}.jpg"
        label_path = f"{output_dir}/labels/equation_{idx}.txt"
        image_label_path = f"{output_dir}/images/equation_{idx}_with_label.jpg"

        cv2.imwrite(image_path, eq.image)
        save_yolo_annotation(eq, label_path)

        # 生成带标注的可视化图
        visualize_sample(image_path, label_path, image_label_path, class_mapping)

        print(f"第 {idx+1} 条方程式已保存: {image_path}")


    print(f"\n=== 数据集生成完成，共 {len(all_equations)} 条方程式 ===")


def generate_dataset_v8():
    """生成V8版本数据集（元素识别用原有类别，标注用新5类别）"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    symbols_dir = os.path.join(base_dir, "raw_symbols")
    output_dir = os.path.join(base_dir, "generated_data_v8")

    # 使用原有的classes.txt进行元素识别
    classes_path = os.path.join(base_dir, "config", "classes.txt")

    print(f"项目根目录: {base_dir}")
    print(f"符号目录: {symbols_dir}")
    print(f"输出目录: {output_dir}")
    print(f"元素识别类别文件: {classes_path}")

    if not os.path.exists(symbols_dir):
        print(f"错误：符号目录不存在 - {symbols_dir}")
        return

    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    # 加载原有的类别映射（用于元素识别）
    original_class_mapping = load_class_mapping(classes_path)
    if not original_class_mapping:
        print("错误：无法加载原有的类别映射，程序退出")
        return
    print(f"成功加载了 {len(original_class_mapping)} 个元素识别类别")

    generator = EquationGeneratorV8(symbols_dir, original_class_mapping)

    # 生成所有方程式
    all_equations = generator.generate_all_equations()

    # 定义新的5类别映射（用于可视化）
    new_class_mapping = {
        'sub': 0,  # 下标
        '+': 1,  # 加号
        'aR': 2,  # arrowR
        'a2': 3,  # arrow2
        'ot': 4  # 其他
    }

    for idx, eq in enumerate(all_equations):
        image_path = f"{output_dir}/images/equation_{idx}.jpg"
        label_path = f"{output_dir}/labels/equation_{idx}.txt"
        image_label_path = f"{output_dir}/images/equation_{idx}_with_label.jpg"

        cv2.imwrite(image_path, eq.image)
        save_yolo_annotation(eq, label_path)

        # 生成带标注的可视化图（使用新的5类别映射）
        visualize_sample_v8(image_path, label_path, image_label_path, new_class_mapping)

        print(f"第 {idx + 1} 条方程式已保存: {image_path}")

    print(f"\n=== 数据集生成完成，共 {len(all_equations)} 条方程式 ===")
    print(f"元素识别使用原有 {len(original_class_mapping)} 个类别")
    print(f"标注使用新的5类别系统: {new_class_mapping}")


def generate_dataset_final():
    """生成V7版本数据集（每条方程式都生成）"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    symbols_dir = os.path.join(base_dir, "raw_symbols")
    output_dir = os.path.join(base_dir, "equations_2")
    classes_path = os.path.join(base_dir, "config", "classes.txt")
    print(f"项目根目录: {base_dir}")
    print(f"符号目录: {symbols_dir}")
    print(f"输出目录: {output_dir}")
    print(f"类别文件: {classes_path}")

    if not os.path.exists(symbols_dir):
        print(f"错误：符号目录不存在 - {symbols_dir}")
        print("请创建 raw_symbols 文件夹并放入化学符号图片")
        return

    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    class_mapping = load_class_mapping(classes_path)
    if not class_mapping:
        print("错误：无法加载类别映射，程序退出")
        return
    print(f"成功加载了 {len(class_mapping)} 个类别")

    generator = EquationGeneratorV7(symbols_dir, class_mapping)

    # 生成所有方程式
    all_equations = generator.generate_all_equations()

    for idx, eq in enumerate(all_equations):
        image_path = f"{output_dir}/images/eq_{idx}.jpg"
        label_path = f"{output_dir}/labels/eq_{idx}.txt"
        image_label_path = f"{output_dir}/images/eq_{idx}_with_label.jpg"

        cv2.imwrite(image_path, eq.image)
        save_yolo_annotation(eq, label_path)

        # 生成带标注的可视化图
        # visualize_sample(image_path, label_path, image_label_path, class_mapping)

        print(f"第 {idx+1} 条方程式已保存: {image_path}")

    print(f"\n=== 数据集生成完成，共 {len(all_equations)} 条方程式 ===")

def generate_dataset_final_v8():
    """生成V8版本数据集（元素识别用原有类别，标注用新5类别）"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    symbols_dir = os.path.join(base_dir, "raw_symbols")
    output_dir = os.path.join(base_dir, "equations_3")
    classes_path = os.path.join(base_dir, "config", "classes.txt")
    print(f"项目根目录: {base_dir}")
    print(f"符号目录: {symbols_dir}")
    print(f"输出目录: {output_dir}")
    print(f"类别文件: {classes_path}")

    if not os.path.exists(symbols_dir):
        print(f"错误：符号目录不存在 - {symbols_dir}")
        print("请创建 raw_symbols 文件夹并放入化学符号图片")
        return

    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    # 加载原有的类别映射（用于元素识别）
    original_class_mapping = load_class_mapping(classes_path)
    if not original_class_mapping:
        print("错误：无法加载原有的类别映射，程序退出")
        return
    print(f"成功加载了 {len(original_class_mapping)} 个元素识别类别")

    generator = EquationGeneratorV8(symbols_dir, original_class_mapping)

    # 生成所有方程式
    all_equations = generator.generate_all_equations()

    # 定义新的5类别映射（用于可视化）
    new_class_mapping = {
        'sub': 0,  # 下标
        '+': 1,  # 加号
        'aR': 2,  # arrowR
        'a2': 3,  # arrow2
        'ot': 4  # 其他
    }
    for idx, eq in enumerate(all_equations):
        image_path = f"{output_dir}/images/equation_{idx}.jpg"
        label_path = f"{output_dir}/labels/equation_{idx}.txt"
        image_label_path = f"{output_dir}/images/equation_{idx}_with_label.jpg"

        cv2.imwrite(image_path, eq.image)
        save_yolo_annotation(eq, label_path)

        # 生成带标注的可视化图（使用新的5类别映射）
        # visualize_sample_v8(image_path, label_path, image_label_path, new_class_mapping)

        print(f"第 {idx + 1} 条方程式已保存: {image_path}")

    print(f"\n=== 数据集生成完成，共 {len(all_equations)} 条方程式 ===")
    print(f"元素识别使用原有 {len(original_class_mapping)} 个类别")
    print(f"标注使用新的5类别系统: {new_class_mapping}")


if __name__ == "__main__":
    # generate_dataset_v1()
    # generate_dataset_v2()
    # generate_dataset_v3()
    # generate_dataset_v4()
    # generate_dataset_v5()
    # generate_dataset_v6()
    # generate_dataset_v7()
    generate_dataset_v8()
    # generate_dataset_final()
    # generate_dataset_final_v8()
