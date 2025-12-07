import cv2
import numpy as np
import os
import random
from typing import Dict
from .chemical_equation import ChemicalEquation


class EquationGeneratorV3:
    def __init__(self, symbols_dir: str, class_mapping: Dict, output_size=(600, 400)):
        self.symbols_dir = symbols_dir
        self.class_mapping = class_mapping  # {'H': 0, 'O': 1, 'plus': 2, ...}
        self.output_size = output_size
        self.symbol_cache = {}  # 缓存加载的符号图片
        
    def load_symbol(self, class_name: str, use_cache: bool = True):
        """Load a random image for a symbol"""
        # 如果允许缓存且缓存中已有图像
        if use_cache and class_name in self.symbol_cache:
            print(f"    使用缓存图像，形状: {self.symbol_cache[class_name].shape}")
            return self.symbol_cache[class_name]

        symbol_dir = os.path.join(self.symbols_dir, class_name)
        print(f"    正在查找目录: {symbol_dir}")
        
        if not os.path.exists(symbol_dir):
            print(f"    目录不存在")
            return None

        images = [f for f in os.listdir(symbol_dir) if f.lower().endswith('.png')]
        print(f"    找到 {len(images)} 个PNG文件: {images}")
        
        if not images:
            print(f"    目录中没有PNG文件")
            return None

        random_image = random.choice(images)
        img_path = os.path.join(symbol_dir, random_image)
        print(f"    尝试加载: {img_path}")

        # 尝试以不同方式加载图像
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"    cv2.IMREAD_UNCHANGED 加载失败，尝试其他方式...")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"    所有加载方式都失败!")
            return None

        print(f"    加载成功，形状: {img.shape}, 维度: {len(img.shape)}")

        # 确保图像是有效的格式
        if len(img.shape) == 2:
            print(f"      这是灰度图像，转换为BGR")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            print(f"      这是BGRA图像")
        elif len(img.shape) == 3 and img.shape[2] == 3:
            print(f"      这是BGR图像")
        else:
            print(f"    未知图像格式: {img.shape}")

        # 存入缓存
        if use_cache:
            self.symbol_cache[class_name] = img

        return img


    def blend_symbol(self, canvas, symbol_img, x, y):
        """Blend symbol with alpha channel to canvas，自动去除白底"""

        # 检查图像维度
        if len(symbol_img.shape) == 2:
            # 灰度图像 -> 转为 BGR
            h, w = symbol_img.shape
            symbol_img = cv2.cvtColor(symbol_img, cv2.COLOR_GRAY2BGR)
        elif len(symbol_img.shape) == 3:
            h, w = symbol_img.shape[:2]
        else:
            print(f"    不支持的图像维度: {symbol_img.shape}")
            return

        # 检查边界
        if y + h > canvas.shape[0] or x + w > canvas.shape[1]:
            print(f"    图像超出画布边界")
            return

        # 如果没有 alpha 通道，则自动检测白底并转为透明
        if symbol_img.shape[2] == 3:
            # 检测接近白色的像素（阈值可调）
            white_thresh = 240
            white_mask = (symbol_img[:, :, 0] > white_thresh) & \
                         (symbol_img[:, :, 1] > white_thresh) & \
                         (symbol_img[:, :, 2] > white_thresh)

            # 创建 alpha 通道
            alpha_channel = np.ones((h, w), dtype=np.uint8) * 255
            alpha_channel[white_mask] = 0  # 白色设为透明

            # 合并为 BGRA
            symbol_img = np.dstack((symbol_img, alpha_channel))

        # ---- Alpha 混合 ----
        if symbol_img.shape[2] == 4:
            alpha = symbol_img[:, :, 3] / 255.0
            symbol_rgb = symbol_img[:, :, :3]

            # Alpha blending
            for c in range(3):
                canvas[y:y + h, x:x + w, c] = (
                        symbol_rgb[:, :, c] * alpha +
                        canvas[y:y + h, x:x + w, c] * (1 - alpha)
                )
        else:
            print(f"    未能正确生成 alpha 通道")
            return

        print(f"    成功混合图像到画布（已自动去除白底）")


    def generate_simple_equation(self) -> ChemicalEquation:
        """Generate simple linear equation: 2H2 + O2 -> 2H2O"""
        equation = ChemicalEquation()

        # Create white canvas
        canvas = np.ones((self.output_size[1], self.output_size[0], 3), dtype=np.uint8) * 255
        
        # 动态计算起始位置，确保有足够空间
        total_elements = 11  # 2H2 + O2 -> 2H2O 有11个元素
        estimated_width_per_element = 60  # 估计每个元素平均宽度
        total_estimated_width = total_elements * estimated_width_per_element
        
        # 如果估计宽度超过画布，调整起始位置
        if total_estimated_width > self.output_size[0] - 100:
            current_x = 30  # 更靠左开始
        else:
            current_x = (self.output_size[0] - total_estimated_width) // 2  # 居中开始
            
        baseline_y = self.output_size[1] // 2

        # Element sequence for: 2H2 + O2 -> 2H2O
        elements_sequence = [
            '2', 'H', '2', '+', 'O', '2', 'arrowR', '2', 'H', '2', 'O'
        ]
        
        print(f"\n=== 开始生成方程式 ===")
        print(f"画布尺寸: {self.output_size}")
        print(f"起始X位置: {current_x}")
        print(f"元素序列: {elements_sequence}")

        for i, element_class in enumerate(elements_sequence):
            print(f"\n处理第 {i+1} 个元素: '{element_class}'")
            
            if element_class not in self.class_mapping:
                print(f"  错误: '{element_class}' 不在 class_mapping 中!")
                continue
            
            symbol_img = self.load_symbol(element_class, use_cache=False)
            
            if symbol_img is None:
                print(f"  无法加载符号: '{element_class}'")
                continue
            else:
                print(f"  成功加载符号，图像形状: {symbol_img.shape}")

            # 处理图像维度
            if len(symbol_img.shape) == 2:
                symbol_img = cv2.cvtColor(symbol_img, cv2.COLOR_GRAY2BGR)
            elif len(symbol_img.shape) != 3:
                print(f"  不支持的图像维度: {symbol_img.shape}")
                continue

            # Random scale - 使用更小的缩放范围节省空间
            scale = random.uniform(0.7, 1.0)  # 从 0.8-1.2 改为 0.7-1.0
            new_width = int(symbol_img.shape[1] * scale)
            new_height = int(symbol_img.shape[0] * scale)
            symbol_img = cv2.resize(symbol_img, (new_width, new_height))
            print(f"  调整后尺寸: {new_width} x {new_height}")

            # Handle subscript
            y_offset = 0
            if element_class in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                y_offset = int(new_height * 0.2)  # 减少下标偏移量
                print(f"  这是数字，应用下标偏移: {y_offset}")

            # Calculate position
            y_position = baseline_y - new_height // 2 + y_offset
            print(f"  放置位置: ({current_x}, {y_position})")

            # Check bounds with better message
            if current_x + new_width < self.output_size[0] - 20:  # 减少边界留白
                # Blend symbol to canvas
                self.blend_symbol(canvas, symbol_img, current_x, y_position)
                print(f"  成功绘制到画布")

                # Calculate YOLO annotation
                x_center = (current_x + new_width // 2) / self.output_size[0]
                y_center = (y_position + new_height // 2) / self.output_size[1]
                width = new_width / self.output_size[0]
                height = new_height / self.output_size[1]

                class_id = self.class_mapping[element_class]
                equation.add_element(element_class, (x_center, y_center, width, height), class_id)
                print(f"  添加标注: {element_class} (ID:{class_id})")

                # 减少元素间距
                current_x += new_width + random.randint(0, 1)  # 从 5-15 改为 3-8
                print(f"  更新X位置: {current_x} (剩余空间: {self.output_size[0] - current_x})")
            else:
                print(f"  超出画布边界! 当前位置: {current_x}, 元素宽度: {new_width}, 画布宽度: {self.output_size[0]}")
                # 即使超出也尝试绘制，但跳过标注
                self.blend_symbol(canvas, symbol_img, current_x, y_position)

        print(f"\n=== 生成完成 ===")
        print(f"最终方程式包含 {len(equation.elements)} 个元素")
        print(f"最终X位置: {current_x}")
        
        equation.image = canvas
        equation.width = self.output_size[0]
        equation.height = self.output_size[1]
        return equation
    



        