import cv2
import numpy as np
import os
import random
from typing import Dict
from .chemical_equation import ChemicalEquation


class EquationGeneratorV5:
    def __init__(self, symbols_dir: str, class_mapping: Dict, output_size=(1100, 150)):
        self.symbols_dir = symbols_dir
        self.class_mapping = class_mapping  # {'H': 0, 'O': 1, 'plus': 2, ...}
        self.output_size = output_size
        self.symbol_cache = {}  # 缓存加载的符号图片

    def load_symbol(self, class_name: str, use_cache: bool = True):
        """Load a random image for a symbol"""
        # 如果允许缓存且缓存中已有图像
        if use_cache and class_name in self.symbol_cache:
            print(f"  使用缓存图像，形状: {self.symbol_cache[class_name].shape}")
            return self.symbol_cache[class_name]

        symbol_dir = os.path.join(self.symbols_dir, class_name)
        print(f"   正在查找目录: {symbol_dir}")

        if not os.path.exists(symbol_dir):
            print(f"    ? 目录不存在")
            return None

        images = [f for f in os.listdir(symbol_dir) if f.lower().endswith('.png')]
        print(f"    找到 {len(images)} 个PNG文件: {images}")

        if not images:
            print(f"    ? 目录中没有PNG文件")
            return None

        random_image = random.choice(images)
        img_path = os.path.join(symbol_dir, random_image)
        print(f"    尝试加载: {img_path}")

        # 尝试以不同方式加载图像
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"    ? cv2.IMREAD_UNCHANGED 加载失败，尝试其他方式...")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            print(f"    ? 所有加载方式都失败!")
            return None

        print(f"    ? 加载成功，形状: {img.shape}, 维度: {len(img.shape)}")

        # 确保图像是有效的格式
        if len(img.shape) == 2:
            print(f"      这是灰度图像，转换为BGR")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            print(f"      这是BGRA图像")
        elif len(img.shape) == 3 and img.shape[2] == 3:
            print(f"      这是BGR图像")
        else:
            print(f"    ? 未知图像格式: {img.shape}")

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
                canvas[y:y+h, x:x+w, c] = (
                    symbol_rgb[:, :, c] * alpha +
                    canvas[y:y+h, x:x+w, c] * (1 - alpha)
                )
        else:
            print(f"    未能正确生成 alpha 通道")
            return

        print(f"    成功混合图像到画布（已自动去除白底）")


    # def generate_simple_equation(self) -> ChemicalEquation:
    #         """Generate simple linear equation: 2H2 + O2 -> 2H2O"""
    #         equation = ChemicalEquation()
    #
    #         canvas = np.ones((self.output_size[1], self.output_size[0], 3), dtype=np.uint8) * 255
    #
    #         total_elements = 11
    #         estimated_width_per_element = 60
    #         total_estimated_width = total_elements * estimated_width_per_element
    #
    #         if total_estimated_width > self.output_size[0] - 100:
    #             current_x = 30
    #         else:
    #             current_x = (self.output_size[0] - total_estimated_width) // 2
    #
    #         baseline_y = self.output_size[1] // 2
    #
    #         elements_sequence = [
    #             ['2', 'H', '2', '+', 'O', '2', 'arrowR', '2', 'H', '2', 'O'],  # 普通化合反应, 2H₂ + O₂ → 2H₂O
    #             ['Zn', '+', '2', 'H', 'Cl', 'arrowR', 'Zn', 'Cl', '2', '+', 'H', '2'],  # 置换反应, Zn + 2HCl → ZnCl₂ + H₂
    #             ['N', 'a', '2', 'C', 'O', '3', '+', '2', 'H', 'C', 'l', 'arrowR',
    #              '2', 'N', 'a', 'C', 'l', '+', 'H', '2', 'O', '+', 'C', 'O', '2'],
    #             # 复分解反应, Na₂CO₃ + 2HCl → 2NaCl + H₂O + CO₂
    #             ['N', '2', '+', '3', 'H', '2', 'arrow2', '2', 'N', 'H', '3', 'arrowU'],  # 可逆反应, N₂ + 3H₂ ↔ 2NH₃
    #             ['Ca', 'C', 'O', '3', 'delta', 'arrowR', 'Ca', 'O', '+', 'C', 'O', '2'],  # 加热反应, CaCO₃ △→ CaO + CO₂
    #             ['2', 'H', '2', 'O', '2', '{', 'M', 'n', 'O', '2', '}', 'arrowR', '2', 'H', '2', 'O', '+', 'O', '2',
    #              'arrowU'],  # 催化条件反应, 2H₂O₂ [MnO₂]→ 2H₂O + O₂
    #             ['A', 'g', '+', 'C', 'l', 'arrowR', 'A', 'g', 'C', 'l', 'arrowD'],  # 离子反应, Ag⁺ + Cl⁻ → AgCl↓
    #             ['2', 'H', '2', 'O', 'arrowR', '2', 'H', '2', 'arrowU', '+', 'O', '2', 'arrowU'],
    #             # 电解反应, 2H₂O → 2H₂ ↑ + O₂ ↑
    #             ['C', 'a', '(', 'O', 'H', ')', '2', 'arrowR', 'C', 'a', 'O', '+', 'H', '2', 'O'],
    #             # 氢氧根 (OH⁻), Ca(OH)₂ → CaO + H₂O
    #             ['Al', '2', '(', 'S', 'O', '4', ')', '3', '+', '3', 'Ba', 'C', 'l', '2', 'arrowR',
    #              '2', 'A', 'l', 'C', 'l', '3', '+', '3', 'B', 'a', '(', 'S', 'O', '4', ')'],
    #             # 硫酸根 (SO₄²⁻), Al₂(SO₄)₃ + 3BaCl₂ → 2AlCl₃ + 3BaSO₄
    #             ['2', 'N', 'H', '4', 'C', 'l', '+', 'C', 'a', '(', 'O', 'H', ')', '2', 'arrowR',
    #              '2', 'N', 'H', '3', 'arrowU', '+', 'C', 'a', 'C', 'l', '2', '+', '2', 'H', '2', 'O']
    #             # 铵盐 (NH₄⁺), 2NH₄Cl + Ca(OH)₂ → 2NH₃↑ + CaCl₂ + 2H₂O
    #         ]
    #
    #         # 分类定义
    #         num_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #         sym_classes = ['arrowR', 'arrowU', 'arrowD', 'arrow2', '+', '-', '=', '(', ')', '[', ']', '{', '}']
    #         # 从 classes.txt 读取所有类别
    #         base_dir = os.path.dirname(os.path.abspath(__file__))
    #         classes_path = os.path.join(base_dir, "..", "config", "classes.txt")
    #         with open(classes_path, 'r', encoding='utf-8') as f:
    #             all_classes = [line.strip() for line in f if line.strip()]
    #         # che_classes 是 all_classes 中不在 num_classes 和 sym_classes 的部分
    #         che_classes = [cls for cls in all_classes if cls not in num_classes and cls not in sym_classes]
    #
    #         print(f"数字类别: {num_classes}")
    #         print(f"符号类别: {sym_classes}")
    #         print(f"化学元素类别: {che_classes}")
    #
    #         print(f"\n=== 开始生成方程式 ===")
    #         print(f"元素序列: {elements_sequence}")
    #
    #         prev_class = None  # 上一个元素名称
    #         prev_type = None  # 上一个元素类别 ('num', 'che', 'sym')
    #
    #         for i, element_class in enumerate(elements_sequence):
    #             print(f"\n处理第 {i + 1} 个元素: '{element_class}'")
    #
    #             if element_class not in self.class_mapping:
    #                 print(f"  ? 错误: '{element_class}' 不在 class_mapping 中!")
    #                 continue
    #
    #             symbol_img = self.load_symbol(element_class, use_cache=False)
    #             if symbol_img is None:
    #                 print(f"  ? 无法加载符号: '{element_class}'")
    #                 continue
    #
    #             if len(symbol_img.shape) == 2:
    #                 symbol_img = cv2.cvtColor(symbol_img, cv2.COLOR_GRAY2BGR)
    #
    #             # ---- 识别当前类别类型 ----
    #             if element_class in num_classes:
    #                 curr_type = 'num'
    #             elif element_class in che_classes:
    #                 curr_type = 'che'
    #             else:
    #                 curr_type = 'sym'
    #
    #             # ---- 判断是否为下标 ----
    #             is_subscript = (curr_type == 'num' and prev_type == 'che')
    #             is_symbol = (curr_type == 'sym')
    #             is_che = (curr_type == 'che')
    #
    #             # ---- 缩放与位置计算 ----
    #             if is_subscript:
    #                 sub_scale = 0.5
    #                 new_width = int(symbol_img.shape[1] * sub_scale)
    #                 new_height = int(symbol_img.shape[0] * sub_scale)
    #                 symbol_img = cv2.resize(symbol_img, (new_width, new_height))
    #                 print(f"  → 作为下标数字，缩放比例: {sub_scale}")
    #                 y_position = baseline_y + int(new_height * 0.2)
    #                 x_position = current_x - int(new_width * 0.4)
    #             else:
    #                 scale = random.uniform(0.8, 1.0)
    #                 new_width = int(symbol_img.shape[1] * scale)
    #                 new_height = int(symbol_img.shape[0] * scale)
    #                 symbol_img = cv2.resize(symbol_img, (new_width, new_height))
    #                 y_position = baseline_y - new_height // 2
    #                 x_position = current_x
    #
    #             # ---- 绘制与标注 ----
    #             if x_position + new_width < self.output_size[0] - 20:
    #                 self.blend_symbol(canvas, symbol_img, x_position, y_position)
    #                 print(f" 绘制成功: {element_class} at ({x_position},{y_position})")
    #
    #                 x_center = (x_position + new_width // 2) / self.output_size[0]
    #                 y_center = (y_position + new_height // 2) / self.output_size[1]
    #                 width = new_width / self.output_size[0]
    #                 height = new_height / self.output_size[1]
    #                 class_id = self.class_mapping[element_class]
    #                 equation.add_element(element_class, (x_center, y_center, width, height), class_id)
    #
    #                 next_class = elements_sequence[i + 1] if i + 1 < len(elements_sequence) else None
    #
    #                 if is_subscript:
    #                     if next_class in che_classes:
    #                         current_x += new_width - random.randint(10, 11)  # 下标+che间隔小
    #                     else:
    #                         current_x += new_width + random.randint(1, 2)  # 下标+非che间隔大
    #                 elif is_symbol:
    #                     current_x += new_width + random.randint(3, 5)  # 普通元素
    #                 elif is_che and next_class in che_classes:
    #                     current_x += new_width - random.randint(2, 3)
    #                 else:
    #                     current_x += new_width + random.randint(0, 1)
    #
    #
    #             else:
    #                 print(f"  超出画布边界，跳过标注")
    #
    #             prev_class = element_class
    #             prev_type = curr_type  # 保存前一个类别类型
    #
    #         print(f"\n=== 生成完成 ===")
    #         print(f"最终方程式包含 {len(equation.elements)} 个元素")
    #
    #         equation.image = canvas
    #         equation.width = self.output_size[0]
    #         equation.height = self.output_size[1]
    #         return equation

    def generate_all_equations(self):
        """
        遍历 elements_sequence 列表，每条方程式生成一张图像和对应标注
        返回：ChemicalEquation 对象列表
        """
        # 所有方程式序列
        elements_sequences = [
            ['2', 'H', '2', '+', 'O', '2', 'arrowR', '2', 'H', '2', 'O'], # 普通化合反应, 2H₂ + O₂ → 2H₂O
            ['Zn', '+', '2', 'H', 'Cl', 'arrowR', 'Zn', 'Cl', '2', '+', 'H', '2', 'arrowU'], # 置换反应, Zn + 2HCl → ZnCl₂ + H₂
            ['Na', '2', 'C', 'O', '3', '+', '2', 'H', 'Cl', 'arrowR', '2', 'Na', 'Cl', '+', 'H', '2', 'O', '+', 'C', 'O', '2', 'arrowU'], # 复分解反应, Na₂CO₃ + 2HCl → 2NaCl + H₂O + CO₂
            ['N', '2', '+', '3', 'H', '2', 'arrow2', '2', 'N', 'H', '3'], # 可逆反应, N₂ + 3H₂ ↔ 2NH₃
            ['Ca', 'C', 'O', '3', '{', 'Delta', '}', 'arrowR', 'Ca', 'O', '+', 'C', 'O', '2'], # 加热反应, CaCO₃ △→ CaO + CO₂
            ['2', 'H', '2', 'O', '2', '{', 'Mn', 'O', '2', '}', 'arrowR', '2', 'H', '2', 'O', '+', 'O', '2', 'arrowU'], # 催化条件反应, 2H₂O₂ [MnO₂]→ 2H₂O + O₂
            ['Ag', '+', '+', 'Cl', '-', 'arrowR', 'Ag', 'Cl', 'arrowD'], # 离子反应, Ag⁺ + Cl⁻ → AgCl↓
            ['2', 'H', '2', 'O', 'arrowR', '2', 'H', '2', 'arrowU', '+', 'O', '2', 'arrowU'], # 电解反应, 2H₂O → 2H₂ ↑ + O₂ ↑
            ['Ca', '(', 'O', 'H', ')', '2', 'arrowR', 'Ca', 'O', '+', 'H', '2', 'O'], # 氢氧根 (OH⁻), Ca(OH)₂ → CaO + H₂O
            ['Al', '2', '(', 'S', 'O', '4', ')', '3', '+', '3', 'Ba', 'Cl', '2', 'arrowR', '2', 'Al', 'Cl', '3', '+', '3', 'Ba', '(', 'S', 'O', '4', ')'], # 硫酸根 (SO₄²⁻), Al₂(SO₄)₃ + 3BaCl₂ → 2AlCl₃ + 3BaSO₄
            ['2', 'N', 'H', '4', 'Cl', '+', 'Ca', '(', 'O', 'H', ')', '2', 'arrowR', '2', 'N', 'H', '3', 'arrowU', '+', 'Ca', 'Cl', '2', '+', '2', 'H', '2', 'O'] # 铵盐 (NH₄⁺), 2NH₄Cl + Ca(OH)₂ → 2NH₃↑ + CaCl₂ + 2H₂O
        ]


        all_equations = []

        # 分类定义
        num_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        sym_classes = ['arrowR', 'arrowU', 'arrowD', 'arrow2', '+', '-', '=', '(', ')', '[', ']', '{', '}']
        # 从 classes.txt 读取所有类别
        base_dir = os.path.dirname(os.path.abspath(__file__))
        classes_path = os.path.join(base_dir, "..", "config", "classes.txt")
        with open(classes_path, 'r', encoding='utf-8') as f:
            all_classes = [line.strip() for line in f if line.strip()]
        # che_classes 是 all_classes 中不在 num_classes 和 sym_classes 的部分
        che_classes = [cls for cls in all_classes if cls not in num_classes and cls not in sym_classes]
        print(f"数字类别: {num_classes}")
        print(f"符号类别: {sym_classes}")
        print(f"化学元素类别: {che_classes}")

        for seq_idx, elements_sequence in enumerate(elements_sequences):
            print(f"\n=== 正在生成第 {seq_idx+1} 条方程式 ===")
            equation = ChemicalEquation()

            canvas = np.ones((self.output_size[1], self.output_size[0], 3), dtype=np.uint8) * 255

            total_elements = len(elements_sequence)
            estimated_width_per_element = 60
            total_estimated_width = total_elements * estimated_width_per_element
            if total_estimated_width > self.output_size[0] - 100:
                current_x = 30
            else:
                current_x = (self.output_size[0] - total_estimated_width) // 2

            baseline_y = self.output_size[1] // 2

            prev_type = None
            in_braces = False
            brace_buffer = []

            for i, element_class in enumerate(elements_sequence):
                if element_class not in self.class_mapping:
                    print(f"  ? 错误: '{element_class}' 不在 class_mapping 中!")
                    continue

                symbol_img = self.load_symbol(element_class, use_cache=False)
                if symbol_img is None:
                    print(f"  ? 无法加载符号: '{element_class}'")
                    continue

                if len(symbol_img.shape) == 2:
                    symbol_img = cv2.cvtColor(symbol_img, cv2.COLOR_GRAY2BGR)

                # ---- 识别当前类别类型 ----
                if element_class in num_classes:
                    curr_type = 'num'
                elif element_class in che_classes:
                    curr_type = 'che'
                else:
                    curr_type = 'sym'

                is_subscript = (curr_type == 'num' and (prev_type == 'che' or prev_type == 'sym' and prev_element in [')', ']']))
                is_symbol = (curr_type == 'sym')
                is_che = (curr_type == 'che')

                # ---- 缩放与位置计算 ----
                if is_subscript:
                    sub_scale = 0.5
                    new_width = int(symbol_img.shape[1] * sub_scale)
                    new_height = int(symbol_img.shape[0] * sub_scale)
                    symbol_img = cv2.resize(symbol_img, (new_width, new_height))
                    y_position = baseline_y + int(new_height * 0.2)
                    x_position = current_x - int(new_width * 0.4)
                else:
                    next_class = elements_sequence[i + 1] if i + 1 < len(elements_sequence) else None
                    if element_class in ['arrowR', 'arrow2']:
                        scale = random.uniform(1.0, 1.2)
                        new_width = int(symbol_img.shape[1] * scale * (1.5 if element_class == 'arrowR' else 1.0))
                        new_height = int(symbol_img.shape[0] * scale * (1.0 if element_class == 'arrowR' else 1.5))
                        y_position = baseline_y - new_height // 2
                    elif element_class in ['+', '-'] and next_class in sym_classes:
                        scale = 0.5
                        new_width = int(symbol_img.shape[1] * scale)
                        new_height = int(symbol_img.shape[0] * scale)
                        y_position = baseline_y - int(new_height * 1)
                    else:
                        scale = random.uniform(0.8, 1.0)
                        new_width = int(symbol_img.shape[1] * scale)
                        new_height = int(symbol_img.shape[0] * scale)
                        y_position = baseline_y - new_height // 2  # 默认基线居中

                    symbol_img = cv2.resize(symbol_img, (new_width, new_height))
                    x_position = current_x

                # ---- {} 条件处理 ----
                # 如果遇到 '{'，记录开始，不绘制
                if element_class == '{':
                    in_braces = True
                    brace_buffer = []
                    continue
                # 如果遇到 '}'，结束{}，把brace_buffer内容绘制在箭头上方或下方
                if element_class == '}':
                    in_braces = False
                    # 基础缩小比例
                    base_scale = 0.7
                    brace_x_pos = x_position
                    for idx, sym in enumerate(brace_buffer):
                        cond_img = self.load_symbol(sym)
                        if cond_img is None:
                            continue

                        # 判断类型
                        if sym in num_classes:
                            curr_type = 'num'
                        elif sym in che_classes:
                            curr_type = 'che'
                        else:
                            curr_type = 'sym'

                        # 下标判断：沿用之前逻辑
                        is_subscript = (curr_type == 'num' and (
                                    prev_type == 'che' or prev_type == 'sym' and prev_element in [')', ']']))
                        if is_subscript:
                            sub_scale = 0.5
                            new_w = int(cond_img.shape[1] * base_scale * sub_scale)
                            new_h = int(cond_img.shape[0] * base_scale * sub_scale)
                            y_offset = int(new_h * 0.2)
                        else:
                            new_w = int(cond_img.shape[1] * base_scale)
                            new_h = int(cond_img.shape[0] * base_scale)
                            y_offset = 0

                        cond_img = cv2.resize(cond_img, (new_w, new_h))
                        cond_y = baseline_y - new_h // 2 - new_height + y_offset  # 放在箭头上方
                        cond_x = brace_x_pos

                        self.blend_symbol(canvas, cond_img, cond_x, cond_y)

                        # 标注
                        x_center = (cond_x + new_w // 2) / self.output_size[0]
                        y_center = (cond_y + new_h // 2) / self.output_size[1]
                        width = new_w / self.output_size[0]
                        height = new_h / self.output_size[1]
                        class_id = self.class_mapping[sym]
                        equation.add_element(sym, (x_center, y_center, width, height), class_id)

                        brace_x_pos += new_w + 2
                        prev_type = curr_type
                        prev_element = sym

                    brace_buffer = []
                    continue
                # 在 {} 内的内容先存起来，不绘制
                if in_braces:
                    brace_buffer.append(element_class)
                    continue

                # ---- 绘制与标注 ----
                if x_position + new_width < self.output_size[0] - 20:
                    self.blend_symbol(canvas, symbol_img, x_position, y_position)
                    print(f" 绘制成功: {element_class} at ({x_position},{y_position})")

                    x_center = (x_position + new_width // 2) / self.output_size[0]
                    y_center = (y_position + new_height // 2) / self.output_size[1]
                    width = new_width / self.output_size[0]
                    height = new_height / self.output_size[1]
                    class_id = self.class_mapping[element_class]
                    equation.add_element(element_class, (x_center, y_center, width, height), class_id)

                    next_class = elements_sequence[i + 1] if i + 1 < len(elements_sequence) else None

                    if is_subscript:
                        if next_class in che_classes:
                            current_x += new_width - random.randint(10, 11)  # 下标+che间隔小
                        elif next_class in [']', ')']:
                            current_x += new_width - random.randint(11, 13)
                        elif next_class in ['(', '[']:
                            current_x += new_width - random.randint(11, 13)
                        else:
                            current_x += new_width - random.randint(1, 2)  # 下标+非che间隔大
                    elif is_symbol and element_class not in ['(', ')', '[', ']']:
                        current_x += new_width + random.randint(3, 5)  # 普通元素
                    elif element_class in ['(', '['] or next_class in [']', ')']:
                        current_x += new_width - random.randint(9, 11)
                    elif element_class in [']', ')'] or next_class in ['(', '[']:
                        current_x += new_width - random.randint(8, 10)
                    elif is_che and next_class in che_classes:
                        current_x += new_width - random.randint(2, 3)
                    else:
                        current_x += new_width +  random.randint(0, 1)


                else:
                    print(f"  超出画布边界，跳过标注")

                prev_type = curr_type
                prev_element = element_class

            equation.image = canvas
            equation.width = self.output_size[0]
            equation.height = self.output_size[1]

            all_equations.append(equation)
            print(f"=== 第 {seq_idx+1} 条方程式生成完成 === 共 {len(equation.elements)} 个元素")

        return all_equations

