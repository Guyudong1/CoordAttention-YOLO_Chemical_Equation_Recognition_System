import cv2
import numpy as np
import os
import random
from typing import Dict
from .chemical_equation_v8 import ChemicalEquation


class EquationGeneratorV8:
    def __init__(self, symbols_dir: str, class_mapping: Dict, output_size=None):
        """
        初始化生成器
        output_size 已弃用，仅保留兼容性；实际画布将根据方程式内容自动调整
        """
        self.symbols_dir = symbols_dir
        self.class_mapping = class_mapping
        self.output_size = output_size  # 已弃用
        self.symbol_cache = {}

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


    def generate_all_equations(self, num_per_equation=1):
        """
        遍历 elements_sequence 列表，每条方程式生成一张图像和对应标注
        返回：ChemicalEquation 对象列表
        (动态画布：先排版计算尺寸，再创建画布并绘制)
        """
        # 所有方程式序列
        elements_sequences = [
            # 基本合成 / 置换 / 复分解
            ['2', 'H', '2', '+', 'O', '2', 'arrowR', '2', 'H', '2', 'O'],  # 水的生成: 2H2 + O2 -> 2H2O
            ['Zn', '+', '2', 'H', 'Cl', 'arrowR', 'Zn', 'Cl', '2', '+', 'H', '2', 'arrowU'],  # Zn + 2HCl -> ZnCl2 + H2
            ['Na', '2', 'C', 'O', '3', '+', '2', 'H', 'Cl', 'arrowR', '2', 'Na', 'Cl', '+', 'H', '2', 'O', '+', 'C',
             'O', '2', 'arrowU'],  # Na2CO3 + 2HCl -> 2NaCl + H2O + CO2
            ['N', '2', '+', '3', 'H', '2', 'arrow2', '2', 'N', 'H', '3'],  # N2 + 3H2 ? 2NH3
            ['C', '+', 'O', '2', 'arrowR', 'C', 'O', '2'],  # 燃烧：碳燃烧 C + O2 -> CO2
            ['2', 'C', 'H', '4', '+', '3', 'O', '2', 'arrowR', '2', 'C', 'O', '2', '+', '4', 'H', '2', 'O'],
            # 甲烷燃烧 2CH4 + 3O2 -> 2CO2 + 4H2O

            # 热分解 / 催化 / 条件
            ['Ca', 'C', 'O', '3', '{', 'Delta', '}', 'arrowR', 'Ca', 'O', '+', 'C', 'O', '2'],  # CaCO3 △→ CaO + CO2
            ['2', 'H', '2', 'O', '2', '{', 'Mn', 'O', '2', '}', 'arrowR', '2', 'H', '2', 'O', '+', 'O', '2',
             'arrowU'],  # 过氧化氢在 MnO2 催化下分解
            ['K', 'Cl', 'O', '3', '{', 'Delta', '}', 'arrowR', 'K', 'Cl', '+', 'O', '2'],
            # KClO3 加热 -> KCl + O2

            # 沉淀 / 离子反应 / 中和
            ['Ag', '+', 'Cl', 'arrowR', 'Ag', 'Cl', 'arrowD'],  # Ag+ + Cl- -> AgCl↓
            ['Na', 'Cl', '+', 'Ag', 'N', 'O', '3', 'arrowR', 'Ag', 'Cl', 'arrowD', '+', 'Na',
             'N', 'O', '3'],  # AgNO3 + NaCl -> AgCl↓ + NaNO3
            ['Na', 'O', 'H', '+', 'H', 'Cl', 'arrowR', 'Na', 'Cl', '+', 'H', '2', 'O'],
            # NaOH + HCl -> NaCl + H2O (中和)
            ['H', '2', 'S', '+', 'Cu', 'S', 'O', '4', 'arrowR', 'Cu', 'S', '+', 'H', '2', 'S', 'O', '4'],
            # H2S 与 CuSO4 型置换 (示例)

            # 气体生成 / 电解 / 氧化还原
            ['2', 'H', '2', 'O', 'arrowR', '2', 'H', '2', 'arrowU', '+', 'O', '2', 'arrowU'],
            # 电解水 2H2O -> 2H2 ↑ + O2 ↑
            ['2', 'K', 'Mn', 'O', '4', '+', '1', '0', 'H', 'Cl', 'arrowR', '2', 'K', 'Cl', '+', '2', 'Mn',
             'Cl', '2', '+', '8', 'H', '2', 'O', '+', '5', 'Cl', '2'],  # KMnO4 + HCl (氧化还原)
            ['Cu', '+', '2', 'N', 'O', '3', 'arrowR', 'Cu', '(', 'N', 'O', '3', ')', '2'],  # Cu + HNO3 (示例，实际有复杂产物)

            # 配位 / 络合 / 催化条件（更复杂的注记）
            ['Cu', 'S', 'O', '4', '+', '4', 'N', 'H', '3', 'arrowR', '{', '[', 'Cu', '(N', 'H', '3', ')', '4',
             ']', '}', '+', 'S', 'O', '4'],  # CuSO4 + 4NH3 -> [Cu(NH3)4] SO4

            # 有机反应 / 燃烧 / 氧化
            ['C', '2', 'H', '5', 'O', 'H', '+', '3', 'O', '2', 'arrowR', '2', 'C', 'O', '2', '+', '3', 'H', '2', 'O'],
            # 乙醇燃烧

            # 更复杂的双替换 / 沉淀 / 碳酸盐 / 酸式盐
            ['Ca', 'C', 'O', '3', '+', '2', 'H', 'Cl', 'arrowR', 'Ca', 'Cl', '2', '+', 'H', '2', 'O',
             '+', 'C', 'O', '2'],
            ['Na', '2', 'S', 'O', '4', '+', 'Ca', 'Cl', '2', 'arrowR', 'Ca', 'S', 'O', '4', '+', '2',
             'Na', 'Cl'],  # Na2SO4 + CaCl2 -> CaSO4 + 2NaCl
            ['Al', '2', '(', 'S', 'O', '4', ')', '3', '+', '3', 'Ba', 'Cl', '2', 'arrowR', '2', 'Al',
             'Cl', '3', '+', '3', 'Ba', '(', 'S', 'O', '4', ')'],  # Al2(SO4)3 + 3BaCl2 -> 2AlCl3 + 3BaSO4

            # 氨 / 铵 / 氨化反应 / 氧化产物
            ['2', 'N', 'H', '4', 'Cl', '+', 'Ca', '(', 'O', 'H', ')', '2', 'arrowR', '2', 'N', 'H', '3',
             'arrowU', '+', 'Ca', 'Cl', '2', '+', '2', 'H', '2', 'O'],
            # 2NH4Cl + Ca(OH)2 -> 2NH3 ↑ + CaCl2 + 2H2O

            # 氧化还原常见例子
            ['2', 'K', 'I', '+', 'Cl', '2', 'arrowR', '2', 'K', 'Cl', '+', 'I', '2'],  # 2KI + Cl2 -> 2KCl + I2
            ['Cl', '2', '+', 'H', '2', 'O', 'arrowR', 'H', 'Cl', '+', 'H', 'O', 'Cl'],
            # Cl2 + H2O -> HCl + HOCl (次氯酸)

            # 可逆 / 平衡及箭头2（垂直双向或可逆）
            ['N', '2', 'O', '4', 'arrow2', '2', 'N', 'O', '2'],  # N2O4 ? 2NO2
            ['2', 'N', 'O', '2', 'arrow2', 'N', '2', 'O', '4'],  # 逆方向（平衡）

            # 气体生成（碳酸盐 + 酸产生二氧化碳）
            ['Mg', 'C', 'O', '3', '+', '2', 'H', 'Cl', 'arrowR', 'Mg', 'Cl', '2', '+', 'H', '2', 'O',
             '+', 'C', 'O', '2'],

            # 金属置换和腐蚀类
            ['Fe', '+', 'Cu', 'S', 'O', '4', 'arrowR', 'Fe', 'S', 'O', '4', '+', 'Cu'],
            # Fe + CuSO4 -> FeSO4 + Cu

            # 氧化剂 / 还原剂反应（示例）
            ['K', '2', 'Cr', '2', 'O', '7', '+', '3', 'S', '+', '4', 'H', '2', 'O', 'arrowR', 'K', '2', 'S', 'O',
             '4', '+', 'Cr', '2', 'S', 'O', '7', '+', '6', 'H'],  # 示意（实际请核对）

            # 碱金属 / 卤素反应例子
            ['2', 'Na', '+', 'Cl', '2', 'arrowR', '2', 'Na', 'Cl'],  # 2Na + Cl2 -> 2NaCl

            # 氧化 / 还原中产生氧气或氢气的例子
            ['2', 'H', '2', 'O', '2', 'arrowR', '2', 'H', '2', 'O', '+', 'O', '2'],

            # 过渡金属催化的复杂示例（含 {} 条件）
            ['C', '2', 'H', '5', 'O', 'H', '+', 'O', '2', '{', 'Pt', '}', 'arrowR', 'C', 'O', '2', '+', 'H', '2', 'O'],

            # 酸性氧化物 / 吸水反应
            ['S', 'O', '3', '+', 'H', '2', 'O', 'arrowR', 'H', '2', 'S', 'O', '4'],

            # 形成盐的常见示例
            ['H', '2', 'S', '+', 'Ba', 'O', 'H', '2', 'arrowR', 'Ba', 'S', '+', 'H', '2', 'O'],
            #  Ba(OH)2 + H2S -> BaS + 2H2O

            # 过氧化物 / 次级氧化
            ['2', 'K', 'O', '2', '+', '2', 'H', '2', 'O', 'arrowR', '2', 'K', 'O', 'H', '+', 'H', '2', 'O', '2'],

            # 补充一些教学常用方程（酸碱、金属氧化）
            ['Ca', 'Cl', '2', '+', '2', 'Na', 'O', 'H', 'arrowR', 'Ca', '(', 'O', 'H', ')', '2', '+', '2',
             'Na', 'Cl'],  # CaCl2 + 2NaOH -> Ca(OH)2 + 2NaCl
            ['Fe', '2', 'O', '3', '+', '3', 'C', 'O', 'arrowR', '2', 'Fe', '+', '3', 'C', 'O', '2'],
            # 铁(III)氧化物被 CO 还原（示例）

            ['2', 'K', 'Br', 'O', '3', '+', '5', 'S', 'O', '2', '+', '3', 'H', '2', 'O', 'arrowR', 'K', '2', 'S', 'O', '4', '+', '2', 'H', 'Br', 'O', '3'],
            # 溴酸钾氧化
            ['C', 'O', '+', 'C', 'arrowR', 'C', 'O', '2'],  # CO + C -> CO2
            ['2', 'H', '2', '+', 'C', 'O', 'arrowR', 'C', 'H', '3', 'O', 'H'],  # CO + 2H2 -> CH3OH
            ['2', 'N', 'O', '+', 'O', '2', 'arrowR', '2', 'N', 'O', '2'],  # 一氧化氮氧化
            ['P', '4', '+', '5', 'O', '2', 'arrowR', '2', 'P', '2', 'O', '5'],  # 磷燃烧
            ['Ca', 'O', '+', 'H', '2', 'O', 'arrowR', 'Ca', '(', 'O', 'H', ')', '2'],  # 熟石灰
            ['Fe', '+', 'S', 'arrowR', 'Fe', 'S'],  # 铁与硫化合
            ['2', 'S', 'O', '2', '+', 'O', '2', 'arrowR', '2', 'S', 'O', '3'],  # 硫氧化
            ['Ca', 'O', '+', 'C', 'arrowR', 'Ca', 'C', 'O', '3'],  # CaO + C -> CaCO3
            ['2', 'H', '2', '+', 'C', 'O', '2', 'arrowR', 'C', 'H', '4', '+', 'O', '2'],  # 甲烷生成
            ['2', 'H', '2', '+', 'S', 'arrowR', '2', 'H', '2', 'S'],  # 制硫化氢
            ['C', 'H', '4', '+', 'Cl', '2', 'arrowR', 'C', 'H', '3', 'Cl', '+', 'H', 'Cl'],  # 氯化甲烷
            ['2', 'N', 'H', '3', '+', '3', 'C', 'u', 'O', 'arrowR', 'N', '2', '+', '3', 'C', 'u', '+', '3', 'H', '2', 'O']  # 氨与氧化铜反应
        ]

        all_equations = []
        # 新的类别映射
        new_class_mapping = {
            'sub': 0,  # 下标
            '+': 1,  # 加号
            'aR': 2,  # arrowR
            'a2': 3,  # arrow2
            'ot': 4  # 其他
        }
        # 颜色映射
        color_mapping = {
            'sub': (0, 0, 255),  # 红色
            '+': (0, 255, 0),  # 绿色
            'aR': (255, 0, 0),  # 蓝色
            'a2': (255, 0, 0),  # 蓝色
            'ot': (0, 0, 0)  # 黑色
        }

        # 分类定义
        num_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        sym_classes = ['arrowR', 'arrowU', 'arrowD', 'arrow2', '+', '-', '=', '(', ')', '[', ']', '{', '}']

        # 从 classes.txt 读取所有类别
        base_dir = os.path.dirname(os.path.abspath(__file__))
        classes_path = os.path.join(base_dir, "..", "config", "classes.txt")
        with open(classes_path, 'r', encoding='utf-8') as f:
            all_classes = [line.strip() for line in f if line.strip()]
        che_classes = [cls for cls in all_classes if cls not in num_classes and cls not in sym_classes]

        print(f"数字类别: {num_classes}")
        print(f"符号类别: {sym_classes}")
        print(f"化学元素类别: {che_classes}")

        for seq_idx, elements_sequence in enumerate(elements_sequences):
            print(f"\n=== 正在生成第 {seq_idx + 1} 条方程式 ===")
            for k in range(num_per_equation):

                print(f"  → 第 {k + 1}/{num_per_equation} 张变体")
                equation = ChemicalEquation()

                # 设置新的类别映射和颜色映射
                equation.new_class_mapping = new_class_mapping
                equation.color_mapping = color_mapping

                # ---- 随机扰动参数（每次生成不同）----
                random_spacing_scale = random.uniform(0.8, 1.2)  # 间距随机缩放
                random_offset_x = random.randint(-5, 5)  # 整体左右偏移
                random_offset_y = random.randint(-5, 5)  # 整体上下偏移
                random_rotate = random.uniform(-2, 2)  # 整体微旋转
                random_blur = random.choice([0, 1, 2])  # 轻微模糊
                random_scale = random.uniform(0.95, 1.05)  # 整体放缩

                # ---- 第一遍：排版计算（仅计算尺寸与相对 x，不创建画布） ----
                positions = []  # 将按顺序保存各单元，单元包含 seq_index
                current_x = 0
                spacing_default = 5
                prev_type = None
                prev_element = None
                in_braces = False
                brace_buffer = []
                brace_groups = []

                max_up = 0
                max_down = 0

                for i, element_class in enumerate(elements_sequence):
                    # 处理 '{' 开始
                    if element_class == '{':
                        in_braces = True
                        brace_buffer = []
                        continue

                    # 处理 '}' 结束 —— 记录 brace buffer 并查找将来绑定的箭头位置
                    if element_class == '}':
                        in_braces = False
                        base_scale = 0.7
                        group_items = []
                        group_width = 0
                        group_max_h = 0
                        # 这里加入下标逻辑
                        for sym in brace_buffer:
                            sym_img = self.load_symbol(sym, use_cache=False)
                            if sym_img is None:
                                continue
                            if len(sym_img.shape) == 2:
                                sym_img = cv2.cvtColor(sym_img, cv2.COLOR_GRAY2BGR)

                            # 类型判断
                            if sym in num_classes:
                                sym_type = 'num'
                            elif sym in che_classes:
                                sym_type = 'che'
                            else:
                                sym_type = 'sym'

                            # 下标判断
                            is_sub = (sym_type == 'num' and (
                                        prev_type == 'che' or (prev_type == 'sym' and prev_element in [')', ']'])))

                            # 缩放处理
                            if is_sub:
                                sub_scale = 0.35
                                new_w = int(sym_img.shape[1] * base_scale * sub_scale)
                                new_h = int(sym_img.shape[0] * base_scale * sub_scale)
                            else:
                                new_w = int(sym_img.shape[1] * base_scale)
                                new_h = int(sym_img.shape[0] * base_scale)

                            group_items.append({
                                "class": sym,
                                "img": sym_img,
                                "width": new_w,
                                "height": new_h,
                                "is_sub": is_sub,
                                "type": sym_type
                            })
                            group_width += new_w + 2
                            group_max_h = max(group_max_h, new_h)
                            prev_type = sym_type
                            prev_element = sym

                        # 寻找绑定箭头
                        anchor_idx = None
                        for j in range(i + 1, len(elements_sequence)):
                            if elements_sequence[j] in ['arrowR', 'arrow2']:
                                anchor_idx = j
                                break
                        brace_groups.append({
                            "items": group_items,
                            "anchor_seq_idx": anchor_idx,
                            "width": group_width,
                            "height": group_max_h + 10
                        })
                        brace_buffer = []
                        continue

                    if in_braces:
                        brace_buffer.append(element_class)
                        continue

                    # 其他符号处理
                    if element_class not in self.class_mapping:
                        positions.append({
                            "kind": "skip",
                            "seq_idx": i,
                            "class": element_class,
                            "x": current_x,
                            "width": 10,
                            "height": 10
                        })
                        current_x += 10 + spacing_default
                        prev_element = element_class
                        continue

                    symbol_img = self.load_symbol(element_class, use_cache=False)
                    if symbol_img is None:
                        positions.append({
                            "kind": "skip",
                            "seq_idx": i,
                            "class": element_class,
                            "x": current_x,
                            "width": 10,
                            "height": 10
                        })
                        current_x += 10 + spacing_default
                        prev_element = element_class
                        continue

                    if len(symbol_img.shape) == 2:
                        symbol_img = cv2.cvtColor(symbol_img, cv2.COLOR_GRAY2BGR)

                    h, w = symbol_img.shape[:2]

                    # 识别类别
                    if element_class in num_classes:
                        curr_type = 'num'
                    elif element_class in che_classes:
                        curr_type = 'che'
                    else:
                        curr_type = 'sym'

                    # 下标判断
                    is_subscript = (curr_type == 'num' and (
                                prev_type == 'che' or (prev_type == 'sym' and prev_element in [')', ']'])))

                    # ---------- 修改点1：处理 + / - 缩小上移 ----------
                    next_class = elements_sequence[i + 1] if i + 1 < len(elements_sequence) else None
                    if element_class in ['+', '-'] and next_class in sym_classes:
                        scale = random.uniform(0.3, 0.5)
                        new_w = max(1, int(w * scale))
                        new_h = max(1, int(h * scale))
                    elif is_subscript:
                        sub_scale = 0.5
                        new_w = max(1, int(w * sub_scale))
                        new_h = max(1, int(h * sub_scale))
                    else:
                        scale = random.uniform(0.8, 1.0)
                        new_w = max(1, int(w * scale))
                        new_h = max(1, int(h * scale))
                    # --------------------------------------------------

                    positions.append({
                        "kind": "symbol",
                        "seq_idx": i,
                        "class": element_class,
                        "img": symbol_img,
                        "width": new_w,
                        "height": new_h,
                        "is_sub": is_subscript,
                        "type": curr_type
                    })

                    # 更新 current_x 与上下估计
                    if is_subscript:
                        max_down = max(max_down, int(new_h * 0.5))
                        current_x += new_w - random.randint(1, 2)
                    else:
                        current_x += new_w + random.randint(0, 5)
                        if element_class in ['arrowR', 'arrow2']:
                            max_up = max(max_up, int(new_h * 0.6))
                    max_up = max(max_up, int(new_h * 0.6))

                    prev_type = curr_type
                    prev_element = element_class

                # 第一遍结束：计算最终画布尺寸
                total_width = max(1, current_x) + 10
                margin = 60
                canvas_w = total_width + margin * 2
                baseline_y = max_up + margin
                canvas_h = baseline_y + max_down + margin
                canvas_h = max(canvas_h, int((max_up + max_down) * 1.2) + 20)
                canvas_w = max(canvas_w, 64)
                canvas_h = max(canvas_h, 32)

                print(
                    f" 排版结果 -> 总宽: {total_width}, max_up: {max_up}, max_down: {max_down}, 画布: {canvas_w}x{canvas_h}")

                # ---- 第二遍：创建画布并绘制 ----
                canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

                # seq_idx -> 绘制位置映射（用于 brace_group 锚定）
                seq_to_drawpos = {}

                draw_x = margin
                for pos in positions:
                    if pos["kind"] == "skip":
                        draw_x += pos["width"] + spacing_default
                        continue

                    if pos["kind"] == "symbol":
                        cls = pos["class"]
                        w = pos["width"]
                        h = pos["height"]
                        is_sub = pos["is_sub"]
                        typ = pos.get("type", None)
                        seq_idx_local = pos["seq_idx"]

                        # resize
                        try:
                            resized_img = cv2.resize(pos["img"], (w, h))
                        except Exception:
                            resized_img = np.ones((h, w, 3), dtype=np.uint8) * 255

                        # ---------- 修改点：箭头水平拉伸 ----------
                        if cls in ['arrowR', 'arrow2']:
                            stretch_factor = 1.9  # 横向拉伸倍数，可调整
                            new_w_stretch = max(1, int(resized_img.shape[1] * stretch_factor))
                            resized_img = cv2.resize(resized_img, (new_w_stretch, h))
                            w = new_w_stretch

                        if cls in ['arrowU', 'arrowD']:
                            stretch_factor = 1.5  # 垂直拉伸倍数，可调整
                            new_h_stretch = max(1, int(resized_img.shape[0] * stretch_factor))
                            resized_img = cv2.resize(resized_img, (w, new_h_stretch))
                            h = new_h_stretch

                        if is_sub:
                            y_pos = baseline_y + int(h * 0.2)
                            x_pos = draw_x - int(w * 0.4)
                        else:
                            y_pos = baseline_y - h // 2
                            x_pos = draw_x

                        # ---------- 新增：+ / - 上移 ----------
                        next_class = elements_sequence[i + 1] if i + 1 < len(elements_sequence) else None
                        if cls in ['+', '-'] and next_class in sym_classes:
                            y_pos -= int(h * 1)
                            x_pos -= int(w * 0.5)

                        # boundary corrections
                        if y_pos < 0:
                            y_pos = 0
                        if x_pos < 0:
                            x_pos = 0
                        if y_pos + h > canvas_h:
                            h = max(1, canvas_h - y_pos)
                            resized_img = resized_img[:h, :, :]
                        if x_pos + w > canvas_w:
                            w = max(1, canvas_w - x_pos)
                            resized_img = resized_img[:, :w, :]

                        self.blend_symbol(canvas, resized_img, x_pos, y_pos)
                        print(f" 绘制成功: {cls} at ({x_pos},{y_pos}) size({w}x{h})")

                        # === 新的标注逻辑 ===
                        # 确定新的类别
                        if is_sub:
                            new_cls = 'sub'
                        elif cls == '+':
                            new_cls = '+'
                        elif cls == 'arrowR':
                            new_cls = 'aR'
                        elif cls == 'arrow2':
                            new_cls = 'a2'
                        else:
                            new_cls = 'ot'

                        # 标注（使用新的类别）
                        x_center = (x_pos + w // 2) / canvas_w
                        y_center = (y_pos + h // 2) / canvas_h
                        width_n = w / canvas_w
                        height_n = h / canvas_h
                        class_id = new_class_mapping.get(new_cls, -1)
                        equation.add_element(
                            original_class=cls,  # 原始类别
                            new_class=new_cls,  # 新类别
                            bbox=(x_center, y_center, width_n, height_n),
                            class_id=class_id,
                            color=color_mapping.get(new_cls, (0, 0, 0))  # 对应的颜色
                        )

                        # 记录绘制位置以便 brace_group 锚定
                        seq_to_drawpos[seq_idx_local] = (x_pos, w)

                        # 如果这个是箭头，检查是否有 brace_group 要绑定到这个箭头（anchor_seq_idx == this seq_idx）
                        if cls in ['arrowR', 'arrow2']:
                            # 找所有 brace_groups with anchor == this seq_idx
                            for bg in brace_groups:
                                if bg["anchor_seq_idx"] == seq_idx_local:
                                    group_w = bg["width"]
                                    group_h = bg["height"]
                                    # 把 group 居中放在箭头正上方
                                    arrow_x, arrow_w = x_pos, w
                                    group_x = arrow_x + (arrow_w // 2) - (group_w // 2)
                                    # 避免越界
                                    if group_x < 0:
                                        group_x = 0
                                    if group_x + group_w > canvas_w:
                                        group_x = max(0, canvas_w - group_w)
                                    group_y = baseline_y - group_h - 6  # 紧靠箭头上方一点（6 像素间隙）

                                    # 绘制 group 内的每个 item（从左到右）
                                    gx = group_x
                                    for item in bg["items"]:
                                        try:
                                            item_img = cv2.resize(item["img"], (item["width"], item["height"]))
                                        except Exception:
                                            item_img = np.ones((item["height"], item["width"], 3), dtype=np.uint8) * 255

                                        if item["is_sub"]:
                                            y_pos_item = group_y + int(item["height"] * 1.4)
                                        else:
                                            y_pos_item = group_y

                                        # 绘制
                                        self.blend_symbol(canvas, item_img, gx, y_pos_item)

                                        # === 新的标注逻辑（brace group中的元素）===
                                        item_cls = item["class"]
                                        if item["is_sub"]:
                                            item_new_cls = 'sub'
                                        elif item_cls == '+':
                                            item_new_cls = '+'
                                        elif item_cls == 'arrowR':
                                            item_new_cls = 'aR'
                                        elif item_cls == 'arrow2':
                                            item_new_cls = 'a2'
                                        else:
                                            item_new_cls = 'ot'

                                        # 标注
                                        actual_y = y_pos_item  # 使用实际的y坐标
                                        actual_h = item["height"]  # 使用实际的高度

                                        # 标注 - 使用实际的绘制位置和尺寸
                                        x_c = (gx + item["width"] // 2) / canvas_w
                                        y_c = (actual_y + actual_h // 2) / canvas_h
                                        w_n = item["width"] / canvas_w
                                        h_n = actual_h / canvas_h
                                        item_class_id = new_class_mapping.get(item_new_cls, -1)
                                        equation.add_element(
                                            original_class=item_cls,
                                            new_class=item_new_cls,
                                            bbox=(x_c, y_c, w_n, h_n),
                                            class_id=item_class_id,
                                            color=color_mapping.get(item_new_cls, (0, 0, 0))
                                        )

                                        gx += item["width"] + 2
                                    # 标记为已绘制，防止重复绘制（将 anchor_seq_idx 设为 None）
                                    bg["anchor_seq_idx"] = None

                        # update draw_x 同原逻辑相近
                        if is_sub:
                            draw_x += max(1, w - random.randint(1, 2))
                        elif cls in ['(', '[']:
                            draw_x += max(1, w - random.randint(9, 11))
                        elif cls in [']', ')']:
                            draw_x += max(1, w - random.randint(8, 10))
                        elif typ == 'sym' and cls not in ['(', ')', '[', ']']:
                            draw_x += w + random.randint(-4, 4)
                        elif typ == 'che' and False:
                            # placeholder (保留结构)
                            draw_x += max(1, w - random.randint(2, 3))
                        else:
                            draw_x += w + random.randint(-5, 5)

                    else:
                        # 备用：不应到这里
                        draw_x += spacing_default

                # 未绑定brace group处理（需要更新标注逻辑）
                for bg in brace_groups:
                    if bg["anchor_seq_idx"] is not None:
                        group_w = bg["width"]
                        group_h = bg["height"]
                        group_x = max(margin, canvas_w - margin - group_w)
                        group_y = baseline_y - group_h - 6
                        gx = group_x
                        for item in bg["items"]:
                            try:
                                item_img = cv2.resize(item["img"], (item["width"], item["height"]))
                            except Exception:
                                item_img = np.ones((item["height"], item["width"], 3), dtype=np.uint8) * 255

                            if item["is_sub"]:
                                y_pos_item = group_y + int(item["height"] * 0.3)
                            else:
                                y_pos_item = group_y

                            self.blend_symbol(canvas, item_img, gx, y_pos_item)

                            # === 新的标注逻辑（未绑定brace group中的元素）===
                            item_cls = item["class"]
                            if item["is_sub"]:
                                item_new_cls = 'sub'
                            elif item_cls == '+':
                                item_new_cls = '+'
                            elif item_cls == 'arrowR':
                                item_new_cls = 'aR'
                            elif item_cls == 'arrow2':
                                item_new_cls = 'a2'
                            else:
                                item_new_cls = 'ot'

                            actual_y = y_pos_item
                            actual_h = item["height"]

                            x_c = (gx + item["width"] // 2) / canvas_w
                            y_c = (actual_y + actual_h // 2) / canvas_h
                            w_n = item["width"] / canvas_w
                            h_n = actual_h / canvas_h
                            item_class_id = new_class_mapping.get(item_new_cls, -1)

                            equation.add_element(
                                original_class=item_cls,
                                new_class=item_new_cls,
                                bbox=(x_c, y_c, w_n, h_n),
                                class_id=item_class_id,
                                color=color_mapping.get(item_new_cls, (0, 0, 0))
                            )
                            gx += item["width"] + 2
                        bg["anchor_seq_idx"] = None

                # 完成此方程绘制
                equation.image = canvas
                equation.width = canvas_w
                equation.height = canvas_h
                all_equations.append(equation)

        return all_equations
