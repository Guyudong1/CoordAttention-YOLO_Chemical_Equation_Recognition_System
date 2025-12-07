import os
import cv2
import numpy as np
from ultralytics import YOLO
from pprint import pprint
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from preprocess import preprocess1

# ===============================================
# 配置区域
# ===============================================
CELL_MODEL_PATH = "Model/chemical_equations_training/cell_only_training_5struct/weights/best.pt"
ELEMENT_MODEL_PATH = "Model/chemical_elements_training/phase2_elements_augmented_12/weights/best.pt"
OUTPUT_FOLDER = "decode_output"

id_to_label = {0: "sub", 1: "+", 2: "aR", 3: "a2", 4: "ot"}
color_mapping = {"sub": (0, 0, 255), "+": (0, 255, 0), "aR": (255, 0, 0), "a2": (255, 0, 0), "ot": (128, 128, 128)}
special_no_merge = {"+", "aR", "a2"}

# ===============================================
# YOLO 辅助函数
# ===============================================
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def filter_big_boxes(boxes, threshold=0.75):
    kept = []
    for i, box in enumerate(boxes):
        keep = True
        area1 = (box[2] - box[0]) * (box[3] - box[1])

        # 遍历已保留的框
        for j in range(len(kept)):
            k_box = kept[j]
            if compute_iou(box, k_box) > threshold:
                area2 = (k_box[2] - k_box[0]) * (k_box[3] - k_box[1])
                if area1 <= area2:
                    keep = False
                    break
                else:
                    kept.pop(j)
                    break

        if keep:
            kept.append(box)

    return kept


def filter_small_boxes(boxes, scores, threshold=0.5):
    order = np.argsort(scores)[::-1]
    kept = []
    used = [False] * len(boxes)
    for i in order:
        if used[i]: continue
        kept.append(i)
        for j in order:
            if used[j] or i == j: continue
            if compute_iou(boxes[i], boxes[j]) > threshold: used[j] = True
    return kept


def draw_big_box(img, box, color):
    x1, y1, x2, y2 = map(int, box)
    overlay = img.copy()
    alpha = 0.35
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
    img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_small_box(img, box, cls_name, ox, oy):
    x1, y1, x2, y2 = map(int, box)
    x1 += ox
    x2 += ox
    y1 += oy
    y2 += oy
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(img, cls_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)


# ===============================================
# 图片 → TOKEN
# ===============================================
def process_single_image(img_path, output_folder, cell_model, element_model):
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    cell_results = cell_model(img)[0]
    boxes = cell_results.boxes.xyxy.cpu().numpy().astype(int)
    classes = cell_results.boxes.cls.cpu().numpy().astype(int)

    print(f"检测到 {len(boxes)} 个大框")

    big_boxes = filter_big_boxes(boxes)
    print(f"过滤后保留 {len(big_boxes)} 个大框")

    big_cls = []
    for bb in big_boxes:
        # 查找匹配的类别
        matched = False
        for i, ori in enumerate(boxes):
            if np.array_equal(bb, ori):
                big_cls.append(id_to_label.get(classes[i], "ot"))
                matched = True
                break

        # 如果没有找到匹配，使用默认类别
        if not matched and len(classes) > 0:
            big_cls.append(id_to_label.get(classes[0], "ot"))

    # 确保长度一致
    if len(big_cls) != len(big_boxes):
        print(f"警告: 大框数量({len(big_boxes)})和类别数量({len(big_cls)})不匹配")
        # 使用默认类别填充
        while len(big_cls) < len(big_boxes):
            big_cls.append("ot")

    for i, box in enumerate(big_boxes):
        if i < len(big_cls):
            draw_big_box(img, box, color_mapping.get(big_cls[i], (128, 128, 128)))
        else:
            draw_big_box(img, box, (128, 128, 128))

    collected = []
    for bi, box in enumerate(big_boxes):
        x1, y1, x2, y2 = box
        crop = img[y1:y2, x1:x2]

        # 确保索引有效
        if bi < len(big_cls):
            struct_label = big_cls[bi]
        else:
            struct_label = "ot"

        result = element_model(crop, conf=0.15)[0]
        elem_boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        elem_ids = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        names = element_model.names

        # 检查是否有检测结果
        if len(elem_boxes) == 0:
            continue

        indices = filter_small_boxes(elem_boxes, scores)

        for idx in indices:
            if idx < len(elem_ids) and idx < len(names):
                elem = names[elem_ids[idx]]
                final = struct_label if struct_label in special_no_merge else f"{struct_label}({elem})"
                ex1, ey1, ex2, ey2 = elem_boxes[idx]
                cx = x1 + (ex1 + ex2) / 2
                cy = y1 + (ey1 + ey2) / 2
                collected.append((final, cx, cy))
                draw_small_box(img, elem_boxes[idx], final, x1, y1)
            else:
                print(f"警告: 索引 {idx} 超出范围 (elem_ids长度: {len(elem_ids)}, names长度: {len(names)})")

    if collected:
        collected.sort(key=lambda x: x[1])
    else:
        print("警告: 没有检测到任何元素")

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, img)
    print(f"图像已保存到: {output_path}")

    return img, collected


# ===============================================
# TOKEN → AST → LaTeX
# ===============================================
def normalize_token(raw):
    text, x, y = raw
    x, y = float(x), float(y)
    if text == "aR": return ("ARROW", "->", x, y)
    if text == "a2": return ("ARROW2", "<->", x, y)
    if text == "+": return ("PLUS", "+ ", x, y)
    if text.startswith("sub(") and text.endswith(")"): return ("SUB", text[4:-1], x, y)
    if text.startswith("ot(") and text.endswith(")"):
        v = text[3:-1]
        v = "O" if v == "0" else ("1" if v == "l" else v)
        return ("OT", v, x, y)
    return ("OT", text, x, y)


def parse_molecules(tokens):
    molecules = []
    current = None
    last_atom = None
    side = "left";
    ast = {"left": [], "right": [], "up": [], "arrows": []}
    for kind, val, x, y in tokens:
        if kind == "ARROW" or kind == "ARROW2":
            if current: ast[side].append(current)
            ast["arrows"].append({"kind": kind, "val": val, "x": x, "y": y})
            side = "right"
            current = None
            last_atom = None
            continue
        if kind == "PLUS":
            if current: ast[side].append(current)
            current = None
            last_atom = None
            continue
        if kind == "OT":
            if val.isdigit() and current is None: current = {"coef": int(val), "atoms": []}; continue
            if current is None: current = {"coef": 1, "atoms": []}
            current["atoms"].append([val, 1])
            last_atom = current["atoms"][-1]
            continue
        if kind == "SUB" and last_atom: last_atom[1] = int(val)
    if current: ast[side].append(current)
    return ast


def parse_up(tokens):
    up = []
    current = None
    last_atom = None
    for kind, val, x, y in tokens:
        if kind == "OT":
            if val.isdigit() and current is None: current = {"coef": int(val), "atoms": []}; continue
            if current is None: current = {"coef": 1, "atoms": []}
            current["atoms"].append([val, 1])
            last_atom = current["atoms"][-1]
            continue
        if kind == "SUB" and last_atom: last_atom[1] = int(val); continue
        if kind in ["PLUS", "ARROW", "ARROW2"]:
            if current: up.append(current); current = None; last_atom = None
    if current: up.append(current)
    return up


def build_ast(tokens):
    # 找到所有箭头的位置
    arrow_tokens = [t for t in tokens if t[0] in ["ARROW", "ARROW2"]]

    if arrow_tokens:
        # 如果有箭头，计算箭头的平均y坐标
        arrow_y = np.mean([y for k, v, x, y in arrow_tokens])
        print(f"箭头平均y坐标: {arrow_y}")

        normal, up = [], []
        for t in tokens:
            k, v, x, y = t
            # 只将比箭头高10像素以上的token归入上方
            if arrow_y is not None and y < arrow_y - 15:
                print(f"Token {t} 在箭头上方 (y={y} < 箭头y-10={arrow_y - 10})")
                up.append(t)
            else:
                normal.append(t)
    else:
        # 没有箭头的情况
        normal = tokens
        up = []

    # 按x坐标排序
    normal.sort(key=lambda t: t[2])
    up.sort(key=lambda t: t[2])

    # 解析正常部分和上方部分
    ast = parse_molecules(normal)
    ast["up"] = parse_up(up)

    return ast


def ast_to_latex(ast):
    def atom_to_str(atom):
        name, count = atom
        if name == "arrowU": return "\\uparrow"
        if name == "arrowD": return "\\downarrow"
        if name.lower() == "delta":
            return "\\Delta" if count == 1 else f"\\Delta_{{{count}}}"
        return name if count == 1 else f"{name}_{{{count}}}"

    def molecule_to_latex(mol):
        coef = mol["coef"]
        atoms = mol["atoms"]
        s = "".join([atom_to_str(a) for a in atoms])
        return f"{coef} {s}" if coef > 1 else s

    left = " + ".join([molecule_to_latex(m) for m in ast["left"]])
    right = " + ".join([molecule_to_latex(m) for m in ast["right"]])

    # 判断箭头类型
    arrow_type = "\\rightarrow"
    if ast["arrows"]:
        kind = ast["arrows"][0]["kind"]
        if kind == "ARROW":
            arrow_type = "\\rightarrow"
        elif kind == "ARROW2":
            arrow_type = "\\leftrightarrow"

    # 上方注释
    up = " + ".join([molecule_to_latex(m) for m in ast.get("up", [])])
    if up:
        if arrow_type == "\\rightarrow":
            arrow = f"\\xrightarrow{{{up}}}"
        else:
            arrow = f"\\xleftrightarrow{{{up}}}"
    else:
        arrow = arrow_type

    return f"${left} {arrow} {right}$"


# ===============================================
# GUI 应用程序
# ===============================================
class ChemicalEquationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("化学方程式识别系统")
        self.root.geometry("1200x800")

        # 初始化变量
        self.img_path = None
        self.original_image = None
        self.preprocessed_image = None  # 新增：预处理后的图像
        self.processed_image = None
        self.cell_model = None
        self.element_model = None
        self.latex_str = ""
        self.latex_img_pil = None  # 存储LaTeX图像的PIL对象

        # 设置样式
        style = ttk.Style()
        style.theme_use('clam')

        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=1)  # 图像行权重
        self.main_frame.rowconfigure(2, weight=2)  # 结果行权重（更大的权重）

        # 创建标题
        title_label = ttk.Label(self.main_frame, text="化学方程式识别系统",
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # 创建左侧控制面板
        self.create_control_panel()

        # 创建图像显示区域
        self.create_image_display()

        # 创建结果显示区域
        self.create_result_display()

        # 初始化模型
        self.load_models()

    def create_control_panel(self):
        """创建左侧控制面板"""
        control_frame = ttk.LabelFrame(self.main_frame, text="控制面板", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10), rowspan=2)

        # 文件选择按钮
        self.select_btn = ttk.Button(control_frame, text="选择图片",
                                     command=self.select_image)
        self.select_btn.grid(row=0, column=0, pady=5, sticky=tk.EW)

        # 文件路径显示
        self.file_label = ttk.Label(control_frame, text="未选择文件", wraplength=200)
        self.file_label.grid(row=1, column=0, pady=5)

        # 预测按钮
        self.predict_btn = ttk.Button(control_frame, text="开始预测",
                                      command=self.predict_equation, state=tk.DISABLED)
        self.predict_btn.grid(row=2, column=0, pady=10, sticky=tk.EW)

        # 进度条
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=3, column=0, pady=5, sticky=tk.EW)

        # 显示模式选择
        self.display_mode = tk.StringVar(value="preprocessed")
        display_frame = ttk.LabelFrame(control_frame, text="显示模式", padding="5")
        display_frame.grid(row=4, column=0, pady=10, sticky=tk.EW)

        ttk.Radiobutton(display_frame, text="预处理图像", variable=self.display_mode,
                        value="preprocessed", command=self.update_display).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(display_frame, text="原始图像", variable=self.display_mode,
                        value="original", command=self.update_display).grid(row=1, column=0, sticky=tk.W)

        # 返回主页按钮
        self.home_btn = ttk.Button(control_frame, text="返回主页",
                                   command=self.reset_app)
        self.home_btn.grid(row=5, column=0, pady=(20, 5), sticky=tk.EW)

        # 退出按钮
        self.exit_btn = ttk.Button(control_frame, text="退出程序",
                                   command=self.root.quit)
        self.exit_btn.grid(row=6, column=0, pady=5, sticky=tk.EW)

        # 设置控制面板列权重
        control_frame.columnconfigure(0, weight=1)
        control_frame.rowconfigure(7, weight=1)  # 添加弹性空间

    def create_image_display(self):
        """创建图像显示区域"""
        image_frame = ttk.LabelFrame(self.main_frame, text="图像预览", padding="10")
        image_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 创建Canvas用于显示图片
        self.image_canvas = tk.Canvas(image_frame, width=500, height=300, bg='white')
        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 添加标签说明
        self.image_label = ttk.Label(image_frame, text="请选择一张化学方程式图片",
                                     font=('Arial', 12))
        self.image_label.grid(row=1, column=0, pady=5)

        # 设置图像框架权重
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

    def create_result_display(self):
        """创建结果显示区域"""
        result_frame = ttk.LabelFrame(self.main_frame, text="识别结果", padding="10")
        result_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))

        # 设置结果框架的行权重
        result_frame.rowconfigure(1, weight=1)  # LaTeX文本区域
        result_frame.rowconfigure(3, weight=2)  # LaTeX图像区域（更大的权重）

        # LaTeX文本显示
        latex_label = ttk.Label(result_frame, text="LaTeX 公式:", font=('Arial', 11, 'bold'))
        latex_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        self.latex_text = tk.Text(result_frame, height=4, width=80,
                                  font=('Courier', 11), wrap=tk.WORD)
        self.latex_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # 添加滚动条
        latex_scroll = ttk.Scrollbar(result_frame, orient="vertical",
                                     command=self.latex_text.yview)
        latex_scroll.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.latex_text.configure(yscrollcommand=latex_scroll.set)

        # LaTeX图像显示区域
        latex_img_label = ttk.Label(result_frame, text="LaTeX 渲染图像:",
                                    font=('Arial', 11, 'bold'))
        latex_img_label.grid(row=2, column=0, sticky=tk.W, pady=(10, 5))

        # 创建Frame作为容器，用于在Canvas中居中显示图像
        self.latex_canvas_frame = ttk.Frame(result_frame)
        self.latex_canvas_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # 创建Canvas用于显示LaTeX图像
        self.latex_canvas = tk.Canvas(self.latex_canvas_frame, bg='white')
        self.latex_canvas.pack(fill=tk.BOTH, expand=True)

        # 添加滚动条
        self.latex_scroll_y = ttk.Scrollbar(self.latex_canvas_frame, orient="vertical", command=self.latex_canvas.yview)
        self.latex_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.latex_canvas.configure(yscrollcommand=self.latex_scroll_y.set)

        # 设置结果框架权重
        result_frame.columnconfigure(0, weight=1)
        self.latex_canvas_frame.rowconfigure(0, weight=1)
        self.latex_canvas_frame.columnconfigure(0, weight=1)

    def load_models(self):
        """加载YOLO模型"""
        try:
            self.progress.start()
            self.root.update()

            # 加载模型
            self.cell_model = YOLO(CELL_MODEL_PATH)
            self.element_model = YOLO(ELEMENT_MODEL_PATH)

            self.progress.stop()
            messagebox.showinfo("成功", "模型加载完成！")
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")

    def select_image(self):
        """选择图片文件并进行预处理"""
        filetypes = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("所有文件", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="选择化学方程式图片",
            filetypes=filetypes
        )

        if filename:
            try:
                # 验证文件是否为图片
                img = Image.open(filename)
                img.verify()

                self.img_path = filename
                self.file_label.config(text=os.path.basename(filename))
                self.predict_btn.config(state=tk.NORMAL)

                # 加载原始图像
                self.original_image = cv2.imread(filename)

                # 进行预处理
                self.preprocessed_image = preprocess1(self.original_image.copy())

                # 显示预处理后的图像
                self.display_preprocessed_image()

            except Exception as e:
                messagebox.showerror("错误", f"无效的图片文件: {str(e)}")

    def display_preprocessed_image(self):
        """显示预处理后的图像"""
        if self.preprocessed_image is not None:
            try:
                # 将BGR转换为RGB
                if len(self.preprocessed_image.shape) == 3:
                    rgb_img = cv2.cvtColor(self.preprocessed_image, cv2.COLOR_BGR2RGB)
                else:
                    rgb_img = cv2.cvtColor(self.preprocessed_image, cv2.COLOR_GRAY2RGB)

                # 转换为PIL Image
                pil_img = Image.fromarray(rgb_img)
                pil_img.thumbnail((500, 300), Image.Resampling.LANCZOS)

                # 转换为PhotoImage
                self.preprocessed_photo = ImageTk.PhotoImage(pil_img)

                # 在Canvas上显示
                self.image_canvas.delete("all")
                self.image_canvas.create_image(250, 150, image=self.preprocessed_photo, anchor=tk.CENTER)
                self.image_label.config(text="预处理后的图像")

            except Exception as e:
                messagebox.showerror("错误", f"无法显示预处理后的图像: {str(e)}")

    def display_original_image(self):
        """显示原始图像"""
        if self.original_image is not None:
            try:
                # 将BGR转换为RGB
                rgb_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

                # 转换为PIL Image
                pil_img = Image.fromarray(rgb_img)
                pil_img.thumbnail((500, 300), Image.Resampling.LANCZOS)

                # 转换为PhotoImage
                self.original_photo = ImageTk.PhotoImage(pil_img)

                # 在Canvas上显示
                self.image_canvas.delete("all")
                self.image_canvas.create_image(250, 150, image=self.original_photo, anchor=tk.CENTER)
                self.image_label.config(text="原始图像")

            except Exception as e:
                messagebox.showerror("错误", f"无法显示原始图像: {str(e)}")

    def update_display(self):
        """根据选择的显示模式更新图像显示"""
        if self.img_path is None:
            return

        mode = self.display_mode.get()
        if mode == "preprocessed" and self.preprocessed_image is not None:
            self.display_preprocessed_image()
        elif mode == "original" and self.original_image is not None:
            self.display_original_image()

    def predict_equation(self):
        """执行预测"""
        if not self.img_path:
            messagebox.showwarning("警告", "请先选择图片文件")
            return

        try:
            # 禁用按钮并启动进度条
            self.predict_btn.config(state=tk.DISABLED)
            self.select_btn.config(state=tk.DISABLED)
            self.progress.start()
            self.root.update()

            # 执行预测
            print(f"\n开始处理图片: {self.img_path}")

            # 使用预处理后的图像进行预测
            if self.preprocessed_image is not None:
                # 保存预处理图像到临时文件
                temp_path = os.path.join(OUTPUT_FOLDER, "temp_preprocessed.png")
                cv2.imwrite(temp_path, self.preprocessed_image)

                # 使用预处理后的图像进行处理
                processed_img, tokens = process_single_image(
                    temp_path,
                    OUTPUT_FOLDER,
                    self.cell_model,
                    self.element_model
                )
            else:
                # 如果没有预处理图像，使用原始图像
                processed_img, tokens = process_single_image(
                    self.img_path,
                    OUTPUT_FOLDER,
                    self.cell_model,
                    self.element_model
                )

            if not tokens:
                messagebox.showwarning("警告", "没有检测到任何化学元素")
                return

            # 显示处理后的图片
            self.display_processed_image(processed_img)

            # 转换为AST和LaTeX
            normalized = [normalize_token(t) for t in tokens]
            ast = build_ast(normalized)
            self.latex_str = ast_to_latex(ast)

            # 显示结果
            self.display_results(ast, self.latex_str)

            self.progress.stop()
            self.predict_btn.config(state=tk.NORMAL)
            self.select_btn.config(state=tk.NORMAL)

            messagebox.showinfo("成功", "预测完成！")

        except Exception as e:
            self.progress.stop()
            self.predict_btn.config(state=tk.NORMAL)
            self.select_btn.config(state=tk.NORMAL)
            messagebox.showerror("错误", f"预测过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def display_processed_image(self, cv2_img):
        """显示处理后的图片（带检测框）"""
        try:
            # 将BGR转换为RGB
            rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

            # 转换为PIL Image
            pil_img = Image.fromarray(rgb_img)
            pil_img.thumbnail((500, 300), Image.Resampling.LANCZOS)

            # 转换为PhotoImage
            self.processed_photo = ImageTk.PhotoImage(pil_img)

            # 在Canvas上显示
            self.image_canvas.delete("all")
            self.image_canvas.create_image(250, 150, image=self.processed_photo, anchor=tk.CENTER)
            self.image_label.config(text="检测结果 (带标注)")

        except Exception as e:
            messagebox.showerror("错误", f"无法显示处理后的图片: {str(e)}")

    def display_results(self, ast, latex_str):
        """显示结果"""
        # 清空之前的内容
        self.latex_text.delete(1.0, tk.END)

        # 显示LaTeX文本
        self.latex_text.insert(1.0, latex_str)

        # 生成并显示LaTeX图像
        self.generate_latex_image(latex_str)

        # 打印AST到控制台
        print("\n========== AST ==========")
        pprint(ast, width=120)
        print("\n========== LaTeX ==========")
        print(latex_str)

    def generate_latex_image(self, latex_str):
        """生成LaTeX图像并显示"""
        try:
            # 创建图形 - 保持原始比例
            plt.rcParams['text.usetex'] = True
            plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb,amsfonts}'
            plt.rcParams['font.size'] = 14

            # 使用原始的图形尺寸，保持比例
            fig, ax = plt.subplots(figsize=(8, 1))
            ax.text(0.5, 0.5, latex_str, fontsize=16, ha='center', va='center')
            ax.axis('off')

            # 调整布局
            plt.tight_layout()

            # 保存到临时文件
            temp_path = os.path.join(OUTPUT_FOLDER, "temp_latex.png")
            plt.savefig(temp_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close()

            # 加载图像并保持原始比例
            self.latex_img_pil = Image.open(temp_path)

            # 在Canvas中显示图像
            self.display_latex_image_in_canvas()

        except Exception as e:
            print(f"生成LaTeX图像时出错: {str(e)}")
            # 如果LaTeX渲染失败，显示文本
            self.latex_canvas.delete("all")
            canvas_width = self.latex_canvas.winfo_width()
            canvas_height = self.latex_canvas.winfo_height()
            if canvas_width < 10 or canvas_height < 10:
                canvas_width, canvas_height = 600, 150
            self.latex_canvas.create_text(canvas_width // 2, canvas_height // 2, text=latex_str,
                                          font=('Arial', 14), anchor=tk.CENTER)

    def display_latex_image_in_canvas(self):
        """在Canvas中显示LaTeX图像（保持原始比例）"""
        if self.latex_img_pil is None:
            return

        # 获取Canvas的尺寸
        canvas_width = self.latex_canvas.winfo_width()
        canvas_height = self.latex_canvas.winfo_height()

        # 如果Canvas还没有大小，使用默认值
        if canvas_width < 10 or canvas_height < 10:
            canvas_width, canvas_height = 600, 150

        # 保持原始图像比例
        img_width, img_height = self.latex_img_pil.size
        img_ratio = img_width / img_height

        # 计算在Canvas中显示的大小（保持比例）
        display_height = min(canvas_height - 20, img_height)
        display_width = int(display_height * img_ratio)

        # 调整大小（如果需要）
        if display_width > canvas_width - 20:
            display_width = canvas_width - 20
            display_height = int(display_width / img_ratio)

        # 调整图像大小
        resized_img = self.latex_img_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)

        # 转换为PhotoImage
        self.latex_photo = ImageTk.PhotoImage(resized_img)

        # 清空Canvas并显示图像
        self.latex_canvas.delete("all")

        # 计算图像在Canvas中的位置（居中）
        x_pos = (canvas_width - display_width) // 2
        y_pos = (canvas_height - display_height) // 2

        # 创建图像
        self.latex_canvas.create_image(x_pos, y_pos, image=self.latex_photo, anchor=tk.NW)

        # 配置Canvas的滚动区域
        self.latex_canvas.configure(scrollregion=self.latex_canvas.bbox("all"))

        # 绑定Canvas大小变化事件
        self.latex_canvas.bind('<Configure>', self.on_latex_canvas_configure)

    def on_latex_canvas_configure(self, event=None):
        """当Canvas大小改变时重新显示图像"""
        if self.latex_img_pil is not None:
            self.display_latex_image_in_canvas()

    def reset_app(self):
        """重置应用状态"""
        self.img_path = None
        self.original_image = None
        self.preprocessed_image = None
        self.processed_image = None
        self.file_label.config(text="未选择文件")
        self.predict_btn.config(state=tk.DISABLED)
        self.latex_img_pil = None
        self.display_mode.set("preprocessed")  # 重置显示模式

        # 清空图像显示
        self.image_canvas.delete("all")
        self.image_label.config(text="请选择一张化学方程式图片")

        # 清空结果
        self.latex_text.delete(1.0, tk.END)
        self.latex_canvas.delete("all")


# ===============================================
# 主函数
# ===============================================
def main():
    # 创建输出文件夹
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 创建主窗口
    root = tk.Tk()

    # 创建应用程序
    app = ChemicalEquationApp(root)

    # 运行主循环
    root.mainloop()


if __name__ == "__main__":
    main()