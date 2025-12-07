import os
import cv2
import numpy as np
from ultralytics import YOLO
from pprint import pprint
import matplotlib.pyplot as plt

# ===============================================
# 配置区域
# ===============================================
CELL_MODEL_PATH = "../Model/chemical_equations_training/cell_only_training_5struct/weights/best.pt"
ELEMENT_MODEL_PATH = "../Model/chemical_elements_training/phase2_elements_augmented_3/weights/best.pt"
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
    inter_x1 = max(x1, x3); inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4); inter_y2 = min(y2, y4)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def filter_big_boxes(boxes, threshold=0.75):
    kept = []
    for box in boxes:
        keep = True
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        for k in kept:
            area2 = (k[2] - k[0]) * (k[3] - k[1])
            if compute_iou(box, k) > threshold:
                if area1 <= area2: keep = False
                else: kept.remove(k)
                break
        if keep: kept.append(box)
    return kept

def filter_small_boxes(boxes, scores, threshold=0.5):
    order = np.argsort(scores)[::-1]
    kept = []
    used = [False]*len(boxes)
    for i in order:
        if used[i]: continue
        kept.append(i)
        for j in order:
            if used[j] or i==j: continue
            if compute_iou(boxes[i], boxes[j])>threshold: used[j]=True
    return kept

def draw_big_box(img, box, color):
    x1, y1, x2, y2 = map(int, box)
    overlay = img.copy()
    alpha = 0.35
    cv2.rectangle(overlay,(x1,y1),(x2,y2),color,-1)
    cv2.rectangle(overlay,(x1,y1),(x2,y2),color,1)
    img[:] = cv2.addWeighted(overlay, alpha, img, 1-alpha,0)

def draw_small_box(img, box, cls_name, ox, oy):
    x1,y1,x2,y2 = map(int,box)
    x1+=ox; x2+=ox; y1+=oy; y2+=oy
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
    cv2.putText(img,cls_name,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)

# ===============================================
# 图片 → TOKEN
# ===============================================
def process_single_image(img_path, output_folder, cell_model, element_model):
    img = cv2.imread(img_path)
    H,W = img.shape[:2]

    cell_results = cell_model(img)[0]
    boxes = cell_results.boxes.xyxy.cpu().numpy().astype(int)
    classes = cell_results.boxes.cls.cpu().numpy().astype(int)
    big_boxes = filter_big_boxes(boxes)
    big_cls = []
    for bb in big_boxes:
        for i, ori in enumerate(boxes):
            if np.array_equal(bb, ori):
                big_cls.append(id_to_label.get(classes[i],"ot"))
                break
    for i, box in enumerate(big_boxes): draw_big_box(img, box, color_mapping[big_cls[i]])

    collected=[]
    for bi, box in enumerate(big_boxes):
        x1,y1,x2,y2 = box
        crop = img[y1:y2, x1:x2]
        struct_label = big_cls[bi]
        result = element_model(crop, conf=0.15)[0]
        elem_boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        elem_ids = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        names = element_model.names
        indices = filter_small_boxes(elem_boxes, scores)
        for idx in indices:
            elem = names[elem_ids[idx]]
            final = struct_label if struct_label in special_no_merge else f"{struct_label}({elem})"
            ex1,ey1,ex2,ey2 = elem_boxes[idx]
            cx = x1 + (ex1+ex2)/2; cy = y1 + (ey1+ey2)/2
            collected.append((final,cx,cy))
            draw_small_box(img, elem_boxes[idx], final, x1, y1)
    collected.sort(key=lambda x: x[1])
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, os.path.basename(img_path)), img)
    return img, collected

# ===============================================
# TOKEN → AST → LaTeX
# ===============================================
def normalize_token(raw):
    text,x,y = raw
    x,y=float(x),float(y)
    if text=="aR": return ("ARROW","->",x,y)
    if text=="a2": return ("ARROW2","<->",x,y)
    if text=="+": return ("PLUS","+ ",x,y)
    if text.startswith("sub(") and text.endswith(")"): return ("SUB",text[4:-1],x,y)
    if text.startswith("ot(") and text.endswith(")"):
        v=text[3:-1]; v="O" if v=="0" else ("1" if v=="l" else v)
        return ("OT",v,x,y)
    return ("OT",text,x,y)

def parse_molecules(tokens):
    molecules=[]; current=None; last_atom=None
    side="left"; ast={"left":[],"right":[],"up":[],"arrows":[]}
    for kind,val,x,y in tokens:
        if kind=="ARROW" or kind=="ARROW2":
            if current: ast[side].append(current)
            ast["arrows"].append({"kind":kind,"val":val,"x":x,"y":y})
            side="right"; current=None; last_atom=None; continue
        if kind=="PLUS":
            if current: ast[side].append(current)
            current=None; last_atom=None; continue
        if kind=="OT":
            if val.isdigit() and current is None: current={"coef":int(val),"atoms":[]}; continue
            if current is None: current={"coef":1,"atoms":[]}
            current["atoms"].append([val,1]); last_atom=current["atoms"][-1]; continue
        if kind=="SUB" and last_atom: last_atom[1]=int(val)
    if current: ast[side].append(current)
    return ast

def parse_up(tokens):
    up=[]; current=None; last_atom=None
    for kind,val,x,y in tokens:
        if kind=="OT":
            if val.isdigit() and current is None: current={"coef":int(val),"atoms":[]}; continue
            if current is None: current={"coef":1,"atoms":[]}
            current["atoms"].append([val,1]); last_atom=current["atoms"][-1]; continue
        if kind=="SUB" and last_atom: last_atom[1]=int(val); continue
        if kind in ["PLUS","ARROW","ARROW2"]:
            if current: up.append(current); current=None; last_atom=None
    if current: up.append(current)
    return up

def build_ast(tokens):
    arrow_y=min([y for k,v,x,y in tokens if k in ["ARROW","ARROW2"]],default=None)
    normal,up=[],[]
    for t in tokens:
        k,v,x,y=t
        if arrow_y is not None and y<arrow_y: up.append(t)
        else: normal.append(t)
    normal.sort(key=lambda t:t[2])
    up.sort(key=lambda t:t[2])
    ast=parse_molecules(normal)
    ast["up"]=parse_up(up)
    return ast

def ast_to_latex(ast):
    def atom_to_str(atom):
        name,count = atom
        if name=="arrowU": return "\\uparrow"
        if name=="arrowD": return "\\downarrow"
        return name if count==1 else f"{name}_{{{count}}}"

    def molecule_to_latex(mol):
        coef = mol["coef"]
        atoms = mol["atoms"]
        s = "".join([atom_to_str(a) for a in atoms])
        return f"{coef} {s}" if coef>1 else s

    left = " + ".join([molecule_to_latex(m) for m in ast["left"]])
    right = " + ".join([molecule_to_latex(m) for m in ast["right"]])

    # 判断箭头类型
    arrow_type = "\\rightarrow"
    if ast["arrows"]:
        kind = ast["arrows"][0]["kind"]
        if kind=="ARROW": arrow_type="\\rightarrow"
        elif kind=="ARROW2": arrow_type="\\leftrightarrow"

    # 上方注释
    up = " + ".join([molecule_to_latex(m) for m in ast.get("up",[])])
    if up:
        if arrow_type=="\\rightarrow":
            arrow = f"\\xrightarrow{{{up}}}"
        else:
            arrow = f"\\xleftrightarrow{{{up}}}"
    else:
        arrow = arrow_type

    return f"${left} {arrow} {right}$"

# ===============================================
# 主函数
# ===============================================
def decode_image(img_path):
    print("Loading YOLO models...")
    cell_model = YOLO(CELL_MODEL_PATH)
    element_model = YOLO(ELEMENT_MODEL_PATH)

    print("\nRunning detection...")
    _, tokens = process_single_image(img_path, OUTPUT_FOLDER, cell_model, element_model)

    print("\n========== TOKEN LIST ==========")
    for t in tokens: print(t)

    normalized=[normalize_token(t) for t in tokens]
    ast=build_ast(normalized)

    print("\n========== AST ==========")
    pprint(ast, width=120)

    latex_str=ast_to_latex(ast)
    print("\n========== LaTeX ==========")
    print(latex_str)

    # ===============================================
    # 显示 LaTeX 图片
    # ===============================================
    plt.rcParams['text.usetex']=True
    plt.rcParams['text.latex.preamble']=r'\usepackage{amsmath}'
    plt.figure(figsize=(8,2))
    plt.text(0.05,0.5,latex_str,fontsize=24)
    plt.axis('off')
    plt.show()

    return tokens, ast, latex_str

# ===============================================
# 示例运行
# ===============================================
if __name__=="__main__":
    IMG_PATH = "../Train/test_img_1/equation_6.jpg"
    decode_image(IMG_PATH)
