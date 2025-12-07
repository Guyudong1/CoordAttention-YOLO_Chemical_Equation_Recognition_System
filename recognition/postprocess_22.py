import re
from pprint import pprint

# ===============================================
# 1. Token 归一化
# ===============================================

def normalize_token(raw):
    text, x, y = raw[0], float(raw[1]), float(raw[2])

    if text == "aR":
        return ("ARROW", "->", x, y)

    if text == "+":
        return ("PLUS", "+", x, y)

    if text.startswith("sub(") and text.endswith(")"):
        return ("SUB", text[4:-1], x, y)

    if text.startswith("ot(") and text.endswith(")"):
        val = text[3:-1]

        # OCR 修正
        if val == "0":
            val = "O"
        if val == "l":
            val = "1"

        return ("OT", val, x, y)

    return ("OT", text, x, y)

# ===============================================
# 2. 分子解析函数（支持 left/right/up）
# ===============================================

def parse_molecules(tokens):
    """解析一组 token，生成分子列表"""
    molecules = []
    current_molecule = None
    last_atom = None

    side = "left"
    ast_side = {"left": [], "right": []}  # 用于普通元素解析

    for kind, value, x, y in tokens:
        if kind == "ARROW":
            # 切换到 right
            if current_molecule:
                ast_side[side].append(current_molecule)
            side = "right"
            current_molecule = None
            last_atom = None
            continue

        if kind == "PLUS":
            if current_molecule:
                ast_side[side].append(current_molecule)
            current_molecule = None
            last_atom = None
            continue

        if kind == "OT":
            if value.isdigit() and current_molecule is None:
                current_molecule = {"coef": int(value), "atoms": []}
                continue
            if current_molecule is None:
                current_molecule = {"coef": 1, "atoms": []}

            current_molecule["atoms"].append([value, 1])
            last_atom = current_molecule["atoms"][-1]
            continue

        if kind == "SUB":
            if last_atom is not None:
                last_atom[1] = int(value)
            continue

    if current_molecule:
        ast_side[side].append(current_molecule)

    return ast_side

def parse_up_molecules(tokens):
    """解析箭头上方的 token，生成 up 列表"""
    up_molecules = []
    current_molecule = None
    last_atom = None

    for kind, value, x, y in tokens:
        if kind == "OT":
            if value.isdigit() and current_molecule is None:
                current_molecule = {"coef": int(value), "atoms": []}
                continue
            if current_molecule is None:
                current_molecule = {"coef": 1, "atoms": []}

            current_molecule["atoms"].append([value, 1])
            last_atom = current_molecule["atoms"][-1]
            continue

        if kind == "SUB":
            if last_atom is not None:
                last_atom[1] = int(value)
            continue

        if kind in ["PLUS", "ARROW"]:
            if current_molecule:
                up_molecules.append(current_molecule)
            current_molecule = None
            last_atom = None

    if current_molecule:
        up_molecules.append(current_molecule)

    return up_molecules

# ===============================================
# 3. Token 分组 + AST 构建
# ===============================================

def parse_tokens_with_up(tokens):
    # 先找到箭头位置
    arrow_y_list = [y for kind, value, x, y in tokens if kind == "ARROW"]
    arrow_y = min(arrow_y_list) if arrow_y_list else None

    normal_tokens = []
    up_tokens = []

    for token in tokens:
        kind, value, x, y = token
        if arrow_y is not None and y < arrow_y - 1e-3:
            up_tokens.append(token)
        else:
            normal_tokens.append(token)

    # 按 x 坐标排序
    normal_tokens.sort(key=lambda t: t[2])
    up_tokens.sort(key=lambda t: t[2])

    # 解析
    ast = {"up": []}

    ast.update(parse_molecules(normal_tokens))
    ast["up"] = parse_up_molecules(up_tokens)

    return ast

# ===============================================
# 4. 读取 TXT token list
# ===============================================

TOKEN_PATTERN = re.compile(
    r"\('(.+?)',\s*(?:np\.float64\()?([0-9\.]+)\)?,\s*(?:np\.float64\()?([0-9\.]+)\)?\)"
)

def load_tokens_from_txt(path):
    raw_tokens = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = TOKEN_PATTERN.match(line)
            if not m:
                print("无法解析行:", line)
                continue

            text, x, y = m.groups()
            raw_tokens.append((text, float(x), float(y)))

    return raw_tokens

# ===============================================
# 5. 主入口
# ===============================================

def decode_from_file(path):
    raw = load_tokens_from_txt(path)
    normalized = [normalize_token(t) for t in raw]
    ast_tree = parse_tokens_with_up(normalized)
    return ast_tree

# ===============================================
# 示例运行
# ===============================================

if __name__ == "__main__":
    FILE_PATH = "post_res2/2.txt"
    ast_result = decode_from_file(FILE_PATH)
    print("========== AST 结果 ==========")
    pprint(ast_result, width=100)
