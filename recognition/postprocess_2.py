import ast as pyast
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
# 2. Token 解析 → AST
# ===============================================

def parse_tokens(tokens):

    tokens = sorted(tokens, key=lambda t: t[2])  # x 排序

    side = "left"
    ast = {"left": [], "right": []}

    current_molecule = None
    last_atom = None

    for kind, value, x, y in tokens:

        if kind == "ARROW":
            if current_molecule:
                ast[side].append(current_molecule)
            side = "right"
            current_molecule = None
            last_atom = None
            continue

        if kind == "PLUS":
            if current_molecule:
                ast[side].append(current_molecule)
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
        ast[side].append(current_molecule)

    return ast


# ===============================================
# 3. 读取 TXT token list
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
# 4. 主入口：读取 txt → AST
# ===============================================

def decode_from_file(path):
    raw = load_tokens_from_txt(path)
    normalized = [normalize_token(t) for t in raw]
    ast_tree = parse_tokens(normalized)
    return ast_tree


# ===============================================
# 示例运行
# ===============================================

if __name__ == "__main__":

    FILE_PATH = "post_res/1.txt"

    ast_result = decode_from_file(FILE_PATH)

    print("========== AST 结果 ==========")
    pprint(ast_result, width=60)   # ★★★ 美化输出 ★★★
