import os
import random
from PIL import Image

def make_big_image_equation_pair(input_dir, output_dir, output_name="merged.jpg"):
    os.makedirs(output_dir, exist_ok=True)

    # 找出所有 equation_X 的编号
    all_ids = []
    for f in os.listdir(input_dir):
        if f.startswith("equation_") and f.endswith((".jpg", ".png", ".jpeg", ".bmp")):
            if "_with_label" not in f:  # 不要带 label 的
                try:
                    number = int(f.split("_")[1].split(".")[0])
                    all_ids.append(number)
                except:
                    continue

    if len(all_ids) < 20:
        raise ValueError("equation_X 原图数量不足 20 张！")

    # 随机选 20 个编号
    selected_ids = random.sample(all_ids, 20)

    processed_imgs = []
    target_height = 160

    # 加载对应图片对：原图 + 带标签图
    for id_ in selected_ids:
        eq_path = os.path.join(input_dir, f"equation_{id_}.png")
        lab_path = os.path.join(input_dir, f"equation_{id_}_with_label.png")

        # 如果不是 png，你可以自动适配
        if not os.path.exists(eq_path):
            eq_path = os.path.join(input_dir, f"equation_{id_}.jpg")
        if not os.path.exists(lab_path):
            lab_path = os.path.join(input_dir, f"equation_{id_}_with_label.jpg")

        img1 = Image.open(eq_path)
        img2 = Image.open(lab_path)

        # 统一高度并等比例缩放
        def resize_keep(img):
            w, h = img.size
            new_w = int(w * (target_height / h))
            return img.resize((new_w, target_height), Image.Resampling.LANCZOS)

        img1 = resize_keep(img1)
        img2 = resize_keep(img2)

        processed_imgs.append((img1, img2))  # 存成一行两个图

    # -------- 拼接为 20 行 × 2 列 --------
    rows = 20
    cols = 2

    # 计算每列最大宽度
    col_widths = [0, 0]
    for pair in processed_imgs:
        col_widths[0] = max(col_widths[0], pair[0].size[0])
        col_widths[1] = max(col_widths[1], pair[1].size[0])

    final_width = sum(col_widths)
    final_height = rows * target_height

    big_img = Image.new("RGB", (final_width, final_height), (255, 255, 255))

    # 粘贴
    y_offset = 0
    for img1, img2 in processed_imgs:
        x_offset = 0

        big_img.paste(img1, (x_offset, y_offset))
        x_offset += col_widths[0]

        big_img.paste(img2, (x_offset, y_offset))

        y_offset += target_height

    save_path = os.path.join(output_dir, output_name)
    big_img.save(save_path)
    print("拼接完成！保存路径：", save_path)


# ------------------ 使用示例 ------------------
input_folder = r"chemical equations\generated_data_v8\images"
output_folder = r"chemical equations\generated_data_v8\images"

make_big_image_equation_pair(input_folder, output_folder)
