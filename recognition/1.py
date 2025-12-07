from preprocess2 import preprocess
import cv2
import os

img_path = r"test_res2/test5.jpg"
img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"无法加载图像：{img_path}")

processed_img = preprocess(img)

# —— 保存到 test_res2 文件夹 ——
save_dir = "test_res2"
os.makedirs(save_dir, exist_ok=True)

# 自动根据原始文件名生成保存路径
save_path = os.path.join(save_dir, "processed_" + os.path.basename(img_path))

# 保存图像
cv2.imwrite(save_path, processed_img)

print(f"已保存预处理后的图片到: {save_path}")


# from preprocess2 import preprocess
# import cv2
#
# img_path = r"test_res2/test2.jpg"
# img = cv2.imread(img_path)
# if img is None:
#     raise ValueError(f"无法加载图像：{img_path}")
#
# processed_img = preprocess(img)
#
# # —— 显示预处理后的图像（不保存） ——
# cv2.imshow("Processed Image", processed_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print("预处理图像已显示。")
