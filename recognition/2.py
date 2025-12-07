from ultralytics import YOLO
import cv2

# 加载训练好的最佳模型
model = YOLO('../Model/chemical_equations_training/cell_only_training_v2/weights/best.pt')
# model = YOLO('../Model/chemical_equations_training/phase2_equations_finetune/weights/best.pt')
# model = YOLO("../Model/chemical_elements_training/phase2_elements_augmented_60x60/weights/best.pt")

# 读取图片
img = cv2.imread('test_res2/processed_test5.jpg')

# 进行预测
results = model(img)

# 显示预测结果
results[0].show()
