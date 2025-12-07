import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def filter_corners_dbscan(corners, eps=15, min_samples=2):
    """
    使用DBSCAN聚类过滤孤立角点
    """
    if corners is None:
        return None, np.array([])

    if len(corners) < 2:
        return corners, np.array([0] * len(corners)) if len(corners) > 0 else np.array([])

    corners_array = np.array([corner.ravel() for corner in corners])

    # 执行DBSCAN聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(corners_array)
    labels = clustering.labels_

    # 找出非噪声点（label != -1）
    non_noise_indices = np.where(labels != -1)[0]

    if len(non_noise_indices) > 0:
        filtered_corners = corners[non_noise_indices]
        filtered_labels = labels[non_noise_indices]
        print(f"DBSCAN过滤: 从 {len(corners)} 个角点中保留 {len(filtered_corners)} 个角点")
        return filtered_corners, filtered_labels
    else:
        print("DBSCAN过滤: 所有角点都被识别为噪声，返回原始角点")
        return corners, labels

def preprocess(img):
    # ==================================初步预处理===================================
    # 转换为灰度图
    img_1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 比例缩放
    scale_percent = 50  # 缩放为原来的20%
    width = int(img_1.shape[1] * scale_percent / 100)
    height = int(img_1.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_1 = cv2.resize(img_1, dim, interpolation=cv2.INTER_AREA)
    # 添加高斯去噪
    img_2 = cv2.GaussianBlur(img_1, (3, 3), 0)  # 高斯核大小为3x3
    # 自适应阈值二值化
    img_3 = cv2.adaptiveThreshold(img_2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 14)
    # 开运算
    img_4 = img_3.copy()
    # kernel_1 = np.ones((2, 2), np.uint8)
    # img_4 = cv2.morphologyEx(img_3, cv2.MORPH_CLOSE, kernel_1)
    kernel_11 = np.ones((3, 3), np.uint8)
    img_4 = cv2.morphologyEx(img_4, cv2.MORPH_OPEN, kernel_11)
    # 将四个图像水平拼接成一行四列
    img_5 = np.hstack([img_1, img_2, img_3, img_4])
    # 调整显示窗口大小以适应拼接后的图像
    # cv2.imshow('Pro', img_5)

    # ==============================Shi-Tomasi 角点检测================================
    img_corner = img_4.copy()
    # 检测参数
    max_corners = 100  # 要检测的最大角点数量
    quality_level = 0.01  # 角点质量水平（0-1之间）
    min_distance = 10  # 角点之间的最小欧式距离
    # 执行角点检测
    corners_1 = cv2.goodFeaturesToTrack(img_corner, max_corners, quality_level, min_distance)

    # ----------------------------------创建显示图像-----------------------------------------
    img_display_original = cv2.resize(img, dim, interpolation=cv2.INTER_AREA).copy()
    img_display_filtered = cv2.resize(img, dim, interpolation=cv2.INTER_AREA).copy()
    # ------------------------绘制原始角点（第一张图）----------------------------
    if corners_1 is not None:
        corners_1_array = np.array([corner.ravel() for corner in corners_1], dtype=np.int32)
        # 在第一张图上绘制所有原始角点（红色）
        for corner in corners_1_array:
            x, y = corner
            cv2.circle(img_display_original, (x, y), 4, (0, 0, 255), -1)  # 红色圆点
            cv2.putText(img_display_original, f'({x},{y})', (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        print(f"原始角点数量: {len(corners_1)}")
    # ------------------------使用DBSCAN去除噪声角点-----------------------------
    if corners_1 is not None:
        corners_2, labels = filter_corners_dbscan(corners_1, eps=20, min_samples=2)
        # 确保labels有正确的长度
        if corners_2 is not None:
            if len(labels) != len(corners_2):
                # 如果长度不匹配，创建对应的标签
                if len(corners_2) == len(corners_1):
                    labels = np.array([0] * len(corners_2))  # 所有点属于同一个簇
                else:
                    labels = np.array([0] * len(corners_2))  # 默认标签
        # 在第二张图上绘制过滤后的角点（用不同颜色显示不同簇）
        if corners_2 is not None:
            corners_2_array = np.array([corner.ravel() for corner in corners_2], dtype=np.int32)
            # 为每个簇分配不同颜色
            colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0),
                      (255, 0, 255), (0, 128, 255), (255, 128, 0), (128, 255, 0)]
            # 获取非噪声点的标签
            non_noise_labels = labels[labels != -1]
            for i, corner in enumerate(corners_2_array):
                x, y = corner
                color_idx = non_noise_labels[i] % len(colors)
                color = colors[color_idx]
                # 绘制角点
                cv2.circle(img_display_filtered, (x, y), 4, color, -1)
                cv2.putText(img_display_filtered, f'C{non_noise_labels[i]}', (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                # 在原始图上也用绿色标记保留的点（便于对比）
                cv2.circle(img_display_original, (x, y), 6, (0, 255, 0), 2)
            print(f"过滤后角点数量: {len(corners_2)}")
            # 在过滤后的图上添加统计信息
            cv2.putText(img_display_filtered, f'Clusters: {len(set(non_noise_labels))}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(img_display_filtered, f'Points: {len(corners_2)}',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        # 在原始图上添加统计信息
        cv2.putText(img_display_original, f'Original Points: {len(corners_1)}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img_display_original, f'Filtered Points: {len(corners_2) if corners_2 is not None else 0}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # 显示两张对比图
    img_6 = np.hstack([img_display_original, img_display_filtered])
    # cv2.imshow('compare', img_6)

    # ==============================边界范围计算和ROI提取================================
    roi_binary = None
    if corners_2 is not None and len(corners_2) > 0:
        corners_array = np.array([corner.ravel() for corner in corners_2], dtype=np.int32)

        # 计算边界矩形
        x_min, y_min = np.min(corners_array, axis=0)
        x_max, y_max = np.max(corners_array, axis=0)

        # 扩大边界（在周围加一圈）
        expand_pixels = 25  # 扩大10个像素
        x_min_expanded = max(0, x_min - expand_pixels)
        y_min_expanded = max(0, y_min - expand_pixels)
        x_max_expanded = min(width, x_max + expand_pixels)
        y_max_expanded = min(height, y_max + expand_pixels)

        # 创建带有边界的显示图像
        img_with_boundaries = cv2.resize(img, dim, interpolation=cv2.INTER_AREA).copy()

        # 绘制角点
        for corner in corners_array:
            x, y = corner
            cv2.circle(img_with_boundaries, (x, y), 3, (0, 0, 255), -1)

        # 绘制原始边界矩形（绿色）
        cv2.rectangle(img_with_boundaries, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # 绘制扩大后的红边框
        cv2.rectangle(img_with_boundaries, (x_min_expanded, y_min_expanded),
                      (x_max_expanded, y_max_expanded), (0, 0, 255), 3)

        # ROI提取
        # 在二值图上分割（使用扩大后的边界）
        img_4_resized = cv2.resize(img_4, dim, interpolation=cv2.INTER_AREA)
        roi_binary = img_4_resized[y_min_expanded:y_max_expanded, x_min_expanded:x_max_expanded]
        # 显示边界和ROI结果
        # cv2.imshow('Expanded_Boundaries', img_with_boundaries)

        if roi_binary.size > 0:
            # cv2.imshow('ROI_res', roi_binary)
            cv2.imwrite('ROI_res.png', roi_binary)

        # 打印边界信息
        print(f"\n=== 边界信息 ===")
        print(f"原始边界矩形: ({x_min}, {y_min}) - ({x_max}, {y_max})")
        print(f"原始边界尺寸: {x_max - x_min} x {y_max - y_min}")
        print(f"扩大后边界: ({x_min_expanded}, {y_min_expanded}) - ({x_max_expanded}, {y_max_expanded})")
        print(f"扩大后尺寸: {x_max_expanded - x_min_expanded} x {y_max_expanded - y_min_expanded}")
        print(f"扩大像素: {expand_pixels}")
    else:
        print("没有有效的角点进行边界计算")

    return roi_binary