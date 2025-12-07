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
    scale_percent = 90  # 缩放为原来的20%
    width = int(img_1.shape[1] * scale_percent / 100)
    height = int(img_1.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_1 = cv2.resize(img_1, dim, interpolation=cv2.INTER_AREA)
    # 添加高斯去噪
    img_2 = img_1.copy()
    # img_2 = cv2.GaussianBlur(img_1, (3, 3), 0)  # 高斯核大小为3x3
    # 自适应阈值二值化
    img_3 = cv2.adaptiveThreshold(img_2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 16)
    # 开运算
    # kernel_1 = np.ones((2, 2), np.uint8)
    # img_4 = cv2.morphologyEx(img_3, cv2.MORPH_CLOSE, kernel_1)
    img_4 = img_3.copy()
    # kernel_11 = np.ones((2, 2), np.uint8)
    # img_4 = cv2.morphologyEx(img_4, cv2.MORPH_OPEN, kernel_11)
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
        expand_pixels = 8  # 扩大10个像素
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

        # if roi_binary.size > 0:
        #     # cv2.imshow('ROI_res', roi_binary)
        #     cv2.imwrite('ROI_res.png', roi_binary)

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


def alpha(img):
    """
    为预处理后的图像添加 alpha 通道：
    - 使用 Otsu 自适应阈值获取前景（墨迹/字符）遮罩
    - 前景为不透明，背景为透明

    参数
    ------
    img : np.ndarray
        输入图像，支持灰度、BGR 或 BGRA。

    返回
    ------
    np.ndarray
        带 alpha 通道的 BGRA 图像（PNG 可直接保存）。
    """
    if img is None:
        return None

    # 若已有 alpha 通道，直接基于其余通道重建（以确保输出格式统一）
    if len(img.shape) == 3 and img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        bgr = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # 灰度输入
        gray = img
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 用 Otsu 自动阈值。对白底黑字的常见情况，使用 THRESH_BINARY_INV 得到前景=255
    _thr, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 组装 BGRA，alpha 取遮罩
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask

    return bgra



def preprocess_to_45x45(img):
    """
    对输入图片进行预处理，输出45x45的等比例缩放结果
    
    参数
    ------
    img : np.ndarray
        输入图像，BGR格式
        
    返回
    ------
    np.ndarray
        45x45的预处理结果，保持原始比例，居中显示在白底上
    """
    if img is None:
        return None
    
    # 预处理
    roi = preprocess(img)
    
    if roi is not None and (not hasattr(roi, 'size') or roi.size > 0):
        # 保持比例缩放：长边缩放到45像素，短边等比例缩放
        h, w = roi.shape[:2]
        max_size = 45
        
        # 计算缩放比例
        if h > w:
            # 高度是长边
            scale = max_size / h
            new_h = max_size
            new_w = int(w * scale)
        else:
            # 宽度是长边
            scale = max_size / w
            new_w = max_size
            new_h = int(h * scale)
        
        # 缩放图像
        roi_scaled = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 确保是3维的（BGR格式）
        if len(roi_scaled.shape) == 2:
            roi_scaled = cv2.cvtColor(roi_scaled, cv2.COLOR_GRAY2BGR)
        
        # 创建45x45的白底画布
        canvas = np.ones((45, 45, 3), dtype=np.uint8) * 255
        
        # 计算居中位置
        start_y = (45 - new_h) // 2
        start_x = (45 - new_w) // 2
        
        # 将缩放后的图像居中粘贴到画布上
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = roi_scaled
        
        return canvas
    else:
        # 如果预处理失败，返回黑色图片
        return np.zeros((45, 45, 3), dtype=np.uint8)


def detect_intersections_and_crops(img, crop_size=45, max_corners=150, quality_level=0.01, min_distance=8, dbscan_eps=8, dbscan_min_samples=1, visualize=False, margin_pixels=5, expand_pixels=5):
    """
    识别图像中的交点/角点，并以每个交点为中心裁剪固定大小(默认45x45)的小图返回。

    参数
    ------
    img : np.ndarray
        输入图像，灰度或BGR。
    crop_size : int
        返回裁剪块的边长（像素）。必须为奇数以便中心对齐，若为偶数将自动+1。
    max_corners : int
        Shi-Tomasi 角点检测的最大角点数量。
    quality_level : float
        角点质量阈值。
    min_distance : int
        角点之间的最小距离。
    dbscan_eps : int
        DBSCAN 聚类半径，用于合并相近角点。
    dbscan_min_samples : int
        DBSCAN 聚类的最小样本数。
    visualize : bool
        是否在原图上可视化检测到的角点（仅显示，不保存）。

    返回
    ------
    crops : List[np.ndarray]
        每个交点对应的 45x45 裁剪图（边界自动白色填充）。
    points : List[Tuple[int, int]]
        对应的交点坐标 (x, y)。
    """
    if img is None:
        return [], []

    # 统一为灰度
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        is_bgr = True
    elif len(img.shape) == 2:
        gray = img
        is_bgr = False
    else:
        # 其他情况（如带alpha），转到BGR再取灰度
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        is_bgr = True

    # 轻度去噪 + 二值化增强角点稳定性
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    # Shi-Tomasi 角点检测（在增强后的灰度上）
    corners = cv2.goodFeaturesToTrack(morph, max_corners, quality_level, min_distance)
    if corners is None or len(corners) == 0:
        return [], []

    # 合并相近角点
    filtered, _labels = filter_corners_dbscan(corners, eps=dbscan_eps, min_samples=dbscan_min_samples)
    if filtered is None or len(filtered) == 0:
        return [], []

    # 裁剪
    if crop_size % 2 == 0:
        crop_size += 1
    half = crop_size // 2

    h, w = gray.shape[:2]
    crops = []
    points = []

    # 为填充准备白色画布模板
    if is_bgr:
        white_pad = np.full((crop_size, crop_size, 3), 255, dtype=img.dtype)
        src_image = img
    else:
        white_pad = np.full((crop_size, crop_size), 255, dtype=img.dtype)
        src_image = gray

    for pt in filtered:
        x, y = [int(v) for v in pt.ravel()]

        # 计算裁剪窗口在原图中的坐标
        x0 = x - half
        y0 = y - half
        x1 = x + half + 1
        y1 = y + half + 1

        # 计算与画布的对齐（用于粘贴）
        pad_x0 = max(0, -x0)
        pad_y0 = max(0, -y0)
        pad_x1 = crop_size - max(0, x1 - w)
        pad_y1 = crop_size - max(0, y1 - h)

        # 与原图相交的合法区域
        src_x0 = max(0, x0)
        src_y0 = max(0, y0)
        src_x1 = min(w, x1)
        src_y1 = min(h, y1)

        # 复制到白色画布
        patch = white_pad.copy()
        patch[pad_y0:pad_y1, pad_x0:pad_x1] = src_image[src_y0:src_y1, src_x0:src_x1]

        # 将字形区域等比放大以尽量撑满 45x45
        # 1) 在patch上定位前景bbox
        if is_bgr:
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            patch_gray = patch
        # 使用Otsu阈值，取黑字为前景
        _thr, mask_patch = cv2.threshold(patch_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(mask_patch > 0))

        if coords.size == 0:
            # 没有明显前景，直接返回原patch（已是45x45）
            norm_patch = patch
        else:
            y_min_c, x_min_c = coords.min(axis=0)
            y_max_c, x_max_c = coords.max(axis=0)
            # 在裁剪前向外扩张边界（留出符号周围笔画空间）
            ph, pw = patch_gray.shape[:2]
            y_min_c = max(0, y_min_c - expand_pixels)
            x_min_c = max(0, x_min_c - expand_pixels)
            y_max_c = min(ph - 1, y_max_c + expand_pixels)
            x_max_c = min(pw - 1, x_max_c + expand_pixels)
            # 裁出扩张后的前景包围框
            content = patch[y_min_c:y_max_c + 1, x_min_c:x_max_c + 1]

            ch, cw = content.shape[:2]
            if ch == 0 or cw == 0:
                norm_patch = patch
            else:
                # 预留固定白边：最长边放大到 (crop_size - 2*margin_pixels)
                target_max = max(1, crop_size - 2 * margin_pixels)
                scale = target_max / float(max(ch, cw))
                new_w = max(1, int(round(cw * scale)))
                new_h = max(1, int(round(ch * scale)))
                interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
                resized = cv2.resize(content, (new_w, new_h), interpolation=interp)

                # 在白底上居中粘贴
                canvas = white_pad.copy()
                off_x = (crop_size - new_w) // 2
                off_y = (crop_size - new_h) // 2
                canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
                norm_patch = canvas

        crops.append(norm_patch)
        points.append((x, y))

    # 可视化（可选）
    if visualize:
        vis = img.copy() if is_bgr else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (x, y) in points:
            cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow('intersections', vis)

    return crops, points


def preprocess1(img):
    # 转换为灰度图
    img_1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 比例缩放
    scale_percent = 100
    width = int(img_1.shape[1] * scale_percent / 100)
    height = int(img_1.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_1 = cv2.resize(img_1, dim, interpolation=cv2.INTER_AREA)

    # ==============================Shi-Tomasi 角点检测================================
    img_corner = img_1.copy()
    # 检测参数
    max_corners = 100  # 要检测的最大角点数量
    quality_level = 0.001  # 角点质量水平（0-1之间）
    min_distance = 1  # 角点之间的最小欧式距离
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
        expand_pixels = 8  # 扩大10个像素
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
        img_4_resized = cv2.resize(img_1, dim, interpolation=cv2.INTER_AREA)
        roi_binary = img_4_resized[y_min_expanded:y_max_expanded, x_min_expanded:x_max_expanded]

    return roi_binary