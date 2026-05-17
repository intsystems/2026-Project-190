import cv2
import numpy as np
import os
import torch
from u_net_binarization import binarize_image, binarize_image_with_loaded_model
from post_processing import crop_line_rectangle
import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import time
import gc
from contextlib import contextmanager
from scipy.ndimage import label
from typing import Tuple

# Используя yolo модель выделяем маску стрниц, бинаризуем изображени поворачивам соотвествующие станницы  разрезам

class YoloMaskNotFoundError(RuntimeError):
    """
    Ошибка отсутствия масок страниц в ответе YOLO.
    """
    pass

def extract_pages_with_yolo(
    image_path,
    model_path,
    output_dir="debug_images",
    conf_threshold=0.3,
    debug=False,
    binary_image=None, 
    return_binary=False,
    yolo_model=None,
    unet_model=None,
    unet_device=None,
    return_page_infos=False,
):
    os.makedirs(output_dir, exist_ok=True)
    model = None
    owns_model = yolo_model is None

    # Загрузка изображения
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить: {image_path}")
    else:
        img = image_path.copy()

    # Бинаризация
    if binary_image is not None:
        binary = binary_image.copy()
    elif unet_model is not None:
        binary = binarize_image_with_loaded_model(
            image=img,
            model=unet_model,
            device=unet_device if unet_device is not None else "cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        binary = binarize_image(img)

    # Гарантируем 2D бинарное изображение.
    if len(binary.shape) == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)
    # Жестко бинаризуем, чтобы убрать полутона после чтения/конвертации.
    binary = np.where(binary < 128, 0, 255).astype(np.uint8)

    if debug:
        cv2.imwrite(os.path.join(output_dir, "01_binary.jpg"), binary)

    # YOLO
    try:
        model = yolo_model if yolo_model is not None else YOLO(model_path)
        results = model(img, conf=conf_threshold, verbose=False)
        if not (results and len(results) > 0 and results[0].masks is not None):
            raise YoloMaskNotFoundError("Маски не найдены")

        annotated = results[0].plot()
        if debug:
            cv2.imwrite(os.path.join(output_dir, "02_yolo_result.jpg"), annotated)

        masks = results[0].masks.data.cpu().numpy()
        orig_h, orig_w = binary.shape
        pages = []
        binary_pages = []
        page_infos = []

        for i, mask in enumerate(masks):
            mask_bin = (mask > 0.5).astype(np.uint8) * 255
            mask_big = cv2.resize(mask_bin, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            if debug:
                cv2.imwrite(os.path.join(output_dir, f"03_mask_big_{i}.jpg"), mask_big)

            contours, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)

            # Вырезаем из цветного и бинарного изображения по одной и той же маске
            page_roi = img[y:y+h, x:x+w]
            binary_roi = binary[y:y+h, x:x+w]
            mask_roi = mask_big[y:y+h, x:x+w]

            # Создаём белый фон размером с вырезанную область
            white_bg = np.full_like(page_roi, 255)   # (h, w, 3) все пиксели = 255
            white_bg_binary = np.full_like(binary_roi, 255)

            # Накладываем цветные пиксели только там, где маска == 255
            mask_3ch = np.stack([mask_roi] * 3, axis=-1)  # (h, w, 3) для поэлементного выбора
            result_page = np.where(mask_3ch == 255, page_roi, white_bg)
            result_binary_page = np.where(mask_roi == 255, binary_roi, white_bg_binary)

            if debug:
                cv2.imwrite(os.path.join(output_dir, f"07_page_{i}.jpg"), result_page)
                cv2.imwrite(os.path.join(output_dir, f"08_binary_page_{i}.jpg"), result_binary_page)
            pages.append(result_page)
            binary_pages.append(result_binary_page)
            page_infos.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
            })

        if return_binary and return_page_infos:
            return pages, binary_pages, page_infos
        if return_binary:
            return pages, binary_pages
        if return_page_infos:
            return pages, page_infos
        return pages
    finally:
        if owns_model:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def correct_perspective(image, debug=False, debug_output_dir="debug_images", return_matrix=False):
    """
    Короткое описание:
        Оценивает угол наклона текста методом горизонтальных проекций.
    Вход:
        image (np.ndarray): исходное изображение (BGR или grayscale).
        debug (bool): сохранять промежуточные результаты.
        debug_output_dir (str): папка для debug-файлов.
    Выход:
        rotated_image (np.ndarray): повернутое исходное изображение.
        rotated_binary (np.ndarray): повернутое бинарное изображение.
    """
    if debug:
        os.makedirs(debug_output_dir, exist_ok=True)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = binarize_image(gray)
    else:
        gray = image.copy()
        # Если вход уже бинарный (0/255), не бинаризуем повторно.
        uniq = np.unique(gray)
        if uniq.size <= 3 and np.all(np.isin(uniq, [0, 255])):
            binary = gray.astype(np.uint8)
        else:
            binary = binarize_image(gray)

    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)
    text_mask = (binary < 128).astype(np.uint8)

    angles = np.linspace(-30.0, 30.0, 120)
    scores = []

    h, w = text_mask.shape
    center = (w // 2, h // 2)

    best_score = -1.0
    for angle in angles:
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated_mask = cv2.warpAffine(
            text_mask,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        profile = np.sum(rotated_mask, axis=1)
        score = float(np.var(profile))
        scores.append(score)

        if score >= best_score:
            best_score = score

    best_idx = int(np.argmax(scores))
    best_angle = float(angles[best_idx])
    best_matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)

    if len(image.shape) == 3:
        rotated_image = cv2.warpAffine(
            image,
            best_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    else:
        rotated_image = cv2.warpAffine(
            image,
            best_matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    rotated_binary = cv2.warpAffine(
        binary,
        best_matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    if debug:
        # График score(angle) через OpenCV.
        graph_h, graph_w = 420, 900
        graph = np.full((graph_h, graph_w, 3), 255, dtype=np.uint8)
        pad_l, pad_r, pad_t, pad_b = 60, 20, 20, 50
        x_min, x_max = float(angles.min()), float(angles.max())
        y_min, y_max = float(min(scores)), float(max(scores))
        y_range = max(y_max - y_min, 1e-9)

        points = []
        for a, s in zip(angles, scores):
            x = int(pad_l + (a - x_min) / (x_max - x_min) * (graph_w - pad_l - pad_r))
            y = int(graph_h - pad_b - (s - y_min) / y_range * (graph_h - pad_t - pad_b))
            points.append((x, y))
        for i in range(1, len(points)):
            cv2.line(graph, points[i - 1], points[i], (255, 0, 0), 2)

        bx, by = points[best_idx]
        cv2.circle(graph, (bx, by), 5, (0, 0, 255), -1)
        cv2.rectangle(graph, (pad_l, pad_t), (graph_w - pad_r, graph_h - pad_b), (0, 0, 0), 1)
        cv2.putText(graph, f"best_angle={best_angle:.2f}", (pad_l, graph_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(debug_output_dir, "projection_scores_plot.jpg"), graph)

    if return_matrix:
        return rotated_image, rotated_binary, best_angle, best_matrix
    return rotated_image, rotated_binary, best_angle

ROBUST_CUTOFF_LOW = 0.3
ROBUST_CUTOFF_HIGH = 0.7


def _connect_nearest_components_by_distance_variance(
    binary: np.ndarray,
    debug: bool = False,
    debug_output_dir: str = "debug_images/image_hyperparameter_estimation",
) -> np.ndarray:
    """
    Соединяет ближайшие компоненты, расстояния между которыми попали в малый класс
    после максимизации межклассовой дисперсии по расстояниям центров масс.
    """
    text_mask = (binary == 0).astype(np.uint8)
    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(text_mask, connectivity=8)
    if num_labels <= 2:
        return binary

    centers = centroids[1:].astype(np.float32)
    nearest_pairs = []
    nearest_distances = []
    for center_idx, center in enumerate(centers):
        deltas = centers - center
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        distances[center_idx] = np.inf
        nearest_idx = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_idx])
        if np.isfinite(nearest_distance):
            nearest_pairs.append((center_idx, nearest_idx, nearest_distance))
            nearest_distances.append(nearest_distance)

    if not nearest_distances:
        return binary

    distances = np.asarray(nearest_distances, dtype=np.float32)
    best_threshold = float(np.median(distances))
    best_variance = -1.0
    for threshold in np.unique(distances):
        left = distances <= float(threshold)
        if not np.any(left) or np.all(left):
            continue
        left_values = distances[left]
        right_values = distances[~left]
        left_weight = float(left_values.size) / float(distances.size)
        right_weight = float(right_values.size) / float(distances.size)
        variance = left_weight * right_weight * (float(np.mean(left_values)) - float(np.mean(right_values))) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = float(threshold)

    connected = binary.copy()
    for center_idx, nearest_idx, nearest_distance in nearest_pairs:
        if nearest_distance > best_threshold:
            continue
        x0, y0 = centers[center_idx]
        x1, y1 = centers[nearest_idx]
        y = int(round((float(y0) + float(y1)) / 2.0))
        x_start = int(round(min(float(x0), float(x1))))
        x_end = int(round(max(float(x0), float(x1))))
        cv2.line(connected, (x_start, y), (x_end, y), 0, 1)

    if debug:
        os.makedirs(debug_output_dir, exist_ok=True)
        with open(os.path.join(debug_output_dir, "nearest_component_distances.txt"), "w", encoding="utf-8") as file:
            file.write(f"threshold={best_threshold:.6f}\n")
            file.write(f"between_class_variance={best_variance:.6f}\n")
            file.write("component_idx\tnearest_component_idx\tdistance\n")
            for center_idx, nearest_idx, nearest_distance in nearest_pairs:
                file.write(f"{center_idx + 1}\t{nearest_idx + 1}\t{nearest_distance:.6f}\n")

        sorted_distances = np.sort(distances)
        graph_h, graph_w = 420, 900
        graph = np.full((graph_h, graph_w, 3), 255, dtype=np.uint8)
        pad_l, pad_r, pad_t, pad_b = 60, 20, 20, 50
        y_max = max(float(np.max(sorted_distances)), 1.0)
        points = []
        denom = max(1, len(sorted_distances) - 1)
        for idx, distance in enumerate(sorted_distances):
            x = int(pad_l + idx / denom * (graph_w - pad_l - pad_r))
            y = int(graph_h - pad_b - float(distance) / y_max * (graph_h - pad_t - pad_b))
            points.append((x, y))
        for idx in range(1, len(points)):
            cv2.line(graph, points[idx - 1], points[idx], (255, 0, 0), 2)
        threshold_y = int(graph_h - pad_b - best_threshold / y_max * (graph_h - pad_t - pad_b))
        cv2.line(graph, (pad_l, threshold_y), (graph_w - pad_r, threshold_y), (0, 0, 255), 1)
        cv2.rectangle(graph, (pad_l, pad_t), (graph_w - pad_r, graph_h - pad_b), (0, 0, 0), 1)
        cv2.putText(graph, f"threshold={best_threshold:.2f}", (pad_l, graph_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(debug_output_dir, "nearest_component_distances_plot.jpg"), graph)
        cv2.imwrite(os.path.join(debug_output_dir, "binary_connected_components.jpg"), connected)

    return connected


def image_hyperparameter_estimation(binary_image: np.ndarray, debug: bool = True) -> int:
    """Оценивает гиперпараметры бинарного изображения"""

    if len(binary_image.shape) == 3:
        binary = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    else:
        binary = binary_image.copy()
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)
    binary = np.where(binary < 128, 0, 255).astype(np.uint8)

    kernel = np.ones((3,3), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=3)
    binary = cv2.dilate(binary, kernel, iterations=3)
    binary = _connect_nearest_components_by_distance_variance(binary, debug=debug)

    if debug:
        output_dir = 'debug_images/image_hyperparameter_estimation'
        
        # Проверка на существование папки, если нет — создаем
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite('debug_images/image_hyperparameter_estimation/img_binary_erode.jpg', binary)

    text_mask = (binary == 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(text_mask, connectivity=8)
    component_widths = []
    component_heights = []
    for label_idx in range(1, num_labels):
        comp_width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        comp_height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        component_widths.append(comp_width)
        component_heights.append(comp_height)
    robust_width = float(np.mean(component_widths[int(len(component_widths) * ROBUST_CUTOFF_LOW): int(len(component_widths) * ROBUST_CUTOFF_HIGH)])) if component_widths else 0.0
    robust_height = float(np.mean(component_heights[int(len(component_heights) * ROBUST_CUTOFF_LOW): int(len(component_heights) * ROBUST_CUTOFF_HIGH)])) if component_heights else 0.0

    return robust_width, robust_height


def select_warp_binary_by_local_angles_hyperparameters(
    binary_image: np.ndarray,
    fallback_grid_rows: int = 20,
    fallback_grid_cols: int = 20,
    fallback_min_cell_text_pixels: int = 32,
    debug: bool = False,
    debug_output_dir: str = "debug_images",
) -> Tuple[int, int, int]:
    """
    Короткое описание:
        Автоматически оценивает размер сетки для warp_binary_by_local_angles.
    Вход:
        binary_image: np.ndarray -- бинарное изображение, где текст равен 0, фон равен 255.
        fallback_grid_rows: int -- число строк сетки, если компоненты не найдены.
        fallback_grid_cols: int -- число столбцов сетки, если компоненты не найдены.
        fallback_min_cell_text_pixels: int -- порог текста.
        debug: bool -- сохранять debug-отчет.
        debug_output_dir: str -- папка для debug-файлов.
    Выход:
        Tuple[int, int, int] -- grid_rows, grid_cols, min_cell_text_pixels.
    """
    # Шаг 1: приводим вход к бинарному виду.
    if len(binary_image.shape) == 3:
        binary = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    else:
        binary = binary_image.copy()
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)
    binary = np.where(binary < 128, 0, 255).astype(np.uint8)
    height, width = binary.shape

    component_width, component_height = image_hyperparameter_estimation(binary, debug=False)
    grid_rows = int(round(float(height) / float(component_height))) if component_height > 0 else int(fallback_grid_rows)
    grid_cols = int(round(float(width) / float(component_width))) if component_width > 0 else int(fallback_grid_cols)
    min_cell_text_pixels = int(fallback_min_cell_text_pixels)

    if debug:
        os.makedirs(debug_output_dir, exist_ok=True)
        with open(os.path.join(debug_output_dir, "warp_local_hyperparameters.txt"), "w", encoding="utf-8") as file:
            file.write(f"component_width={component_width:.6f}\n")
            file.write(f"component_height={component_height:.6f}\n")
            file.write(f"estimated_grid_rows={grid_rows}\n")
            file.write(f"estimated_grid_cols={grid_cols}\n")
            file.write(f"estimated_min_cell_text_pixels={min_cell_text_pixels}\n")

    return 3, 3, min_cell_text_pixels


def _estimate_rotation_padding(height: int, width: int, angle_deg: float, safety_px: int = 2) -> Tuple[int, int]:
    """
    Короткое описание:
        Оценивает минимальный белый padding, чтобы поворот изображения не обрезал углы.
    Вход:
        height: int -- высота изображения.
        width: int -- ширина изображения.
        angle_deg: float -- угол поворота в градусах.
        safety_px: int -- небольшой запас в пикселях.
    Выход:
        Tuple[int, int] -- padding по y и x.
    """
    angle_rad = abs(np.deg2rad(float(angle_deg)))
    sin_a = abs(float(np.sin(angle_rad)))
    cos_a = abs(float(np.cos(angle_rad)))
    rotated_width = float(height) * sin_a + float(width) * cos_a
    rotated_height = float(height) * cos_a + float(width) * sin_a
    pad_x = int(np.ceil(max(0.0, rotated_width - float(width)) / 2.0)) + int(safety_px)
    pad_y = int(np.ceil(max(0.0, rotated_height - float(height)) / 2.0)) + int(safety_px)
    return max(0, pad_y), max(0, pad_x)


def warp_binary_by_local_angles(
    binary_image: np.ndarray,
    grid_rows: int = 20,
    grid_cols: int = 20,
    min_cell_text_pixels: int = 32,
    max_local_angle_delta: float = 8.0,
    hyperparameter_selection: bool = False,
    debug: bool = False,
    debug_output_dir: str = "debug_images",
    return_transform_sequence: bool = False,
) -> np.ndarray:
    """
    Короткое описание:
        Выпрямляет бинарное изображение по локальным углам, оцененным на сетке.
    Вход:
        binary_image: np.ndarray -- бинарное изображение, где текст равен 0, фон равен 255.
        grid_rows: int -- число ячеек сетки по вертикали.
        grid_cols: int -- число ячеек сетки по горизонтали.
        min_cell_text_pixels: int -- минимум черных пикселей для оценки угла в ячейке.
        max_local_angle_delta: float -- максимальное отличие локального угла от глобального.
        hyperparameter_selection: bool -- автоматически подобрать grid_rows, grid_cols и min_cell_text_pixels.
        debug: bool -- сохранять debug-карты.
        debug_output_dir: str -- папка для debug-файлов.
        return_transform_sequence: bool -- вернуть карту преобразования output -> input.
    Выход:
        np.ndarray -- выпрямленное бинарное изображение.
    """
    # Шаг 1: приводим вход к строгому бинарному виду.
    if debug:
        os.makedirs(debug_output_dir, exist_ok=True)

    if len(binary_image.shape) == 3:
        binary = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    else:
        binary = binary_image.copy()
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)
    binary = np.where(binary < 128, 0, 255).astype(np.uint8)
    original_height, original_width = binary.shape

    # Шаг 1.5: добавляем не фиксированные 15%, а минимальное поле под реальный поворот.
    _, _, preliminary_global_angle = correct_perspective(
        binary,
        debug=False,
    )
    pad_y, pad_x = _estimate_rotation_padding(
        original_height,
        original_width,
        preliminary_global_angle,
        safety_px=2,
    )
    if pad_y > 0 or pad_x > 0:
        binary = cv2.copyMakeBorder(
            binary,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            cv2.BORDER_CONSTANT,
            value=255,
        )

    # Шаг 2: сначала убираем общий наклон, потом оцениваем локальные остаточные углы.
    _, corrected_binary, global_angle, global_matrix = correct_perspective(
        binary,
        debug=False,
        return_matrix=True,
    )
    corrected_binary = np.where(corrected_binary < 128, 0, 255).astype(np.uint8)
    height, width = corrected_binary.shape

    # Шаг 2.5: если включен автоподбор, оцениваем параметры уже после грубого выравнивания.
    if hyperparameter_selection:
        grid_rows, grid_cols, min_cell_text_pixels = select_warp_binary_by_local_angles_hyperparameters(
            corrected_binary,
            fallback_grid_rows=grid_rows,
            fallback_grid_cols=grid_cols,
            fallback_min_cell_text_pixels=min_cell_text_pixels,
            debug=debug,
            debug_output_dir=debug_output_dir,
        )

    coord_x, coord_y = np.meshgrid(
        np.arange(binary.shape[1], dtype=np.float32) - float(pad_x),
        np.arange(binary.shape[0], dtype=np.float32) - float(pad_y),
    )
    coord_x = cv2.warpAffine(
        coord_x,
        global_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1,
    )
    coord_y = cv2.warpAffine(
        coord_y,
        global_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1,
    )

    y_bounds = np.linspace(0, height, max(1, grid_rows) + 1, dtype=np.int32)
    x_bounds = np.linspace(0, width, max(1, grid_cols) + 1, dtype=np.int32)
    angle_map = np.zeros((max(1, grid_rows), max(1, grid_cols)), dtype=np.float32)

    # Шаг 3: в каждой ячейке оцениваем локальный угол через correct_perspective.
    for gy in range(angle_map.shape[0]):
        for gx in range(angle_map.shape[1]):
            y0, y1 = int(y_bounds[gy]), int(y_bounds[gy + 1])
            x0, x1 = int(x_bounds[gx]), int(x_bounds[gx + 1])
            cell = corrected_binary[y0:y1, x0:x1]

            if int(np.sum(cell == 0)) < min_cell_text_pixels:
                angle_map[gy, gx] = 0.0
                continue

            try:
                _, _, local_angle = correct_perspective(cell, debug=False)
            except Exception:
                local_angle = 0.0

            local_angle = float(np.clip(
                float(local_angle),
                -abs(float(max_local_angle_delta)),
                abs(float(max_local_angle_delta)),
            ))
            angle_map[gy, gx] = local_angle

    # Шаг 4: строим плотную карту углов и интегрируем ее в карту вертикальных смещений.
    dense_angles = cv2.resize(
        angle_map,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    ).astype(np.float32)
    slopes = np.tan(np.deg2rad(dense_angles)).astype(np.float32)
    displacement_y = np.zeros((height, width), dtype=np.float32)
    for x in range(1, width):
        displacement_y[:, x] = displacement_y[:, x - 1] + slopes[:, x - 1]

    # Шаг 5: центрируем смещения по каждой строке, чтобы не уводить страницу вверх или вниз.
    displacement_y -= np.mean(displacement_y, axis=1, keepdims=True)

    # Шаг 6: remap читает исходную точку для каждого пикселя результата.
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x
    map_y = grid_y + displacement_y
    warped = cv2.remap(
        corrected_binary,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    warped = np.where(warped < 128, 0, 255).astype(np.uint8)
    coord_x = cv2.remap(
        coord_x,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1,
    )
    coord_y = cv2.remap(
        coord_y,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1,
    )

    # Шаг 7: еще раз убираем возможный остаточный глобальный наклон после локального warp.
    _, warped, final_global_angle, final_global_matrix = correct_perspective(
        warped,
        debug=False,
        return_matrix=True,
    )
    warped = np.where(warped < 128, 0, 255).astype(np.uint8)
    coord_x = cv2.warpAffine(
        coord_x,
        final_global_matrix,
        (warped.shape[1], warped.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1,
    )
    coord_y = cv2.warpAffine(
        coord_y,
        final_global_matrix,
        (warped.shape[1], warped.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1,
    )

    # Шаг 8: обрезаем добавленные поля, но явно не режем черные пиксели,
    # если после финального поворота они вылезли за стандартную рамку.
    crop_y0 = pad_y
    crop_y1 = pad_y + original_height
    crop_x0 = pad_x
    crop_x1 = pad_x + original_width
    black_y, black_x = np.where(warped < 128)
    if black_y.size > 0:
        crop_y0 = min(crop_y0, int(np.min(black_y)))
        crop_y1 = max(crop_y1, int(np.max(black_y)) + 1)
        crop_x0 = min(crop_x0, int(np.min(black_x)))
        crop_x1 = max(crop_x1, int(np.max(black_x)) + 1)
    crop_y0 = max(0, crop_y0)
    crop_y1 = min(warped.shape[0], crop_y1)
    crop_x0 = max(0, crop_x0)
    crop_x1 = min(warped.shape[1], crop_x1)
    warped = warped[crop_y0:crop_y1, crop_x0:crop_x1]
    coord_x = coord_x[crop_y0:crop_y1, crop_x0:crop_x1]
    coord_y = coord_y[crop_y0:crop_y1, crop_x0:crop_x1]

    # Шаг 9: сохраняем debug-карты для проверки локальных углов и смещений.
    if debug:
        angle_min = float(np.min(dense_angles))
        angle_max = float(np.max(dense_angles))
        angle_range = max(angle_max - angle_min, 1e-6)
        angle_vis = ((dense_angles - angle_min) / angle_range * 255.0).astype(np.uint8)
        angle_vis = cv2.applyColorMap(angle_vis, cv2.COLORMAP_JET)

        disp_min = float(np.min(displacement_y))
        disp_max = float(np.max(displacement_y))
        disp_range = max(disp_max - disp_min, 1e-6)
        disp_vis = ((displacement_y - disp_min) / disp_range * 255.0).astype(np.uint8)
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)

        grid_vis = cv2.cvtColor(corrected_binary, cv2.COLOR_GRAY2BGR)
        for y in y_bounds:
            cv2.line(grid_vis, (0, int(y)), (width - 1, int(y)), (0, 0, 255), 1)
        for x in x_bounds:
            cv2.line(grid_vis, (int(x), 0), (int(x), height - 1), (0, 0, 255), 1)
        for gy in range(angle_map.shape[0]):
            for gx in range(angle_map.shape[1]):
                y_mid = int((y_bounds[gy] + y_bounds[gy + 1]) // 2)
                x_mid = int((x_bounds[gx] + x_bounds[gx + 1]) // 2)
                cv2.putText(
                    grid_vis,
                    f"{float(angle_map[gy, gx]):.1f}",
                    (x_mid - 18, y_mid),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imwrite(os.path.join(debug_output_dir, "warp_local_corrected_binary.jpg"), corrected_binary)
        cv2.imwrite(os.path.join(debug_output_dir, "warp_local_grid_angles.jpg"), grid_vis)
        cv2.imwrite(os.path.join(debug_output_dir, "warp_local_angle_map.jpg"), angle_vis)
        cv2.imwrite(os.path.join(debug_output_dir, "warp_local_displacement_y.jpg"), disp_vis)
        cv2.imwrite(os.path.join(debug_output_dir, "warp_local_result.jpg"), warped)

    if return_transform_sequence:
        transform_sequence = {
            "name": "warp_binary_by_local_angles",
            "input_shape": (int(original_height), int(original_width)),
            "output_shape": (int(warped.shape[0]), int(warped.shape[1])),
            "output_to_input_x": coord_x.astype(np.float32),
            "output_to_input_y": coord_y.astype(np.float32),
            "global_angle": float(global_angle),
            "final_global_angle": float(final_global_angle),
            "safe_crop": (int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)),
        }
        return warped, transform_sequence
    return warped


def _enforce_bijective_vertical_map(map_y: np.ndarray,
                                    min_vertical_step: float = 0.01) -> np.ndarray:
    """
    Короткое описание:
        Делает vertical backward-map строго монотонной по y в каждом столбце.
    Вход:
        map_y: np.ndarray -- карта исходных y-координат для cv2.remap.
        min_vertical_step: float -- минимальный положительный шаг между соседними y.
    Выход:
        np.ndarray -- монотонная карта, покрывающая диапазон [0, H - 1].
    """
    height, width = map_y.shape
    base_y = np.arange(height, dtype=np.float32)
    bijective_map_y = np.zeros_like(map_y, dtype=np.float32)
    min_vertical_step = max(1e-6, float(min_vertical_step))

    for x in range(width):
        column = map_y[:, x].astype(np.float32).copy()

        # Запрещаем fold-over: каждая следующая точка должна идти ниже предыдущей.
        for y in range(1, height):
            column[y] = max(column[y], column[y - 1] + min_vertical_step)

        col_min = float(column[0])
        col_max = float(column[-1])
        if col_max - col_min < 1e-9:
            bijective_map_y[:, x] = base_y
            continue

        # Растягиваем обратно на весь диапазон высоты, чтобы верх/низ не схлопывались.
        column = (column - col_min) / (col_max - col_min) * float(height - 1)
        bijective_map_y[:, x] = column.astype(np.float32)

    return bijective_map_y


def warp_binary_by_local_angles_bijection(
    binary_image: np.ndarray,
    grid_rows: int = 10,
    grid_cols: int = 16,
    min_cell_text_pixels: int = 32,
    max_local_angle_delta: float = 12.0,
    min_vertical_step: float = 0.01,
    debug: bool = False,
    debug_output_dir: str = "debug_images",
    return_transform_sequence: bool = False,
) -> np.ndarray:
    """
    Короткое описание:
        Биективная версия локального выпрямления: запрещает fold-over по вертикали.
    Вход:
        binary_image: np.ndarray -- бинарное изображение, где текст равен 0, фон равен 255.
        grid_rows: int -- число ячеек сетки по вертикали.
        grid_cols: int -- число ячеек сетки по горизонтали.
        min_cell_text_pixels: int -- минимум черных пикселей для оценки угла в ячейке.
        max_local_angle_delta: float -- максимальное отличие локального угла от глобального.
        min_vertical_step: float -- минимальный шаг map_y между соседними строками.
        debug: bool -- сохранять debug-карты.
        debug_output_dir: str -- папка для debug-файлов.
        return_transform_sequence: bool -- вернуть карту преобразования output -> input.
    Выход:
        np.ndarray -- выпрямленное бинарное изображение.
    """
    # Шаг 1: приводим вход к строгому бинарному виду и добавляем поля.
    if debug:
        os.makedirs(debug_output_dir, exist_ok=True)

    if len(binary_image.shape) == 3:
        binary = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    else:
        binary = binary_image.copy()
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)
    binary = np.where(binary < 128, 0, 255).astype(np.uint8)
    original_height, original_width = binary.shape
    pad_y = max(1, int(original_height * 0.15))
    pad_x = max(1, int(original_width * 0.15))
    binary = cv2.copyMakeBorder(
        binary,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    # Шаг 2: убираем общий наклон, затем оцениваем локальные остаточные углы.
    _, corrected_binary, global_angle, global_matrix = correct_perspective(
        binary,
        debug=False,
        return_matrix=True,
    )
    corrected_binary = np.where(corrected_binary < 128, 0, 255).astype(np.uint8)
    height, width = corrected_binary.shape

    coord_x, coord_y = np.meshgrid(
        np.arange(binary.shape[1], dtype=np.float32) - float(pad_x),
        np.arange(binary.shape[0], dtype=np.float32) - float(pad_y),
    )
    coord_x = cv2.warpAffine(
        coord_x,
        global_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1,
    )
    coord_y = cv2.warpAffine(
        coord_y,
        global_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1,
    )

    y_bounds = np.linspace(0, height, max(1, grid_rows) + 1, dtype=np.int32)
    x_bounds = np.linspace(0, width, max(1, grid_cols) + 1, dtype=np.int32)
    angle_map = np.zeros((max(1, grid_rows), max(1, grid_cols)), dtype=np.float32)

    # Шаг 3: оцениваем угол в каждой ячейке.
    for gy in range(angle_map.shape[0]):
        for gx in range(angle_map.shape[1]):
            y0, y1 = int(y_bounds[gy]), int(y_bounds[gy + 1])
            x0, x1 = int(x_bounds[gx]), int(x_bounds[gx + 1])
            cell = corrected_binary[y0:y1, x0:x1]

            if int(np.sum(cell == 0)) < min_cell_text_pixels:
                angle_map[gy, gx] = 0.0
                continue

            try:
                _, _, local_angle = correct_perspective(cell, debug=False)
            except Exception:
                local_angle = 0.0

            angle_map[gy, gx] = float(np.clip(
                float(local_angle),
                -abs(float(max_local_angle_delta)),
                abs(float(max_local_angle_delta)),
            ))

    # Шаг 4: строим поле смещений по локальным углам.
    dense_angles = cv2.resize(
        angle_map,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    ).astype(np.float32)
    slopes = np.tan(np.deg2rad(dense_angles)).astype(np.float32)
    displacement_y = np.zeros((height, width), dtype=np.float32)
    for x in range(1, width):
        displacement_y[:, x] = displacement_y[:, x - 1] + slopes[:, x - 1]
    displacement_y -= np.mean(displacement_y, axis=1, keepdims=True)

    # Шаг 5: делаем backward-map биективной по вертикали.
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    raw_map_y = grid_y + displacement_y
    map_x = grid_x
    map_y = _enforce_bijective_vertical_map(
        raw_map_y,
        min_vertical_step=min_vertical_step,
    )

    # Шаг 6: применяем remap. Геометрия map_y уже строго монотонна по каждому x.
    warped = cv2.remap(
        corrected_binary,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    warped = np.where(warped < 128, 0, 255).astype(np.uint8)
    coord_x = cv2.remap(
        coord_x,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1,
    )
    coord_y = cv2.remap(
        coord_y,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1,
    )

    # Шаг 7: финально убираем остаточный глобальный наклон.
    _, warped, final_global_angle, final_global_matrix = correct_perspective(
        warped,
        debug=False,
        return_matrix=True,
    )
    warped = np.where(warped < 128, 0, 255).astype(np.uint8)
    coord_x = cv2.warpAffine(
        coord_x,
        final_global_matrix,
        (warped.shape[1], warped.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1,
    )
    coord_y = cv2.warpAffine(
        coord_y,
        final_global_matrix,
        (warped.shape[1], warped.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1,
    )

    # Шаг 8: безопасный crop без срезания черных пикселей.
    crop_y0 = pad_y
    crop_y1 = pad_y + original_height
    crop_x0 = pad_x
    crop_x1 = pad_x + original_width
    black_y, black_x = np.where(warped < 128)
    if black_y.size > 0:
        crop_y0 = min(crop_y0, int(np.min(black_y)))
        crop_y1 = max(crop_y1, int(np.max(black_y)) + 1)
        crop_x0 = min(crop_x0, int(np.min(black_x)))
        crop_x1 = max(crop_x1, int(np.max(black_x)) + 1)
    crop_y0 = max(0, crop_y0)
    crop_y1 = min(warped.shape[0], crop_y1)
    crop_x0 = max(0, crop_x0)
    crop_x1 = min(warped.shape[1], crop_x1)
    warped = warped[crop_y0:crop_y1, crop_x0:crop_x1]
    coord_x = coord_x[crop_y0:crop_y1, crop_x0:crop_x1]
    coord_y = coord_y[crop_y0:crop_y1, crop_x0:crop_x1]

    # Шаг 9: сохраняем debug биективной карты.
    if debug:
        angle_min = float(np.min(dense_angles))
        angle_max = float(np.max(dense_angles))
        angle_range = max(angle_max - angle_min, 1e-6)
        angle_vis = ((dense_angles - angle_min) / angle_range * 255.0).astype(np.uint8)
        angle_vis = cv2.applyColorMap(angle_vis, cv2.COLORMAP_JET)

        disp_min = float(np.min(displacement_y))
        disp_max = float(np.max(displacement_y))
        disp_range = max(disp_max - disp_min, 1e-6)
        disp_vis = ((displacement_y - disp_min) / disp_range * 255.0).astype(np.uint8)
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)

        map_min = float(np.min(map_y))
        map_max = float(np.max(map_y))
        map_range = max(map_max - map_min, 1e-6)
        map_vis = ((map_y - map_min) / map_range * 255.0).astype(np.uint8)
        map_vis = cv2.applyColorMap(map_vis, cv2.COLORMAP_VIRIDIS)

        vertical_jacobian = np.diff(map_y, axis=0)
        jac_min = float(np.min(vertical_jacobian)) if vertical_jacobian.size else 0.0
        jac_max = float(np.max(vertical_jacobian)) if vertical_jacobian.size else 0.0
        jac_range = max(jac_max - jac_min, 1e-6)
        jac_vis = ((vertical_jacobian - jac_min) / jac_range * 255.0).astype(np.uint8)
        jac_vis = cv2.applyColorMap(jac_vis, cv2.COLORMAP_TURBO)

        cv2.imwrite(os.path.join(debug_output_dir, "warp_bijection_corrected_binary.jpg"), corrected_binary)
        cv2.imwrite(os.path.join(debug_output_dir, "warp_bijection_angle_map.jpg"), angle_vis)
        cv2.imwrite(os.path.join(debug_output_dir, "warp_bijection_displacement_y.jpg"), disp_vis)
        cv2.imwrite(os.path.join(debug_output_dir, "warp_bijection_map_y.jpg"), map_vis)
        cv2.imwrite(os.path.join(debug_output_dir, "warp_bijection_vertical_jacobian.jpg"), jac_vis)
        cv2.imwrite(os.path.join(debug_output_dir, "warp_bijection_result.jpg"), warped)

    if return_transform_sequence:
        transform_sequence = {
            "name": "warp_binary_by_local_angles_bijection",
            "input_shape": (int(original_height), int(original_width)),
            "output_shape": (int(warped.shape[0]), int(warped.shape[1])),
            "output_to_input_x": coord_x.astype(np.float32),
            "output_to_input_y": coord_y.astype(np.float32),
            "global_angle": float(global_angle),
            "final_global_angle": float(final_global_angle),
            "min_vertical_step": float(min_vertical_step),
            "safe_crop": (int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)),
        }
        return warped, transform_sequence
    return warped


# Значит это кореция шума - бесполезная вешь u-net как и otse работает ещё более фиговие (размывается разница между
# текстом и синими линиями) - в общем такое

def normalize_illumination(image, clip_limit=2.0, tile_grid_size=(8,8), gamma=0.3):
    """
    Выравнивает неравномерную освещённость и ослабляет засвеченные участки.

    Параметры:
        image: входное изображение (BGR или grayscale)
        clip_limit: порог ограничения контраста для CLAHE (больше – сильнее контраст)
        tile_grid_size: размер сетки (число тайлов по высоте и ширине)
        gamma: показатель гамма-коррекции (<1 осветляет тени, >1 затемняет)

    Возвращает:
        normalized: изображение с нормализованной освещённостью (uint8)
    """
    # Если изображение цветное, переводим в LAB и работаем только с каналом яркости
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_norm = _apply_clahe_and_gamma(l, clip_limit, tile_grid_size, gamma)
        lab_norm = cv2.merge((l_norm, a, b))
        normalized = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
    else:
        normalized = _apply_clahe_and_gamma(image, clip_limit, tile_grid_size, gamma)

    return normalized

def _apply_clahe_and_gamma(channel, clip_limit, tile_grid_size, gamma):
    # CLAHE – выравнивание локальной гистограммы
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    equalized = clahe.apply(channel)

    # Гамма-коррекция для подавления засветов
    look_up = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    corrected = cv2.LUT(equalized, look_up)
    return corrected

if __name__ == "__main__":
    # 1_18.JPG 3_51.JPG ru_hw2022_1_IMG_7886.JPG 77_742.JPG 1_11.JPG
    img_path = '/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code/debug_images/compare_hpp_dbnetpp/011_1/00_input.jpg'

    # Бинаризация страницы локальным otsu после нормализации освещения
    # img = cv2.imread(img_path)
    # img_corrected = normalize_illumination(img, clip_limit=4, gamma=0.2)
    # cv2.imwrite(f'debug_images/normalize_illumination.jpg', img_corrected)
    # binary = binarize_local_otsu_by_regions(img_corrected, debug=True, debug_output_dir='debug_images')

    #Нормализация освещения + бинаризация страницы
    # img = cv2.imread(img_path)
    # pages = extract_pages_with_yolo(
    #     image_path=img_path,
    #     model_path='models/yolo_detect_notebook/yolo_detect_notebook_1_(1-architecture).pt',
    #     output_dir='debug_images',
    #     return_binary = False
    # )
    # for idx, page in enumerate(pages):
    #     img_corrected = normalize_illumination(page, clip_limit=4, gamma=0.2)
    #     cv2.imwrite(f'debug_images/normalize_illumination.jpg', img_corrected)
    #     binary = binarize_local_otsu_by_regions(img_corrected, debug=True, debug_output_dir='debug_images')

    # Измененния искревления страницы
    img_path = normalize_illumination(cv2.imread(img_path), clip_limit=4, gamma=0.2)
    
    start_time = time.time()
    pages, binary_pages = extract_pages_with_yolo(
        image_path=img_path,
        model_path='models/yolo_segment_notebook/yolo_segment_notebook_3_(2-architecture).pt',
        output_dir='debug_images',
        conf_threshold=0.5,
        return_binary = True
    )
    end_time = time.time()
    print('Время выполнения extract_pages_with_yolo:', end_time - start_time)
    for idx, page in enumerate(binary_pages):
        start_time = time.time()
        _, binary_final, _ = correct_perspective(page, debug=False)
        end_time = time.time()
        print('Время выполнения correct_perspective:', end_time - start_time)

        start_time = time.time()
        robust_width, robust_height = image_hyperparameter_estimation(page)
        end_time = time.time()
        print('Время выполнения image_hyperparameter_estimation:', end_time - start_time, "Робастная ширина:", robust_width, "Робастная высота:", robust_height)


        start_time = time.time()
        warp_binary = warp_binary_by_local_angles(page, debug=True, hyperparameter_selection=True)
        end_time = time.time()
        print('Время выполнения warp_binary_by_local_angles:', end_time - start_time)
        cv2.imwrite(f'debug_images/img_binary_page_{idx}.jpg', page)
        cv2.imwrite(f'debug_images/img_binary_final_page_{idx}.jpg', binary_final)
        cv2.imwrite(f'debug_images/img_warp_binary_page_{idx}.jpg', warp_binary)
    #Исправление наклона
    # start_time = time.time()
    # img_final, binary_final = correct_perspective(img, debug=True)
    # print(find_split(binary_final))
    # end_time = time.time()
    # print(end_time - start_time)
    # cv2.imwrite('debug_images/img_final.jpg', img_final)
    # cv2.imwrite('debug_images/binary_final.jpg', binary_final)
    # cv2.imwrite('debug_images/img_corrected.jpg', img)

    # pages, binary_pages = extract_pages_with_yolo(
    #     image_path=img_path,
    #     model_path='models/yolo_detect_notebook/yolo_detect_notebook_1_(1-architecture).pt',
    #     output_dir='debug_images',
    #     conf_threshold=0.8,
    #     return_binary = True
    # )
    # for idx, page in enumerate(binary_pages):
    #     _, binary_final = correct_perspective(page, debug=True)
    #     cv2.imwrite(f'debug_images/img_binary_page{idx}.jpg', page)
    #     cv2.imwrite(f'debug_images/img_binary_final{idx}.jpg', binary_final)

    # for idx, page in enumerate(binary_pages):
    #     cv2.imwrite(f'debug_images/img_binary_final{idx}.jpg', page)
