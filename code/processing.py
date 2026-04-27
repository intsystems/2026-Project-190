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

LOCAL_OTSU_WHITE_THRESHOLD = 245  # Порог белого фона для fallback в почти пустых областях.
LOCAL_OTSU_LINE_KERNEL_FACTOR = 0.04  # Доля размера изображения для поиска длинных линий.
LOCAL_OTSU_MIN_LINE_KERNEL = 25  # Минимальная длина ядра для поиска линий клеток.
LOCAL_OTSU_GRID_DILATE_SIZE = 3  # Расширение найденной сетки перед выбеливанием.
LOCAL_OTSU_MIN_STD = 2.0  # Минимальная дисперсия яркости, ниже которой область считаем почти однотонной.
LOCAL_OTSU_GRID_SIZE = (3, 3)  # Количество областей по x и y для локального Otsu.
LOCAL_OTSU_REMOVE_GRID_LINES = True  # Удалять длинные линии клеток после локального Otsu.
def binarize_local_otsu_by_regions(
    image: np.ndarray,
    grid_size: Tuple[int, int] = LOCAL_OTSU_GRID_SIZE,
    remove_grid_lines: bool = LOCAL_OTSU_REMOVE_GRID_LINES,
    debug: bool = False,
    debug_output_dir: str = "debug_images",
) -> np.ndarray:
    """
    Короткое описание:
        Переводит изображение в серый формат и применяет Otsu отдельно в каждой области сетки.
    Вход:
        image (np.ndarray): входное цветное или серое изображение.
        grid_size (Tuple[int, int]): количество областей по x и y.
        remove_grid_lines (bool): удалять длинные горизонтальные и вертикальные линии.
        debug (bool): сохранять debug-изображения.
        debug_output_dir (str): папка для debug-файлов.
    Выход:
        np.ndarray: бинарное изображение uint8, где фон белый, текст черный.
    """
    if image is None:
        raise ValueError("image не должен быть None")
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    grid_x, grid_y = grid_size
    if grid_x <= 0 or grid_y <= 0:
        raise ValueError("grid_size должен содержать положительные значения")

    height, width = gray.shape
    binary = np.full((height, width), 255, dtype=np.uint8)
    thresholds = []

    # Шаг 1: применяем Otsu независимо в каждой ячейке сетки.
    for y_idx in range(grid_y):
        y1 = int(round(y_idx * height / grid_y))
        y2 = int(round((y_idx + 1) * height / grid_y))
        for x_idx in range(grid_x):
            x1 = int(round(x_idx * width / grid_x))
            x2 = int(round((x_idx + 1) * width / grid_x))
            region = gray[y1:y2, x1:x2]
            if region.size == 0:
                continue

            if float(np.std(region)) < LOCAL_OTSU_MIN_STD:
                threshold = LOCAL_OTSU_WHITE_THRESHOLD
                region_binary = np.where(region < threshold, 0, 255).astype(np.uint8)
            else:
                threshold, region_binary = cv2.threshold(
                    region,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )
            binary[y1:y2, x1:x2] = region_binary
            thresholds.append((x1, y1, x2, y2, float(threshold)))

    binary_before_grid_removal = binary.copy()
    grid_mask = np.zeros_like(binary)
    horizontal_lines = np.zeros_like(binary)
    vertical_lines = np.zeros_like(binary)

    # Шаг 2: ищем длинные линии клеток морфологией и выбеливаем их.
    if remove_grid_lines:
        foreground = np.where(binary < 128, 255, 0).astype(np.uint8)
        horizontal_kernel_width = max(LOCAL_OTSU_MIN_LINE_KERNEL, int(width * LOCAL_OTSU_LINE_KERNEL_FACTOR))
        vertical_kernel_height = max(LOCAL_OTSU_MIN_LINE_KERNEL, int(height * LOCAL_OTSU_LINE_KERNEL_FACTOR))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_width, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_height))

        horizontal_lines = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, vertical_kernel)
        grid_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)

        if LOCAL_OTSU_GRID_DILATE_SIZE > 1:
            dilate_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (LOCAL_OTSU_GRID_DILATE_SIZE, LOCAL_OTSU_GRID_DILATE_SIZE),
            )
            grid_mask = cv2.dilate(grid_mask, dilate_kernel, iterations=1)
        binary[grid_mask > 0] = 255

    # Шаг 3: сохраняем debug-карты для проверки локальных порогов и найденной сетки.
    if debug:
        os.makedirs(debug_output_dir, exist_ok=True)
        debug_grid = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2, threshold in thresholds:
            cv2.rectangle(debug_grid, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
            cv2.putText(
                debug_grid,
                f"{threshold:.0f}",
                (x1 + 5, min(y1 + 25, y2 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(os.path.join(debug_output_dir, "local_otsu_gray.jpg"), gray)
        cv2.imwrite(os.path.join(debug_output_dir, "local_otsu_binary_before_grid_removal.jpg"), binary_before_grid_removal)
        cv2.imwrite(os.path.join(debug_output_dir, "local_otsu_binary.jpg"), binary)
        cv2.imwrite(os.path.join(debug_output_dir, "local_otsu_grid.jpg"), debug_grid)
        cv2.imwrite(os.path.join(debug_output_dir, "local_otsu_horizontal_lines.jpg"), horizontal_lines)
        cv2.imwrite(os.path.join(debug_output_dir, "local_otsu_vertical_lines.jpg"), vertical_lines)
        cv2.imwrite(os.path.join(debug_output_dir, "local_otsu_grid_lines_mask.jpg"), grid_mask)

    return binary


def clean_binary_opening_closing(binary_image: np.ndarray) -> np.ndarray:
    """
    Короткое описание:
        Очищает бинарное изображение через opening и closing.
    Вход:
        binary_image (np.ndarray): бинарное изображение, где фон белый, текст черный.
    Выход:
        np.ndarray: очищенное бинарное изображение uint8.
    """
    # Шаг 1: приводим изображение к одноканальному uint8 и жестко бинаризуем.
    if binary_image is None:
        raise ValueError("binary_image не должен быть None")
    if len(binary_image.shape) == 3:
        binary = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    else:
        binary = binary_image.copy()
    if binary.dtype != np.uint8:
        binary = np.clip(binary, 0, 255).astype(np.uint8)
    binary = np.where(binary < 128, 0, 255).astype(np.uint8)

    # Шаг 2: инвертируем, чтобы морфология работала по черному тексту как по foreground.
    foreground = 255 - binary

    # Шаг 3: opening убирает мелкий шум, closing склеивает маленькие разрывы.
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, opening_kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, closing_kernel, iterations=1)

    # Шаг 4: возвращаем формат фон белый, текст черный.
    return 255 - closed


def extract_pages_with_yolo(
    image_path,
    model_path,
    output_dir="debug_images",
    conf_threshold=0.7,
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

ROBUST_CUTOFF_LOW = 0.15
ROBUST_CUTOFF_HIGH = 0.95
def image_hyperparameter_estimation(binary_image: np.ndarray, fallback_num_lines: int = 20) -> int:
    """Оценивает гиперпараметры бинарного изображения"""

    if len(binary_image.shape) == 3:
        binary = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    else:
        binary = binary_image.copy()
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)
    binary = np.where(binary < 128, 0, 255).astype(np.uint8)

    # Шаг 2: HPP-порог выбираем по максимальной межклассовой дисперсии.
    hpp = np.sum(binary == 0, axis=1).astype(np.float32)
    hpp_max = float(np.max(hpp)) if hpp.size else 0.0
    best_threshold = 0.0
    best_variance = -1.0
    num_lines = int(fallback_num_lines)
    if hpp_max > 0:
        thresholds = np.linspace(
            0.05 * hpp_max,
            0.95 * hpp_max,
            WARP_AUTO_HPP_THRESHOLD_COUNT,
            dtype=np.float32,
        )
        for threshold in thresholds:
            text_rows = hpp >= float(threshold)
            if not np.any(text_rows) or np.all(text_rows):
                continue
            background_values = hpp[~text_rows]
            text_values = hpp[text_rows]
            w0 = float(background_values.size) / float(hpp.size)
            w1 = float(text_values.size) / float(hpp.size)
            variance = w0 * w1 * (float(np.mean(text_values)) - float(np.mean(background_values))) ** 2
            if variance > best_variance:
                best_variance = variance
                best_threshold = float(threshold)

        if best_variance >= 0:
            estimated_rows = _count_true_regions(hpp >= best_threshold)
            if estimated_rows > 0:
                num_lines = estimated_rows

    # Шаг 3: по связным компонентам оцениваем среднюю ширину слова/буквенного куска.
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

    return num_lines, robust_width, robust_height


def _count_true_regions(mask: np.ndarray) -> int:
    """
    Короткое описание:
        Считает количество непрерывных True-участков в одномерной маске.
    Вход:
        mask: np.ndarray -- одномерная bool-маска.
    Выход:
        int -- количество найденных участков.
    """
    count = 0
    in_region = False
    for value in mask.astype(bool):
        if value and not in_region:
            count += 1
            in_region = True
        elif not value:
            in_region = False
    return count


WARP_AUTO_HPP_THRESHOLD_COUNT = 64  # Число кандидатов порога HPP для автоподбора строк.
WARP_AUTO_MIN_GRID_ROWS = 2  # Нижняя граница числа строк сетки локального warp.
WARP_AUTO_MAX_GRID_ROWS = 80  # Верхняя граница числа строк сетки локального warp.
WARP_AUTO_MIN_GRID_COLS = 2  # Нижняя граница числа столбцов сетки локального warp.
WARP_AUTO_MAX_GRID_COLS = 120  # Верхняя граница числа столбцов сетки локального warp.
WARP_AUTO_MIN_CELL_TEXT_PIXELS = 8  # Нижняя граница min_cell_text_pixels.
WARP_AUTO_MAX_CELL_TEXT_PIXELS = 512  # Верхняя граница min_cell_text_pixels.
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
        Автоматически подбирает сетку и порог текста для warp_binary_by_local_angles.
    Вход:
        binary_image: np.ndarray -- бинарное изображение, где текст равен 0, фон равен 255.
        fallback_grid_rows: int -- число строк сетки, если HPP не смог оценить строки.
        fallback_grid_cols: int -- число столбцов сетки, если компоненты не найдены.
        fallback_min_cell_text_pixels: int -- порог текста, если плотность не оценена.
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
    _, width = binary.shape

    # Шаг 2: HPP-порог выбираем по максимальной межклассовой дисперсии.
    hpp = np.sum(binary == 0, axis=1).astype(np.float32)
    hpp_max = float(np.max(hpp)) if hpp.size else 0.0
    best_threshold = 0.0
    best_variance = -1.0
    grid_rows = int(fallback_grid_rows)
    if hpp_max > 0:
        thresholds = np.linspace(
            0.05 * hpp_max,
            0.95 * hpp_max,
            WARP_AUTO_HPP_THRESHOLD_COUNT,
            dtype=np.float32,
        )
        for threshold in thresholds:
            text_rows = hpp >= float(threshold)
            if not np.any(text_rows) or np.all(text_rows):
                continue
            background_values = hpp[~text_rows]
            text_values = hpp[text_rows]
            w0 = float(background_values.size) / float(hpp.size)
            w1 = float(text_values.size) / float(hpp.size)
            variance = w0 * w1 * (float(np.mean(text_values)) - float(np.mean(background_values))) ** 2
            if variance > best_variance:
                best_variance = variance
                best_threshold = float(threshold)

        if best_variance >= 0:
            estimated_rows = _count_true_regions(hpp >= best_threshold)
            if estimated_rows > 0:
                grid_rows = estimated_rows
    grid_rows = int(np.clip(grid_rows, WARP_AUTO_MIN_GRID_ROWS, WARP_AUTO_MAX_GRID_ROWS))

    # Шаг 3: по связным компонентам оцениваем среднюю ширину слова/буквенного куска.
    text_mask = (binary == 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(text_mask, connectivity=8)
    component_widths = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        comp_width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        component_widths.append(comp_width)

    grid_cols = int(fallback_grid_cols)
    avg_component_width = 0.0
    if component_widths:
        avg_component_width = float(np.mean(component_widths))
        if avg_component_width > 0:
            grid_cols = int(round(float(width) / avg_component_width))
    grid_cols = int(np.clip(grid_cols, WARP_AUTO_MIN_GRID_COLS, WARP_AUTO_MAX_GRID_COLS))

    # Шаг 4: по плотности текста оцениваем, сколько черных пикселей должно быть в ячейке.
    black_pixels = int(np.sum(binary == 0))
    white_pixels = int(np.sum(binary == 255))
    total_pixels = max(1, black_pixels + white_pixels)
    black_ratio = float(black_pixels) / float(total_pixels)
    white_black_ratio = float(white_pixels) / float(max(1, black_pixels))
    avg_black_per_cell = float(black_pixels) / float(max(1, grid_rows * grid_cols))
    min_cell_text_pixels = int(round(avg_black_per_cell * 0.35))
    if min_cell_text_pixels <= 0:
        min_cell_text_pixels = int(fallback_min_cell_text_pixels)
    min_cell_text_pixels = int(np.clip(
        min_cell_text_pixels,
        WARP_AUTO_MIN_CELL_TEXT_PIXELS,
        WARP_AUTO_MAX_CELL_TEXT_PIXELS,
    ))

    # Шаг 5: сохраняем короткий отчет, чтобы было видно, почему выбраны такие значения.
    if debug:
        os.makedirs(debug_output_dir, exist_ok=True)
        with open(os.path.join(debug_output_dir, "warp_local_hyperparameters.txt"), "w", encoding="utf-8") as file:
            file.write(f"hpp_best_threshold={best_threshold:.6f}\n")
            file.write(f"hpp_best_variance={best_variance:.6f}\n")
            file.write(f"estimated_grid_rows={grid_rows}\n")
            file.write(f"component_count={len(component_widths)}\n")
            file.write(f"avg_component_width={avg_component_width:.6f}\n")
            file.write(f"estimated_grid_cols={grid_cols}\n")
            file.write(f"black_pixels={black_pixels}\n")
            file.write(f"white_pixels={white_pixels}\n")
            file.write(f"black_ratio={black_ratio:.6f}\n")
            file.write(f"white_black_ratio={white_black_ratio:.6f}\n")
            file.write(f"avg_black_per_cell={avg_black_per_cell:.6f}\n")
            file.write(f"estimated_min_cell_text_pixels={min_cell_text_pixels}\n")

    return int(grid_rows / 4), int(grid_cols / 4), min_cell_text_pixels # / 4 / 8


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
    max_local_angle_delta: float = 15.0,
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
        with open(os.path.join(debug_output_dir, "warp_local_angles.txt"), "w", encoding="utf-8") as file:
            file.write(f"global_angle={float(global_angle):.6f}\n")
            file.write(f"preliminary_global_angle={float(preliminary_global_angle):.6f}\n")
            file.write(f"final_global_angle={float(final_global_angle):.6f}\n")
            file.write(f"hyperparameter_selection={bool(hyperparameter_selection)}\n")
            file.write(f"grid_rows={int(grid_rows)}\n")
            file.write(f"grid_cols={int(grid_cols)}\n")
            file.write(f"min_cell_text_pixels={int(min_cell_text_pixels)}\n")
            file.write(f"adaptive_pad_y={int(pad_y)}\n")
            file.write(f"adaptive_pad_x={int(pad_x)}\n")
            file.write(f"safe_crop=({crop_x0}, {crop_y0}) - ({crop_x1}, {crop_y1})\n")
            for gy in range(angle_map.shape[0]):
                row_values = " ".join(f"{float(value):.4f}" for value in angle_map[gy])
                file.write(f"row_{gy:02d}: {row_values}\n")

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
        with open(os.path.join(debug_output_dir, "warp_bijection_angles.txt"), "w", encoding="utf-8") as file:
            file.write(f"global_angle={float(global_angle):.6f}\n")
            file.write(f"final_global_angle={float(final_global_angle):.6f}\n")
            file.write(f"min_vertical_step={float(min_vertical_step):.6f}\n")
            file.write(f"min_vertical_jacobian={float(jac_min):.6f}\n")
            file.write(f"safe_crop=({crop_x0}, {crop_y0}) - ({crop_x1}, {crop_y1})\n")
            for gy in range(angle_map.shape[0]):
                row_values = " ".join(f"{float(value):.4f}" for value in angle_map[gy])
                file.write(f"row_{gy:02d}: {row_values}\n")

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
    img_path = '/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code/datasets/HWR200/hw_dataset/184/reuse7/ФотоСветлое/2.jpg'

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
    start_time = time.time()
    pages, binary_pages = extract_pages_with_yolo(
        image_path=img_path,
        model_path='models/yolo_segment_notebook/yolo_segment_notebook_3_(2-architecture).pt',
        output_dir='debug_images',
        conf_threshold=0.8,
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
        num_lines, robust_width, robust_height = image_hyperparameter_estimation(page)
        end_time = time.time()
        print('Время выполнения image_hyperparameter_estimation:', end_time - start_time, "Число строк:", num_lines, "Робастная ширина:", robust_width, "Робастная высота:", robust_height)


        start_time = time.time()
        warp_binary = warp_binary_by_local_angles(page, debug=True, hyperparameter_selection=True)
        end_time = time.time()
        clean_binary = clean_binary_opening_closing(warp_binary)
        print('Время выполнения warp_binary_by_local_angles:', end_time - start_time)
        cv2.imwrite(f'debug_images/img_binary_page_{idx}.jpg', page)
        cv2.imwrite(f'debug_images/img_binary_final_page_{idx}.jpg', binary_final)
        cv2.imwrite(f'debug_images/img_warp_binary_page_{idx}.jpg', warp_binary)
        cv2.imwrite(f'debug_images/img_clean_binary_page_{idx}.jpg', clean_binary)

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
