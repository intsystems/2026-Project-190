import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from processing import warp_binary_by_local_angles


# Корень проекта.
PROJECT_ROOT = Path(__file__).resolve().parent

# Куда сохраняется итоговый json с метриками.
OUTPUT_JSON_PATH = PROJECT_ROOT / "grade_warp_binary_by_local_angles_results.json"

# Куда сохраняются debug-картинки и внутренний debug метода.
DEBUG_DIR = PROJECT_ROOT / "debug_images" / "grade_warp_binary_by_local_angles"

# Фиксируем seed, чтобы эксперимент был воспроизводимым.
RANDOM_SEED = 42

# Размер синтетической страницы.
IMAGE_WIDTH = 700
IMAGE_HEIGHT = int(1.414 * IMAGE_WIDTH)

# Сколько синтетических страниц делать на каждый тип искривления.
SAMPLES_PER_DISTORTION = 5

# Число строк на синтетической странице.
MIN_LINES = 5
MAX_LINES = 30

# Параметры текста.
TEXT_LEFT_MARGIN = 35
TEXT_RIGHT_MARGIN = 35
TEXT_TOP_MARGIN = 45
TEXT_LINE_GAP = 38
LINE_GROUPS_TOP_MARGIN = 85
LINE_GROUPS_TEXT_LINE_GAP = 105
LINE_GROUPS_MIN_LINES = 3
LINE_GROUPS_MAX_LINES = 3
LINE_GROUPS_MAX_WORDS = 5
LINE_GROUPS_MAX_TEXT_WIDTH = 460
TEXT_FONT_SCALE = 0.9
TEXT_THICKNESS = 2

# Насколько расширять gt-область строки при расчете centerline-метрик.
LINE_METRIC_PADDING = 16

# Значение ошибки, если centerline-метрика не смогла быть посчитана.
FAILED_LINE_METRIC_VALUE = float(IMAGE_HEIGHT)

# Максимальный сдвиг в пикселях для best-shift IoU/Dice.
BEST_SHIFT_MAX_PIXELS = 12

# Параметры тестируемого метода.
METHOD_GRID_ROWS = 10
METHOD_GRID_COLS = 16
METHOD_MIN_CELL_TEXT_PIXELS = 24
METHOD_MAX_LOCAL_ANGLE_DELTA = 10.0

# Включает сохранение подробного debug.
DEBUG = True


DISTORTION_CONFIGS = [
    {
        "name": "sine_easy",
        "type": "sine",
        "amplitude": 6.0,
        "period": 480.0,
        "phase": 0.3,
    },
    {
        "name": "sine_hard",
        "type": "sine",
        "amplitude": 18.0,
        "period": 360.0,
        "phase": 1.2,
    },
    {
        "name": "parabolic",
        "type": "parabolic",
        "amplitude": 22.0,
    },
    {
        "name": "hard_parabolic",
        "type": "parabolic",
        "amplitude": 34.0,
    },
    {
        "name": "piecewise_linear",
        "type": "piecewise_linear",
        "max_shift": 22.0,
        "control_points": 7,
    },
    {
        "name": "combined",
        "type": "combined",
        "sine_amplitude": 12.0,
        "parabolic_amplitude": 15.0,
        "tilt": 10.0,
        "period": 420.0,
    },
    {
        "name": "line_groups_angles",
        "type": "line_groups_angles",
        "angles": [-2.0, 1.5, -1.0],
        "group_blur": 21,
    },
]


def create_synthetic_page(sample_idx: int,
                          distortion_type: str = "") -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Короткое описание:
        Создает синтетическую бинарную страницу с несколькими строками текста.
    Вход:
        sample_idx: int -- индекс примера для воспроизводимого разнообразия текста.
        distortion_type: str -- тип искривления для настройки плотности строк.
    Выход:
        Tuple[np.ndarray, List[np.ndarray]] -- чистая страница и список масок строк.
    """
    # Шаг 1: создаем белую страницу и генератор случайных строк.
    page = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8) * 255
    line_masks: List[np.ndarray] = []
    rng = random.Random(RANDOM_SEED + sample_idx)
    words = [
        "alpha", "beta", "gamma", "delta", "line", "warp", "text",
        "paper", "school", "method", "metric", "binary", "angle",
    ]

    # Шаг 2: рисуем строки разной длины отдельными масками.
    if distortion_type == "line_groups_angles":
        num_lines = LINE_GROUPS_MAX_LINES
        y_positions = [
            LINE_GROUPS_TOP_MARGIN + line_idx * LINE_GROUPS_TEXT_LINE_GAP
            for line_idx in range(num_lines)
        ]
    else:
        num_lines = rng.randint(MIN_LINES, MAX_LINES)
        y_positions = [TEXT_TOP_MARGIN + line_idx * TEXT_LINE_GAP for line_idx in range(num_lines)]

    for y in y_positions:
        if y >= IMAGE_HEIGHT - TEXT_TOP_MARGIN:
            break

        if distortion_type == "line_groups_angles":
            word_count = rng.randint(3, LINE_GROUPS_MAX_WORDS)
            max_width = LINE_GROUPS_MAX_TEXT_WIDTH
        else:
            word_count = rng.randint(5, 10)
            max_width = IMAGE_WIDTH - TEXT_LEFT_MARGIN - TEXT_RIGHT_MARGIN
        text = " ".join(rng.choice(words) for _ in range(word_count))
        while True:
            text_size, _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_FONT_SCALE,
                TEXT_THICKNESS,
            )
            if text_size[0] <= max_width or word_count <= 3:
                break
            word_count -= 1
            text = " ".join(text.split()[:word_count])

        line_mask = np.ones_like(page) * 255
        cv2.putText(
            line_mask,
            text,
            (TEXT_LEFT_MARGIN, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_FONT_SCALE,
            0,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )
        line_mask = np.where(line_mask < 128, 0, 255).astype(np.uint8)
        page[line_mask == 0] = 0
        line_masks.append(line_mask)

    return page, line_masks


def build_displacement_map(config: Dict[str, float],
                           height: int,
                           width: int,
                           sample_idx: int) -> np.ndarray:
    """
    Короткое описание:
        Строит известную карту вертикального искривления для синтетического примера.
    Вход:
        config: Dict[str, float] -- параметры типа искривления.
        height: int -- высота изображения.
        width: int -- ширина изображения.
        sample_idx: int -- индекс примера.
    Выход:
        np.ndarray -- карта вертикальных смещений размера H x W.
    """
    # Шаг 1: готовим координаты и случайный генератор.
    x = np.arange(width, dtype=np.float32)
    x_norm = (x - width / 2.0) / max(width / 2.0, 1.0)
    rng = np.random.default_rng(RANDOM_SEED + sample_idx * 17)
    distortion_type = str(config["type"])

    # Шаг 2: считаем 1D-смещение по x.
    if distortion_type == "sine":
        amplitude = float(config["amplitude"])
        period = float(config["period"])
        phase = float(config["phase"]) + sample_idx * 0.4
        displacement = amplitude * np.sin(2.0 * np.pi * x / period + phase)
    elif distortion_type == "parabolic":
        amplitude = float(config["amplitude"])
        displacement = amplitude * (x_norm ** 2 - np.mean(x_norm ** 2))
    elif distortion_type == "piecewise_linear":
        control_points = int(config["control_points"])
        max_shift = float(config["max_shift"])
        control_x = np.linspace(0, width - 1, control_points)
        control_y = rng.uniform(-max_shift, max_shift, size=control_points)
        displacement = np.interp(x, control_x, control_y)
    elif distortion_type == "combined":
        sine = float(config["sine_amplitude"]) * np.sin(
            2.0 * np.pi * x / float(config["period"]) + sample_idx * 0.25
        )
        parabola = float(config["parabolic_amplitude"]) * (x_norm ** 2 - np.mean(x_norm ** 2))
        tilt = float(config["tilt"]) * x_norm
        displacement = sine + parabola + tilt
    elif distortion_type == "line_groups_angles":
        angles = np.array(config["angles"], dtype=np.float32)
        group_count = len(angles)
        group_bounds = np.linspace(0, height, group_count + 1, dtype=np.int32)
        displacement_map = np.zeros((height, width), dtype=np.float32)
        x_centered = x - width / 2.0

        for group_idx, angle in enumerate(angles):
            y0 = int(group_bounds[group_idx])
            y1 = int(group_bounds[group_idx + 1])
            slope = float(np.tan(np.deg2rad(float(angle))))
            displacement_map[y0:y1, :] = slope * x_centered.reshape(1, -1)

        blur_size = int(config["group_blur"])
        if blur_size > 1:
            if blur_size % 2 == 0:
                blur_size += 1
            displacement_map = cv2.GaussianBlur(displacement_map, (1, blur_size), 0)
        return displacement_map.astype(np.float32)
    else:
        raise ValueError(f"Неизвестный тип искривления: {distortion_type}")

    # Шаг 3: расширяем 1D-смещение на всю страницу.
    displacement_map = np.tile(displacement.reshape(1, -1), (height, 1)).astype(np.float32)
    return displacement_map


def apply_vertical_warp(binary: np.ndarray, displacement_y: np.ndarray) -> np.ndarray:
    """
    Короткое описание:
        Применяет вертикальное искривление к бинарному изображению.
    Вход:
        binary: np.ndarray -- бинарное изображение.
        displacement_y: np.ndarray -- карта вертикальных смещений.
    Выход:
        np.ndarray -- искривленное бинарное изображение.
    """
    # Шаг 1: строим reverse-map для cv2.remap.
    height, width = binary.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x
    map_y = grid_y - displacement_y.astype(np.float32)

    # Шаг 2: переносим пиксели с nearest-neighbor, чтобы сохранить 0/255.
    warped = cv2.remap(
        binary,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return np.where(warped < 128, 0, 255).astype(np.uint8)


def binary_mask(binary: np.ndarray) -> np.ndarray:
    """
    Короткое описание:
        Переводит бинарное изображение в булеву маску текста.
    Вход:
        binary: np.ndarray -- бинарное изображение.
    Выход:
        np.ndarray -- булева маска текста.
    """
    return binary < 128


def fit_binary_to_shape(binary: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Короткое описание:
        Приводит бинарное изображение к нужному размеру через центральный crop/pad белым фоном.
    Вход:
        binary: np.ndarray -- исходное бинарное изображение.
        target_shape: Tuple[int, int] -- целевой размер (height, width).
    Выход:
        np.ndarray -- бинарное изображение целевого размера.
    """
    # Шаг 1: создаем белый холст gt-размера.
    target_height, target_width = target_shape
    fitted = np.ones((target_height, target_width), dtype=np.uint8) * 255
    source = np.where(binary < 128, 0, 255).astype(np.uint8)
    source_height, source_width = source.shape

    # Шаг 2: центрально совмещаем source и target, лишнее аккуратно кропаем.
    src_y0 = max(0, (source_height - target_height) // 2)
    src_x0 = max(0, (source_width - target_width) // 2)
    dst_y0 = max(0, (target_height - source_height) // 2)
    dst_x0 = max(0, (target_width - source_width) // 2)
    copy_height = min(source_height - src_y0, target_height - dst_y0)
    copy_width = min(source_width - src_x0, target_width - dst_x0)

    if copy_height > 0 and copy_width > 0:
        fitted[dst_y0:dst_y0 + copy_height, dst_x0:dst_x0 + copy_width] = source[
            src_y0:src_y0 + copy_height,
            src_x0:src_x0 + copy_width,
        ]
    return fitted


def pad_binary_to_shape(binary: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Короткое описание:
        Дополняет бинарное изображение белым фоном до target_shape без обрезания.
    Вход:
        binary: np.ndarray -- исходное бинарное изображение.
        target_shape: Tuple[int, int] -- целевой размер (height, width), не меньше исходного.
    Выход:
        np.ndarray -- бинарное изображение на белом холсте целевого размера.
    """
    # Шаг 1: готовим строгую бинарную картинку и белый холст.
    source = np.where(binary < 128, 0, 255).astype(np.uint8)
    source_height, source_width = source.shape
    target_height, target_width = target_shape
    padded = np.ones((target_height, target_width), dtype=np.uint8) * 255

    # Шаг 2: кладем изображение по центру, ничего не кропаем.
    dst_y0 = max(0, (target_height - source_height) // 2)
    dst_x0 = max(0, (target_width - source_width) // 2)
    padded[dst_y0:dst_y0 + source_height, dst_x0:dst_x0 + source_width] = source
    return padded


def prepare_same_shape_pair(first: np.ndarray, second: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Короткое описание:
        Переносит две бинарные картинки на общий белый холст без потери пикселей.
    Вход:
        first: np.ndarray -- первая бинарная картинка.
        second: np.ndarray -- вторая бинарная картинка.
    Выход:
        Tuple[np.ndarray, np.ndarray] -- картинки одинакового размера.
    """
    common_shape = (
        max(first.shape[0], second.shape[0]),
        max(first.shape[1], second.shape[1]),
    )
    return pad_binary_to_shape(first, common_shape), pad_binary_to_shape(second, common_shape)


def calculate_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Короткое описание:
        Считает IoU черных пикселей двух бинарных изображений.
    Вход:
        pred: np.ndarray -- предсказанное бинарное изображение.
        target: np.ndarray -- целевое бинарное изображение.
    Выход:
        float -- значение IoU.
    """
    pred, target = prepare_same_shape_pair(pred, target)
    pred_mask = binary_mask(pred)
    target_mask = binary_mask(target)
    union = int(np.sum(pred_mask | target_mask))
    if union == 0:
        return 1.0
    intersection = int(np.sum(pred_mask & target_mask))
    return float(intersection / union)


def calculate_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Короткое описание:
        Считает Dice/F1 по черным пикселям.
    Вход:
        pred: np.ndarray -- предсказанное бинарное изображение.
        target: np.ndarray -- целевое бинарное изображение.
    Выход:
        float -- значение Dice.
    """
    pred, target = prepare_same_shape_pair(pred, target)
    pred_mask = binary_mask(pred)
    target_mask = binary_mask(target)
    denominator = int(np.sum(pred_mask) + np.sum(target_mask))
    if denominator == 0:
        return 1.0
    intersection = int(np.sum(pred_mask & target_mask))
    return float(2.0 * intersection / denominator)


def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Короткое описание:
        Сдвигает булеву маску без циклического переноса.
    Вход:
        mask: np.ndarray -- исходная булева маска.
        dx: int -- сдвиг по X.
        dy: int -- сдвиг по Y.
    Выход:
        np.ndarray -- сдвинутая булева маска.
    """
    # Шаг 1: готовим пустую маску того же размера.
    shifted = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape

    # Шаг 2: считаем пересечение исходной и целевой области.
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)

    # Шаг 3: если пересечения нет, возвращаем пустую маску.
    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return shifted

    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return shifted


def calculate_iou_from_masks(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """
    Короткое описание:
        Считает IoU по готовым булевым маскам.
    Вход:
        pred_mask: np.ndarray -- предсказанная маска.
        target_mask: np.ndarray -- целевая маска.
    Выход:
        float -- значение IoU.
    """
    union = int(np.sum(pred_mask | target_mask))
    if union == 0:
        return 1.0
    intersection = int(np.sum(pred_mask & target_mask))
    return float(intersection / union)


def calculate_dice_from_masks(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """
    Короткое описание:
        Считает Dice по готовым булевым маскам.
    Вход:
        pred_mask: np.ndarray -- предсказанная маска.
        target_mask: np.ndarray -- целевая маска.
    Выход:
        float -- значение Dice.
    """
    denominator = int(np.sum(pred_mask) + np.sum(target_mask))
    if denominator == 0:
        return 1.0
    intersection = int(np.sum(pred_mask & target_mask))
    return float(2.0 * intersection / denominator)


def calculate_best_shift_metrics(pred: np.ndarray,
                                 target: np.ndarray,
                                 max_shift: int = BEST_SHIFT_MAX_PIXELS) -> Dict[str, float]:
    """
    Короткое описание:
        Считает лучшие IoU и Dice при небольшом глобальном сдвиге результата.
    Вход:
        pred: np.ndarray -- предсказанное бинарное изображение.
        target: np.ndarray -- целевое бинарное изображение.
        max_shift: int -- максимальный перебираемый сдвиг по X и Y.
    Выход:
        Dict[str, float] -- best_shift_iou, best_shift_dice и лучший сдвиг.
    """
    # Шаг 1: готовим маски и начальные значения.
    pred, target = prepare_same_shape_pair(pred, target)
    pred_mask = binary_mask(pred)
    target_mask = binary_mask(target)
    best_iou = -1.0
    best_dice = -1.0
    best_iou_shift = (0, 0)
    best_dice_shift = (0, 0)

    # Шаг 2: перебираем сдвиги и выбираем лучшие значения метрик.
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = shift_mask(pred_mask, dx=dx, dy=dy)
            iou = calculate_iou_from_masks(shifted, target_mask)
            dice = calculate_dice_from_masks(shifted, target_mask)
            if iou > best_iou:
                best_iou = iou
                best_iou_shift = (dx, dy)
            if dice > best_dice:
                best_dice = dice
                best_dice_shift = (dx, dy)

    return {
        "best_shift_iou": float(best_iou),
        "best_shift_dice": float(best_dice),
        "best_shift_iou_dx": float(best_iou_shift[0]),
        "best_shift_iou_dy": float(best_iou_shift[1]),
        "best_shift_dice_dx": float(best_dice_shift[0]),
        "best_shift_dice_dy": float(best_dice_shift[1]),
    }


def calculate_chamfer_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Короткое описание:
        Считает симметричное Chamfer-расстояние между черными пикселями.
    Вход:
        pred: np.ndarray -- предсказанное бинарное изображение.
        target: np.ndarray -- целевое бинарное изображение.
    Выход:
        float -- среднее Chamfer-расстояние.
    """
    # Шаг 1: готовим маски и обрабатываем пустые случаи.
    pred, target = prepare_same_shape_pair(pred, target)
    pred_mask = binary_mask(pred)
    target_mask = binary_mask(target)
    if not np.any(pred_mask) and not np.any(target_mask):
        return 0.0
    if not np.any(pred_mask) or not np.any(target_mask):
        return float(max(pred.shape))

    # Шаг 2: distanceTransform дает расстояние до ближайшего нулевого пикселя.
    target_distance_input = np.where(target_mask, 0, 255).astype(np.uint8)
    pred_distance_input = np.where(pred_mask, 0, 255).astype(np.uint8)
    dist_to_target = cv2.distanceTransform(target_distance_input, cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform(pred_distance_input, cv2.DIST_L2, 3)

    pred_to_target = float(np.mean(dist_to_target[pred_mask]))
    target_to_pred = float(np.mean(dist_to_pred[target_mask]))
    return float((pred_to_target + target_to_pred) / 2.0)


def get_line_band(line_mask: np.ndarray) -> Tuple[int, int]:
    """
    Короткое описание:
        Возвращает вертикальный диапазон строки с padding.
    Вход:
        line_mask: np.ndarray -- бинарная маска одной строки.
    Выход:
        Tuple[int, int] -- границы y0, y1.
    """
    ys = np.where(binary_mask(line_mask))[0]
    if len(ys) == 0:
        return 0, line_mask.shape[0] - 1
    y0 = max(0, int(np.min(ys)) - LINE_METRIC_PADDING)
    y1 = min(line_mask.shape[0] - 1, int(np.max(ys)) + LINE_METRIC_PADDING)
    return y0, y1


def calculate_centerline(mask: np.ndarray,
                         y0: int,
                         y1: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Короткое описание:
        Считает средний y текста по каждому столбцу внутри полосы строки.
    Вход:
        mask: np.ndarray -- булева маска текста.
        y0: int -- верхняя граница полосы.
        y1: int -- нижняя граница полосы.
    Выход:
        Tuple[np.ndarray, np.ndarray] -- x-координаты и center-y значения.
    """
    # Шаг 1: собираем центры по x там, где есть текст.
    xs: List[int] = []
    centers: List[float] = []
    for x_idx in range(mask.shape[1]):
        local_ys = np.where(mask[y0:y1 + 1, x_idx])[0]
        if len(local_ys) == 0:
            continue
        xs.append(x_idx)
        centers.append(float(np.mean(local_ys + y0)))

    return np.array(xs, dtype=np.int32), np.array(centers, dtype=np.float32)


def calculate_line_metrics(candidate: np.ndarray,
                           target: np.ndarray,
                           line_masks: List[np.ndarray]) -> Dict[str, float]:
    """
    Короткое описание:
        Считает метрики положения и ровности строк по известным gt-строкам.
    Вход:
        candidate: np.ndarray -- проверяемое бинарное изображение.
        target: np.ndarray -- идеальное бинарное изображение.
        line_masks: List[np.ndarray] -- gt-маски отдельных строк.
    Выход:
        Dict[str, float] -- centerline MAE и средняя остаточная кривизна.
    """
    # Шаг 1: считаем ошибку center-y по каждой строке.
    candidate_mask = binary_mask(candidate)
    target_mask = binary_mask(target)
    mae_values: List[float] = []
    curvature_values: List[float] = []

    for line_mask in line_masks:
        y0, y1 = get_line_band(line_mask)
        target_xs, target_centers = calculate_centerline(target_mask, y0, y1)
        candidate_xs, candidate_centers = calculate_centerline(candidate_mask, y0, y1)
        if len(target_xs) == 0 or len(candidate_xs) == 0:
            continue

        common_xs = np.intersect1d(target_xs, candidate_xs)
        if len(common_xs) < 5:
            continue

        target_map = dict(zip(target_xs.tolist(), target_centers.tolist()))
        candidate_map = dict(zip(candidate_xs.tolist(), candidate_centers.tolist()))
        target_common = np.array([target_map[int(x)] for x in common_xs], dtype=np.float32)
        candidate_common = np.array([candidate_map[int(x)] for x in common_xs], dtype=np.float32)
        mae_values.append(float(np.mean(np.abs(candidate_common - target_common))))
        curvature_values.append(float(np.std(candidate_common)))

    # Шаг 2: возвращаем средние значения по строкам.
    if len(mae_values) == 0:
        return {
            "line_center_mae": FAILED_LINE_METRIC_VALUE,
            "line_curvature_std": FAILED_LINE_METRIC_VALUE,
        }
    return {
        "line_center_mae": float(np.mean(mae_values)),
        "line_curvature_std": float(np.mean(curvature_values)),
    }


def calculate_metrics(candidate: np.ndarray,
                      target: np.ndarray,
                      line_masks: List[np.ndarray]) -> Dict[str, float]:
    """
    Короткое описание:
        Считает полный набор метрик качества выпрямления.
    Вход:
        candidate: np.ndarray -- проверяемое бинарное изображение.
        target: np.ndarray -- идеальное бинарное изображение.
        line_masks: List[np.ndarray] -- gt-маски отдельных строк.
    Выход:
        Dict[str, float] -- словарь метрик.
    """
    # Шаг 1: если safe-crop метода расширил картинку, переносим все на общий белый холст.
    common_shape = (
        max(candidate.shape[0], target.shape[0]),
        max(candidate.shape[1], target.shape[1]),
    )
    candidate = pad_binary_to_shape(candidate, common_shape)
    target = pad_binary_to_shape(target, common_shape)
    line_masks = [pad_binary_to_shape(line_mask, common_shape) for line_mask in line_masks]

    # Шаг 2: считаем пиксельные и геометрические метрики.
    metrics = {
        "iou": calculate_iou(candidate, target),
        "dice": calculate_dice(candidate, target),
        "chamfer_distance": calculate_chamfer_distance(candidate, target),
    }
    metrics.update(calculate_best_shift_metrics(candidate, target))
    metrics.update(calculate_line_metrics(candidate, target, line_masks))
    return metrics


def calculate_improvements(before_metrics: Dict[str, float],
                           after_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Короткое описание:
        Считает улучшение метода относительно искривленного входа.
    Вход:
        before_metrics: Dict[str, float] -- метрики до исправления.
        after_metrics: Dict[str, float] -- метрики после исправления.
    Выход:
        Dict[str, float] -- словарь улучшений.
    """
    # Шаг 1: для похожести больше лучше, для ошибок меньше лучше.
    return {
        "iou_delta": float(after_metrics["iou"] - before_metrics["iou"]),
        "dice_delta": float(after_metrics["dice"] - before_metrics["dice"]),
        "best_shift_iou_delta": float(after_metrics["best_shift_iou"] - before_metrics["best_shift_iou"]),
        "best_shift_dice_delta": float(after_metrics["best_shift_dice"] - before_metrics["best_shift_dice"]),
        "chamfer_delta": float(before_metrics["chamfer_distance"] - after_metrics["chamfer_distance"]),
        "line_center_mae_delta": float(before_metrics["line_center_mae"] - after_metrics["line_center_mae"]),
        "line_curvature_std_delta": float(before_metrics["line_curvature_std"] - after_metrics["line_curvature_std"]),
    }


def save_debug_images(clean: np.ndarray,
                      warped: np.ndarray,
                      restored: np.ndarray,
                      displacement_y: np.ndarray,
                      sample_dir: Path) -> None:
    """
    Короткое описание:
        Сохраняет debug-картинки одного синтетического примера.
    Вход:
        clean: np.ndarray -- идеальная страница.
        warped: np.ndarray -- искривленная страница.
        restored: np.ndarray -- результат метода.
        displacement_y: np.ndarray -- карта синтетического искривления.
        sample_dir: Path -- папка примера.
    Выход:
        None
    """
    # Шаг 1: сохраняем основные изображения.
    sample_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sample_dir / "00_clean_gt.png"), clean)
    cv2.imwrite(str(sample_dir / "01_warped_input.png"), warped)
    cv2.imwrite(str(sample_dir / "02_restored_output.png"), restored)

    # Шаг 2: сохраняем карту различий и карту смещений.
    clean_for_diff, restored_for_diff = prepare_same_shape_pair(clean, restored)
    diff = np.zeros((clean_for_diff.shape[0], clean_for_diff.shape[1], 3), dtype=np.uint8)
    clean_mask = binary_mask(clean_for_diff)
    restored_mask = binary_mask(restored_for_diff)
    diff[clean_mask & restored_mask] = (0, 0, 0)
    diff[clean_mask & (~restored_mask)] = (0, 0, 255)
    diff[(~clean_mask) & restored_mask] = (255, 0, 0)
    diff[(~clean_mask) & (~restored_mask)] = (255, 255, 255)
    cv2.imwrite(str(sample_dir / "03_diff_gt_output.png"), diff)

    disp_min = float(np.min(displacement_y))
    disp_max = float(np.max(displacement_y))
    disp_range = max(disp_max - disp_min, 1e-6)
    disp_vis = ((displacement_y - disp_min) / disp_range * 255.0).astype(np.uint8)
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(sample_dir / "04_synthetic_displacement.png"), disp_vis)


def run_single_case(config: Dict[str, float],
                    sample_idx: int) -> Dict[str, object]:
    """
    Короткое описание:
        Создает один пример, искривляет его, исправляет методом и считает метрики.
    Вход:
        config: Dict[str, float] -- конфигурация искривления.
        sample_idx: int -- индекс примера.
    Выход:
        Dict[str, object] -- подробный результат по примеру.
    """
    # Шаг 1: создаем gt и синтетическое искривление.
    clean, line_masks = create_synthetic_page(sample_idx, distortion_type=str(config["type"]))
    displacement_y = build_displacement_map(config, IMAGE_HEIGHT, IMAGE_WIDTH, sample_idx)
    warped = apply_vertical_warp(clean, displacement_y)

    # Шаг 2: запускаем проверяемый метод.
    case_name = str(config["name"])
    sample_name = f"{case_name}_{sample_idx:03d}"
    sample_dir = DEBUG_DIR / sample_name
    restored = warp_binary_by_local_angles(
        warped,
        grid_rows=METHOD_GRID_ROWS,
        grid_cols=METHOD_GRID_COLS,
        min_cell_text_pixels=METHOD_MIN_CELL_TEXT_PIXELS,
        max_local_angle_delta=METHOD_MAX_LOCAL_ANGLE_DELTA,
        debug=DEBUG,
        debug_output_dir=str(sample_dir / "method_debug"),
    )

    # Шаг 3: считаем метрики до и после исправления.
    before_metrics = calculate_metrics(warped, clean, line_masks)
    after_metrics = calculate_metrics(restored, clean, line_masks)
    improvements = calculate_improvements(before_metrics, after_metrics)

    # Шаг 4: сохраняем debug примера.
    if DEBUG:
        save_debug_images(clean, warped, restored, displacement_y, sample_dir)

    return {
        "sample_name": sample_name,
        "distortion_config": config,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "improvements": improvements,
        "debug_dir": str(sample_dir),
    }


def aggregate_results(results: List[Dict[str, object]]) -> Dict[str, object]:
    """
    Короткое описание:
        Агрегирует метрики по всем примерам и по типам искривлений.
    Вход:
        results: List[Dict[str, object]] -- список результатов отдельных примеров.
    Выход:
        Dict[str, object] -- агрегированные средние метрики.
    """
    # Шаг 1: готовим списки по каждому набору метрик.
    metric_groups = ["before_metrics", "after_metrics", "improvements"]
    aggregate: Dict[str, object] = {}

    for group in metric_groups:
        keys = list(results[0][group].keys())
        aggregate[group] = {
            key: float(np.mean([float(item[group][key]) for item in results]))
            for key in keys
        }

    # Шаг 2: отдельно агрегируем по типам искривления.
    by_distortion: Dict[str, object] = {}
    for config in DISTORTION_CONFIGS:
        name = str(config["name"])
        items = [item for item in results if item["distortion_config"]["name"] == name]
        by_distortion[name] = {}
        for group in metric_groups:
            keys = list(items[0][group].keys())
            by_distortion[name][group] = {
                key: float(np.mean([float(item[group][key]) for item in items]))
                for key in keys
            }

    aggregate["by_distortion"] = by_distortion
    return aggregate


def main() -> None:
    """
    Короткое описание:
        Запускает генерацию синтетического датасета и оценку warp_binary_by_local_angles.
    Вход:
        None
    Выход:
        None
    """
    # Шаг 1: готовим папки и seed.
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if DEBUG:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # Шаг 2: запускаем все типы искривлений.
    results: List[Dict[str, object]] = []
    total_cases = len(DISTORTION_CONFIGS) * SAMPLES_PER_DISTORTION
    progress = tqdm(total=total_cases, desc="Grade warp_binary_by_local_angles")
    for config in DISTORTION_CONFIGS:
        for local_idx in range(SAMPLES_PER_DISTORTION):
            sample_idx = len(results)
            result = run_single_case(config, sample_idx)
            results.append(result)
            progress.update(1)
    progress.close()

    # Шаг 3: считаем средние метрики и сохраняем JSON.
    report = {
        "method": "warp_binary_by_local_angles",
        "samples_per_distortion": SAMPLES_PER_DISTORTION,
        "image_size": {
            "height": IMAGE_HEIGHT,
            "width": IMAGE_WIDTH,
        },
        "method_params": {
            "grid_rows": METHOD_GRID_ROWS,
            "grid_cols": METHOD_GRID_COLS,
            "min_cell_text_pixels": METHOD_MIN_CELL_TEXT_PIXELS,
            "max_local_angle_delta": METHOD_MAX_LOCAL_ANGLE_DELTA,
        },
        "results": results,
        "average": aggregate_results(results),
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
