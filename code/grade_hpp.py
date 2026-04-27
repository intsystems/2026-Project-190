import gc
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from hpp_method import LineSegmentation
from u_net_binarization import load_unet_model, binarize_image_with_loaded_model


# Корень проекта.
PROJECT_ROOT = Path(__file__).resolve().parent

# Папка с исходными изображениями и polygon-разметкой строк.
IMAGES_DIR = PROJECT_ROOT / "datasets" / "school_notebooks_RU" / "images_base"
LABELS_DIR = PROJECT_ROOT / "datasets" / "school_notebooks_RU" / "images_detect_lines"

# Вес U-Net модели для бинаризации текста при построении GT class-matrix.
UNET_MODEL_PATH = PROJECT_ROOT / "models" / "u_net" / "unet_binarization_3_(6-architecture).pth"

# Куда сохраняется итоговая оценка.
OUTPUT_JSON_PATH = PROJECT_ROOT / "grade_hpp_results.json"

# Куда сохраняется debug.
DEBUG_DIR = PROJECT_ROOT / "debug_images" / "grade_hpp"

# Ровно те файлы, которые сравнивались в comparison_yolo_hpp.py.
LABEL_NAMES = ["1_11.txt", "1_19.txt", "8_109.txt", "2786.txt", "2796.txt"]

# Допустимые расширения изображений.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")

# Минимальная площадь строки в пикселях.
MIN_TEXT_PIXELS = 5

# IoU-порог, при котором строка считается правильно найденной.
LINE_IOU_THRESHOLD = 0.30

# Вероятность правильного класса для deterministic cross entropy.
CE_CORRECT_PROB = 0.999

# Максимальный сдвиг pred class-matrix при поиске лучшего наложения по текстовым пикселям.
MAX_MASK_ALIGNMENT_SHIFT = 200

# Включает debug-картинки матриц классов.
DEBUG = True


def find_image_path(stem: str) -> Path:
    """
    Короткое описание:
        Находит изображение по имени txt-файла разметки.
    Вход:
        stem: str -- имя файла без расширения.
    Выход:
        Path -- путь к изображению.
    """
    for extension in IMAGE_EXTENSIONS:
        image_path = IMAGES_DIR / f"{stem}{extension}"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"Не найдено изображение для {stem}")


def get_label_paths() -> List[Path]:
    """
    Короткое описание:
        Возвращает список txt-разметок для оценки.
    Вход:
        None
    Выход:
        List[Path] -- пути к label-файлам.
    """
    return [
        LABELS_DIR / (label_name if label_name.endswith(".txt") else f"{label_name}.txt")
        for label_name in LABEL_NAMES
    ]


def read_yolo_polygons(label_path: Path, image_width: int, image_height: int) -> List[np.ndarray]:
    """
    Короткое описание:
        Читает YOLO-seg полигоны строк и переводит координаты в пиксели.
    Вход:
        label_path: Path -- путь к txt-разметке.
        image_width: int -- ширина изображения.
        image_height: int -- высота изображения.
    Выход:
        List[np.ndarray] -- список полигонов формы N x 2.
    """
    polygons: List[np.ndarray] = []
    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()
            if len(values) < 7:
                continue
            coords = np.array(values[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] = np.clip(coords[:, 0] * image_width, 0, image_width - 1)
            coords[:, 1] = np.clip(coords[:, 1] * image_height, 0, image_height - 1)
            polygons.append(coords.astype(np.int32))
    return polygons


def build_target_class_matrix(polygons: List[np.ndarray], binary: np.ndarray) -> np.ndarray:
    """
    Короткое описание:
        Строит GT class-matrix: 0 фон, 1 первая строка, 2 вторая строка и т.д.
    Вход:
        polygons: List[np.ndarray] -- полигоны строк.
        binary: np.ndarray -- бинарное изображение, где текст равен 0.
    Выход:
        np.ndarray -- матрица классов int32.
    """
    target = np.zeros(binary.shape[:2], dtype=np.int32)
    black_pixels = binary < 128

    for class_idx, polygon in enumerate(polygons, start=1):
        polygon_mask = np.zeros(binary.shape[:2], dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [polygon], 255)
        line_mask = np.logical_and(polygon_mask == 255, black_pixels)
        if int(np.sum(line_mask)) >= MIN_TEXT_PIXELS:
            target[line_mask] = class_idx
    return target


def crop_to_nonzero_bbox(class_matrix: np.ndarray) -> np.ndarray:
    """
    Короткое описание:
        Обрезает class-matrix по bbox всех ненулевых классов.
    Вход:
        class_matrix: np.ndarray -- матрица классов.
    Выход:
        np.ndarray -- обрезанная матрица или исходная, если классов нет.
    """
    ys, xs = np.where(class_matrix > 0)
    if len(xs) == 0:
        return class_matrix
    return class_matrix[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1]


def fit_class_matrix_to_shape(class_matrix: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Короткое описание:
        Центрированно pad/crop матрицу классов до target_shape.
    Вход:
        class_matrix: np.ndarray -- исходная матрица классов.
        target_shape: Tuple[int, int] -- целевой размер H x W.
    Выход:
        np.ndarray -- матрица нужного размера.
    """
    target_height, target_width = target_shape
    source_height, source_width = class_matrix.shape[:2]
    fitted = np.zeros((target_height, target_width), dtype=np.int32)

    src_y0 = max(0, (source_height - target_height) // 2)
    src_x0 = max(0, (source_width - target_width) // 2)
    dst_y0 = max(0, (target_height - source_height) // 2)
    dst_x0 = max(0, (target_width - source_width) // 2)
    copy_height = min(source_height - src_y0, target_height - dst_y0)
    copy_width = min(source_width - src_x0, target_width - dst_x0)

    if copy_height > 0 and copy_width > 0:
        fitted[dst_y0:dst_y0 + copy_height, dst_x0:dst_x0 + copy_width] = class_matrix[
            src_y0:src_y0 + copy_height,
            src_x0:src_x0 + copy_width,
        ]
    return fitted


def prepare_pair_for_metrics(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Короткое описание:
        Приводит pred/target class-matrix к общей форме для сравнения.
    Вход:
        pred: np.ndarray -- предсказанная матрица.
        target: np.ndarray -- GT-матрица.
    Выход:
        Tuple[np.ndarray, np.ndarray] -- матрицы одинакового размера.
    """
    if pred.shape == target.shape:
        return pred.astype(np.int32), target.astype(np.int32)

    pred_crop = crop_to_nonzero_bbox(pred)
    target_crop = crop_to_nonzero_bbox(target)
    common_shape = (
        max(pred_crop.shape[0], target_crop.shape[0]),
        max(pred_crop.shape[1], target_crop.shape[1]),
    )
    return (
        fit_class_matrix_to_shape(pred_crop, common_shape),
        fit_class_matrix_to_shape(target_crop, common_shape),
    )


def shift_class_matrix(class_matrix: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Короткое описание:
        Сдвигает class-matrix без циклического переноса.
    Вход:
        class_matrix: np.ndarray -- исходная матрица классов.
        dx: int -- сдвиг по X.
        dy: int -- сдвиг по Y.
    Выход:
        np.ndarray -- сдвинутая матрица классов.
    """
    shifted = np.zeros_like(class_matrix, dtype=np.int32)
    height, width = class_matrix.shape

    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)

    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return shifted

    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = class_matrix[src_y0:src_y1, src_x0:src_x1]
    return shifted


def align_pred_by_text_intersection(pred: np.ndarray,
                                    target: np.ndarray,
                                    max_shift: int = MAX_MASK_ALIGNMENT_SHIFT) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Короткое описание:
        Ищет глобальный сдвиг pred, максимизирующий пересечение текстовых пикселей.
    Вход:
        pred: np.ndarray -- предсказанная class-matrix.
        target: np.ndarray -- GT class-matrix.
        max_shift: int -- максимальный сдвиг по X/Y.
    Выход:
        Tuple[np.ndarray, Dict[str, float]] -- сдвинутая pred и debug-метрики сдвига.
    """
    # Шаг 1: работаем только с бинарным фактом текста: class > 0.
    pred_text = (pred > 0).astype(np.uint8)
    target_text = (target > 0).astype(np.uint8)
    pred_pixels = int(np.sum(pred_text))
    target_pixels = int(np.sum(target_text))
    if pred_pixels == 0 or target_pixels == 0:
        return pred, {
            "alignment_dx": 0.0,
            "alignment_dy": 0.0,
            "alignment_intersection": 0.0,
            "alignment_intersection_before": 0.0,
            "alignment_intersection_gain": 0.0,
        }

    # Шаг 2: берем bbox pred как шаблон, чтобы matchTemplate считал сумму пересечения.
    pred_y, pred_x = np.where(pred_text)
    y0 = int(pred_y.min())
    y1 = int(pred_y.max()) + 1
    x0 = int(pred_x.min())
    x1 = int(pred_x.max()) + 1
    pred_template = pred_text[y0:y1, x0:x1].astype(np.float32)

    # Шаг 3: вырезаем из target область, соответствующую всем допустимым сдвигам шаблона.
    max_shift = max(0, int(max_shift))
    padded_target = np.pad(
        target_text,
        ((max_shift, max_shift), (max_shift, max_shift)),
        mode="constant",
        constant_values=0,
    ).astype(np.float32)
    search_y0 = y0
    search_x0 = x0
    search_y1 = y0 + 2 * max_shift + pred_template.shape[0]
    search_x1 = x0 + 2 * max_shift + pred_template.shape[1]
    target_search = padded_target[search_y0:search_y1, search_x0:search_x1]

    # Шаг 4: максимум matchTemplate == максимум количества совпавших текстовых пикселей.
    response = cv2.matchTemplate(target_search, pred_template, cv2.TM_CCORR)
    _, max_value, _, max_location = cv2.minMaxLoc(response)
    best_dx = int(max_location[0] - max_shift)
    best_dy = int(max_location[1] - max_shift)
    aligned = shift_class_matrix(pred, dx=best_dx, dy=best_dy)

    intersection_before = int(np.sum(np.logical_and(pred > 0, target > 0)))
    intersection_after = int(max_value)
    return aligned, {
        "alignment_dx": float(best_dx),
        "alignment_dy": float(best_dy),
        "alignment_intersection": float(intersection_after),
        "alignment_intersection_before": float(intersection_before),
        "alignment_intersection_gain": float(intersection_after - intersection_before),
        "alignment_max_shift": float(max_shift),
        "alignment_pred_text_pixels": float(pred_pixels),
        "alignment_target_text_pixels": float(target_pixels),
    }


def class_matrix_to_color(class_matrix: np.ndarray) -> np.ndarray:
    """
    Короткое описание:
        Переводит class-matrix в цветную debug-картинку.
    Вход:
        class_matrix: np.ndarray -- матрица классов.
    Выход:
        np.ndarray -- BGR-визуализация.
    """
    image = np.ones((*class_matrix.shape, 3), dtype=np.uint8) * 255
    rng = np.random.default_rng(42)
    max_class = int(np.max(class_matrix))
    colors = rng.integers(0, 255, size=(max_class + 1, 3), dtype=np.uint8)
    colors[0] = np.array([255, 255, 255], dtype=np.uint8)
    for class_idx in range(1, max_class + 1):
        image[class_matrix == class_idx] = colors[class_idx].tolist()
    return image


def save_debug_matrices(stem: str, pred: np.ndarray, target: np.ndarray) -> None:
    """
    Короткое описание:
        Сохраняет debug-визуализацию pred/target/diff class-matrix.
    Вход:
        stem: str -- имя примера.
        pred: np.ndarray -- предсказанная матрица.
        target: np.ndarray -- GT-матрица.
    Выход:
        None
    """
    if not DEBUG:
        return
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DEBUG_DIR / f"{stem}_pred_classes.png"), class_matrix_to_color(pred))
    cv2.imwrite(str(DEBUG_DIR / f"{stem}_target_classes.png"), class_matrix_to_color(target))

    diff = np.ones((*target.shape, 3), dtype=np.uint8) * 255
    diff[pred == target] = (230, 230, 230)
    diff[np.logical_and(target > 0, pred == target)] = (0, 180, 0)
    diff[np.logical_and(target > 0, pred != target)] = (0, 0, 255)
    diff[np.logical_and(target == 0, pred > 0)] = (255, 0, 0)
    cv2.imwrite(str(DEBUG_DIR / f"{stem}_diff_classes.png"), diff)


def deterministic_cross_entropy(pred: np.ndarray,
                                target: np.ndarray,
                                text_only: bool = False) -> float:
    """
    Короткое описание:
        Считает CE для deterministic class-map через почти one-hot вероятности.
    Вход:
        pred: np.ndarray -- предсказанная матрица.
        target: np.ndarray -- GT-матрица.
        text_only: bool -- считать только по GT-тексту.
    Выход:
        float -- средняя cross entropy.
    """
    mask = target > 0 if text_only else np.ones(target.shape, dtype=bool)
    if int(np.sum(mask)) == 0:
        return 0.0
    correct = pred[mask] == target[mask]
    wrong_prob = max(1e-9, (1.0 - CE_CORRECT_PROB) / max(1, int(max(np.max(pred), np.max(target)))))
    losses = np.where(correct, -np.log(CE_CORRECT_PROB), -np.log(wrong_prob))
    return float(np.mean(losses))


def per_class_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Короткое описание:
        Считает precision/recall/F1/IoU для каждого класса GT и фона.
    Вход:
        pred: np.ndarray -- предсказанная матрица.
        target: np.ndarray -- GT-матрица.
    Выход:
        Dict[str, Dict[str, float]] -- метрики по классам.
    """
    result: Dict[str, Dict[str, float]] = {}
    classes = sorted(set(np.unique(pred).tolist()) | set(np.unique(target).tolist()))

    for class_idx in classes:
        pred_mask = pred == class_idx
        target_mask = target == class_idx
        tp = int(np.sum(np.logical_and(pred_mask, target_mask)))
        fp = int(np.sum(np.logical_and(pred_mask, ~target_mask)))
        fn = int(np.sum(np.logical_and(~pred_mask, target_mask)))
        union = tp + fp + fn
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        iou = tp / union if union > 0 else 1.0
        result[str(int(class_idx))] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }
    return result


def line_detection_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Короткое описание:
        Считает, сколько строк правильно найдено через Hungarian matching по IoU.
    Вход:
        pred: np.ndarray -- предсказанная матрица.
        target: np.ndarray -- GT-матрица.
    Выход:
        Dict[str, float] -- количество строк и precision/recall/F1 по строкам.
    """
    pred_classes = [int(value) for value in np.unique(pred) if int(value) > 0]
    target_classes = [int(value) for value in np.unique(target) if int(value) > 0]
    if not pred_classes or not target_classes:
        correct = 0
    else:
        iou_matrix = np.zeros((len(target_classes), len(pred_classes)), dtype=np.float32)
        for target_idx, target_class in enumerate(target_classes):
            target_mask = target == target_class
            for pred_idx, pred_class in enumerate(pred_classes):
                pred_mask = pred == pred_class
                intersection = int(np.sum(np.logical_and(target_mask, pred_mask)))
                union = int(np.sum(np.logical_or(target_mask, pred_mask)))
                iou_matrix[target_idx, pred_idx] = intersection / union if union > 0 else 0.0
        row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)
        correct = int(np.sum(iou_matrix[row_ind, col_ind] >= LINE_IOU_THRESHOLD))

    pred_count = len(pred_classes)
    target_count = len(target_classes)
    precision = correct / pred_count if pred_count > 0 else 0.0
    recall = correct / target_count if target_count > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "target_lines": float(target_count),
        "predicted_lines": float(pred_count),
        "correctly_detected_lines": float(correct),
        "line_precision": float(precision),
        "line_recall": float(recall),
        "line_f1": float(f1),
        "line_iou_threshold": float(LINE_IOU_THRESHOLD),
    }


def evaluate_image(label_path: Path, unet_model, unet_device) -> Dict[str, object]:
    """
    Короткое описание:
        Строит GT/pred class-matrix и считает метрики для одного изображения.
    Вход:
        label_path: Path -- путь к txt-разметке.
        unet_model: Any -- U-Net модель.
        unet_device: Any -- устройство.
    Выход:
        Dict[str, object] -- результат оценки.
    """
    image_path = find_image_path(label_path.stem)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    binary = binarize_image_with_loaded_model(image, unet_model, unet_device)
    image_height, image_width = binary.shape[:2]
    polygons = read_yolo_polygons(label_path, image_width, image_height)
    target_matrix = build_target_class_matrix(polygons, binary)

    segmenter = LineSegmentation(debug=False)
    hpp_result = segmenter.segment_lines(str(image_path), return_class_matrix=True)
    if len(hpp_result) < 3 or hpp_result[2] is None:
        pred_matrix = np.zeros_like(target_matrix, dtype=np.int32)
    else:
        pred_matrix = hpp_result[2]

    pred_metric, target_metric = prepare_pair_for_metrics(pred_matrix, target_matrix)
    pred_metric, alignment_metrics = align_pred_by_text_intersection(pred_metric, target_metric)
    save_debug_matrices(label_path.stem, pred_metric, target_metric)

    metrics = {
        "cross_entropy": deterministic_cross_entropy(pred_metric, target_metric, text_only=False),
        "cross_entropy_text_only": deterministic_cross_entropy(pred_metric, target_metric, text_only=True),
        "pixel_accuracy": float(np.mean(pred_metric == target_metric)),
        "text_pixel_accuracy": float(np.mean(pred_metric[target_metric > 0] == target_metric[target_metric > 0]))
        if int(np.sum(target_metric > 0)) > 0 else 0.0,
    }
    metrics.update(alignment_metrics)
    metrics.update(line_detection_metrics(pred_metric, target_metric))

    result = {
        "image_path": str(image_path),
        "label_path": str(label_path),
        "target_shape": [int(target_matrix.shape[0]), int(target_matrix.shape[1])],
        "pred_shape": [int(pred_matrix.shape[0]), int(pred_matrix.shape[1])],
        "metric_shape": [int(pred_metric.shape[0]), int(pred_metric.shape[1])],
        "metrics": metrics,
        "per_class": per_class_metrics(pred_metric, target_metric),
    }

    del image, binary, polygons, target_matrix, pred_matrix, pred_metric, target_metric
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def average_results(results: Dict[str, Dict[str, object]]) -> Dict[str, float]:
    """
    Короткое описание:
        Усредняет верхнеуровневые метрики по изображениям без ошибок.
    Вход:
        results: Dict[str, Dict[str, object]] -- результаты по файлам.
    Выход:
        Dict[str, float] -- средние значения.
    """
    metric_values: Dict[str, List[float]] = {}
    for result in results.values():
        if "metrics" not in result:
            continue
        for key, value in result["metrics"].items():
            metric_values.setdefault(key, []).append(float(value))
    return {
        key: float(np.mean(values))
        for key, values in metric_values.items()
        if values
    }


def main() -> None:
    """
    Короткое описание:
        Запускает оценку HPP class-matrix и сохраняет JSON.
    Вход:
        None
    Выход:
        None
    """
    label_paths = get_label_paths()
    unet_model, unet_device = load_unet_model(str(UNET_MODEL_PATH))

    results: Dict[str, Dict[str, object]] = {}
    for label_path in tqdm(label_paths, desc="grade_hpp"):
        try:
            results[label_path.stem] = evaluate_image(label_path, unet_model, unet_device)
        except Exception as exc:
            results[label_path.stem] = {"error": str(exc)}

    output = {
        "label_names": LABEL_NAMES,
        "average_metrics": average_results(results),
        "results": results,
    }
    OUTPUT_JSON_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
