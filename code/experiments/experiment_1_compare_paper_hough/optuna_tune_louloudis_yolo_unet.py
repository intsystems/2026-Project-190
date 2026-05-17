"""
Короткое описание:
    Подбирает гиперпараметры экспериментального Louloudis + YOLO + U-Net pipeline
    на первых 20 изображениях HWR200 test split.
Вход:
    Все пути и параметры заданы константами в начале файла.
Выход:
    JSON-отчеты Optuna, best-summary и debug-картинки предсказаний.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import optuna
from shapely.geometry import Polygon
from tqdm import tqdm


# Шаг 1: задаем корни проекта и добавляем их в sys.path для локальных импортов.
EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import louloudis_text_line_detection_exact as louloudis
from u_net_binarization import load_unet_model


# Split и ground truth.
SPLIT_PATH = PROJECT_ROOT / "handwriting-recognition" / "splits" / "test.txt"
HWR200_IMAGE_ROOT = PROJECT_ROOT / "datasets" / "HWR200" / "hw_dataset"
HWR200_LABELS_PATH = PROJECT_ROOT / "datasets" / "HWR200" / "labels.txt"

# Куда складывать все результаты.
OUTPUT_DIR = PROJECT_ROOT / "debug_images" / "experiment_1_compare_paper_hough" / "optuna_louloudis_yolo_unet"
TRIALS_JSONL_PATH = OUTPUT_DIR / "trials.jsonl"
BEST_SUMMARY_PATH = OUTPUT_DIR / "best_summary.json"
BEST_VISUALIZATION_DIR = OUTPUT_DIR / "best_predictions"

# Размер выборки и число итераций из задачи.
N_IMAGES = 20
N_TRIALS = 40
RANDOM_SEED = 42

# Метрика: IoU-сопоставление полигонов с порогом 0.5 и hmean precision/recall.
IOU_THRESHOLD = 0.5

# Визуализация: GT зеленым, prediction красным.
GT_COLOR_BGR = (0, 180, 0)
PRED_COLOR_BGR = (0, 0, 255)
PAGE_BBOX_COLOR_BGR = (255, 150, 0)
PANEL_BACKGROUND_BGR = (245, 245, 245)
PANEL_TEXT_BGR = (20, 20, 20)

# Фиксированные не-статейные/инфраструктурные параметры.
# Все основные гиперпараметры, явно присутствующие в статье Louloudis 2008,
# ниже подбираются в suggest_params().
FIXED_METHOD_PARAMS = {
    "UNET_THRESHOLD": 0.5,
    "MIN_COMPONENT_AREA": 1,
    "MIN_COMPONENT_HEIGHT_FOR_AH": 1,
    "NEW_LINE_DISTANCE_TOLERANCE_FACTOR": 0.25,
    "NEW_LINE_GROUPING_AH_FACTOR": 0.8,
}

# Глобальные модели, чтобы Optuna не перегружала веса 50 * 20 раз.
YOLO_MODEL = None
UNET_MODEL = None
UNET_DEVICE = None
DATASET_ROWS: List[Dict[str, Any]] = []


def read_first_split_paths() -> List[str]:
    """
    Короткое описание:
        Читает первые N_IMAGES непустых строк из test split.
    Вход:
        None
    Выход:
        List[str] -- относительные пути изображений внутри HWR200/hw_dataset.
    """
    # Шаг 1: читаем split и берем первые непустые пути.
    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        paths = [line.strip() for line in file if line.strip()]
    return paths[:N_IMAGES]


def load_hwr200_labels() -> Dict[str, List[np.ndarray]]:
    """
    Короткое описание:
        Загружает labels.txt HWR200 в словарь path -> polygons.
    Вход:
        None
    Выход:
        Dict[str, List[np.ndarray]] -- полигоны строк в координатах исходных изображений.
    """
    # Шаг 1: парсим каждую строку labels.txt.
    labels: Dict[str, List[np.ndarray]] = {}
    with open(HWR200_LABELS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue
            relative_path, payload = line.split("\t", 1)
            objects = json.loads(payload)
            polygons = []
            for item in objects:
                points = np.asarray(item.get("points", []), dtype=np.float32)
                if points.shape[0] >= 3:
                    polygons.append(points.reshape(-1, 2))
            labels[relative_path] = polygons
    return labels


def build_dataset_rows() -> List[Dict[str, Any]]:
    """
    Короткое описание:
        Собирает первые N_IMAGES изображений, их абсолютные пути и GT-полигоны.
    Вход:
        None
    Выход:
        List[Dict[str, Any]] -- строки датасета для оптимизации.
    """
    # Шаг 1: читаем split и labels.
    split_paths = read_first_split_paths()
    labels = load_hwr200_labels()
    rows: List[Dict[str, Any]] = []

    # Шаг 2: проверяем наличие картинок и разметки.
    for relative_path in split_paths:
        image_path = HWR200_IMAGE_ROOT / relative_path
        if not image_path.exists():
            print(f"[WARN] Нет изображения: {image_path}")
            continue
        gt_polygons = labels.get(relative_path, [])
        if len(gt_polygons) == 0:
            print(f"[WARN] Нет GT-полигонов: {relative_path}")
        rows.append(
            {
                "relative_path": relative_path,
                "image_path": image_path,
                "gt_polygons": gt_polygons,
            }
        )
    return rows


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Короткое описание:
        Задает пространство поиска Optuna для гиперпараметров метода.
    Вход:
        trial: optuna.Trial -- текущий trial.
    Выход:
        Dict[str, Any] -- словарь гиперпараметров.
    """
    # Шаг 1: подбираем все статейные гиперпараметры из louloudis_text_line_detection_exact.py.
    params = dict(FIXED_METHOD_PARAMS)
    params.update(
        {
            # Оценка AH.
            "AH_HISTOGRAM_BIN_WIDTH": trial.suggest_int("AH_HISTOGRAM_BIN_WIDTH", 1, 4),

            # Subset-разбиение из статьи (формулы (1)-(3)).
            "SUBSET1_MIN_HEIGHT_FACTOR": trial.suggest_float("SUBSET1_MIN_HEIGHT_FACTOR", 0.30, 0.80),
            "SUBSET1_MAX_HEIGHT_FACTOR": trial.suggest_float("SUBSET1_MAX_HEIGHT_FACTOR", 2.20, 4.20),
            "SUBSET1_MIN_WIDTH_FACTOR": trial.suggest_float("SUBSET1_MIN_WIDTH_FACTOR", 0.20, 0.80),
            "SUBSET2_MIN_HEIGHT_FACTOR": trial.suggest_float("SUBSET2_MIN_HEIGHT_FACTOR", 2.2, 4.2),
            "SUBSET3_MAX_HEIGHT_FACTOR": trial.suggest_float("SUBSET3_MAX_HEIGHT_FACTOR", 2.2, 4.2),
            "SUBSET3_SMALL_HEIGHT_FACTOR": trial.suggest_float("SUBSET3_SMALL_HEIGHT_FACTOR", 0.20, 0.80),
            "SUBSET3_NARROW_WIDTH_FACTOR": trial.suggest_float("SUBSET3_NARROW_WIDTH_FACTOR", 0.20, 0.80),

            # Hough-параметры из статьи.
            "HOUGH_THETA_MIN_DEG": trial.suggest_int("HOUGH_THETA_MIN_DEG", 75, 92),
            "HOUGH_THETA_MAX_DEG": trial.suggest_int("HOUGH_THETA_MAX_DEG", 88, 105),
            "HOUGH_THETA_STEP_DEG": trial.suggest_int("HOUGH_THETA_STEP_DEG", 1, 2),
            "HOUGH_RHO_STEP_AH_FACTOR": trial.suggest_float("HOUGH_RHO_STEP_AH_FACTOR", 0.10, 0.35),
            "HOUGH_RHO_NEIGHBOR_CELLS": trial.suggest_int("HOUGH_RHO_NEIGHBOR_CELLS", 2, 10),
            "HOUGH_MIN_VOTES_N1": trial.suggest_int("HOUGH_MIN_VOTES_N1", 3, 12),
            "HOUGH_SECONDARY_VOTES_N2": trial.suggest_int("HOUGH_SECONDARY_VOTES_N2", 6, 20),
            "HOUGH_DOMINANT_ANGLE_TOLERANCE_DEG": trial.suggest_float("HOUGH_DOMINANT_ANGLE_TOLERANCE_DEG", 0.5, 5.0),
            "HOUGH_COMPONENT_MIN_BLOCK_FRACTION": trial.suggest_float("HOUGH_COMPONENT_MIN_BLOCK_FRACTION", 0.30, 0.75),

            # Пост-обработка новых линий из статьи.
            "NEW_LINE_DISTANCE_FACTOR": trial.suggest_float("NEW_LINE_DISTANCE_FACTOR", 0.6, 1.2),

            # Разделение Subset 2 (зона и локальное удаление junction).
            "JUNCTION_REMOVAL_NEIGHBORHOOD": trial.suggest_int("JUNCTION_REMOVAL_NEIGHBORHOOD", 1, 5, step=2),
            "SUBSET2_ZONE_TOP_FACTOR": trial.suggest_float("SUBSET2_ZONE_TOP_FACTOR", 0.30, 0.70),
            "SUBSET2_ZONE_BOTTOM_FACTOR": trial.suggest_float("SUBSET2_ZONE_BOTTOM_FACTOR", 1.20, 1.80),
        }
    )

    # Шаг 2: валидируем зависимости параметров.
    if params["SUBSET1_MAX_HEIGHT_FACTOR"] <= params["SUBSET1_MIN_HEIGHT_FACTOR"]:
        params["SUBSET1_MAX_HEIGHT_FACTOR"] = params["SUBSET1_MIN_HEIGHT_FACTOR"] + 0.1
    if params["HOUGH_THETA_MAX_DEG"] <= params["HOUGH_THETA_MIN_DEG"]:
        params["HOUGH_THETA_MAX_DEG"] = params["HOUGH_THETA_MIN_DEG"] + max(1, params["HOUGH_THETA_STEP_DEG"])
    if params["HOUGH_SECONDARY_VOTES_N2"] <= params["HOUGH_MIN_VOTES_N1"]:
        params["HOUGH_SECONDARY_VOTES_N2"] = params["HOUGH_MIN_VOTES_N1"] + 1
    if params["SUBSET2_ZONE_BOTTOM_FACTOR"] <= params["SUBSET2_ZONE_TOP_FACTOR"]:
        params["SUBSET2_ZONE_BOTTOM_FACTOR"] = params["SUBSET2_ZONE_TOP_FACTOR"] + 0.2
    return params


def apply_louloudis_params(params: Dict[str, Any]) -> None:
    """
    Короткое описание:
        Применяет гиперпараметры к экспериментальному модулю Louloudis.
    Вход:
        params: Dict[str, Any] -- словарь гиперпараметров.
    Выход:
        None
    """
    # Шаг 1: все одноименные ключи переносим в globals модуля.
    for name, value in params.items():
        setattr(louloudis, name, value)


def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Короткое описание:
        Доводит словарь параметров до валидных ограничений метода.
    Вход:
        params: Dict[str, Any] -- параметры Optuna.
    Выход:
        Dict[str, Any] -- нормализованные параметры.
    """
    # Шаг 1: добавляем фиксированные параметры, которых нет в study.best_params.
    normalized = dict(FIXED_METHOD_PARAMS)
    normalized.update(params)
    # Шаг 1.1: подстраховка для старых best_summary без новых ключей.
    for key in [
        "SUBSET1_MIN_HEIGHT_FACTOR",
        "SUBSET1_MAX_HEIGHT_FACTOR",
        "SUBSET1_MIN_WIDTH_FACTOR",
        "AH_HISTOGRAM_BIN_WIDTH",
        "SUBSET2_MIN_HEIGHT_FACTOR",
        "SUBSET3_MAX_HEIGHT_FACTOR",
        "SUBSET3_SMALL_HEIGHT_FACTOR",
        "SUBSET3_NARROW_WIDTH_FACTOR",
        "HOUGH_THETA_MIN_DEG",
        "HOUGH_THETA_MAX_DEG",
        "HOUGH_THETA_STEP_DEG",
        "HOUGH_RHO_STEP_AH_FACTOR",
        "HOUGH_RHO_NEIGHBOR_CELLS",
        "HOUGH_MIN_VOTES_N1",
        "HOUGH_SECONDARY_VOTES_N2",
        "HOUGH_DOMINANT_ANGLE_TOLERANCE_DEG",
        "HOUGH_COMPONENT_MIN_BLOCK_FRACTION",
        "NEW_LINE_DISTANCE_FACTOR",
        "JUNCTION_REMOVAL_NEIGHBORHOOD",
        "SUBSET2_ZONE_TOP_FACTOR",
        "SUBSET2_ZONE_BOTTOM_FACTOR",
    ]:
        normalized.setdefault(key, getattr(louloudis, key))

    # Шаг 2: сохраняем ограничения зависимостей.
    if normalized["SUBSET1_MAX_HEIGHT_FACTOR"] <= normalized["SUBSET1_MIN_HEIGHT_FACTOR"]:
        normalized["SUBSET1_MAX_HEIGHT_FACTOR"] = normalized["SUBSET1_MIN_HEIGHT_FACTOR"] + 0.1
    if normalized["HOUGH_THETA_MAX_DEG"] <= normalized["HOUGH_THETA_MIN_DEG"]:
        normalized["HOUGH_THETA_MAX_DEG"] = normalized["HOUGH_THETA_MIN_DEG"] + max(1, int(normalized["HOUGH_THETA_STEP_DEG"]))
    if normalized["HOUGH_SECONDARY_VOTES_N2"] <= normalized["HOUGH_MIN_VOTES_N1"]:
        normalized["HOUGH_SECONDARY_VOTES_N2"] = normalized["HOUGH_MIN_VOTES_N1"] + 1
    if normalized["SUBSET2_ZONE_BOTTOM_FACTOR"] <= normalized["SUBSET2_ZONE_TOP_FACTOR"]:
        normalized["SUBSET2_ZONE_BOTTOM_FACTOR"] = normalized["SUBSET2_ZONE_TOP_FACTOR"] + 0.2
    return normalized


def class_matrix_to_polygons(
    class_matrix: np.ndarray,
    x_offset: int,
    y_offset: int,
) -> List[np.ndarray]:
    """
    Короткое описание:
        Переводит class-matrix строк в line-полигоны по контурам масок классов.
    Вход:
        class_matrix: np.ndarray -- матрица классов строк, 0 фон.
        x_offset: int -- смещение crop страницы по X в исходном изображении.
        y_offset: int -- смещение crop страницы по Y в исходном изображении.
    Выход:
        List[np.ndarray] -- предсказанные полигоны строк в координатах исходного изображения.
    """
    # Шаг 1: для каждого класса строки строим ориентированный minAreaRect, как раньше.
    polygons: List[np.ndarray] = []
    max_class = int(np.max(class_matrix)) if class_matrix.size > 0 else 0
    for class_index in range(1, max_class + 1):
        class_mask = (class_matrix == class_index).astype(np.uint8)
        if int(np.sum(class_mask)) == 0:
            continue
        ys, xs = np.where(class_mask > 0)
        if len(xs) < 3:
            continue
        points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect).astype(np.float32)
        box[:, 0] += float(x_offset)
        box[:, 1] += float(y_offset)
        polygons.append(box)
    return polygons


def safe_polygon(points: np.ndarray) -> Polygon:
    """
    Короткое описание:
        Создает валидный shapely Polygon из массива точек.
    Вход:
        points: np.ndarray -- точки полигона.
    Выход:
        Polygon -- валидный polygon или пустая геометрия.
    """
    # Шаг 1: исправляем самопересечения через buffer(0).
    polygon = Polygon(np.asarray(points, dtype=np.float32).reshape(-1, 2))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def polygon_iou(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    Короткое описание:
        Считает IoU двух полигонов.
    Вход:
        predicted: np.ndarray -- предсказанный полигон.
        target: np.ndarray -- GT полигон.
    Выход:
        float -- IoU.
    """
    # Шаг 1: считаем площади intersection/union.
    pred_poly = safe_polygon(predicted)
    gt_poly = safe_polygon(target)
    if pred_poly.is_empty or gt_poly.is_empty:
        return 0.0
    intersection = pred_poly.intersection(gt_poly).area
    union = pred_poly.area + gt_poly.area - intersection
    return float(intersection / union) if union > 0 else 0.0


def match_polygons(predicted: List[np.ndarray], targets: List[np.ndarray]) -> Dict[str, float]:
    """
    Короткое описание:
        Greedy IoU-сопоставление полигонов при IOU_THRESHOLD.
    Вход:
        predicted: List[np.ndarray] -- предсказанные полигоны.
        targets: List[np.ndarray] -- GT полигоны.
    Выход:
        Dict[str, float] -- tp/fp/fn/precision/recall/hmean.
    """
    # Шаг 1: жадно матчим каждый prediction с лучшим еще свободным GT.
    matched_targets = [False] * len(targets)
    tp = 0
    fp = 0
    for pred_poly in predicted:
        best_iou = 0.0
        best_index = -1
        for target_index, target_poly in enumerate(targets):
            if matched_targets[target_index]:
                continue
            iou = polygon_iou(pred_poly, target_poly)
            if iou > best_iou:
                best_iou = iou
                best_index = target_index
        if best_iou >= IOU_THRESHOLD and best_index >= 0:
            matched_targets[best_index] = True
            tp += 1
        else:
            fp += 1

    # Шаг 2: считаем precision/recall/hmean.
    fn = matched_targets.count(False)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    hmean = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "hmean": float(hmean),
    }


def evaluate_one_image(row: Dict[str, Any], params: Dict[str, Any], debug_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Короткое описание:
        Запускает YOLO + U-Net + Louloudis для одного изображения и считает метрику.
    Вход:
        row: Dict[str, Any] -- строка датасета.
        params: Dict[str, Any] -- гиперпараметры.
        debug_dir: Optional[Path] -- папка debug для одного изображения или None.
    Выход:
        Dict[str, Any] -- prediction polygons, page info и метрики.
    """
    debug_enabled = debug_dir is not None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Шаг 1: читаем изображение и запускаем pipeline с уже загруженными моделями.
    start_time = time.perf_counter()
    class_matrix, lines, page_info = louloudis.run_on_image(
        str(row["image_path"]),
        debug=debug_enabled,
        debug_output_dir=str(debug_dir if debug_dir is not None else OUTPUT_DIR / "tmp_no_debug"),
        yolo_model=YOLO_MODEL,
        unet_model=UNET_MODEL,
        unet_device=UNET_DEVICE,
        return_page_info=True,
        use_tqdm=False,
    )
    runtime_sec = time.perf_counter() - start_time

    # Шаг 2: переводим class-matrix в полигоны исходного изображения.
    bbox = page_info["bbox"]
    predicted_polygons = class_matrix_to_polygons(
        class_matrix=class_matrix,
        x_offset=int(bbox["x"]),
        y_offset=int(bbox["y"]),
    )
    metrics = match_polygons(predicted_polygons, row["gt_polygons"])
    metrics["pred_count"] = float(len(predicted_polygons))
    metrics["gt_count"] = float(len(row["gt_polygons"]))

    return {
        "relative_path": row["relative_path"],
        "predicted_polygons": predicted_polygons,
        "class_matrix": class_matrix,
        "page_info": page_info,
        "metrics": metrics,
        "line_count": len(lines),
        "runtime_sec": float(runtime_sec),
    }


def aggregate_metrics(per_image_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Короткое описание:
        Агрегирует tp/fp/fn по всем изображениям и считает global hmean.
    Вход:
        per_image_results: List[Dict[str, Any]] -- результаты по изображениям.
    Выход:
        Dict[str, float] -- precision/recall/hmean и счетчики.
    """
    # Шаг 1: суммируем счетчики.
    tp = sum(float(item["metrics"]["tp"]) for item in per_image_results)
    fp = sum(float(item["metrics"]["fp"]) for item in per_image_results)
    fn = sum(float(item["metrics"]["fn"]) for item in per_image_results)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    hmean = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "hmean": float(hmean),
    }


def aggregate_runtime(per_image_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Короткое описание:
        Агрегирует время run_on_image по изображениям.
    Вход:
        per_image_results: List[Dict[str, Any]] -- результаты по изображениям.
    Выход:
        Dict[str, float] -- total/mean/median/min/max runtime.
    """
    runtimes = [float(item["runtime_sec"]) for item in per_image_results if item.get("runtime_sec") is not None]
    if len(runtimes) == 0:
        return {
            "total_runtime_sec": 0.0,
            "mean_runtime_sec": 0.0,
            "median_runtime_sec": 0.0,
            "min_runtime_sec": 0.0,
            "max_runtime_sec": 0.0,
        }
    values = np.asarray(runtimes, dtype=np.float64)
    return {
        "total_runtime_sec": float(np.sum(values)),
        "mean_runtime_sec": float(np.mean(values)),
        "median_runtime_sec": float(np.median(values)),
        "min_runtime_sec": float(np.min(values)),
        "max_runtime_sec": float(np.max(values)),
    }


def objective(trial: optuna.Trial) -> float:
    """
    Короткое описание:
        Objective Optuna: максимизирует global hmean.
    Вход:
        trial: optuna.Trial -- текущая итерация.
    Выход:
        float -- hmean.
    """
    # Шаг 1: применяем trial-гиперпараметры.
    trial_start_time = time.perf_counter()
    params = suggest_params(trial)
    apply_louloudis_params(params)

    # Шаг 2: считаем метрику по первым N_IMAGES.
    per_image_results = []
    for row in DATASET_ROWS:
        per_image_results.append(evaluate_one_image(row, params))
    metrics = aggregate_metrics(per_image_results)
    runtime = aggregate_runtime(per_image_results)
    trial_runtime_sec = time.perf_counter() - trial_start_time

    # Шаг 3: сохраняем trial в JSONL и возвращаем hmean.
    record = {
        "trial": int(trial.number),
        "params": params,
        "metrics": metrics,
        "runtime": runtime,
        "trial_runtime_sec": float(trial_runtime_sec),
        "per_image_metrics": [
            {
                "relative_path": item["relative_path"],
                "metrics": item["metrics"],
                "line_count": int(item["line_count"]),
                "runtime_sec": float(item["runtime_sec"]),
                "page_bbox": item["page_info"]["bbox"],
                "page_confidence": item["page_info"].get("confidence"),
            }
            for item in per_image_results
        ],
    }
    with open(TRIALS_JSONL_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    trial.set_user_attr("precision", metrics["precision"])
    trial.set_user_attr("recall", metrics["recall"])
    trial.set_user_attr("tp", metrics["tp"])
    trial.set_user_attr("fp", metrics["fp"])
    trial.set_user_attr("fn", metrics["fn"])
    trial.set_user_attr("trial_runtime_sec", trial_runtime_sec)
    trial.set_user_attr("mean_runtime_sec", runtime["mean_runtime_sec"])
    trial.set_user_attr("median_runtime_sec", runtime["median_runtime_sec"])
    return float(metrics["hmean"])


def draw_polygons_for_debug(
    image: np.ndarray,
    gt_polygons: List[np.ndarray],
    predicted_polygons: List[np.ndarray],
    page_info: Dict[str, Any],
) -> np.ndarray:
    """
    Короткое описание:
        Рисует GT, prediction и bbox найденной страницы на копии изображения.
    Вход:
        image: np.ndarray -- BGR изображение.
        gt_polygons: List[np.ndarray] -- GT полигоны.
        predicted_polygons: List[np.ndarray] -- prediction полигоны.
        page_info: Dict[str, Any] -- информация YOLO crop.
    Выход:
        np.ndarray -- debug-изображение BGR.
    """
    # Шаг 1: рисуем GT зеленым и prediction красным.
    canvas = image.copy()
    for polygon in gt_polygons:
        cv2.polylines(canvas, [np.asarray(polygon, dtype=np.int32)], True, GT_COLOR_BGR, 2)
    for polygon in predicted_polygons:
        cv2.polylines(canvas, [np.asarray(polygon, dtype=np.int32)], True, PRED_COLOR_BGR, 2)

    # Шаг 2: рисуем bbox страницы и легенду.
    bbox = page_info["bbox"]
    cv2.rectangle(
        canvas,
        (int(bbox["x"]), int(bbox["y"])),
        (int(bbox["x"] + bbox["w"]), int(bbox["y"] + bbox["h"])),
        PAGE_BBOX_COLOR_BGR,
        2,
    )
    cv2.putText(canvas, "GT green | Pred red | YOLO page blue", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
    cv2.putText(canvas, "GT green | Pred red | YOLO page blue", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
    return canvas


def color_for_index(index: int, total: int, saturation: float = 0.85, value: float = 0.95) -> Tuple[int, int, int]:
    """
    Короткое описание:
        Возвращает стабильный BGR-цвет для индекса полигона.
    Вход:
        index: int -- индекс полигона.
        total: int -- общее число полигонов.
        saturation: float -- насыщенность HSV.
        value: float -- яркость HSV.
    Выход:
        Tuple[int, int, int] -- цвет BGR.
    """
    # Шаг 1: генерируем цвет через OpenCV HSV.
    hue = int(round(179.0 * (index % max(total, 1)) / max(total, 1)))
    hsv = np.array([[[hue, int(255 * saturation), int(255 * value)]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_colored_polygons(
    image: np.ndarray,
    polygons: List[np.ndarray],
    thickness: int = 2,
    label_prefix: str = "",
) -> np.ndarray:
    """
    Короткое описание:
        Рисует каждый полигон своим цветом.
    Вход:
        image: np.ndarray -- BGR изображение.
        polygons: List[np.ndarray] -- список полигонов.
        thickness: int -- толщина линии.
        label_prefix: str -- префикс подписи номера.
    Выход:
        np.ndarray -- изображение с полигонами.
    """
    # Шаг 1: рисуем контуры и номера.
    canvas = image.copy()
    total = len(polygons)
    for index, polygon in enumerate(polygons, start=1):
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
        if len(points) < 3:
            continue
        color = color_for_index(index - 1, total)
        cv2.polylines(canvas, [points], True, color, thickness)
        x, y = points[0]
        if label_prefix:
            cv2.putText(
                canvas,
                f"{label_prefix}{index}",
                (int(x), max(18, int(y) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
    return canvas


def polygons_to_colored_mask(shape: Tuple[int, int], polygons: List[np.ndarray]) -> np.ndarray:
    """
    Короткое описание:
        Создает цветную маску полигонов, где каждый полигон имеет свой цвет.
    Вход:
        shape: Tuple[int, int] -- размер H, W.
        polygons: List[np.ndarray] -- список полигонов.
    Выход:
        np.ndarray -- BGR маска.
    """
    # Шаг 1: заполняем полигоны на светлом фоне.
    height, width = shape
    mask_image = np.full((height, width, 3), 255, dtype=np.uint8)
    total = len(polygons)
    for index, polygon in enumerate(polygons):
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
        if len(points) < 3:
            continue
        color = color_for_index(index, total)
        cv2.fillPoly(mask_image, [points], color)
        cv2.polylines(mask_image, [points], True, (0, 0, 0), 2)
    return mask_image


def put_panel_title(image: np.ndarray, title: str) -> np.ndarray:
    """
    Короткое описание:
        Добавляет заголовок в левый верхний угол панели.
    Вход:
        image: np.ndarray -- BGR изображение.
        title: str -- заголовок.
    Выход:
        np.ndarray -- изображение с заголовком.
    """
    # Шаг 1: рисуем подложку и текст.
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 520), 42), (255, 255, 255), -1)
    cv2.putText(canvas, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, PANEL_TEXT_BGR, 2)
    return canvas


def resize_for_panel(image: np.ndarray, panel_size: Tuple[int, int]) -> np.ndarray:
    """
    Короткое описание:
        Масштабирует изображение в фиксированную панель с сохранением пропорций.
    Вход:
        image: np.ndarray -- BGR изображение.
        panel_size: Tuple[int, int] -- размер панели W, H.
    Выход:
        np.ndarray -- BGR панель.
    """
    # Шаг 1: вписываем изображение в панель.
    panel_width, panel_height = panel_size
    height, width = image.shape[:2]
    scale = min(panel_width / max(width, 1), panel_height / max(height, 1))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    panel = np.full((panel_height, panel_width, 3), PANEL_BACKGROUND_BGR, dtype=np.uint8)
    y0 = (panel_height - new_height) // 2
    x0 = (panel_width - new_width) // 2
    panel[y0:y0 + new_height, x0:x0 + new_width] = resized
    return panel


def make_detailed_debug_panel(
    image: np.ndarray,
    gt_polygons: List[np.ndarray],
    predicted_polygons: List[np.ndarray],
    page_info: Dict[str, Any],
) -> np.ndarray:
    """
    Короткое описание:
        Собирает 2 x 2 debug-панель: фото, GT-маска, Pred-маска, overlay.
    Вход:
        image: np.ndarray -- исходное BGR изображение.
        gt_polygons: List[np.ndarray] -- GT полигоны.
        predicted_polygons: List[np.ndarray] -- предсказанные полигоны.
        page_info: Dict[str, Any] -- bbox страницы.
    Выход:
        np.ndarray -- итоговая 2 x 2 панель.
    """
    # Шаг 1: готовим четыре изображения.
    original_panel = put_panel_title(image, "1. Original image")

    gt_mask = polygons_to_colored_mask(image.shape[:2], gt_polygons)
    gt_mask = draw_colored_polygons(gt_mask, gt_polygons, thickness=2, label_prefix="G")
    gt_mask = put_panel_title(gt_mask, "2. Ground truth mask")

    pred_mask = polygons_to_colored_mask(image.shape[:2], predicted_polygons)
    pred_mask = draw_colored_polygons(pred_mask, predicted_polygons, thickness=2, label_prefix="P")
    pred_mask = put_panel_title(pred_mask, "3. Predicted mask")

    overlay = image.copy()
    bbox = page_info["bbox"]
    cv2.rectangle(
        overlay,
        (int(bbox["x"]), int(bbox["y"])),
        (int(bbox["x"] + bbox["w"]), int(bbox["y"] + bbox["h"])),
        PAGE_BBOX_COLOR_BGR,
        3,
    )
    overlay = draw_colored_polygons(overlay, gt_polygons, thickness=3, label_prefix="G")
    overlay = draw_colored_polygons(overlay, predicted_polygons, thickness=2, label_prefix="P")
    overlay = put_panel_title(overlay, "4. GT + prediction rectangles")

    # Шаг 2: собираем панели одного размера.
    panel_size = (900, 700)
    panels = [
        resize_for_panel(original_panel, panel_size),
        resize_for_panel(gt_mask, panel_size),
        resize_for_panel(pred_mask, panel_size),
        resize_for_panel(overlay, panel_size),
    ]
    top = np.hstack([panels[0], panels[1]])
    bottom = np.hstack([panels[2], panels[3]])
    return np.vstack([top, bottom])


def class_matrix_to_colored_original(
    image_shape: Tuple[int, int],
    class_matrix: np.ndarray,
    page_info: Dict[str, Any],
) -> np.ndarray:
    """
    Короткое описание:
        Переносит цветную class_matrix из crop страницы в координаты исходного изображения.
    Вход:
        image_shape: Tuple[int, int] -- размер исходного изображения H, W.
        class_matrix: np.ndarray -- class_matrix crop страницы.
        page_info: Dict[str, Any] -- bbox crop страницы.
    Выход:
        np.ndarray -- цветная BGR-сегментация class_matrix в исходных координатах.
    """
    # Шаг 1: создаем белый холст исходного размера.
    height, width = image_shape
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    max_class = int(np.max(class_matrix)) if class_matrix.size > 0 else 0
    if max_class == 0:
        return canvas

    # Шаг 2: раскрашиваем каждый класс строки своим цветом.
    colored_crop = np.full((*class_matrix.shape, 3), 255, dtype=np.uint8)
    for class_index in range(1, max_class + 1):
        color = color_for_index(class_index - 1, max_class)
        colored_crop[class_matrix == class_index] = color

    # Шаг 3: вставляем crop обратно в координаты исходника.
    bbox = page_info["bbox"]
    x0 = int(bbox["x"])
    y0 = int(bbox["y"])
    x1 = min(width, x0 + colored_crop.shape[1])
    y1 = min(height, y0 + colored_crop.shape[0])
    crop_w = max(0, x1 - x0)
    crop_h = max(0, y1 - y0)
    if crop_w > 0 and crop_h > 0:
        canvas[y0:y1, x0:x1] = colored_crop[:crop_h, :crop_w]
    return canvas


def class_matrix_overlay_original(
    image: np.ndarray,
    class_matrix: np.ndarray,
    page_info: Dict[str, Any],
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Короткое описание:
        Накладывает class_matrix на исходное изображение.
    Вход:
        image: np.ndarray -- исходное BGR изображение.
        class_matrix: np.ndarray -- class_matrix crop страницы.
        page_info: Dict[str, Any] -- bbox crop страницы.
        alpha: float -- прозрачность цветной сегментации.
    Выход:
        np.ndarray -- overlay BGR.
    """
    # Шаг 1: переносим class_matrix в исходные координаты.
    segmentation = class_matrix_to_colored_original(image.shape[:2], class_matrix, page_info)
    overlay = image.copy()
    foreground = np.any(segmentation != 255, axis=2)
    overlay[foreground] = cv2.addWeighted(image[foreground], 1.0 - alpha, segmentation[foreground], alpha, 0)
    return overlay


def save_per_image_debug_folder(
    image_index: int,
    row: Dict[str, Any],
    result: Dict[str, Any],
    output_root: Path,
) -> Dict[str, str]:
    """
    Короткое описание:
        Сохраняет подробные debug-файлы одного изображения в отдельную папку.
    Вход:
        image_index: int -- индекс изображения.
        row: Dict[str, Any] -- строка датасета.
        result: Dict[str, Any] -- результат evaluate_one_image.
        output_root: Path -- корневая папка best_predictions.
    Выход:
        Dict[str, str] -- пути к основным сохраненным файлам.
    """
    # Шаг 1: создаем папку изображения.
    folder_name = f"{image_index:02d}_{row['relative_path'].replace('/', '__')}"
    folder_name = Path(folder_name).with_suffix("").name
    image_dir = output_root / folder_name
    image_dir.mkdir(parents=True, exist_ok=True)

    # Шаг 2: читаем исходник и сохраняем базовые представления.
    image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(row["image_path"])

    gt_mask = polygons_to_colored_mask(image.shape[:2], row["gt_polygons"])
    gt_mask = draw_colored_polygons(gt_mask, row["gt_polygons"], thickness=2, label_prefix="G")

    pred_mask = polygons_to_colored_mask(image.shape[:2], result["predicted_polygons"])
    pred_mask = draw_colored_polygons(pred_mask, result["predicted_polygons"], thickness=2, label_prefix="P")

    segmentation_before_minrect = class_matrix_to_colored_original(
        image_shape=image.shape[:2],
        class_matrix=result["class_matrix"],
        page_info=result["page_info"],
    )
    segmentation_overlay = class_matrix_overlay_original(
        image=image,
        class_matrix=result["class_matrix"],
        page_info=result["page_info"],
    )

    rectangles_overlay = draw_colored_polygons(image, row["gt_polygons"], thickness=3, label_prefix="G")
    rectangles_overlay = draw_colored_polygons(rectangles_overlay, result["predicted_polygons"], thickness=2, label_prefix="P")
    bbox = result["page_info"]["bbox"]
    cv2.rectangle(
        rectangles_overlay,
        (int(bbox["x"]), int(bbox["y"])),
        (int(bbox["x"] + bbox["w"]), int(bbox["y"] + bbox["h"])),
        PAGE_BBOX_COLOR_BGR,
        3,
    )

    panel = make_detailed_debug_panel(
        image=image,
        gt_polygons=row["gt_polygons"],
        predicted_polygons=result["predicted_polygons"],
        page_info=result["page_info"],
    )

    paths = {
        "original": str(image_dir / "00_original.jpg"),
        "ground_truth_mask": str(image_dir / "01_ground_truth_mask.png"),
        "predicted_mask_minrect": str(image_dir / "02_predicted_mask_minrect.png"),
        "segmentation_before_minrect": str(image_dir / "03_segmentation_before_minrect.png"),
        "segmentation_before_minrect_overlay": str(image_dir / "04_segmentation_before_minrect_overlay.jpg"),
        "gt_pred_rectangles": str(image_dir / "05_gt_pred_rectangles.jpg"),
        "four_panel": str(image_dir / "06_four_panel.jpg"),
    }
    cv2.imwrite(paths["original"], image)
    cv2.imwrite(paths["ground_truth_mask"], gt_mask)
    cv2.imwrite(paths["predicted_mask_minrect"], pred_mask)
    cv2.imwrite(paths["segmentation_before_minrect"], segmentation_before_minrect)
    cv2.imwrite(paths["segmentation_before_minrect_overlay"], segmentation_overlay)
    cv2.imwrite(paths["gt_pred_rectangles"], rectangles_overlay)
    cv2.imwrite(paths["four_panel"], panel)

    # Шаг 3: сохраняем численный отчет по изображению.
    report = {
        "relative_path": row["relative_path"],
        "image_path": str(row["image_path"]),
        "metrics": result["metrics"],
        "line_count": int(result["line_count"]),
        "page_bbox": result["page_info"]["bbox"],
        "page_confidence": result["page_info"].get("confidence"),
        "files": paths,
        "method_debug_dir": str(image_dir / "method_debug"),
    }
    with open(image_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    return paths


def evaluate_and_save_best(best_params: Dict[str, Any], best_value: float, study_runtime_sec: Optional[float] = None) -> None:
    """
    Короткое описание:
        Перезапускает best params, сохраняет summary и картинки предсказаний.
    Вход:
        best_params: Dict[str, Any] -- лучшие параметры Optuna.
        best_value: float -- лучшее значение hmean.
    Выход:
        None
    """
    # Шаг 1: применяем лучшие параметры и готовим папку.
    best_params = normalize_params(best_params)
    apply_louloudis_params(best_params)
    BEST_VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    per_image_results = []
    visualization_files: Dict[str, Dict[str, str]] = {}

    # Шаг 2: сохраняем визуализацию для каждого изображения.
    for image_index, row in enumerate(tqdm(DATASET_ROWS, desc="Best predictions")):
        folder_name = f"{image_index:02d}_{row['relative_path'].replace('/', '__')}"
        folder_name = Path(folder_name).with_suffix("").name
        method_debug_dir = BEST_VISUALIZATION_DIR / folder_name / "method_debug"
        result = evaluate_one_image(row, best_params, debug_dir=method_debug_dir)
        per_image_results.append(result)
        visualization_files[row["relative_path"]] = save_per_image_debug_folder(
            image_index=image_index,
            row=row,
            result=result,
            output_root=BEST_VISUALIZATION_DIR,
        )

    # Шаг 3: сохраняем подробный summary.
    metrics = aggregate_metrics(per_image_results)
    runtime = aggregate_runtime(per_image_results)
    summary = {
        "optimized_metric": "hmean = 2 * precision * recall / (precision + recall)",
        "iou_threshold": IOU_THRESHOLD,
        "n_images": len(DATASET_ROWS),
        "n_trials": N_TRIALS,
        "study_runtime_sec": None if study_runtime_sec is None else float(study_runtime_sec),
        "best_runtime_recomputed": runtime,
        "runtime_scope": "run_on_image: YOLO page crop + U-Net binarization + Louloudis method logic",
        "tuned_params": [
            "SUBSET1_MIN_HEIGHT_FACTOR",
            "SUBSET1_MAX_HEIGHT_FACTOR",
            "SUBSET1_MIN_WIDTH_FACTOR",
            "HOUGH_MIN_VOTES_N1",
            "HOUGH_SECONDARY_VOTES_N2",
            "HOUGH_RHO_NEIGHBOR_CELLS",
            "HOUGH_COMPONENT_MIN_BLOCK_FRACTION",
            "HOUGH_RHO_STEP_AH_FACTOR",
        ],
        "fixed_method_params": FIXED_METHOD_PARAMS,
        "best_value_from_study": float(best_value),
        "best_metrics_recomputed": metrics,
        "best_params": best_params,
        "split_path": str(SPLIT_PATH),
        "labels_path": str(HWR200_LABELS_PATH),
        "yolo_model_path": str(louloudis.YOLO_PAGE_SEGMENTATION_MODEL_PATH),
        "unet_model_path": str(louloudis.UNET_BINARIZATION_MODEL_PATH),
        "visualization_dir": str(BEST_VISUALIZATION_DIR),
        "per_image": [
            {
                "relative_path": item["relative_path"],
                "metrics": item["metrics"],
                "line_count": int(item["line_count"]),
                "runtime_sec": float(item["runtime_sec"]),
                "page_bbox": item["page_info"]["bbox"],
                "page_confidence": item["page_info"].get("confidence"),
                "visualization_files": visualization_files.get(item["relative_path"], {}),
                "predicted_polygons": [poly.astype(float).tolist() for poly in item["predicted_polygons"]],
            }
            for item in per_image_results
        ],
    }
    with open(BEST_SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    # Шаг 4: подробный вывод в консоль.
    print("\n[OK] Optuna завершена")
    print(f"[OK] Лучший hmean: {metrics['hmean']:.6f}")
    print(f"[OK] Precision: {metrics['precision']:.6f}")
    print(f"[OK] Recall: {metrics['recall']:.6f}")
    print(f"[OK] TP/FP/FN: {metrics['tp']:.0f}/{metrics['fp']:.0f}/{metrics['fn']:.0f}")
    print(f"[OK] Runtime mean/median: {runtime['mean_runtime_sec']:.3f}s / {runtime['median_runtime_sec']:.3f}s")
    if study_runtime_sec is not None:
        print(f"[OK] Study runtime: {study_runtime_sec:.3f}s")
    print("[OK] Лучшие гиперпараметры:")
    for key in sorted(best_params):
        print(f"  {key}: {best_params[key]}")
    print(f"[OK] Summary: {BEST_SUMMARY_PATH}")
    print(f"[OK] Картинки: {BEST_VISUALIZATION_DIR}")


def main() -> None:
    """
    Короткое описание:
        Загружает данные/модели, запускает Optuna и сохраняет лучший результат.
    Вход:
        None
    Выход:
        None
    """
    global YOLO_MODEL, UNET_MODEL, UNET_DEVICE, DATASET_ROWS

    # Шаг 1: готовим выходную папку и данные.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if TRIALS_JSONL_PATH.exists():
        TRIALS_JSONL_PATH.unlink()
    DATASET_ROWS = build_dataset_rows()
    if len(DATASET_ROWS) == 0:
        raise RuntimeError("Не найдено ни одного изображения для оптимизации")

    # Шаг 2: загружаем модели один раз.
    from ultralytics import YOLO

    YOLO_MODEL = YOLO(str(louloudis.YOLO_PAGE_SEGMENTATION_MODEL_PATH))
    UNET_MODEL, UNET_DEVICE = load_unet_model(
        model_path=str(louloudis.UNET_BINARIZATION_MODEL_PATH),
        device=louloudis.UNET_DEVICE,
    )

    # Шаг 3: запускаем Optuna на N_TRIALS итераций.
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study_start_time = time.perf_counter()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    study_runtime_sec = time.perf_counter() - study_start_time

    # Шаг 4: сохраняем лучший результат и визуализации.
    evaluate_and_save_best(normalize_params(study.best_params), study.best_value, study_runtime_sec=study_runtime_sec)


if __name__ == "__main__":
    main()
