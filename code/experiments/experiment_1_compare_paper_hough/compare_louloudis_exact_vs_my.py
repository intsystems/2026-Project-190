"""
Короткое описание:
    Сравнивает louloudis_text_line_detection_exact и my_louloudis_text_line_detection_exact
    на первых изображениях test split с лучшими гиперпараметрами experiment_1.
Вход:
    Константы путей и количества изображений заданы в начале файла.
Выход:
    JSON-отчет и подробный консольный вывод о том, какой метод лучше по hmean.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPLIT_PATH = PROJECT_ROOT / "handwriting-recognition" / "splits" / "my_split_with_manual_train" / "test.txt"
HWR200_IMAGE_ROOT = PROJECT_ROOT / "datasets" / "HWR200" / "hw_dataset"
HWR200_LABELS_PATH = PROJECT_ROOT / "datasets" / "HWR200" / "labels_DBNet++_sync_test_manual.txt"
BEST_SUMMARY_PATH = PROJECT_ROOT / "debug_images" / "experiment_1_compare_paper_hough" / "optuna_louloudis_yolo_unet" / "best_summary.json"
OUTPUT_DIR = PROJECT_ROOT / "debug_images" / "experiment_1_compare_paper_hough" / "louloudis_exact"
OUTPUT_JSON_PATH = OUTPUT_DIR / "comparison_summary.json"

EXACT_MODULE_PATH = EXPERIMENT_DIR / "louloudis_text_line_detection_exact.py"
MY_MODULE_PATH = EXPERIMENT_DIR / "my_louloudis_text_line_detection_exact.py"

# Количество изображений для сравнения:
# - None: использовать весь split test.txt
# - int: использовать первые N изображений
COMPARE_N_IMAGES = None
IOU_THRESHOLD = 0.5
TARGET_VIS_DIR = OUTPUT_DIR / "targets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Louloudis exact/my methods separately or together.")
    parser.add_argument(
        "--method",
        choices=["exact", "my", "both"],
        default="both",
        help="exact: только louloudis_text_line_detection_exact; my: только my_; both: старое сравнение.",
    )
    return parser.parse_args()


def output_json_path_for_method(method: str) -> Path:
    if method == "exact":
        return OUTPUT_DIR / "louloudis_text_line_detection_exact_summary.json"
    if method == "my":
        return OUTPUT_DIR / "my_louloudis_text_line_detection_exact_summary.json"
    return OUTPUT_JSON_PATH


def load_module(module_name: str, module_path: Path) -> Any:
    """
    Короткое описание:
        Загружает python-файл как отдельный модуль.
    Вход:
        module_name: str -- имя модуля.
        module_path: Path -- путь к .py файлу.
    Выход:
        Any -- загруженный модуль.
    """
    # Шаг 1: загружаем модуль без конфликта имен.
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось загрузить модуль: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_split_paths() -> List[str]:
    """
    Короткое описание:
        Читает пути из split с учетом COMPARE_N_IMAGES.
    Вход:
        None
    Выход:
        List[str] -- относительные пути.
    """
    # Шаг 1: читаем непустые строки.
    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        paths = [line.strip() for line in file if line.strip()]
    if COMPARE_N_IMAGES is None:
        return paths
    return paths[: int(COMPARE_N_IMAGES)]


def safe_name(relative_path: str) -> str:
    """
    Короткое описание:
        Строит безопасное имя папки/файла из относительного пути изображения.
    Вход:
        relative_path: str -- относительный путь из split.
    Выход:
        str -- безопасное имя.
    """
    return (
        relative_path
        .replace("\\", "__")
        .replace("/", "__")
        .replace(" ", "_")
    )


def resolve_existing_image_path(relative_path: str) -> Path:
    """
    Короткое описание:
        Возвращает существующий путь изображения, если исходный путь битый.
    Вход:
        relative_path: str -- путь из split.
    Выход:
        Path -- существующий путь; если не найден, возвращается исходный.
    """
    original = HWR200_IMAGE_ROOT / relative_path
    if original.exists():
        return original

    # Частый кейс в split: *.png.jpg. Пробуем убрать последний суффикс.
    if relative_path.lower().endswith(".png.jpg"):
        candidate = HWR200_IMAGE_ROOT / relative_path[:-4]
        if candidate.exists():
            return candidate
    if relative_path.lower().endswith(".jpg.jpg"):
        candidate = HWR200_IMAGE_ROOT / relative_path[:-4]
        if candidate.exists():
            return candidate

    return original


def draw_polygons(image: np.ndarray, polygons: List[np.ndarray], color: tuple, thickness: int = 2) -> np.ndarray:
    """
    Короткое описание:
        Рисует полигоны на копии изображения.
    Вход:
        image: np.ndarray -- исходное изображение.
        polygons: List[np.ndarray] -- полигоны.
        color: tuple -- BGR цвет.
        thickness: int -- толщина линии.
    Выход:
        np.ndarray -- изображение с полигонами.
    """
    result = image.copy()
    for polygon in polygons:
        points = np.round(np.asarray(polygon, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(result, [points], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return result


def save_target_visualization(row: Dict[str, Any]) -> None:
    """
    Короткое описание:
        Сохраняет визуализацию таргета из labels: GT-полигоны на исходном изображении.
    Вход:
        row: Dict[str, Any] -- данные одного изображения.
    Выход:
        None
    """
    image_path = Path(row["image_path"])
    if not image_path.exists():
        return
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    base = safe_name(row["relative_path"])
    item_dir = TARGET_VIS_DIR / base
    item_dir.mkdir(parents=True, exist_ok=True)

    overlay = draw_polygons(image, row["gt_polygons"], color=(0, 200, 0), thickness=2)
    cv2.imwrite(str(item_dir / "00_original.jpg"), image)
    cv2.imwrite(str(item_dir / "01_target_gt_polygons.jpg"), overlay)

    info = {
        "relative_path": row["relative_path"],
        "gt_count": int(len(row["gt_polygons"])),
        "image_path": str(row["image_path"]),
    }
    with open(item_dir / "target_info.json", "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2, ensure_ascii=False)


def load_labels() -> Dict[str, List[np.ndarray]]:
    """
    Короткое описание:
        Загружает GT-полигоны HWR200.
    Вход:
        None
    Выход:
        Dict[str, List[np.ndarray]] -- path -> polygons.
    """
    # Шаг 1: парсим labels.txt.
    labels: Dict[str, List[np.ndarray]] = {}
    with open(HWR200_LABELS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue
            relative_path, payload = line.split("\t", 1)
            objects = json.loads(payload)
            labels[relative_path] = [
                np.asarray(item["points"], dtype=np.float32).reshape(-1, 2)
                for item in objects
                if len(item.get("points", [])) >= 3
            ]
    return labels


def load_best_params() -> Dict[str, Any]:
    """
    Короткое описание:
        Загружает лучшие параметры Optuna из best_summary.json.
    Вход:
        None
    Выход:
        Dict[str, Any] -- параметры.
    """
    # Шаг 1: читаем best_summary, если он есть.
    if not BEST_SUMMARY_PATH.exists():
        return {}
    summary = json.loads(BEST_SUMMARY_PATH.read_text(encoding="utf-8"))
    params = dict(summary.get("best_params", {}))
    # Удаляем legacy-параметры shrink/swell, они больше не используются.
    for key in list(params.keys()):
        if key.startswith("POSTPROCESS_"):
            params.pop(key, None)
    return params


def apply_params(module: Any, params: Dict[str, Any]) -> None:
    """
    Короткое описание:
        Применяет параметры к модулю метода.
    Вход:
        module: Any -- модуль метода.
        params: Dict[str, Any] -- параметры.
    Выход:
        None
    """
    # Шаг 1: переносим только те параметры, которые реально есть в модуле.
    for name, value in params.items():
        if hasattr(module, name):
            setattr(module, name, value)


def class_matrix_to_polygons(class_matrix: np.ndarray, x_offset: int, y_offset: int) -> List[np.ndarray]:
    """
    Короткое описание:
        Переводит class_matrix в полигоны по контурам масок классов.
    Вход:
        class_matrix: np.ndarray -- class matrix, 0 фон.
        x_offset: int -- смещение crop страницы по X.
        y_offset: int -- смещение crop страницы по Y.
    Выход:
        List[np.ndarray] -- полигоны в координатах исходного изображения.
    """
    # Шаг 1: каждый класс строки превращаем в ориентированный minAreaRect, как раньше.
    polygons = []
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


def render_class_matrix(class_matrix: np.ndarray) -> np.ndarray:
    """
    Короткое описание:
        Визуализирует class-matrix цветами по ID строки.
    Вход:
        class_matrix: np.ndarray -- матрица классов, 0 фон.
    Выход:
        np.ndarray -- BGR визуализация.
    """
    height, width = class_matrix.shape[:2]
    result = np.full((height, width, 3), 255, dtype=np.uint8)
    max_class = int(np.max(class_matrix)) if class_matrix.size > 0 else 0
    if max_class <= 0:
        return result
    rng = np.random.default_rng(12345)
    colors = rng.integers(40, 230, size=(max_class + 1, 3), dtype=np.uint8)
    for class_index in range(1, max_class + 1):
        result[class_matrix == class_index] = colors[class_index]
    return result


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    """
    Короткое описание:
        Считает IoU двух полигонов.
    Вход:
        first: np.ndarray -- первый полигон.
        second: np.ndarray -- второй полигон.
    Выход:
        float -- IoU.
    """
    # Шаг 1: создаем валидные polygon и считаем intersection/union.
    first_poly = Polygon(np.asarray(first, dtype=np.float32).reshape(-1, 2))
    second_poly = Polygon(np.asarray(second, dtype=np.float32).reshape(-1, 2))
    if not first_poly.is_valid:
        first_poly = first_poly.buffer(0)
    if not second_poly.is_valid:
        second_poly = second_poly.buffer(0)
    if first_poly.is_empty or second_poly.is_empty:
        return 0.0
    intersection = first_poly.intersection(second_poly).area
    union = first_poly.area + second_poly.area - intersection
    return float(intersection / union) if union > 0 else 0.0


def match_polygons(predicted: List[np.ndarray], target: List[np.ndarray]) -> Dict[str, Any]:
    """
    Короткое описание:
        Greedy IoU matching при IOU_THRESHOLD.
    Вход:
        predicted: List[np.ndarray] -- предсказанные полигоны.
        target: List[np.ndarray] -- GT полигоны.
    Выход:
        Dict[str, float] -- tp/fp/fn/precision/recall/hmean.
    """
    # Шаг 1: сопоставляем каждое предсказание с лучшим свободным GT.
    matched = [False] * len(target)
    matched_pred = [False] * len(predicted)
    pairs: List[Dict[str, Any]] = []
    tp = 0
    fp = 0
    for pred_index, pred in enumerate(predicted):
        best_iou = 0.0
        best_index = -1
        for index, gt in enumerate(target):
            if matched[index]:
                continue
            iou = polygon_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_iou >= IOU_THRESHOLD and best_index >= 0:
            matched[best_index] = True
            matched_pred[pred_index] = True
            pairs.append(
                {
                    "pred_index": int(pred_index),
                    "gt_index": int(best_index),
                    "iou": float(best_iou),
                }
            )
            tp += 1
        else:
            fp += 1
    fn = matched.count(False)
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
        "pred_count": float(len(predicted)),
        "gt_count": float(len(target)),
        "matches": pairs,
        "pred_matched": [bool(value) for value in matched_pred],
        "gt_matched": [bool(value) for value in matched],
    }


def save_method_target_visuals(
    row: Dict[str, Any],
    method_name: str,
    class_matrix: np.ndarray,
    predicted_polygons: List[np.ndarray],
    metrics: Dict[str, Any],
) -> None:
    """
    Короткое описание:
        Сохраняет визуализацию результатов метода: final_class_matrix и TP/FP/FN overlay.
    Вход:
        row: Dict[str, Any] -- данные изображения.
        method_name: str -- имя метода.
        class_matrix: np.ndarray -- class-matrix.
        predicted_polygons: List[np.ndarray] -- предсказанные полигоны.
        metrics: Dict[str, Any] -- детальные метрики с matched-флагами.
    Выход:
        None
    """
    image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        return
    base = safe_name(row["relative_path"])
    method_dir = TARGET_VIS_DIR / base / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    # Визуализация final_class_matrix через реально построенные полигоны предсказания.
    final_polygons_vis = np.full_like(image, 255)
    rng = np.random.default_rng(12345)
    colors = rng.integers(40, 230, size=(max(1, len(predicted_polygons)), 3), dtype=np.uint8)
    for index, polygon in enumerate(predicted_polygons):
        points = np.round(np.asarray(polygon, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        color = tuple(int(value) for value in colors[index % len(colors)])
        cv2.polylines(final_polygons_vis, [points], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(str(method_dir / "00_final_class_matrix.png"), final_polygons_vis)

    # Показываем только то, что реально сматчилось: пары GT<->Prediction.
    overlay = image.copy()
    gt_polygons = row["gt_polygons"]
    for match in metrics.get("matches", []):
        pred_index = int(match["pred_index"])
        gt_index = int(match["gt_index"])
        if 0 <= pred_index < len(predicted_polygons):
            pred_points = np.round(np.asarray(predicted_polygons[pred_index], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pred_points], isClosed=True, color=(0, 200, 0), thickness=2, lineType=cv2.LINE_AA)
        if 0 <= gt_index < len(gt_polygons):
            gt_points = np.round(np.asarray(gt_polygons[gt_index], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [gt_points], isClosed=True, color=(255, 140, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.imwrite(str(method_dir / "01_matches_tp_fp_fn.jpg"), overlay)
    info = {
        "relative_path": row["relative_path"],
        "method": method_name,
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "hmean": metrics["hmean"],
        "legend": {
            "matched_pred": "green",
            "matched_gt": "orange",
        },
    }
    with open(method_dir / "result_info.json", "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2, ensure_ascii=False)


def aggregate(per_image: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Короткое описание:
        Агрегирует метрики по всем изображениям.
    Вход:
        per_image: List[Dict[str, Any]] -- результаты по изображениям.
    Выход:
        Dict[str, float] -- global precision/recall/hmean.
    """
    # Шаг 1: суммируем счетчики.
    tp = sum(item["metrics"]["tp"] for item in per_image)
    fp = sum(item["metrics"]["fp"] for item in per_image)
    fn = sum(item["metrics"]["fn"] for item in per_image)
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


def aggregate_runtime(per_image: List[Dict[str, Any]]) -> Dict[str, float]:
    runtimes = [float(item["runtime_sec"]) for item in per_image if item.get("runtime_sec") is not None]
    if not runtimes:
        return {
            "count": 0.0,
            "total_runtime_sec": 0.0,
            "mean_runtime_sec": 0.0,
            "median_runtime_sec": 0.0,
            "min_runtime_sec": 0.0,
            "max_runtime_sec": 0.0,
        }
    values = np.asarray(runtimes, dtype=np.float64)
    return {
        "count": float(len(runtimes)),
        "total_runtime_sec": float(np.sum(values)),
        "mean_runtime_sec": float(np.mean(values)),
        "median_runtime_sec": float(np.median(values)),
        "min_runtime_sec": float(np.min(values)),
        "max_runtime_sec": float(np.max(values)),
    }


def evaluate_module(
    module: Any,
    rows: List[Dict[str, Any]],
    yolo_model: Any,
    unet_model: Any,
    unet_device: Any,
    method_name: str,
) -> Dict[str, Any]:
    """
    Короткое описание:
        Оценивает один модуль метода на наборе изображений.
    Вход:
        module: Any -- модуль метода.
        rows: List[Dict[str, Any]] -- строки датасета.
        yolo_model: Any -- загруженная YOLO.
        unet_model: Any -- загруженная U-Net.
        unet_device: Any -- устройство U-Net.
    Выход:
        Dict[str, Any] -- aggregate и per-image результаты.
    """
    # Шаг 1: прогоняем изображения.
    per_image = []
    skipped_missing = 0
    skipped_runtime = 0
    for row in tqdm(rows, desc=f"Evaluate {module.__name__}"):
        image_path = Path(row["image_path"])
        if (not image_path.exists()) or (cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None):
            skipped_missing += 1
            continue
        try:
            start_time = time.perf_counter()
            class_matrix, lines, page_info = module.run_on_image(
                str(image_path),
                debug=False,
                yolo_model=yolo_model,
                unet_model=unet_model,
                unet_device=unet_device,
                return_page_info=True,
                use_tqdm=False,
            )
            runtime_sec = time.perf_counter() - start_time
        except Exception:
            skipped_runtime += 1
            continue
        bbox = page_info["bbox"]
        predicted = class_matrix_to_polygons(class_matrix, int(bbox["x"]), int(bbox["y"]))
        metrics = match_polygons(predicted, row["gt_polygons"])
        save_method_target_visuals(row, method_name, class_matrix, predicted, metrics)
        per_image.append(
            {
                "relative_path": row["relative_path"],
                "metrics": metrics,
                "runtime_sec": float(runtime_sec),
                "line_count": int(len(lines)),
                "page_bbox": bbox,
                "page_confidence": page_info.get("confidence"),
            }
        )
    return {
        "aggregate": aggregate(per_image),
        "runtime": aggregate_runtime(per_image),
        "runtime_scope": "module.run_on_image: YOLO page crop + U-Net binarization + method logic",
        "per_image": per_image,
        "skipped_missing_or_unreadable": int(skipped_missing),
        "skipped_runtime_errors": int(skipped_runtime),
    }


def main() -> None:
    """
    Короткое описание:
        Точка входа сравнения двух методов.
    Вход:
        None
    Выход:
        None
    """
    args = parse_args()

    # Шаг 1: грузим модули, параметры и данные.
    exact_module = load_module("louloudis_exact_compare", EXACT_MODULE_PATH) if args.method in {"exact", "both"} else None
    my_module = load_module("my_louloudis_compare", MY_MODULE_PATH) if args.method in {"my", "both"} else None
    best_params = load_best_params()
    if exact_module is not None:
        apply_params(exact_module, best_params)
    if my_module is not None:
        apply_params(my_module, best_params)

    labels = load_labels()
    rows = []
    split_paths = read_split_paths()
    for relative_path in split_paths:
        image_path = resolve_existing_image_path(relative_path)
        rows.append(
            {
                "relative_path": relative_path,
                "image_path": image_path,
                "gt_polygons": labels.get(relative_path, []),
            }
        )

    # Шаг 1.1: сохраняем target-визуализации, чтобы было видно ожидание разметки labels.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_VIS_DIR.mkdir(parents=True, exist_ok=True)
    for row in tqdm(rows, desc="Save target GT overlays"):
        save_target_visualization(row)

    # Шаг 2: грузим внешние модели один раз.
    from ultralytics import YOLO
    from u_net_binarization import load_unet_model

    reference_module = exact_module if exact_module is not None else my_module
    yolo_model = YOLO(str(reference_module.YOLO_PAGE_SEGMENTATION_MODEL_PATH))
    unet_model, unet_device = load_unet_model(str(reference_module.UNET_BINARIZATION_MODEL_PATH), device=reference_module.UNET_DEVICE)

    # Шаг 3: оцениваем выбранные методы.
    results: Dict[str, Any] = {}
    if exact_module is not None:
        results["louloudis_text_line_detection_exact"] = evaluate_module(
            exact_module,
            rows,
            yolo_model,
            unet_model,
            unet_device,
            method_name="louloudis_text_line_detection_exact",
        )
    if my_module is not None:
        results["my_louloudis_text_line_detection_exact"] = evaluate_module(
            my_module,
            rows,
            yolo_model,
            unet_model,
            unet_device,
            method_name="my_louloudis_text_line_detection_exact",
        )

    # Шаг 4: выбираем победителя только если оценивали оба.
    winner = None
    if len(results) == 2:
        exact_hmean = results["louloudis_text_line_detection_exact"]["aggregate"]["hmean"]
        my_hmean = results["my_louloudis_text_line_detection_exact"]["aggregate"]["hmean"]
        winner = "my_louloudis_text_line_detection_exact" if my_hmean > exact_hmean else "louloudis_text_line_detection_exact"
    report = {
        "metric": f"polygon IoU greedy matching, threshold={IOU_THRESHOLD}",
        "method_mode": args.method,
        "n_images": len(rows),
        "best_params": best_params,
        "winner": winner,
        "results": results,
    }

    # Шаг 5: сохраняем и печатаем подробный результат.
    output_json_path = output_json_path_for_method(args.method)
    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("[OK] Оценка завершена")
    for method_name, result in results.items():
        aggregate = result["aggregate"]
        runtime = result["runtime"]
        print(f"[OK] {method_name}: hmean={aggregate['hmean']:.6f}, precision={aggregate['precision']:.6f}, recall={aggregate['recall']:.6f}")
        print(f"[OK] {method_name}: runtime mean/median={runtime['mean_runtime_sec']:.3f}s / {runtime['median_runtime_sec']:.3f}s")
    if winner is not None:
        print(f"[OK] Winner: {winner}")
    print(f"[OK] Report: {output_json_path}")
    print(f"[OK] Target overlays: {TARGET_VIS_DIR}")


if __name__ == "__main__":
    main()
