"""
Короткое описание:
    Подбирает три гиперпараметра Das/Panda HPP + seam carving line segmentation
    на первых 20 изображениях HWR200 test split.
Выход:
    trials.jsonl, best_summary.json и debug-картинки best-предсказаний.
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


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import das_panda_hpp_seam_exact as hpp_seam
from u_net_binarization import load_unet_model


SPLIT_PATH = PROJECT_ROOT / "handwriting-recognition" / "splits" / "test.txt"
HWR200_IMAGE_ROOT = PROJECT_ROOT / "datasets" / "HWR200" / "hw_dataset"
HWR200_LABELS_PATH = PROJECT_ROOT / "datasets" / "HWR200" / "labels.txt"

OUTPUT_DIR = PROJECT_ROOT / "debug_images" / "experiment_2_compare_paper_hpp" / "optuna_das_panda_hpp_seam"
TRIALS_JSONL_PATH = OUTPUT_DIR / "trials.jsonl"
BEST_SUMMARY_PATH = OUTPUT_DIR / "best_summary.json"
BEST_VISUALIZATION_DIR = OUTPUT_DIR / "best_predictions"

N_IMAGES = 20
N_TRIALS = 40
RANDOM_SEED = 42
IOU_THRESHOLD = 0.5

GT_COLOR_BGR = (0, 180, 0)
PRED_COLOR_BGR = (0, 0, 255)
PAGE_BBOX_COLOR_BGR = (255, 150, 0)
PANEL_BACKGROUND_BGR = (245, 245, 245)
PANEL_TEXT_BGR = (20, 20, 20)

YOLO_MODEL = None
UNET_MODEL = None
UNET_DEVICE = None
DATASET_ROWS: List[Dict[str, Any]] = []


def read_first_split_paths() -> List[str]:
    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        paths = [line.strip() for line in file if line.strip()]
    return paths[:N_IMAGES]


def load_hwr200_labels() -> Dict[str, List[np.ndarray]]:
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


def resolve_existing_image_path(relative_path: str) -> Optional[Path]:
    image_path = HWR200_IMAGE_ROOT / relative_path
    if image_path.exists():
        return image_path
    parent = image_path.parent
    if not parent.exists():
        return None
    stem = image_path.stem
    suffix = image_path.suffix.lower()
    for candidate in parent.iterdir():
        if not candidate.is_file():
            continue
        if candidate.name == image_path.name:
            return candidate
        if candidate.stem == stem and candidate.suffix.lower() in {suffix, ".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            return candidate
    return None


def build_dataset_rows() -> List[Dict[str, Any]]:
    split_paths = read_first_split_paths()
    labels = load_hwr200_labels()
    rows: List[Dict[str, Any]] = []
    for relative_path in split_paths:
        image_path = resolve_existing_image_path(relative_path)
        if image_path is None:
            print(f"[WARN] Нет изображения: {HWR200_IMAGE_ROOT / relative_path}")
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
    return {
        "HPP_ZSCORE_THRESHOLD": trial.suggest_float("HPP_ZSCORE_THRESHOLD", -1.0, 2.5),
        "REGION_MERGE_GAP_ROWS": trial.suggest_int("REGION_MERGE_GAP_ROWS", 0, 80),
        "MIN_TEXT_REGION_HEIGHT": trial.suggest_int("MIN_TEXT_REGION_HEIGHT", 1, 80),
    }


def apply_hpp_params(params: Dict[str, Any]) -> None:
    for name, value in params.items():
        setattr(hpp_seam, name, value)


def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "HPP_ZSCORE_THRESHOLD": getattr(hpp_seam, "HPP_ZSCORE_THRESHOLD"),
        "REGION_MERGE_GAP_ROWS": getattr(hpp_seam, "REGION_MERGE_GAP_ROWS"),
        "MIN_TEXT_REGION_HEIGHT": getattr(hpp_seam, "MIN_TEXT_REGION_HEIGHT"),
    }
    normalized.update(params)
    normalized["REGION_MERGE_GAP_ROWS"] = int(normalized["REGION_MERGE_GAP_ROWS"])
    normalized["MIN_TEXT_REGION_HEIGHT"] = int(normalized["MIN_TEXT_REGION_HEIGHT"])
    normalized["HPP_ZSCORE_THRESHOLD"] = float(normalized["HPP_ZSCORE_THRESHOLD"])
    return normalized


def class_matrix_to_polygons(
    class_matrix: np.ndarray,
    x_offset: int,
    y_offset: int,
) -> List[np.ndarray]:
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
    polygon = Polygon(np.asarray(points, dtype=np.float32).reshape(-1, 2))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def polygon_iou(predicted: np.ndarray, target: np.ndarray) -> float:
    pred_poly = safe_polygon(predicted)
    gt_poly = safe_polygon(target)
    if pred_poly.is_empty or gt_poly.is_empty:
        return 0.0
    intersection = pred_poly.intersection(gt_poly).area
    union = pred_poly.area + gt_poly.area - intersection
    return float(intersection / union) if union > 0 else 0.0


def match_polygons(predicted: List[np.ndarray], targets: List[np.ndarray]) -> Dict[str, float]:
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
    debug_enabled = debug_dir is not None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    class_matrix, lines, page_info = hpp_seam.run_on_image(
        str(row["image_path"]),
        debug=debug_enabled,
        debug_output_dir=str(debug_dir if debug_dir is not None else OUTPUT_DIR / "tmp_no_debug"),
        yolo_model=YOLO_MODEL,
        unet_model=UNET_MODEL,
        unet_device=UNET_DEVICE,
        return_page_info=True,
    )
    runtime_sec = time.perf_counter() - start_time

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
    trial_start_time = time.perf_counter()
    params = normalize_params(suggest_params(trial))
    apply_hpp_params(params)
    per_image_results = []
    for row in DATASET_ROWS:
        per_image_results.append(evaluate_one_image(row, params))
    metrics = aggregate_metrics(per_image_results)
    runtime = aggregate_runtime(per_image_results)
    trial_runtime_sec = time.perf_counter() - trial_start_time

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


def color_for_index(index: int, total: int, saturation: float = 0.85, value: float = 0.95) -> Tuple[int, int, int]:
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
    canvas = image.copy()
    total = len(polygons)
    for index, polygon in enumerate(polygons, start=1):
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
        if len(points) < 3:
            continue
        color = color_for_index(index - 1, total)
        cv2.polylines(canvas, [points], True, color, thickness)
        if label_prefix:
            x, y = points[0]
            cv2.putText(canvas, f"{label_prefix}{index}", (int(x), max(18, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return canvas


def polygons_to_colored_mask(shape: Tuple[int, int], polygons: List[np.ndarray]) -> np.ndarray:
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


def class_matrix_to_colored_original(
    image_shape: Tuple[int, int],
    class_matrix: np.ndarray,
    page_info: Dict[str, Any],
) -> np.ndarray:
    height, width = image_shape
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    max_class = int(np.max(class_matrix)) if class_matrix.size > 0 else 0
    if max_class == 0:
        return canvas
    colored_crop = np.full((*class_matrix.shape, 3), 255, dtype=np.uint8)
    for class_index in range(1, max_class + 1):
        color = color_for_index(class_index - 1, max_class)
        colored_crop[class_matrix == class_index] = color
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
    segmentation = class_matrix_to_colored_original(image.shape[:2], class_matrix, page_info)
    overlay = image.copy()
    foreground = np.any(segmentation != 255, axis=2)
    overlay[foreground] = cv2.addWeighted(image[foreground], 1.0 - alpha, segmentation[foreground], alpha, 0)
    return overlay


def resize_for_panel(image: np.ndarray, panel_size: Tuple[int, int]) -> np.ndarray:
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


def put_panel_title(image: np.ndarray, title: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 560), 42), (255, 255, 255), -1)
    cv2.putText(canvas, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, PANEL_TEXT_BGR, 2)
    return canvas


def make_detailed_debug_panel(
    image: np.ndarray,
    gt_polygons: List[np.ndarray],
    predicted_polygons: List[np.ndarray],
    page_info: Dict[str, Any],
) -> np.ndarray:
    original_panel = put_panel_title(image, "1. Original image")
    gt_mask = polygons_to_colored_mask(image.shape[:2], gt_polygons)
    gt_mask = draw_colored_polygons(gt_mask, gt_polygons, thickness=2, label_prefix="G")
    gt_mask = put_panel_title(gt_mask, "2. Ground truth mask")
    pred_mask = polygons_to_colored_mask(image.shape[:2], predicted_polygons)
    pred_mask = draw_colored_polygons(pred_mask, predicted_polygons, thickness=2, label_prefix="P")
    pred_mask = put_panel_title(pred_mask, "3. Predicted minRect mask")
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
    overlay = put_panel_title(overlay, "4. GT + prediction")
    panel_size = (900, 700)
    panels = [
        resize_for_panel(original_panel, panel_size),
        resize_for_panel(gt_mask, panel_size),
        resize_for_panel(pred_mask, panel_size),
        resize_for_panel(overlay, panel_size),
    ]
    return np.vstack([np.hstack([panels[0], panels[1]]), np.hstack([panels[2], panels[3]])])


def save_per_image_debug_folder(
    image_index: int,
    row: Dict[str, Any],
    result: Dict[str, Any],
    output_root: Path,
) -> Dict[str, str]:
    folder_name = f"{image_index:02d}_{row['relative_path'].replace('/', '__')}"
    folder_name = Path(folder_name).with_suffix("").name
    image_dir = output_root / folder_name
    image_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(row["image_path"])

    gt_mask = polygons_to_colored_mask(image.shape[:2], row["gt_polygons"])
    gt_mask = draw_colored_polygons(gt_mask, row["gt_polygons"], thickness=2, label_prefix="G")
    pred_mask = polygons_to_colored_mask(image.shape[:2], result["predicted_polygons"])
    pred_mask = draw_colored_polygons(pred_mask, result["predicted_polygons"], thickness=2, label_prefix="P")
    segmentation = class_matrix_to_colored_original(image.shape[:2], result["class_matrix"], result["page_info"])
    segmentation_overlay = class_matrix_overlay_original(image, result["class_matrix"], result["page_info"])
    overlay = draw_colored_polygons(image, row["gt_polygons"], thickness=3, label_prefix="G")
    overlay = draw_colored_polygons(overlay, result["predicted_polygons"], thickness=2, label_prefix="P")
    bbox = result["page_info"]["bbox"]
    cv2.rectangle(overlay, (int(bbox["x"]), int(bbox["y"])), (int(bbox["x"] + bbox["w"]), int(bbox["y"] + bbox["h"])), PAGE_BBOX_COLOR_BGR, 3)
    panel = make_detailed_debug_panel(image, row["gt_polygons"], result["predicted_polygons"], result["page_info"])

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
    cv2.imwrite(paths["segmentation_before_minrect"], segmentation)
    cv2.imwrite(paths["segmentation_before_minrect_overlay"], segmentation_overlay)
    cv2.imwrite(paths["gt_pred_rectangles"], overlay)
    cv2.imwrite(paths["four_panel"], panel)

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
    best_params = normalize_params(best_params)
    apply_hpp_params(best_params)
    BEST_VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    per_image_results = []
    visualization_files: Dict[str, Dict[str, str]] = {}
    for image_index, row in enumerate(tqdm(DATASET_ROWS, desc="Best predictions")):
        folder_name = f"{image_index:02d}_{row['relative_path'].replace('/', '__')}"
        folder_name = Path(folder_name).with_suffix("").name
        method_debug_dir = BEST_VISUALIZATION_DIR / folder_name / "method_debug"
        result = evaluate_one_image(row, best_params, debug_dir=method_debug_dir)
        per_image_results.append(result)
        visualization_files[row["relative_path"]] = save_per_image_debug_folder(image_index, row, result, BEST_VISUALIZATION_DIR)

    metrics = aggregate_metrics(per_image_results)
    runtime = aggregate_runtime(per_image_results)
    summary = {
        "optimized_metric": "hmean = 2 * precision * recall / (precision + recall)",
        "iou_threshold": IOU_THRESHOLD,
        "n_images": len(DATASET_ROWS),
        "n_trials": N_TRIALS,
        "study_runtime_sec": None if study_runtime_sec is None else float(study_runtime_sec),
        "best_runtime_recomputed": runtime,
        "runtime_scope": "run_on_image: YOLO page crop + U-Net binarization + Das/Panda HPP seam logic",
        "tuned_params": [
            "HPP_ZSCORE_THRESHOLD",
            "REGION_MERGE_GAP_ROWS",
            "MIN_TEXT_REGION_HEIGHT",
        ],
        "best_value_from_study": float(best_value),
        "best_metrics_recomputed": metrics,
        "best_params": best_params,
        "split_path": str(SPLIT_PATH),
        "labels_path": str(HWR200_LABELS_PATH),
        "yolo_model_path": str(hpp_seam.YOLO_PAGE_SEGMENTATION_MODEL_PATH),
        "unet_model_path": str(hpp_seam.UNET_BINARIZATION_MODEL_PATH),
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
    global YOLO_MODEL, UNET_MODEL, UNET_DEVICE, DATASET_ROWS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if TRIALS_JSONL_PATH.exists():
        TRIALS_JSONL_PATH.unlink()
    DATASET_ROWS = build_dataset_rows()
    if len(DATASET_ROWS) == 0:
        raise RuntimeError("Не найдено ни одного изображения для оптимизации")

    from ultralytics import YOLO

    YOLO_MODEL = YOLO(str(hpp_seam.YOLO_PAGE_SEGMENTATION_MODEL_PATH))
    UNET_MODEL, UNET_DEVICE = load_unet_model(
        model_path=str(hpp_seam.UNET_BINARIZATION_MODEL_PATH),
        device=hpp_seam.UNET_DEVICE,
    )

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study_start_time = time.perf_counter()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    study_runtime_sec = time.perf_counter() - study_start_time
    evaluate_and_save_best(normalize_params(study.best_params), study.best_value, study_runtime_sec=study_runtime_sec)


if __name__ == "__main__":
    main()
