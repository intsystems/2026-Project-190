"""
Короткое описание:
    Сравнивает das_panda_hpp_seam_exact и my_das_panda_hpp_seam_exact
    на test split HWR200.
Выход:
    comparison_summary.json и debug-визуализации в debug_images/experiment_2_compare_paper_hpp.
"""

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm

from post_processing import class_matrix_to_postprocessed_polygons, class_matrix_to_center_mass_cropped_polygons

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

SPLIT_PATH = PROJECT_ROOT / "handwriting-recognition" / "splits" / "my_split_with_only_test_train" / "test.txt"
HWR200_IMAGE_ROOT = PROJECT_ROOT / "datasets" / "HWR200" / "hw_dataset"
HWR200_LABELS_PATH = PROJECT_ROOT / "datasets" / "HWR200" / "labels_with_manual_test_and_paddle_ocr_train.txt"

NUM_POSTPROCESSING_METHODS = 3

BEST_SUMMARY_PATH = PROJECT_ROOT / "debug_images" / "experiment_2_compare_paper_hpp" / "optuna_das_panda_hpp_seam" / "best_summary.json"
OUTPUT_DIR = PROJECT_ROOT / "debug_images" / "experiment_2_hpp_methods" / "evaluate_hpp_methods"
OUTPUT_JSON_PATH = OUTPUT_DIR / "comparison_summary.json"
TARGET_VIS_DIR = OUTPUT_DIR / "targets"

EXACT_MODULE_PATH = EXPERIMENT_DIR / "das_panda_hpp_seam_exact.py"
MY_MODULE_PATH = EXPERIMENT_DIR / "my_das_panda_hpp_seam_exact.py"
MY_SMALL_MODULE_PATH = EXPERIMENT_DIR / "my_small_size_das_panda_hpp_seam_exact.py"

# None: весь test.txt. Число: первые N.
EVALUATE_N_IMAGES = None
IOU_THRESHOLD = 0.5
CLEAR_OUTPUT_DIR_ON_START = True
SKIPPED_INPUT_ROWS: List[Dict[str, str]] = []



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Das-Panda exact/my methods separately or together.")
    parser.add_argument(
        "--method",
        choices=["exact", "my", "my_small"],
        default="exact",
        help="exact: только das_panda_hpp_seam_exact; my: только my_; my_small: только my_small",
    )
    return parser.parse_args()


def output_json_path_for_method(method: str) -> Path:
    if method == "exact":
        return OUTPUT_DIR / "das_panda_hpp_seam_exact_summary.json"
    if method == "my":
        return OUTPUT_DIR / "my_das_panda_hpp_seam_exact_summary.json"
    if method == "my_small":
        return OUTPUT_DIR / "my_small_size_das_panda_hpp_seam_exact_summary.json"
    return OUTPUT_JSON_PATH


def load_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось загрузить модуль: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_split_paths() -> List[str]:
    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        paths = [line.strip() for line in file if line.strip()]
    if EVALUATE_N_IMAGES is None:
        return paths
    return paths[: int(EVALUATE_N_IMAGES)]


def safe_name(relative_path: str) -> str:
    return relative_path.replace("\\", "__").replace("/", "__").replace(" ", "_")


def resolve_existing_image_path(relative_path: str) -> Optional[Path]:
    original = HWR200_IMAGE_ROOT / relative_path
    if original.exists():
        return original
    if relative_path.lower().endswith(".png.jpg") or relative_path.lower().endswith(".jpg.jpg"):
        candidate = HWR200_IMAGE_ROOT / relative_path[:-4]
        if candidate.exists():
            return candidate
    parent = original.parent
    if not parent.exists():
        return None
    stem = original.stem
    for candidate in parent.iterdir():
        if candidate.is_file() and candidate.stem == stem and candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            return candidate
    return None


def load_labels() -> Dict[str, List[np.ndarray]]:
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


def build_rows() -> List[Dict[str, Any]]:
    global SKIPPED_INPUT_ROWS
    labels = load_labels()
    rows: List[Dict[str, Any]] = []
    SKIPPED_INPUT_ROWS = []
    for relative_path in read_split_paths():
        image_path = resolve_existing_image_path(relative_path)
        if image_path is None:
            SKIPPED_INPUT_ROWS.append(
                {
                    "relative_path": relative_path,
                    "reason": "image_path_not_found",
                    "expected_path": str(HWR200_IMAGE_ROOT / relative_path),
                }
            )
            continue
        rows.append(
            {
                "relative_path": relative_path,
                "image_path": image_path,
                "gt_polygons": labels.get(relative_path, []),
            }
        )
    return rows


def load_best_params() -> Dict[str, Any]:
    if not BEST_SUMMARY_PATH.exists():
        print(f"[WARN] Best params не найдены: {BEST_SUMMARY_PATH}")
        return {}
    summary = json.loads(BEST_SUMMARY_PATH.read_text(encoding="utf-8"))
    return dict(summary.get("best_params", {}))


def apply_params(module: Any, params: Dict[str, Any]) -> List[str]:
    applied = []

    def apply_to(target: Any, prefix: str = "") -> None:
        if target is None:
            return

        for name, value in params.items():
            if hasattr(target, name):
                setattr(target, name, value)
                applied_name = f"{prefix}{name}" if prefix else name
                if applied_name not in applied:
                    applied.append(applied_name)

    apply_to(module)

    nested_exact = getattr(module, "exact", None)
    apply_to(nested_exact, "exact.")

    nested_base_my = getattr(module, "base_my", None)
    apply_to(nested_base_my, "base_my.")

    nested_base_my_exact = getattr(nested_base_my, "exact", None)
    apply_to(nested_base_my_exact, "base_my.exact.")

    return applied

def class_matrix_to_polygons(class_matrix: np.ndarray, x_offset: int, y_offset: int) -> List[np.ndarray]:
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


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
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
            pairs.append({"pred_index": int(pred_index), "gt_index": int(best_index), "iou": float(best_iou)})
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


def draw_polygons(image: np.ndarray, polygons: List[np.ndarray], color: tuple, thickness: int = 2) -> np.ndarray:
    result = image.copy()
    for polygon in polygons:
        points = np.round(np.asarray(polygon, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(result, [points], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return result


def save_target_visualization(row: Dict[str, Any]) -> None:
    image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        return
    item_dir = TARGET_VIS_DIR / safe_name(row["relative_path"])
    item_dir.mkdir(parents=True, exist_ok=True)
    overlay = draw_polygons(image, row["gt_polygons"], color=(0, 200, 0), thickness=2)
    cv2.imwrite(str(item_dir / "00_original.jpg"), image)
    cv2.imwrite(str(item_dir / "01_target_gt_polygons.jpg"), overlay)
    with open(item_dir / "target_info.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "relative_path": row["relative_path"],
                "gt_count": int(len(row["gt_polygons"])),
                "image_path": str(row["image_path"]),
            },
            file,
            indent=2,
            ensure_ascii=False,
        )


def save_method_visuals(
    row: Dict[str, Any],
    method_name: str,
    predicted_polygons: List[np.ndarray],
    metrics: Dict[str, Any],
    i: int = 0
) -> None:
    image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        return
    method_dir = TARGET_VIS_DIR / safe_name(row["relative_path"]) / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    pred_canvas = np.full_like(image, 255)
    rng = np.random.default_rng(12345)
    colors = rng.integers(40, 230, size=(max(1, len(predicted_polygons)), 3), dtype=np.uint8)
    for index, polygon in enumerate(predicted_polygons):
        points = np.round(np.asarray(polygon, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(pred_canvas, [points], True, tuple(int(v) for v in colors[index % len(colors)]), 2, cv2.LINE_AA)
    cv2.imwrite(str(method_dir / f"00_final_class_matrix_method_{i}.png"), pred_canvas)

    matched_overlay = image.copy()
    for match in metrics.get("matches", []):
        pred_index = int(match["pred_index"])
        gt_index = int(match["gt_index"])
        if 0 <= pred_index < len(predicted_polygons):
            pred_points = np.round(np.asarray(predicted_polygons[pred_index], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(matched_overlay, [pred_points], True, (0, 200, 0), 2, cv2.LINE_AA)
        if 0 <= gt_index < len(row["gt_polygons"]):
            gt_points = np.round(np.asarray(row["gt_polygons"][gt_index], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(matched_overlay, [gt_points], True, (255, 140, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(method_dir / f"01_matches_tp_fp_fn_method_{i}.jpg"), matched_overlay)

    with open(method_dir / "result_info.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "relative_path": row["relative_path"],
                "method": method_name,
                "metrics": {key: value for key, value in metrics.items() if key not in {"matches", "pred_matched", "gt_matched"}},
                "matches": metrics.get("matches", []),
                "legend": {"matched_pred": "green", "matched_gt": "orange"},
            },
            file,
            indent=2,
            ensure_ascii=False,
        )



def evaluate_module(
    module: Any,
    rows: List[Dict[str, Any]],
    yolo_model: Any,
    unet_model: Any,
    unet_device: Any,
    method_name = "my_small_size_das_panda_hpp_seam_exact"
) -> Dict[str, Any]:
    per_image = []
    skipped_missing = 0
    skipped_runtime = 0

    for row in tqdm(rows, desc=f"Evaluate {method_name}"):
        if cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR) is None:
            skipped_missing += 1
            continue
        try:
            start_time = time.perf_counter()
            class_matrix, lines, page_info = module.run_on_image(
                str(row["image_path"]),
                debug=False,
                yolo_model=yolo_model,
                unet_model=unet_model,
                unet_device=unet_device,
                return_page_info=True,
            )
            runtime_sec = time.perf_counter() - start_time
        except Exception as error:
            skipped_runtime += 1
            error_metrics = {
                "tp": 0.0,
                "fp": 0.0,
                "fn": float(len(row["gt_polygons"])),
                "precision": 0.0,
                "recall": 0.0,
                "hmean": 0.0,
                "pred_count": 0.0,
                "gt_count": float(len(row["gt_polygons"])),
                "matches": [],
                "pred_matched": [],
                "gt_matched": [False] * len(row["gt_polygons"]),
            }

            per_image.append(
                {
                    "relative_path": row["relative_path"],
                    "error": repr(error),
                    "all_metrics": [dict(error_metrics) for _ in range(NUM_POSTPROCESSING_METHODS)],
                    "segment_method_time": None,
                    "post_processing_times": [None] * NUM_POSTPROCESSING_METHODS,
                    "method_timing_seconds": None,
                    "line_count": 0,
                    "page_bbox": None,
                    "page_confidence": None,
                }
            )
            continue

        bbox = page_info["bbox"]

        # пробуем разные мтеоды посто обработки
        predicted_method = []
        post_processing_times = []

        start_time = time.perf_counter()
        predicted = class_matrix_to_polygons(class_matrix, int(bbox["x"]), int(bbox["y"]))
        post_processing_times.append(time.perf_counter() - start_time)
        predicted_method.append(predicted)
        
        start_time = time.perf_counter()
        predicted = class_matrix_to_postprocessed_polygons(
            class_matrix=class_matrix,
            x_offset=int(bbox["x"]),
            y_offset=int(bbox["y"]),
            tall_factor=1.2,
        )
        post_processing_times.append(time.perf_counter() - start_time)
        predicted_method.append(predicted)

        start_time = time.perf_counter()
        predicted = class_matrix_to_center_mass_cropped_polygons(
            class_matrix=class_matrix,
            x_offset=int(bbox["x"]),
            y_offset=int(bbox["y"]),
            min_component_area=6,
        )
        post_processing_times.append(time.perf_counter() - start_time)
        predicted_method.append(predicted)

        # start_time = time.perf_counter()
        # predicted = class_matrix_to_pca_detection_boxes(
        #     class_matrix=class_matrix,
        #     x_offset=int(bbox["x"]),
        #     y_offset=int(bbox["y"]),
        #     min_component_area=6,
        #     min_points=10,
        # )
        # post_processing_times.append(time.perf_counter() - start_time)
        # predicted_method.append(predicted)

        # for i in [0.98]:
        #     start_time = time.perf_counter()
        #     predicted = class_matrix_to_top_polygons(
        #         class_matrix=class_matrix,
        #         x_offset=int(bbox["x"]),
        #         y_offset=int(bbox["y"]),
        #         min_component_area=3,
        #         keep_pixel_fraction=i,
        #     )
        #     post_processing_times.append(time.perf_counter() - start_time)
        #     predicted_method.append(predicted)

        # for i in [0.98]:
        #     start_time = time.perf_counter()

        #     predicted = class_matrix_to_pca_top_polygons(
        #         class_matrix=class_matrix,
        #         x_offset=int(bbox["x"]),
        #         y_offset=int(bbox["y"]),
        #         min_component_area=3,
        #         keep_pixel_fraction=i,
        #         min_points=10,
        #         min_height=2.0,
        #     )

        #     post_processing_times.append(time.perf_counter() - start_time)
        #     predicted_method.append(predicted)

        # start_time = time.perf_counter()

        # safe_name = row["relative_path"].replace("/", "_").replace("\\", "_")

        # predicted = class_matrix_to_morph_pca_polygons_debug(
        #     class_matrix=class_matrix,
        #     x_offset=int(bbox["x"]),
        #     y_offset=int(bbox["y"]),
        #     min_component_area=6,
        #     min_points=3,
        #     kernel_size=3,
        #     morph_iterations=2,
        #     min_height=2.0,
        #     debug_dir=OUTPUT_DIR / "morph_pca_debug",
        #     debug_name=safe_name,
        # )

        # post_processing_times.append(time.perf_counter() - start_time)
        # predicted_method.append(predicted)

        metrics = []
        for i in range(0, len(predicted_method)):
            metrics_i = match_polygons(predicted_method[i], row["gt_polygons"])
            save_method_visuals(row, method_name + f"_method_{i}", predicted_method[i], metrics_i, i = i)
            metrics.append(metrics_i)
        per_image.append(
            {
                "relative_path": row["relative_path"],
                "all_metrics": metrics,
                "segment_method_time": float(runtime_sec),
                "post_processing_times": [float(x) for x in post_processing_times],
                "method_timing_seconds": page_info.get("timing_seconds"),
                "line_count": int(len(lines)),
                "page_bbox": bbox,
                "page_confidence": page_info.get("confidence"),
            }
        )

    aggregates = []
    def aggregate(per_image: List[Dict[str, Any]], i : int) -> Dict[str, float]:
        tp = sum(item["all_metrics"][i]["tp"] for item in per_image)
        fp = sum(item["all_metrics"][i]["fp"] for item in per_image)
        fn = sum(item["all_metrics"][i]["fn"] for item in per_image)
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
    
    def aggregate_scalar_runtime(per_image, time_name: str) -> Dict[str, float]:
        runtimes = [
            float(item[time_name])
            for item in per_image
            if time_name in item and item[time_name] is not None
        ]
        return summarize_runtimes(runtimes, time_name)


    def aggregate_indexed_runtime(per_image, i: int, time_name: str) -> Dict[str, float]:
        runtimes = [
            float(item[time_name][i])
            for item in per_image
            if time_name in item
            and item[time_name] is not None
            and len(item[time_name]) > i
            and item[time_name][i] is not None
        ]
        return summarize_runtimes(runtimes, time_name)


    def summarize_runtimes(runtimes, time_name: str) -> Dict[str, float]:
        if not runtimes:
            return {
                "count": 0.0,
                f"total_{time_name}_sec": 0.0,
                f"mean_{time_name}_sec": 0.0,
                f"median_{time_name}_sec": 0.0,
                f"min_{time_name}_sec": 0.0,
                f"max_{time_name}_sec": 0.0,
            }

        values = np.asarray(runtimes, dtype=np.float64)
        return {
            "count": float(len(runtimes)),
            f"total_{time_name}_sec": float(np.sum(values)),
            f"mean_{time_name}_sec": float(np.mean(values)),
            f"median_{time_name}_sec": float(np.median(values)),
            f"min_{time_name}_sec": float(np.min(values)),
            f"max_{time_name}_sec": float(np.max(values)),
        }

    num_methods = NUM_POSTPROCESSING_METHODS # количесвто метовод
    for i in range(0, num_methods):
        aggregates.append(aggregate(per_image, i))

    post_processing_times = []
    for i in range(0, num_methods):
        post_processing_times.append(aggregate_indexed_runtime(per_image, i, time_name = "post_processing_times"))

    return {
        "method": method_name,
        "aggregates": aggregates,
        "segment_method_time": aggregate_scalar_runtime(per_image, time_name="segment_method_time"),
        "post_processing_times" : post_processing_times,
        "runtime_scope": f"module.run_on_image: YOLO page crop + U-Net binarization + {method_name} logic",
        "per_image": per_image,
        "skipped_missing": int(skipped_missing),
        "skipped_runtime": int(skipped_runtime),
    }


def main() -> None:
    args = parse_args()
    if CLEAR_OUTPUT_DIR_ON_START and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_VIS_DIR.mkdir(parents=True, exist_ok=True)

    exact_module = load_module("das_panda_hpp_exact_compare", EXACT_MODULE_PATH) if args.method in {"exact"} else None
    my_module = load_module("my_das_panda_hpp_compare", MY_MODULE_PATH) if args.method in {"my"} else None
    my_small_module = load_module("my_small_size_das_panda_hpp_seam_exact", MY_SMALL_MODULE_PATH) if args.method in {"my_small"} else None
    
    module = None
    module_name = None
    module_path = None
    if exact_module is not None:
        module = exact_module
        module_name = "das_panda_hpp_exact_compare"
        module_path = EXACT_MODULE_PATH
    if my_module is not None:
        module = my_module
        module_name = "my_das_panda_hpp_compare"
        module_path = MY_MODULE_PATH
    if my_small_module is not None:
        module = my_small_module
        module_name = "my_small_size_das_panda_hpp_seam_exact"
        module_path = MY_SMALL_MODULE_PATH

    if module is None:
        raise RuntimeError(f"Module was not loaded for method={args.method}")

    best_params = load_best_params()
    applied_params = apply_params(module, best_params) if module is not None else []
    print(f"[OK] Applied params: {applied_params}")
    print(f"[OK] Best params path: {BEST_SUMMARY_PATH}")
    print(f"[OK] Best params loaded: {best_params}")

    rows = build_rows()
    for row in tqdm(rows, desc="Save target GT overlays"):
        save_target_visualization(row)

    from ultralytics import YOLO
    from u_net_binarization import load_unet_model

    yolo_model = YOLO(str(module.YOLO_PAGE_SEGMENTATION_MODEL_PATH))
    unet_model, unet_device = load_unet_model(
        model_path=str(module.UNET_BINARIZATION_MODEL_PATH),
        device=module.UNET_DEVICE,
    )

    result = evaluate_module(module, rows, yolo_model, unet_model, unet_device, module_name)
    summary = {
        "evaluate_n_images": EVALUATE_N_IMAGES,
        "rows_count": len(rows),
        "skipped_input_rows": SKIPPED_INPUT_ROWS,
        "iou_threshold": IOU_THRESHOLD,
        "module_path": str(module_path),
        "applied_params": applied_params,
        "best_params": best_params,
        "result": result,
    }
    output_json_path = output_json_path_for_method(args.method)

    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"[OK] Summary: {output_json_path}")

    aggregates = result["aggregates"]
    post_processing_times = result["post_processing_times"]
    segment_method_time = result["segment_method_time"]
    for i in range(0, len(aggregates)):
        print()
        print(f"Method_{i}:")
        print(
            f"[OK] {module_name} method_{i} hmean: "
            f"{aggregates[i]['hmean']:.6f}, precision: {aggregates[i]['precision']:.6f}, recall: {aggregates[i]['recall']:.6f}"
        )
        print(f"[OK] TP/FP/FN: {aggregates[i]['tp']:.0f}/{aggregates[i]['fp']:.0f}/{aggregates[i]['fn']:.0f}")
        print(f"[OK] Post processing times mean/median: {post_processing_times[i]['mean_post_processing_times_sec']:.3f}s / {post_processing_times[i]['median_post_processing_times_sec']:.3f}s")
        print()

    print("Time segment block")
    print(f"[OK] Runtime mean/median: {segment_method_time['mean_segment_method_time_sec']:.3f}s / {segment_method_time['median_segment_method_time_sec']:.3f}s")
    print(f"[OK] Summary: {output_json_path}")
    print(f"[OK] Targets/debug: {TARGET_VIS_DIR}")



if __name__ == "__main__":
    main()
