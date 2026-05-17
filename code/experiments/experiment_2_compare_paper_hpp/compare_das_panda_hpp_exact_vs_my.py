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


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

SPLIT_PATH = PROJECT_ROOT / "handwriting-recognition" / "splits/my_split_with_manual_train" / "test.txt"
HWR200_IMAGE_ROOT = PROJECT_ROOT / "datasets" / "HWR200" / "hw_dataset"
HWR200_LABELS_PATH = PROJECT_ROOT / "datasets" / "HWR200" / "labels.txt"

BEST_SUMMARY_PATH = PROJECT_ROOT / "debug_images" / "experiment_2_compare_paper_hpp" / "optuna_das_panda_hpp_seam" / "best_summary.json"
OUTPUT_DIR = PROJECT_ROOT / "debug_images" / "experiment_2_compare_paper_hpp" / "compare_das_panda_hpp_exact_vs_my"
OUTPUT_JSON_PATH = OUTPUT_DIR / "comparison_summary.json"
TARGET_VIS_DIR = OUTPUT_DIR / "targets"

EXACT_MODULE_PATH = EXPERIMENT_DIR / "das_panda_hpp_seam_exact.py"
MY_MODULE_PATH = EXPERIMENT_DIR / "my_das_panda_hpp_seam_exact.py"

# None: весь test.txt. Число: первые N.
COMPARE_N_IMAGES = None
IOU_THRESHOLD = 0.5
CLEAR_OUTPUT_DIR_ON_START = True
SKIPPED_INPUT_ROWS: List[Dict[str, str]] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Das-Panda exact/my methods separately or together.")
    parser.add_argument(
        "--method",
        choices=["exact", "my", "both"],
        default="both",
        help="exact: только das_panda_hpp_seam_exact; my: только my_; both: старое сравнение.",
    )
    return parser.parse_args()


def output_json_path_for_method(method: str) -> Path:
    if method == "exact":
        return OUTPUT_DIR / "das_panda_hpp_seam_exact_summary.json"
    if method == "my":
        return OUTPUT_DIR / "my_das_panda_hpp_seam_exact_summary.json"
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
    if COMPARE_N_IMAGES is None:
        return paths
    return paths[: int(COMPARE_N_IMAGES)]


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
    module_file = Path(getattr(module, "__file__", ""))
    is_my_hpp = module_file.name == "my_das_panda_hpp_seam_exact.py"
    for name, value in params.items():
        if is_my_hpp and name in {"HPP_ZSCORE_THRESHOLD", "REGION_MERGE_GAP_ROWS", "MIN_TEXT_REGION_HEIGHT"}:
            continue
        if hasattr(module, name):
            setattr(module, name, value)
            applied.append(name)
    # my_das_panda_hpp_seam_exact импортирует exact-модуль внутрь себя.
    # Пробрасываем параметры туда тоже, чтобы summary/debug и вспомогательные
    # функции видели тот же набор best-параметров.
    nested_exact = getattr(module, "exact", None)
    if nested_exact is not None:
        for name, value in params.items():
            if is_my_hpp and name in {"HPP_ZSCORE_THRESHOLD", "REGION_MERGE_GAP_ROWS", "MIN_TEXT_REGION_HEIGHT"}:
                continue
            if hasattr(nested_exact, name):
                setattr(nested_exact, name, value)
                nested_name = f"exact.{name}"
                if nested_name not in applied:
                    applied.append(nested_name)
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
    cv2.imwrite(str(method_dir / "00_final_class_matrix.png"), pred_canvas)

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
    cv2.imwrite(str(method_dir / "01_matches_tp_fp_fn.jpg"), matched_overlay)

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


def aggregate(per_image: List[Dict[str, Any]]) -> Dict[str, float]:
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
    runtimes = [
        float(item["runtime_sec"])
        for item in per_image
        if "runtime_sec" in item and item["runtime_sec"] is not None
    ]
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
            per_image.append(
                {
                    "relative_path": row["relative_path"],
                    "error": repr(error),
                    "runtime_sec": None,
                    "metrics": {
                        "tp": 0.0,
                        "fp": 0.0,
                        "fn": float(len(row["gt_polygons"])),
                        "precision": 0.0,
                        "recall": 0.0,
                        "hmean": 0.0,
                    },
                    "line_count": 0,
                }
            )
            continue
        bbox = page_info["bbox"]
        predicted = class_matrix_to_polygons(class_matrix, int(bbox["x"]), int(bbox["y"]))
        metrics = match_polygons(predicted, row["gt_polygons"])
        save_method_visuals(row, method_name, predicted, metrics)
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
        "method": method_name,
        "aggregate": aggregate(per_image),
        "runtime": aggregate_runtime(per_image),
        "runtime_scope": "module.run_on_image: YOLO page crop + U-Net binarization + method logic",
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

    exact_module = load_module("das_panda_hpp_exact_compare", EXACT_MODULE_PATH) if args.method in {"exact", "both"} else None
    my_module = load_module("my_das_panda_hpp_compare", MY_MODULE_PATH) if args.method in {"my", "both"} else None
    best_params = load_best_params()
    exact_applied_params = apply_params(exact_module, best_params) if exact_module is not None else {}
    my_applied_params = apply_params(my_module, best_params) if my_module is not None else {}
    print(f"[OK] Best params path: {BEST_SUMMARY_PATH}")
    print(f"[OK] Best params loaded: {best_params}")
    print(f"[OK] Applied to exact: {exact_applied_params}")
    print(f"[OK] Applied to my: {my_applied_params}")

    rows = build_rows()
    for row in tqdm(rows, desc="Save target GT overlays"):
        save_target_visualization(row)

    from ultralytics import YOLO
    from u_net_binarization import load_unet_model

    reference_module = exact_module if exact_module is not None else my_module
    yolo_model = YOLO(str(reference_module.YOLO_PAGE_SEGMENTATION_MODEL_PATH))
    unet_model, unet_device = load_unet_model(
        model_path=str(reference_module.UNET_BINARIZATION_MODEL_PATH),
        device=reference_module.UNET_DEVICE,
    )

    results: Dict[str, Any] = {}
    if exact_module is not None:
        results["das_panda_hpp_seam_exact"] = evaluate_module(exact_module, rows, yolo_model, unet_model, unet_device, "das_panda_hpp_seam_exact")
    if my_module is not None:
        results["my_das_panda_hpp_seam_exact"] = evaluate_module(my_module, rows, yolo_model, unet_model, unet_device, "my_das_panda_hpp_seam_exact")

    winner = None
    if len(results) == 2:
        exact_h = results["das_panda_hpp_seam_exact"]["aggregate"]["hmean"]
        my_h = results["my_das_panda_hpp_seam_exact"]["aggregate"]["hmean"]
        winner = "my_das_panda_hpp_seam_exact" if my_h > exact_h else "das_panda_hpp_seam_exact"
    summary = {
        "method_mode": args.method,
        "compare_n_images": COMPARE_N_IMAGES,
        "rows_count": len(rows),
        "skipped_input_rows": SKIPPED_INPUT_ROWS,
        "iou_threshold": IOU_THRESHOLD,
        "best_summary_path": str(BEST_SUMMARY_PATH),
        "best_params_applied": best_params,
        "best_params_applied_to_exact": exact_applied_params,
        "best_params_applied_to_my": my_applied_params,
        "winner": winner,
        "results": results,
    }
    output_json_path = output_json_path_for_method(args.method)
    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    for method_name, result in results.items():
        aggregate = result["aggregate"]
        runtime = result["runtime"]
        print(f"[OK] {method_name}: hmean={aggregate['hmean']:.6f}, precision={aggregate['precision']:.6f}, recall={aggregate['recall']:.6f}")
        print(f"[OK] {method_name}: runtime mean/median={runtime['mean_runtime_sec']:.3f}s / {runtime['median_runtime_sec']:.3f}s")
    if winner is not None:
        print(f"[OK] Winner: {winner}")
    print(f"[OK] Summary: {output_json_path}")
    print(f"[OK] Targets/debug: {TARGET_VIS_DIR}")


if __name__ == "__main__":
    main()
