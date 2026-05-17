"""
Короткое описание:
    Запускает DBNet++ на HWR200 test split и считает те же polygon IoU метрики:
    precision, recall, hmean при IoU >= 0.5.
Выход:
    debug_images/experiment_3_test_DBNet++/dbnetpp_test/summary.json
    и debug-визуализации для первых DEBUG_FIRST_IMAGES.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from shapely.geometry import Polygon
from tqdm import tqdm


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
HWR_REPO_ROOT = PROJECT_ROOT / "handwriting-recognition"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HWR_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(HWR_REPO_ROOT))

from src.model import build_model
from src.postprocess import PostprocessConfig, decode_prob_map
from src.utils import preprocess_image_pil


SPLIT_PATH = PROJECT_ROOT / "handwriting-recognition" / "splits" / "test.txt"
HWR200_IMAGE_ROOT = PROJECT_ROOT / "datasets" / "HWR200" / "hw_dataset"
HWR200_LABELS_PATH = PROJECT_ROOT / "datasets" / "HWR200" / "labels_DBNet++_sync_test_manual.txt"

DBNET_CKPT_PATH = PROJECT_ROOT / "models" / "DBNet++" / "DBNet++.pt"
DBNET_CFG_PATH = HWR_REPO_ROOT / "config.yaml"
DBNET_IMAGE_SIZE = 640
DEVICE = "cuda"

OUTPUT_DIR = PROJECT_ROOT / "debug_images" / "experiment_3_test_DBNet++" / "dbnetpp_test"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
PREDICTIONS_JSONL_PATH = OUTPUT_DIR / "predictions.jsonl"
VIS_DIR = OUTPUT_DIR / "visualizations"
SINGLE_OUTPUT_DIR = OUTPUT_DIR / "single_image"

# None: весь test split. Int: первые N.
TEST_N_IMAGES = None
DEBUG_FIRST_IMAGES = 10
IOU_THRESHOLD = 0.5


def read_split_paths() -> List[str]:
    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        paths = [line.strip() for line in file if line.strip()]
    if TEST_N_IMAGES is None:
        return paths
    return paths[: int(TEST_N_IMAGES)]


def resolve_existing_image_path(relative_path: str) -> Optional[Path]:
    image_path = HWR200_IMAGE_ROOT / relative_path
    if image_path.exists():
        return image_path
    if relative_path.lower().endswith(".png.jpg") or relative_path.lower().endswith(".jpg.jpg"):
        candidate = HWR200_IMAGE_ROOT / relative_path[:-4]
        if candidate.exists():
            return candidate
    parent = image_path.parent
    if not parent.exists():
        return None
    stem = image_path.stem
    for candidate in parent.iterdir():
        if candidate.is_file() and candidate.stem == stem and candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            return candidate
    return None


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


def build_rows() -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    labels = load_hwr200_labels()
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for relative_path in read_split_paths():
        image_path = resolve_existing_image_path(relative_path)
        if image_path is None:
            skipped.append(
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
    return rows, skipped


def load_dbnet_model() -> Tuple[Any, Any, torch.device]:
    if not DBNET_CKPT_PATH.exists():
        raise FileNotFoundError(f"Не найден checkpoint DBNet++: {DBNET_CKPT_PATH}")
    if not DBNET_CFG_PATH.exists():
        raise FileNotFoundError(f"Не найден config DBNet++: {DBNET_CFG_PATH}")

    cfg = OmegaConf.load(DBNET_CFG_PATH)
    device = torch.device(DEVICE if DEVICE == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    checkpoint = torch.load(DBNET_CKPT_PATH, map_location="cpu", weights_only=False)
    weights = checkpoint["ema"] if checkpoint.get("ema") is not None else checkpoint["model"]
    model.load_state_dict(weights)
    model.to(device).eval()
    return model, cfg, device


@torch.no_grad()
def predict_dbnet_polygons_on_image(
    image_rgb: Image.Image,
    model: Any,
    cfg: Any,
    device: torch.device,
) -> Tuple[List[np.ndarray], List[float]]:
    tensor, meta = preprocess_image_pil(image_rgb, image_size=DBNET_IMAGE_SIZE)
    tensor = tensor.to(device)
    output = model(tensor)
    prob = output["prob"][0, 0].float().cpu().numpy()
    post_cfg = PostprocessConfig(
        thresh=cfg.postprocess.thresh,
        box_thresh=cfg.postprocess.box_thresh,
        unclip_ratio=cfg.postprocess.unclip_ratio,
        max_candidates=cfg.postprocess.max_candidates,
        min_size=cfg.postprocess.min_size,
    )
    return decode_prob_map(
        prob,
        post_cfg,
        scale=meta["scale"],
        pad=(meta["pad_left"], meta["pad_top"]),
        original_size=(meta["orig_w"], meta["orig_h"]),
    )


def predict_dbnet_polygons(
    image_bgr: np.ndarray,
    model: Any,
    cfg: Any,
    device: torch.device,
) -> Tuple[List[np.ndarray], List[float]]:
    image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return predict_dbnet_polygons_on_image(image_rgb, model, cfg, device)


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


def match_polygons(predicted: List[np.ndarray], targets: List[np.ndarray]) -> Dict[str, Any]:
    matched_targets = [False] * len(targets)
    matched_predictions = [False] * len(predicted)
    matches: List[Dict[str, Any]] = []
    tp = 0
    fp = 0
    for pred_index, pred_poly in enumerate(predicted):
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
            matched_predictions[pred_index] = True
            matches.append({"pred_index": int(pred_index), "gt_index": int(best_index), "iou": float(best_iou)})
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
        "pred_count": float(len(predicted)),
        "gt_count": float(len(targets)),
        "matches": matches,
        "pred_matched": matched_predictions,
        "gt_matched": matched_targets,
    }


def aggregate_metrics(per_image: List[Dict[str, Any]]) -> Dict[str, float]:
    tp = sum(float(item["metrics"]["tp"]) for item in per_image)
    fp = sum(float(item["metrics"]["fp"]) for item in per_image)
    fn = sum(float(item["metrics"]["fn"]) for item in per_image)
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


def color_for_index(index: int, total: int, saturation: float = 0.85, value: float = 0.95) -> Tuple[int, int, int]:
    hue = int(round(179.0 * (index % max(total, 1)) / max(total, 1)))
    hsv = np.array([[[hue, int(255 * saturation), int(255 * value)]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_polygons(
    image: np.ndarray,
    polygons: List[np.ndarray],
    label_prefix: str,
    thickness: int = 2,
) -> np.ndarray:
    canvas = image.copy()
    total = len(polygons)
    for index, polygon in enumerate(polygons, start=1):
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
        if len(points) < 3:
            continue
        color = color_for_index(index - 1, total)
        cv2.polylines(canvas, [points.reshape(-1, 1, 2)], True, color, thickness, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{label_prefix}{index}",
            (int(points[0, 0]), max(18, int(points[0, 1]) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def safe_name(relative_path: str) -> str:
    return relative_path.replace("\\", "__").replace("/", "__").replace(" ", "_")


def resolve_input_image_path(image_arg: str) -> Path:
    image_path = Path(image_arg)
    if image_path.exists():
        return image_path
    resolved = resolve_existing_image_path(image_arg)
    if resolved is not None:
        return resolved
    raise FileNotFoundError(
        f"Не найдено изображение: {image_arg}. "
        f"Можно передать полный путь или путь относительно {HWR200_IMAGE_ROOT}"
    )


def save_debug_visuals(
    row: Dict[str, Any],
    image_index: int,
    predicted_polygons: List[np.ndarray],
    scores: List[float],
    metrics: Dict[str, Any],
) -> Dict[str, str]:
    image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        return {}
    item_dir = VIS_DIR / f"{image_index:04d}_{safe_name(row['relative_path'])}"
    item_dir.mkdir(parents=True, exist_ok=True)

    gt_vis = draw_polygons(image, row["gt_polygons"], "G", thickness=2)
    pred_vis = draw_polygons(image, predicted_polygons, "P", thickness=2)
    overlay = image.copy()
    overlay = draw_polygons(overlay, row["gt_polygons"], "G", thickness=3)
    overlay = draw_polygons(overlay, predicted_polygons, "P", thickness=2)

    matched_overlay = image.copy()
    for match in metrics.get("matches", []):
        pred_index = int(match["pred_index"])
        gt_index = int(match["gt_index"])
        if 0 <= pred_index < len(predicted_polygons):
            points = np.asarray(predicted_polygons[pred_index], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(matched_overlay, [points], True, (0, 200, 0), 2, cv2.LINE_AA)
        if 0 <= gt_index < len(row["gt_polygons"]):
            points = np.asarray(row["gt_polygons"][gt_index], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(matched_overlay, [points], True, (255, 140, 0), 2, cv2.LINE_AA)

    paths = {
        "original": str(item_dir / "00_original.jpg"),
        "gt_polygons": str(item_dir / "01_gt_polygons.jpg"),
        "dbnet_polygons": str(item_dir / "02_dbnet_polygons.jpg"),
        "gt_vs_dbnet": str(item_dir / "03_gt_vs_dbnet.jpg"),
        "matches": str(item_dir / "04_matches.jpg"),
        "prediction_json": str(item_dir / "prediction.json"),
    }
    cv2.imwrite(paths["original"], image)
    cv2.imwrite(paths["gt_polygons"], gt_vis)
    cv2.imwrite(paths["dbnet_polygons"], pred_vis)
    cv2.imwrite(paths["gt_vs_dbnet"], overlay)
    cv2.imwrite(paths["matches"], matched_overlay)
    with open(paths["prediction_json"], "w", encoding="utf-8") as file:
        json.dump(
            {
                "relative_path": row["relative_path"],
                "metrics": metrics,
                "scores": [float(score) for score in scores],
                "predicted_polygons": [polygon.astype(float).tolist() for polygon in predicted_polygons],
                "gt_polygons": [polygon.astype(float).tolist() for polygon in row["gt_polygons"]],
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    return paths


def save_single_prediction(
    image_path: Path,
    predicted_polygons: List[np.ndarray],
    scores: List[float],
    runtime_sec: float,
) -> Path:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)

    item_dir = SINGLE_OUTPUT_DIR / safe_name(str(image_path))
    item_dir.mkdir(parents=True, exist_ok=True)

    pred_vis = draw_polygons(image, predicted_polygons, "P", thickness=2)
    cv2.imwrite(str(item_dir / "00_original.jpg"), image)
    cv2.imwrite(str(item_dir / "01_predicted_polygons.jpg"), pred_vis)

    prediction_path = item_dir / "prediction.json"
    with open(prediction_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "image_path": str(image_path),
                "runtime_sec": float(runtime_sec),
                "scores": [float(score) for score in scores],
                "predicted_polygons": [polygon.astype(float).tolist() for polygon in predicted_polygons],
                "visualization": str(item_dir / "01_predicted_polygons.jpg"),
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    return prediction_path


def evaluate_image(
    row: Dict[str, Any],
    model: Any,
    cfg: Any,
    device: torch.device,
) -> Dict[str, Any]:
    image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(row["image_path"])
    start_time = time.perf_counter()
    predicted_polygons, scores = predict_dbnet_polygons(image, model, cfg, device)
    runtime_sec = time.perf_counter() - start_time
    metrics = match_polygons(predicted_polygons, row["gt_polygons"])
    return {
        "relative_path": row["relative_path"],
        "image_path": str(row["image_path"]),
        "metrics": metrics,
        "runtime_sec": float(runtime_sec),
        "scores": [float(score) for score in scores],
        "predicted_polygons": predicted_polygons,
    }


def run_single_image(image_arg: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SINGLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_path = resolve_input_image_path(image_arg)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)

    model, cfg, device = load_dbnet_model()
    start_time = time.perf_counter()
    predicted_polygons, scores = predict_dbnet_polygons(image, model, cfg, device)
    runtime_sec = time.perf_counter() - start_time

    prediction_path = save_single_prediction(
        image_path=image_path,
        predicted_polygons=predicted_polygons,
        scores=scores,
        runtime_sec=runtime_sec,
    )

    print(f"[OK] image: {image_path}")
    print(f"[OK] polygons: {len(predicted_polygons)}")
    print(f"[OK] runtime_sec: {runtime_sec:.6f}")
    print(f"[OK] prediction_json: {prediction_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DBNet++ on HWR200 test split or one image.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Полный путь к изображению или путь относительно datasets/HWR200/hw_dataset.",
    )
    args = parser.parse_args()
    if args.image is not None:
        run_single_image(args.image)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    if PREDICTIONS_JSONL_PATH.exists():
        PREDICTIONS_JSONL_PATH.unlink()

    rows, skipped_input_rows = build_rows()
    if len(rows) == 0:
        raise RuntimeError("Не найдено изображений для теста DBNet++")

    model, cfg, device = load_dbnet_model()
    per_image: List[Dict[str, Any]] = []
    skipped_runtime: List[Dict[str, str]] = []

    for image_index, row in enumerate(tqdm(rows, desc="Evaluate DBNet++", unit="image")):
        try:
            result = evaluate_image(row, model, cfg, device)
        except Exception as error:
            skipped_runtime.append({"relative_path": row["relative_path"], "error": repr(error)})
            continue

        debug_files = {}
        if image_index < DEBUG_FIRST_IMAGES:
            debug_files = save_debug_visuals(
                row=row,
                image_index=image_index,
                predicted_polygons=result["predicted_polygons"],
                scores=result["scores"],
                metrics=result["metrics"],
            )

        compact = {
            "relative_path": result["relative_path"],
            "image_path": result["image_path"],
            "metrics": result["metrics"],
            "runtime_sec": result["runtime_sec"],
            "scores": result["scores"],
            "debug_files": debug_files,
            "predicted_polygons": [polygon.astype(float).tolist() for polygon in result["predicted_polygons"]],
        }
        per_image.append(compact)
        with open(PREDICTIONS_JSONL_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps(compact, ensure_ascii=False) + "\n")

    metrics = aggregate_metrics(per_image)
    runtime = aggregate_runtime(per_image)
    summary = {
        "method": "DBNet++ direct full-image inference",
        "optimized_metric": "hmean = 2 * precision * recall / (precision + recall)",
        "iou_threshold": IOU_THRESHOLD,
        "test_n_images": TEST_N_IMAGES,
        "rows_count": len(rows),
        "evaluated_count": len(per_image),
        "skipped_input_rows": skipped_input_rows,
        "skipped_runtime": skipped_runtime,
        "metrics": metrics,
        "runtime": runtime,
        "dbnet_checkpoint": str(DBNET_CKPT_PATH),
        "dbnet_config": str(DBNET_CFG_PATH),
        "dbnet_image_size": int(DBNET_IMAGE_SIZE),
        "device": str(device),
        "split_path": str(SPLIT_PATH),
        "labels_path": str(HWR200_LABELS_PATH),
        "predictions_jsonl": str(PREDICTIONS_JSONL_PATH),
        "visualizations_dir": str(VIS_DIR),
        "debug_first_images": int(DEBUG_FIRST_IMAGES),
        "per_image": per_image,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n[OK] DBNet++ test завершен")
    print(f"[OK] hmean: {metrics['hmean']:.6f}")
    print(f"[OK] precision: {metrics['precision']:.6f}")
    print(f"[OK] recall: {metrics['recall']:.6f}")
    print(f"[OK] TP/FP/FN: {metrics['tp']:.0f}/{metrics['fp']:.0f}/{metrics['fn']:.0f}")
    print(f"[OK] runtime mean/median: {runtime['mean_runtime_sec']:.3f}s / {runtime['median_runtime_sec']:.3f}s")
    print(f"[OK] Summary: {SUMMARY_PATH}")
    print(f"[OK] Visualizations: {VIS_DIR}")
    


if __name__ == "__main__":
    main()
