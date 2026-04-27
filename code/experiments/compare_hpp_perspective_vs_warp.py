import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import grade_hpp
from hpp_method import LineSegmentation
from u_net_binarization import load_unet_model, binarize_image_with_loaded_model


OUTPUT_JSON_PATH = PROJECT_ROOT / "experiments" / "compare_hpp_perspective_vs_warp_results.json"
DEBUG_DIR = PROJECT_ROOT / "debug_images" / "experiments" / "compare_hpp_perspective_vs_warp"
YOLO_PAGE_MODEL_PATH = PROJECT_ROOT / "models" / "yolo_detect_notebook" / "yolo_detect_notebook_1_(1-architecture).pt"
SAVE_VISUAL_DEBUG = True

LABEL_NAMES = [
    # "22_236.txt",
    # "ru_hw2022_2_IMG_7791.txt",
    # "ru_hw2022_2_IMG_7790.txt",
    # "ru_hw2022_4_IMG_7354.txt",
    "ru_hw2022_6_IMG_7175.txt",
    # "2196.txt"
    # "1_11.txt",
    # "1_19.txt",
    # "8_109.txt",
    # "2786.txt",
    # "2796.txt",
    # "1_18.txt",
    # "1_23.txt",
    # "2910.txt",
    # "2812.txt",
    # "2013.txt",
    # "2015.txt",
    # "2016.txt",
    # "2884.txt",
]


def get_label_paths() -> List[Path]:
    """
    Короткое описание:
        Возвращает фиксированный список label-файлов для эксперимента.
    Вход:
        None
    Выход:
        List[Path] -- пути к txt-разметкам.
    """
    return [
        grade_hpp.LABELS_DIR / (name if name.endswith(".txt") else f"{name}.txt")
        for name in LABEL_NAMES
    ]


def build_target_matrix(label_path: Path, unet_model, unet_device) -> np.ndarray:
    """
    Короткое описание:
        Строит GT class-matrix по polygon-разметке и общей U-Net.
    Вход:
        label_path: Path -- путь к txt-разметке.
        unet_model: Any -- загруженная U-Net.
        unet_device: Any -- устройство U-Net.
    Выход:
        np.ndarray -- target class-matrix.
    """
    image_path = grade_hpp.find_image_path(label_path.stem)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")
    binary = binarize_image_with_loaded_model(image, unet_model, unet_device)
    polygons = grade_hpp.read_yolo_polygons(label_path, binary.shape[1], binary.shape[0])
    target_matrix = grade_hpp.build_target_class_matrix(polygons, binary)
    del image, binary, polygons
    return target_matrix


def calculate_order_invariant_text_iou(pred: np.ndarray,
                                       target: np.ndarray) -> Tuple[Dict[str, float], Dict[str, object]]:
    """
    Короткое описание:
        Считает IoU строк без привязки к номерам классов.
    Вход:
        pred: np.ndarray -- предсказанная class-matrix.
        target: np.ndarray -- GT class-matrix.
    Выход:
        Tuple[Dict[str, float], Dict[str, object]] -- средние IoU и подробности по GT-классам.
    """
    # Шаг 1: номера строк у HPP и GT могут идти в разном порядке, поэтому сопоставляем
    # классы геометрически: какая pred-строка сильнее всего пересекается с какой GT-строкой.
    pred_classes = [int(value) for value in np.unique(pred) if int(value) > 0]
    target_classes = [int(value) for value in np.unique(target) if int(value) > 0]
    if not target_classes:
        text_iou = 1.0 if not pred_classes else 0.0
        return (
            {"text_iou": float(text_iou), "text_iou_weighted": float(text_iou)},
            {"text_iou_per_class": {}, "text_iou_matches": []},
        )
    if not pred_classes:
        return (
            {"text_iou": 0.0, "text_iou_weighted": 0.0},
            {
                "text_iou_per_class": {
                    str(target_class): {
                        "iou": 0.0,
                        "matched_pred_class": 0.0,
                        "target_pixels": float(np.sum(target == target_class)),
                        "pred_pixels": 0.0,
                        "intersection": 0.0,
                        "union": float(np.sum(target == target_class)),
                    }
                    for target_class in target_classes
                },
                "text_iou_matches": [],
            },
        )

    intersection_matrix = np.zeros((len(target_classes), len(pred_classes)), dtype=np.float32)
    iou_matrix = np.zeros((len(target_classes), len(pred_classes)), dtype=np.float32)
    for target_idx, target_class in enumerate(target_classes):
        target_mask = target == target_class
        for pred_idx, pred_class in enumerate(pred_classes):
            pred_mask = pred == pred_class
            intersection = float(np.sum(np.logical_and(target_mask, pred_mask)))
            union = float(np.sum(np.logical_or(target_mask, pred_mask)))
            intersection_matrix[target_idx, pred_idx] = intersection
            iou_matrix[target_idx, pred_idx] = intersection / union if union > 0.0 else 0.0

    row_indices, col_indices = linear_sum_assignment(-intersection_matrix)
    target_to_pred: Dict[int, int] = {}
    matches: List[Dict[str, float]] = []
    for row_idx, col_idx in zip(row_indices, col_indices):
        intersection = float(intersection_matrix[row_idx, col_idx])
        iou = float(iou_matrix[row_idx, col_idx])
        if intersection <= 0.0:
            continue
        target_class = target_classes[int(row_idx)]
        pred_class = pred_classes[int(col_idx)]
        target_to_pred[target_class] = pred_class
        matches.append({
            "target_class": float(target_class),
            "pred_class": float(pred_class),
            "intersection": intersection,
            "iou": iou,
        })

    # Шаг 2: считаем IoU отдельно для каждой GT-строки уже после найденного соответствия.
    per_class: Dict[str, Dict[str, float]] = {}
    iou_sum = 0.0
    weighted_sum = 0.0
    target_pixel_sum = 0.0
    for target_class in target_classes:
        target_mask = target == target_class
        matched_pred_class = target_to_pred.get(target_class)
        pred_mask = pred == matched_pred_class if matched_pred_class is not None else np.zeros_like(target_mask)
        intersection = float(np.sum(np.logical_and(target_mask, pred_mask)))
        union = float(np.sum(np.logical_or(target_mask, pred_mask)))
        target_pixels = float(np.sum(target_mask))
        pred_pixels = float(np.sum(pred_mask))
        iou = intersection / union if union > 0.0 else 0.0

        per_class[str(target_class)] = {
            "iou": float(iou),
            "matched_pred_class": float(matched_pred_class or 0),
            "target_pixels": target_pixels,
            "pred_pixels": pred_pixels,
            "intersection": intersection,
            "union": union,
        }
        iou_sum += float(iou)
        weighted_sum += iou * target_pixels
        target_pixel_sum += target_pixels

    # Делим строго на количество GT-строк, чтобы лишние/перепутанные pred-классы
    # не меняли знаменатель средней оценки по изображению.
    target_line_count = len(target_classes)
    text_iou = float(iou_sum / target_line_count) if target_line_count > 0 else 0.0
    text_iou_weighted = float(weighted_sum / target_pixel_sum) if target_pixel_sum > 0.0 else 0.0
    return (
        {
            "text_iou": text_iou,
            "text_iou_weighted": text_iou_weighted,
            "text_iou_target_lines": float(target_line_count),
        },
        {
            "text_iou_per_class": per_class,
            "text_iou_matches": matches,
            "target_classes": target_classes,
            "pred_classes": pred_classes,
            "intersection_matrix": intersection_matrix.tolist(),
            "iou_matrix": iou_matrix.tolist(),
        },
    )


def _matched_class_colors(matches: List[Dict[str, float]]) -> Dict[int, Tuple[int, int, int]]:
    """
    Короткое описание:
        Создает стабильные цвета для matched GT-классов.
    """
    rng = np.random.default_rng(42)
    colors: Dict[int, Tuple[int, int, int]] = {}
    for match in matches:
        target_class = int(match["target_class"])
        colors[target_class] = tuple(int(value) for value in rng.integers(30, 245, size=3))
    return colors


def save_mask_matching_debug(stem: str,
                             variant_name: str,
                             pred: np.ndarray,
                             target: np.ndarray,
                             text_iou_details: Dict[str, object],
                             alignment_metrics: Dict[str, float]) -> None:
    """
    Короткое описание:
        Сохраняет визуальный debug того, как pred-классы сопоставлены GT-классам.
    Вход:
        stem: str -- имя изображения.
        variant_name: str -- имя варианта HPP.
        pred: np.ndarray -- pred class-matrix после приведения формы и alignment.
        target: np.ndarray -- target class-matrix после приведения формы.
        text_iou_details: Dict[str, object] -- детали matching из calculate_order_invariant_text_iou.
        alignment_metrics: Dict[str, float] -- метрики глобального сдвига pred.
    Выход:
        None
    """
    if not SAVE_VISUAL_DEBUG:
        return

    output_dir = DEBUG_DIR / variant_name / stem
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "pred_metric.npy", pred)
    np.save(output_dir / "target_metric.npy", target)

    matches = text_iou_details.get("text_iou_matches", [])
    colors = _matched_class_colors(matches)
    target_to_pred = {
        int(match["target_class"]): int(match["pred_class"])
        for match in matches
    }

    height, width = target.shape
    target_vis = np.ones((height, width, 3), dtype=np.uint8) * 255
    pred_vis = np.ones((height, width, 3), dtype=np.uint8) * 255
    overlay = np.ones((height, width, 3), dtype=np.uint8) * 255

    for target_class, pred_class in target_to_pred.items():
        color = colors[target_class]
        target_mask = target == target_class
        pred_mask = pred == pred_class
        intersection = np.logical_and(target_mask, pred_mask)
        target_only = np.logical_and(target_mask, ~pred_mask)
        pred_only = np.logical_and(pred_mask, ~target_mask)

        target_vis[target_mask] = color
        pred_vis[pred_mask] = color
        overlay[intersection] = color
        overlay[target_only] = tuple(max(0, int(channel * 0.45)) for channel in color)
        overlay[pred_only] = tuple(min(255, int(channel * 0.45 + 140)) for channel in color)

    unmatched_target = np.logical_and(target > 0, overlay[:, :, 0] == 255)
    unmatched_target = np.logical_and(unmatched_target, overlay[:, :, 1] == 255)
    unmatched_target = np.logical_and(unmatched_target, overlay[:, :, 2] == 255)
    matched_pred_classes = set(target_to_pred.values())
    unmatched_pred = pred > 0
    for pred_class in matched_pred_classes:
        unmatched_pred = np.logical_and(unmatched_pred, pred != pred_class)
    overlay[unmatched_target] = (0, 0, 180)
    overlay[unmatched_pred] = (180, 0, 0)

    cv2.imwrite(str(output_dir / "target_matched_colors.png"), target_vis)
    cv2.imwrite(str(output_dir / "pred_matched_colors.png"), pred_vis)
    cv2.imwrite(str(output_dir / "matched_overlay.png"), overlay)

    log_path = output_dir / "matching_debug.txt"
    with log_path.open("w", encoding="utf-8") as file:
        file.write(f"stem={stem}\n")
        file.write(f"variant={variant_name}\n")
        file.write(f"shape={target.shape}\n")
        file.write(f"alignment_metrics={json.dumps(alignment_metrics, ensure_ascii=False)}\n")
        file.write(f"target_classes={text_iou_details.get('target_classes', [])}\n")
        file.write(f"pred_classes={text_iou_details.get('pred_classes', [])}\n")
        file.write("matches target_class -> pred_class:\n")
        for match in matches:
            file.write(
                f"  target={int(match['target_class'])} pred={int(match['pred_class'])} "
                f"intersection={float(match['intersection']):.0f} iou={float(match['iou']):.6f}\n"
            )
        file.write("per_class:\n")
        for target_class, info in text_iou_details.get("text_iou_per_class", {}).items():
            file.write(f"  target={target_class}: {json.dumps(info, ensure_ascii=False)}\n")
        file.write("intersection_matrix rows=target_classes cols=pred_classes:\n")
        for row in text_iou_details.get("intersection_matrix", []):
            file.write("  " + " ".join(f"{float(value):.0f}" for value in row) + "\n")
        file.write("iou_matrix rows=target_classes cols=pred_classes:\n")
        for row in text_iou_details.get("iou_matrix", []):
            file.write("  " + " ".join(f"{float(value):.6f}" for value in row) + "\n")


def evaluate_hpp_variant(label_path: Path,
                         target_matrix: np.ndarray,
                         page_yolo_model: YOLO,
                         variant_name: str,
                         use_warp_binary_by_local_angles: bool,
                         use_bijection_warp: bool = False) -> Dict[str, object]:
    """
    Короткое описание:
        Запускает один вариант HPP и считает метрики class-matrix.
    Вход:
        label_path: Path -- путь к txt-разметке.
        target_matrix: np.ndarray -- GT class-matrix.
        page_yolo_model: YOLO -- общая модель поиска страниц.
        variant_name: str -- имя варианта для debug.
        use_warp_binary_by_local_angles: bool -- True local warp, False correct_perspective.
        use_bijection_warp: bool -- True bijection local warp, False обычный local warp.
    Выход:
        Dict[str, object] -- metrics и формы матриц.
    """
    image_path = grade_hpp.find_image_path(label_path.stem)
    segmenter = LineSegmentation(
        debug=False,
        page_yolo_model=page_yolo_model,
        use_warp_binary_by_local_angles=use_warp_binary_by_local_angles,
        use_bijection_warp=use_bijection_warp,
    )
    hpp_result = segmenter.segment_lines(str(image_path), return_class_matrix=True)
    if len(hpp_result) < 3 or hpp_result[2] is None:
        pred_matrix = np.zeros_like(target_matrix, dtype=np.int32)
    else:
        pred_matrix = hpp_result[2]

    pred_metric, target_metric = grade_hpp.prepare_pair_for_metrics(pred_matrix, target_matrix)
    pred_metric, alignment_metrics = grade_hpp.align_pred_by_text_intersection(pred_metric, target_metric)
    text_iou_metrics, text_iou_details = calculate_order_invariant_text_iou(pred_metric, target_metric)
    save_mask_matching_debug(
        label_path.stem,
        variant_name,
        pred_metric,
        target_metric,
        text_iou_details,
        alignment_metrics,
    )

    metrics = {
        "cross_entropy": grade_hpp.deterministic_cross_entropy(pred_metric, target_metric, text_only=False),
        "cross_entropy_text_only": grade_hpp.deterministic_cross_entropy(pred_metric, target_metric, text_only=True),
        "pixel_accuracy": float(np.mean(pred_metric == target_metric)),
        "text_pixel_accuracy": float(np.mean(pred_metric[target_metric > 0] == target_metric[target_metric > 0]))
        if int(np.sum(target_metric > 0)) > 0 else 0.0,
    }
    metrics.update(alignment_metrics)
    metrics.update(grade_hpp.line_detection_metrics(pred_metric, target_metric))
    metrics.update(text_iou_metrics)

    return {
        "pred_shape": [int(pred_matrix.shape[0]), int(pred_matrix.shape[1])],
        "target_shape": [int(target_matrix.shape[0]), int(target_matrix.shape[1])],
        "metrics": metrics,
        "per_class": grade_hpp.per_class_metrics(pred_metric, target_metric),
        "order_invariant_text_iou": text_iou_details,
    }


def average_metrics(results: Dict[str, Dict[str, object]]) -> Dict[str, float]:
    """
    Короткое описание:
        Усредняет metrics по изображениям без ошибок.
    Вход:
        results: Dict[str, Dict[str, object]] -- результаты варианта.
    Выход:
        Dict[str, float] -- средние метрики.
    """
    values: Dict[str, List[float]] = {}
    for result in results.values():
        if "metrics" not in result:
            continue
        for key, value in result["metrics"].items():
            values.setdefault(key, []).append(float(value))
    return {key: float(np.mean(items)) for key, items in values.items() if items}


def main() -> None:
    """
    Короткое описание:
        Сравнивает HPP с correct_perspective и HPP с warp_binary_by_local_angles_bijection.
    Вход:
        None
    Выход:
        None
    """
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    label_paths = get_label_paths()
    unet_model, unet_device = load_unet_model(str(grade_hpp.UNET_MODEL_PATH))
    page_yolo_model = YOLO(str(YOLO_PAGE_MODEL_PATH))

    results = {
        "correct_perspective": {},
        "warp_binary_by_local_angles_bijection": {},
    }

    for label_path in tqdm(label_paths, desc="compare_hpp_perspective_vs_warp"):
        try:
            target_matrix = build_target_matrix(label_path, unet_model, unet_device)
            results["correct_perspective"][label_path.stem] = evaluate_hpp_variant(
                label_path,
                target_matrix,
                page_yolo_model,
                variant_name="correct_perspective",
                use_warp_binary_by_local_angles=False,
            )
            results["warp_binary_by_local_angles_bijection"][label_path.stem] = evaluate_hpp_variant(
                label_path,
                target_matrix,
                page_yolo_model,
                variant_name="warp_binary_by_local_angles_bijection",
                use_warp_binary_by_local_angles=True,
                use_bijection_warp=True,
            )
            del target_matrix
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            results["correct_perspective"][label_path.stem] = {"error": str(exc)}
            results["warp_binary_by_local_angles_bijection"][label_path.stem] = {"error": str(exc)}

    output = {
        "label_names": LABEL_NAMES,
        "main_metric": "cross_entropy",
        "average_metrics": {
            "correct_perspective": average_metrics(results["correct_perspective"]),
            "warp_binary_by_local_angles_bijection": average_metrics(
                results["warp_binary_by_local_angles_bijection"],
            ),
        },
        "results": results,
    }
    OUTPUT_JSON_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
