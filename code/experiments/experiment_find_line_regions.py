import gc
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from types import MethodType
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_closing, gaussian_filter1d
from tqdm import tqdm
from ultralytics import YOLO


# Делаем импорт из корня проекта независимо от того, откуда запущен скрипт.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import grade_hpp
from hpp_method import LineSegmentation
from u_net_binarization import load_unet_model, binarize_image_with_loaded_model


# Куда сохраняем результаты эксперимента.
OUTPUT_JSON_PATH = PROJECT_ROOT / "experiments" / "find_line_regions_experiment_results.json"
DEBUG_DIR = PROJECT_ROOT / "debug_images" / "experiments" / "find_line_regions"

# Модель поиска страниц. Загружается один раз на весь эксперимент.
YOLO_PAGE_MODEL_PATH = PROJECT_ROOT / "models" / "yolo_detect_notebook" / "yolo_detect_notebook_1_(1-architecture).pt"

# Общие параметры постобработки 1D TEXT/GAP маски.
SMOOTH_SIGMA = 2.0
MIN_TEXT_REGION_HEIGHT = 4
MAX_GAP_TO_MERGE = 3
CLOSING_STRUCTURE_SIZE = 5

# Метод 1: два порога.
HYSTERESIS_LOW_THRESHOLD = 0.18
HYSTERESIS_HIGH_THRESHOLD = 0.40

# Метод 2: динамическая разметка TEXT/GAP.
DP_SWITCH_PENALTY = 0.18

# Метод 3: Otsu.
OTSU_SMOOTH_SIGMA = 1.5

# Распараллеливаем три метода. YOLO-модель общая, поэтому доступ к ней защищен lock.
MAX_WORKERS = 3
YOLO_MODEL_LOCK = Lock()


def _regions_from_mask(text_mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Короткое описание:
        Превращает 1D TEXT/GAP маску в интервалы строк и чистит короткие артефакты.
    Вход:
        text_mask: np.ndarray -- булева маска TEXT по y.
    Выход:
        List[Tuple[int, int]] -- интервалы строк.
    """
    # Шаг 1: закрываем мелкие дырки внутри строки.
    structure = np.ones(CLOSING_STRUCTURE_SIZE, dtype=bool)
    text_mask = binary_closing(text_mask.astype(bool), structure=structure)

    # Шаг 2: выделяем интервалы True.
    regions: List[Tuple[int, int]] = []
    in_region = False
    start = 0
    for y_idx, is_text in enumerate(text_mask):
        if is_text and not in_region:
            start = y_idx
            in_region = True
        elif not is_text and in_region:
            regions.append((start, y_idx - 1))
            in_region = False
    if in_region:
        regions.append((start, len(text_mask) - 1))

    # Шаг 3: удаляем слишком короткие ложные сегменты.
    regions = [
        (start, end)
        for start, end in regions
        if end - start + 1 >= MIN_TEXT_REGION_HEIGHT
    ]

    # Шаг 4: склеиваем соседние сегменты, если gap между ними очень маленький.
    merged: List[Tuple[int, int]] = []
    for start, end in regions:
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start - prev_end <= MAX_GAP_TO_MERGE:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _find_line_regions_hysteresis(self,
                                  normalized_hpp: np.ndarray,
                                  debug_filename: str = None) -> List[Tuple[int, int]]:
    """
    Короткое описание:
        Метод 1: smoothing + два порога + закрытие дыр + удаление коротких сегментов.
    """
    smooth = gaussian_filter1d(normalized_hpp.astype(np.float32), sigma=SMOOTH_SIGMA)
    sure_text = smooth >= HYSTERESIS_HIGH_THRESHOLD
    candidate_text = smooth >= HYSTERESIS_LOW_THRESHOLD

    text_mask = np.zeros_like(candidate_text, dtype=bool)
    y = 0
    while y < len(candidate_text):
        if not candidate_text[y]:
            y += 1
            continue
        start = y
        while y < len(candidate_text) and candidate_text[y]:
            y += 1
        end = y - 1
        if np.any(sure_text[start:end + 1]):
            text_mask[start:end + 1] = True

    return _regions_from_mask(text_mask)


def _find_line_regions_dp(self,
                          normalized_hpp: np.ndarray,
                          debug_filename: str = None) -> List[Tuple[int, int]]:
    """
    Короткое описание:
        Метод 2: Viterbi/DP разметка TEXT/GAP со штрафом за частые переключения.
    """
    profile = gaussian_filter1d(normalized_hpp.astype(np.float32), sigma=SMOOTH_SIGMA)
    n = len(profile)
    if n == 0:
        return []

    # state 0 = GAP, state 1 = TEXT.
    dp = np.zeros((n, 2), dtype=np.float32)
    parent = np.zeros((n, 2), dtype=np.int32)
    gap_cost = profile
    text_cost = 1.0 - profile

    dp[0, 0] = gap_cost[0]
    dp[0, 1] = text_cost[0]
    for y_idx in range(1, n):
        for state in (0, 1):
            emission = gap_cost[y_idx] if state == 0 else text_cost[y_idx]
            stay_cost = dp[y_idx - 1, state]
            switch_cost = dp[y_idx - 1, 1 - state] + DP_SWITCH_PENALTY
            if stay_cost <= switch_cost:
                dp[y_idx, state] = stay_cost + emission
                parent[y_idx, state] = state
            else:
                dp[y_idx, state] = switch_cost + emission
                parent[y_idx, state] = 1 - state

    states = np.zeros(n, dtype=np.int32)
    states[-1] = int(np.argmin(dp[-1]))
    for y_idx in range(n - 2, -1, -1):
        states[y_idx] = parent[y_idx + 1, states[y_idx + 1]]
    return _regions_from_mask(states == 1)


def _find_line_regions_otsu(self,
                            normalized_hpp: np.ndarray,
                            debug_filename: str = None) -> List[Tuple[int, int]]:
    """
    Короткое описание:
        Метод 3: Otsu-порог для смеси TEXT/GAP + чистка интервалов.
    """
    smooth = gaussian_filter1d(normalized_hpp.astype(np.float32), sigma=OTSU_SMOOTH_SIGMA)
    profile_u8 = np.clip(smooth * 255.0, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(profile_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_mask = profile_u8 >= threshold
    return _regions_from_mask(text_mask)


FIND_LINE_REGION_METHODS = {
    "hysteresis_two_thresholds": _find_line_regions_hysteresis,
    "dp_switch_penalty": _find_line_regions_dp,
    "otsu_mixture_threshold": _find_line_regions_otsu,
}


def save_debug_matrices_to_dir(stem: str,
                               pred: np.ndarray,
                               target: np.ndarray,
                               output_dir: Path) -> None:
    """
    Короткое описание:
        Сохраняет pred/target/diff class-matrix в указанную папку без глобальных переменных.
    Вход:
        stem: str -- имя примера.
        pred: np.ndarray -- предсказанная матрица.
        target: np.ndarray -- GT-матрица.
        output_dir: Path -- папка debug.
    Выход:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{stem}_pred_classes.png"), grade_hpp.class_matrix_to_color(pred))
    cv2.imwrite(str(output_dir / f"{stem}_target_classes.png"), grade_hpp.class_matrix_to_color(target))

    diff = np.ones((*target.shape, 3), dtype=np.uint8) * 255
    diff[pred == target] = (230, 230, 230)
    diff[np.logical_and(target > 0, pred == target)] = (0, 180, 0)
    diff[np.logical_and(target > 0, pred != target)] = (0, 0, 255)
    diff[np.logical_and(target == 0, pred > 0)] = (255, 0, 0)
    cv2.imwrite(str(output_dir / f"{stem}_diff_classes.png"), diff)


def evaluate_with_method(method_name: str,
                         method_function,
                         label_paths: List[Path],
                         target_cache: Dict[str, np.ndarray],
                         page_yolo_model: YOLO) -> Dict[str, object]:
    """
    Короткое описание:
        Запускает grade_hpp-оценку для одного варианта _find_line_regions.
    Вход:
        method_name: str -- имя метода.
        method_function: Callable -- реализация _find_line_regions.
        label_paths: List[Path] -- список файлов разметки.
        target_cache: Dict[str, np.ndarray] -- заранее построенные GT class-matrix.
        page_yolo_model: YOLO -- общая YOLO-модель страниц.
    Выход:
        Dict[str, object] -- результаты метода.
    """
    method_results: Dict[str, Dict[str, object]] = {}
    for label_path in tqdm(label_paths, desc=method_name, position=0, leave=False):
        try:
            image_path = grade_hpp.find_image_path(label_path.stem)
            segmenter = LineSegmentation(debug=False, page_yolo_model=page_yolo_model)
            segmenter._find_line_regions = MethodType(method_function, segmenter)

            # YOLO-модель одна на весь процесс, поэтому защищаем инференс lock-ом.
            with YOLO_MODEL_LOCK:
                hpp_result = segmenter.segment_lines(str(image_path), return_class_matrix=True)

            if len(hpp_result) < 3 or hpp_result[2] is None:
                pred_matrix = np.zeros_like(target_cache[label_path.stem], dtype=np.int32)
            else:
                pred_matrix = hpp_result[2]

            target_matrix = target_cache[label_path.stem]
            pred_metric, target_metric = grade_hpp.prepare_pair_for_metrics(pred_matrix, target_matrix)
            pred_metric, alignment_metrics = grade_hpp.align_pred_by_text_intersection(pred_metric, target_metric)

            metrics = {
                "cross_entropy": grade_hpp.deterministic_cross_entropy(pred_metric, target_metric, text_only=False),
                "cross_entropy_text_only": grade_hpp.deterministic_cross_entropy(pred_metric, target_metric, text_only=True),
                "pixel_accuracy": float(np.mean(pred_metric == target_metric)),
                "text_pixel_accuracy": float(np.mean(pred_metric[target_metric > 0] == target_metric[target_metric > 0]))
                if int(np.sum(target_metric > 0)) > 0 else 0.0,
            }
            metrics.update(alignment_metrics)
            metrics.update(grade_hpp.line_detection_metrics(pred_metric, target_metric))

            method_results[label_path.stem] = {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "target_shape": [int(target_matrix.shape[0]), int(target_matrix.shape[1])],
                "pred_shape": [int(pred_matrix.shape[0]), int(pred_matrix.shape[1])],
                "metrics": metrics,
                "per_class": grade_hpp.per_class_metrics(pred_metric, target_metric),
            }

            if grade_hpp.DEBUG:
                method_debug_dir = DEBUG_DIR / method_name
                save_debug_matrices_to_dir(label_path.stem, pred_metric, target_metric, method_debug_dir)

            del pred_matrix, target_matrix, pred_metric, target_metric
            gc.collect()
        except Exception as exc:
            method_results[label_path.stem] = {"error": str(exc)}

    return {
        "method_name": method_name,
        "average_metrics": grade_hpp.average_results(method_results),
        "results": method_results,
    }


def build_target_cache(label_paths: List[Path], unet_model, unet_device) -> Dict[str, np.ndarray]:
    """
    Короткое описание:
        Один раз строит GT class-matrix для всех изображений через общую U-Net.
    Вход:
        label_paths: List[Path] -- файлы разметки.
        unet_model: Any -- загруженная U-Net.
        unet_device: Any -- устройство U-Net.
    Выход:
        Dict[str, np.ndarray] -- stem -> target class-matrix.
    """
    target_cache: Dict[str, np.ndarray] = {}
    for label_path in tqdm(label_paths, desc="build_target_cache"):
        image_path = grade_hpp.find_image_path(label_path.stem)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Не удалось прочитать изображение: {image_path}")
        binary = binarize_image_with_loaded_model(image, unet_model, unet_device)
        polygons = grade_hpp.read_yolo_polygons(label_path, binary.shape[1], binary.shape[0])
        target_cache[label_path.stem] = grade_hpp.build_target_class_matrix(polygons, binary)
        del image, binary, polygons
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return target_cache


def main() -> None:
    """
    Короткое описание:
        Сравнивает три экспериментальных метода _find_line_regions и сохраняет JSON.
    Вход:
        None
    Выход:
        None
    """
    os.makedirs(OUTPUT_JSON_PATH.parent, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    label_paths = grade_hpp.get_label_paths()

    # U-Net и YOLO грузятся один раз на весь эксперимент.
    unet_model, unet_device = load_unet_model(str(grade_hpp.UNET_MODEL_PATH))
    page_yolo_model = YOLO(str(YOLO_PAGE_MODEL_PATH))

    target_cache = build_target_cache(label_paths, unet_model, unet_device)

    all_results: Dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(FIND_LINE_REGION_METHODS))) as executor:
        futures = {
            executor.submit(
                evaluate_with_method,
                method_name,
                method_function,
                label_paths,
                target_cache,
                page_yolo_model,
            ): method_name
            for method_name, method_function in FIND_LINE_REGION_METHODS.items()
        }
        for future in as_completed(futures):
            method_name = futures[future]
            try:
                all_results[method_name] = future.result()
            except Exception as exc:
                all_results[method_name] = {"error": str(exc)}

    output = {
        "label_names": grade_hpp.LABEL_NAMES,
        "methods": list(FIND_LINE_REGION_METHODS.keys()),
        "parameters": {
            "smooth_sigma": SMOOTH_SIGMA,
            "min_text_region_height": MIN_TEXT_REGION_HEIGHT,
            "max_gap_to_merge": MAX_GAP_TO_MERGE,
            "closing_structure_size": CLOSING_STRUCTURE_SIZE,
            "hysteresis_low_threshold": HYSTERESIS_LOW_THRESHOLD,
            "hysteresis_high_threshold": HYSTERESIS_HIGH_THRESHOLD,
            "dp_switch_penalty": DP_SWITCH_PENALTY,
            "otsu_smooth_sigma": OTSU_SMOOTH_SIGMA,
            "max_mask_alignment_shift": grade_hpp.MAX_MASK_ALIGNMENT_SHIFT,
        },
        "results": all_results,
    }
    OUTPUT_JSON_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
