"""
Короткое описание:
    Сравнивает HPP-сегментацию строк и связку YOLO->DBNet++ на одном изображении HWR200.
Вход:
    TARGET_IMAGE_PATH: Path -- изображение для первого debug-прогона.
    HWR200_ROOT: Path -- корень датасета HWR200.
    HWR200_LABELS_TXT: Path -- PaddleOCR-style разметка строк HWR200.
    DBNET_CKPT_PATH: Path -- checkpoint DBNet++.
Выход:
    debug_images/compare_hpp_dbnetpp/* -- изображения и json-отчет сравнения.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


CODE_ROOT = Path(__file__).resolve().parents[1]
HWR_REPO_ROOT = CODE_ROOT / "handwriting-recognition"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(HWR_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(HWR_REPO_ROOT))

from hpp_method import LineSegmentation
from processing import extract_pages_with_yolo
from u_net_binarization import binarize_image
from src.model import build_model
from src.postprocess import PostprocessConfig, decode_prob_map
from src.utils import preprocess_image_pil


HWR200_ROOT = CODE_ROOT / "datasets" / "HWR200"
HWR200_IMAGE_ROOT = HWR200_ROOT / "hw_dataset"
HWR200_LABELS_TXT = HWR200_ROOT / "labels.txt"
GOOD_IMAGES_TXT = HWR200_ROOT / "good_images_with_detect_lines.txt"

PAGE_YOLO_MODEL_PATH = CODE_ROOT / "models" / "yolo_segment_notebook" / "yolo_segment_notebook_1_(1-architecture).pt"
DBNET_CKPT_PATH = CODE_ROOT / "models" / "DBNet++" / "DBNet++.pt"
DBNET_CFG_PATH = HWR_REPO_ROOT / "config.yaml"

DEBUG_OUTPUT_DIR = CODE_ROOT / "debug_images" / "compare_hpp_dbnetpp"
TARGET_IMAGE_PATH = None  # None: взять первое существующее изображение из good_images_with_detect_lines.txt.
MIN_GT_SCORE = 0.0
DEVICE = "cuda"
UNET_MODEL_PATH = CODE_ROOT / "models" / "u_net" / "unet_binarization_3_(6-architecture).pth"
RUN_ALL_GOOD_IMAGES = True
DEBUG_FIRST_IMAGES = 50
IOU_THRESHOLD = 0.5
BCE_EPS = 1e-6


def read_first_existing_image() -> Path:
    """
    Короткое описание:
        Берет первое существующее изображение из списка good_images_with_detect_lines.txt.
    Вход:
        отсутствует.
    Выход:
        Path -- путь к изображению.
    """
    if TARGET_IMAGE_PATH is not None:
        target = Path(TARGET_IMAGE_PATH)
        if not target.exists():
            raise FileNotFoundError(target)
        return target

    with open(GOOD_IMAGES_TXT, "r", encoding="utf-8") as file:
        for line in file:
            path = Path(line.strip())
            if path.exists():
                return path
    raise FileNotFoundError(f"Не найдено существующих изображений в {GOOD_IMAGES_TXT}")


def read_good_images() -> list[Path]:
    """
    Короткое описание:
        Читает список изображений для большого сравнения.
    Вход:
        отсутствует.
    Выход:
        list[Path] -- существующие пути из good_images_with_detect_lines.txt.
    """
    if TARGET_IMAGE_PATH is not None:
        return [read_first_existing_image()]
    images = []
    with open(GOOD_IMAGES_TXT, "r", encoding="utf-8") as file:
        for line in file:
            path = Path(line.strip())
            if path.exists():
                images.append(path)
    return images


def read_hwr200_labels(labels_txt: Path) -> dict[str, list[dict[str, Any]]]:
    """
    Короткое описание:
        Читает PaddleOCR-style labels.txt в словарь rel_path -> boxes.
    Вход:
        labels_txt: Path -- путь к labels.txt.
    Выход:
        dict[str, list[dict[str, Any]]] -- разметка по относительным путям.
    """
    labels = {}
    with open(labels_txt, "r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue
            rel_path, payload = line.split("\t", 1)
            labels[rel_path] = json.loads(payload)
    return labels


def image_path_to_label_rel(image_path: Path) -> str:
    """
    Короткое описание:
        Переводит абсолютный путь HWR200/hw_dataset к относительному пути labels.txt.
    Вход:
        image_path: Path -- путь к изображению.
    Выход:
        str -- относительный путь внутри hw_dataset.
    """
    return str(image_path.resolve().relative_to(HWR200_IMAGE_ROOT.resolve())).replace("\\", "/")


def strict_otsu_text_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Короткое описание:
        Строит грубую маску текста через Otsu, чтобы сравнивать полигоны по черным пикселям.
    Вход:
        image_bgr: np.ndarray -- исходное BGR-изображение.
    Выход:
        np.ndarray -- bool-маска текста.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary < 128


def unet_text_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Короткое описание:
        Строит маску текста через ту же U-Net-бинаризацию, что используется в HPP.
    Вход:
        image_bgr: np.ndarray -- исходное BGR-изображение.
    Выход:
        np.ndarray -- bool-маска текста.
    """
    binary = binarize_image(
        image_bgr,
        model_path=str(UNET_MODEL_PATH),
        target_size=(3000, 3000),
        device=DEVICE if torch.cuda.is_available() else "cpu",
    )
    if len(binary.shape) == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    binary = np.where(binary < 128, 0, 255).astype(np.uint8)
    return binary < 128


def polygons_to_text_masks(polygons: list[np.ndarray], text_mask: np.ndarray) -> list[np.ndarray]:
    """
    Короткое описание:
        Переводит полигоны строк в маски текста внутри этих полигонов.
    Вход:
        polygons: list[np.ndarray] -- список полигонов в координатах исходного изображения.
        text_mask: np.ndarray -- bool-маска текста.
    Выход:
        list[np.ndarray] -- bool-маски строк.
    """
    masks = []
    height, width = text_mask.shape
    for polygon in polygons:
        poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(poly) < 3:
            continue
        poly[:, 0] = np.clip(poly[:, 0], 0, width - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, height - 1)
        polygon_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [poly.astype(np.int32)], 1)
        mask = (polygon_mask > 0) & text_mask
        if np.any(mask):
            masks.append(mask)
    return masks


def class_matrix_to_masks(class_matrix: np.ndarray) -> list[np.ndarray]:
    """
    Короткое описание:
        Переводит class_matrix HPP в список bool-масок строк.
    Вход:
        class_matrix: np.ndarray -- матрица классов строк, 0 означает фон.
    Выход:
        list[np.ndarray] -- bool-маски строк.
    """
    masks = []
    for class_idx in range(1, int(np.max(class_matrix)) + 1):
        mask = class_matrix == class_idx
        if np.any(mask):
            masks.append(mask)
    return masks


def masks_to_union(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """
    Короткое описание:
        Объединяет список масок строк в одну bool-маску.
    Вход:
        masks: list[np.ndarray] -- маски строк.
        shape: tuple[int, int] -- размер итоговой маски.
    Выход:
        np.ndarray -- объединенная bool-маска.
    """
    union = np.zeros(shape, dtype=bool)
    for mask in masks:
        union |= mask.astype(bool)
    return union


def masks_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Короткое описание:
        Считает IoU двух bool-масок.
    Вход:
        mask_a: np.ndarray -- первая маска.
        mask_b: np.ndarray -- вторая маска.
    Выход:
        float -- IoU.
    """
    inter = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return float(inter) / float(max(union, 1))


def pixel_binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    """
    Короткое описание:
        Считает пиксельные precision, recall, F1, IoU и cross-entropy.
    Вход:
        pred_mask: np.ndarray -- предсказанная bool-маска текста строк.
        gt_mask: np.ndarray -- целевая bool-маска текста строк.
    Выход:
        dict[str, float] -- набор пиксельных метрик.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    iou = tp / max(tp + fp + fn, 1)

    pred_prob = np.where(pred, 1.0 - BCE_EPS, BCE_EPS).astype(np.float32)
    gt_float = gt.astype(np.float32)
    bce = -np.mean(gt_float * np.log(pred_prob) + (1.0 - gt_float) * np.log(1.0 - pred_prob))

    return {
        "pixel_tp": float(tp),
        "pixel_fp": float(fp),
        "pixel_fn": float(fn),
        "pixel_tn": float(tn),
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "pixel_f1": float(f1),
        "pixel_iou": float(iou),
        "cross_entropy": float(bce),
    }


def greedy_match_metrics(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], iou_threshold: float = 0.5) -> dict[str, Any]:
    """
    Короткое описание:
        Сравнивает предсказанные и целевые строки greedy matching по IoU.
    Вход:
        pred_masks: list[np.ndarray] -- маски метода.
        gt_masks: list[np.ndarray] -- целевые маски.
        iou_threshold: float -- порог совпадения.
    Выход:
        dict[str, Any] -- precision, recall, hmean и пары сопоставления.
    """
    pairs = []
    for pred_idx, pred in enumerate(pred_masks):
        for gt_idx, gt in enumerate(gt_masks):
            pairs.append((masks_iou(pred, gt), pred_idx, gt_idx))
    pairs.sort(reverse=True, key=lambda item: item[0])

    used_pred = set()
    used_gt = set()
    matches = []
    for iou, pred_idx, gt_idx in pairs:
        if iou < iou_threshold:
            break
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append({"pred_idx": int(pred_idx), "gt_idx": int(gt_idx), "iou": float(iou)})

    tp = len(matches)
    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    hmean = 2 * precision * recall / max(precision + recall, 1e-9)
    best_ious = [max([masks_iou(pred, gt) for gt in gt_masks] or [0.0]) for pred in pred_masks]
    matched_ious = [item["iou"] for item in matches]
    count_error = abs(len(pred_masks) - len(gt_masks))

    return {
        "count_pred": int(len(pred_masks)),
        "count_gt": int(len(gt_masks)),
        "count_error": int(count_error),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "hmean": float(hmean),
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else 0.0,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "min_matched_iou": float(np.min(matched_ious)) if matched_ious else 0.0,
        "matches": matches,
    }


def load_dbnet_model() -> tuple[Any, Any, torch.device]:
    """
    Короткое описание:
        Загружает DBNet++ и postprocess-конфиг.
    Вход:
        отсутствует.
    Выход:
        tuple[Any, Any, torch.device] -- model, cfg, device.
    """
    cfg = OmegaConf.load(DBNET_CFG_PATH)
    device = torch.device(DEVICE if torch.cuda.is_available() or DEVICE == "cpu" else "cpu")
    model = build_model(cfg)
    checkpoint = torch.load(DBNET_CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["ema"] if checkpoint.get("ema") is not None else checkpoint["model"])
    model.to(device).eval()
    return model, cfg, device


@torch.no_grad()
def predict_dbnet_polygons_on_image(image_rgb: Image.Image, model: Any, cfg: Any, device: torch.device) -> tuple[list[np.ndarray], list[float]]:
    """
    Короткое описание:
        Запускает DBNet++ на одном RGB-изображении и возвращает полигоны в его координатах.
    Вход:
        image_rgb: Image.Image -- RGB-изображение.
        model: Any -- DBNet++.
        cfg: Any -- конфиг DBNet++.
        device: torch.device -- устройство.
    Выход:
        tuple[list[np.ndarray], list[float]] -- полигоны и score.
    """
    tensor, meta = preprocess_image_pil(image_rgb, image_size=cfg.data.image_size)
    tensor = tensor.to(device)
    out = model(tensor)
    prob = out["prob"][0, 0].float().cpu().numpy()
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


def predict_yolo_dbnet_polygons(image_path: Path, dbnet_model: Any, dbnet_cfg: Any, device: torch.device) -> tuple[list[np.ndarray], list[float], list[dict[str, int]]]:
    """
    Короткое описание:
        Запускает YOLO page detector, затем DBNet++ на каждой найденной странице.
    Вход:
        image_path: Path -- исходное изображение.
        dbnet_model: Any -- DBNet++.
        dbnet_cfg: Any -- конфиг DBNet++.
        device: torch.device -- устройство.
    Выход:
        tuple[list[np.ndarray], list[float], list[dict[str, int]]] -- полигоны, score, bbox страниц.
    """
    page_yolo_model = YOLO(str(PAGE_YOLO_MODEL_PATH))
    return predict_yolo_dbnet_polygons_with_loaded_yolo(
        image_path=image_path,
        page_yolo_model=page_yolo_model,
        dbnet_model=dbnet_model,
        dbnet_cfg=dbnet_cfg,
        device=device,
    )


def predict_yolo_dbnet_polygons_with_loaded_yolo(
    image_path: Path,
    page_yolo_model: Any,
    dbnet_model: Any,
    dbnet_cfg: Any,
    device: torch.device,
) -> tuple[list[np.ndarray], list[float], list[dict[str, int]]]:
    """
    Короткое описание:
        Запускает загруженный YOLO page detector, затем DBNet++ на каждой найденной странице.
    Вход:
        image_path: Path -- исходное изображение.
        page_yolo_model: Any -- заранее загруженная YOLO-модель страниц.
        dbnet_model: Any -- DBNet++.
        dbnet_cfg: Any -- конфиг DBNet++.
        device: torch.device -- устройство.
    Выход:
        tuple[list[np.ndarray], list[float], list[dict[str, int]]] -- полигоны, score, bbox страниц.
    """
    pages, _, page_infos = extract_pages_with_yolo(
        image_path=str(image_path),
        model_path=str(PAGE_YOLO_MODEL_PATH),
        output_dir=str(DEBUG_OUTPUT_DIR),
        conf_threshold=0.8,
        return_binary=True,
        return_page_infos=True,
        yolo_model=page_yolo_model,
    )

    all_polygons = []
    all_scores = []
    for page, page_info in zip(pages, page_infos):
        page_rgb = Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB))
        polygons, scores = predict_dbnet_polygons_on_image(page_rgb, dbnet_model, dbnet_cfg, device)
        for polygon, score in zip(polygons, scores):
            restored = np.asarray(polygon, dtype=np.float32).copy()
            restored[:, 0] += float(page_info["x"])
            restored[:, 1] += float(page_info["y"])
            all_polygons.append(restored)
            all_scores.append(float(score))

    return all_polygons, all_scores, page_infos


def draw_masks_overlay(image_bgr: np.ndarray, masks: list[np.ndarray], output_path: Path, title: str) -> None:
    """
    Короткое описание:
        Сохраняет цветную визуализацию набора масок поверх изображения.
    Вход:
        image_bgr: np.ndarray -- исходное изображение.
        masks: list[np.ndarray] -- bool-маски.
        output_path: Path -- путь сохранения.
        title: str -- подпись.
    Выход:
        None
    """
    vis = image_bgr.copy()
    rng = np.random.default_rng(42)
    overlay = np.zeros_like(vis)
    for mask in masks:
        color = rng.integers(40, 255, size=3, dtype=np.uint8)
        overlay[mask] = color.tolist()
    vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
    cv2.putText(vis, title, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite(str(output_path), vis)


def draw_polygons_overlay(image_bgr: np.ndarray, polygons: list[np.ndarray], output_path: Path, color: tuple[int, int, int], title: str) -> None:
    """
    Короткое описание:
        Сохраняет полигоны поверх изображения.
    Вход:
        image_bgr: np.ndarray -- исходное изображение.
        polygons: list[np.ndarray] -- полигоны.
        output_path: Path -- путь сохранения.
        color: tuple[int, int, int] -- BGR-цвет.
        title: str -- подпись.
    Выход:
        None
    """
    vis = image_bgr.copy()
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=3)
    cv2.putText(vis, title, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
    cv2.imwrite(str(output_path), vis)


def compare_one_image(
    image_path: Path,
    labels: dict[str, list[dict[str, Any]]],
    hpp: LineSegmentation,
    page_yolo_model: Any,
    dbnet_model: Any,
    dbnet_cfg: Any,
    device: torch.device,
    debug: bool,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Короткое описание:
        Сравнивает HPP и YOLO->DBNet++ на одном изображении.
    Вход:
        image_path: Path -- путь к изображению.
        labels: dict[str, list[dict[str, Any]]] -- разметка HWR200.
        hpp: LineSegmentation -- HPP-сегментатор.
        page_yolo_model: Any -- YOLO-модель страниц.
        dbnet_model: Any -- DBNet++.
        dbnet_cfg: Any -- конфиг DBNet++.
        device: torch.device -- устройство.
        debug: bool -- сохранять изображения отладки.
        output_dir: Path -- папка отладки для изображения.
    Выход:
        dict[str, Any] -- отчет по изображению.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    rel_path = image_path_to_label_rel(image_path)
    if rel_path not in labels:
        raise FileNotFoundError(f"Нет разметки в labels.txt для {rel_path}")

    gt_boxes = [box for box in labels[rel_path] if float(box.get("score") or 1.0) >= MIN_GT_SCORE]
    gt_polygons = [np.asarray(box["points"], dtype=np.float32) for box in gt_boxes]
    text_mask = unet_text_mask(image_bgr)
    gt_masks = polygons_to_text_masks(gt_polygons, text_mask)
    gt_union = masks_to_union(gt_masks, text_mask.shape)

    if debug:
        cv2.imwrite(str(output_dir / "00_input.jpg"), image_bgr)
        cv2.imwrite(str(output_dir / "01_unet_text_mask.png"), (text_mask.astype(np.uint8) * 255))
        draw_polygons_overlay(image_bgr, gt_polygons, output_dir / "02_gt_polygons.jpg", (0, 255, 0), f"GT polygons: {len(gt_polygons)}")
        draw_masks_overlay(image_bgr, gt_masks, output_dir / "03_gt_text_masks.jpg", f"GT text masks: {len(gt_masks)}")

    # Шаг 1: HPP.
    _, _, hpp_class_matrix = hpp.segment_lines(str(image_path), return_class_matrix=True)
    hpp_masks = class_matrix_to_masks(hpp_class_matrix)
    hpp_union = masks_to_union(hpp_masks, text_mask.shape)
    hpp_metrics = greedy_match_metrics(hpp_masks, gt_masks, iou_threshold=IOU_THRESHOLD)
    hpp_metrics.update(pixel_binary_metrics(hpp_union, gt_union))
    if debug:
        draw_masks_overlay(image_bgr, hpp_masks, output_dir / "04_hpp_masks.jpg", f"HPP masks: {len(hpp_masks)}")

    # Шаг 2: YOLO -> DBNet++.
    dbnet_polygons, dbnet_scores, page_infos = predict_yolo_dbnet_polygons_with_loaded_yolo(
        image_path=image_path,
        page_yolo_model=page_yolo_model,
        dbnet_model=dbnet_model,
        dbnet_cfg=dbnet_cfg,
        device=device,
    )
    dbnet_masks = polygons_to_text_masks(dbnet_polygons, text_mask)
    dbnet_union = masks_to_union(dbnet_masks, text_mask.shape)
    dbnet_metrics = greedy_match_metrics(dbnet_masks, gt_masks, iou_threshold=IOU_THRESHOLD)
    dbnet_metrics.update(pixel_binary_metrics(dbnet_union, gt_union))
    if debug:
        draw_polygons_overlay(image_bgr, dbnet_polygons, output_dir / "05_dbnet_polygons.jpg", (0, 0, 255), f"YOLO->DBNet++ polygons: {len(dbnet_polygons)}")
        draw_masks_overlay(image_bgr, dbnet_masks, output_dir / "06_dbnet_text_masks.jpg", f"YOLO->DBNet++ masks: {len(dbnet_masks)}")

    return {
        "image_path": str(image_path),
        "label_rel_path": rel_path,
        "image_shape": list(image_bgr.shape),
        "target": {
            "polygons": len(gt_polygons),
            "text_masks": len(gt_masks),
            "min_score": MIN_GT_SCORE,
            "text_mask": "u_net",
        },
        "hpp": hpp_metrics,
        "yolo_dbnetpp": {
            **dbnet_metrics,
            "page_infos": page_infos,
            "scores": dbnet_scores,
        },
        "debug_files": {
            "input": str(output_dir / "00_input.jpg"),
            "unet_text_mask": str(output_dir / "01_unet_text_mask.png"),
            "gt_polygons": str(output_dir / "02_gt_polygons.jpg"),
            "gt_text_masks": str(output_dir / "03_gt_text_masks.jpg"),
            "hpp_masks": str(output_dir / "04_hpp_masks.jpg"),
            "dbnet_polygons": str(output_dir / "05_dbnet_polygons.jpg"),
            "dbnet_text_masks": str(output_dir / "06_dbnet_text_masks.jpg"),
        },
    }


def summarize_method_reports(image_reports: list[dict[str, Any]], method_name: str) -> dict[str, float]:
    """
    Короткое описание:
        Считает средние значения всех числовых метрик метода.
    Вход:
        image_reports: list[dict[str, Any]] -- отчеты по изображениям.
        method_name: str -- имя метода в отчете.
    Выход:
        dict[str, float] -- средние метрики.
    """
    ignore_keys = {"matches", "page_infos", "scores"}
    values: dict[str, list[float]] = {}
    for report in image_reports:
        method = report[method_name]
        for key, value in method.items():
            if key in ignore_keys:
                continue
            if isinstance(value, (int, float)):
                values.setdefault(key, []).append(float(value))

    summary = {}
    for key, vals in values.items():
        summary[f"mean_{key}"] = float(np.mean(vals)) if vals else 0.0
        summary[f"std_{key}"] = float(np.std(vals)) if vals else 0.0
    return summary


def main() -> None:
    """
    Короткое описание:
        Запускает большой прогон сравнения HPP и YOLO->DBNet++ по good_images_with_detect_lines.txt.
    Вход:
        отсутствует.
    Выход:
        None
    """
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = read_hwr200_labels(HWR200_LABELS_TXT)
    images = read_good_images() if RUN_ALL_GOOD_IMAGES else [read_first_existing_image()]
    dbnet_model, dbnet_cfg, device = load_dbnet_model()
    page_yolo_model = YOLO(str(PAGE_YOLO_MODEL_PATH))

    image_reports = []
    failed = []
    for idx, image_path in enumerate(tqdm(images, desc="Compare HPP vs DBNet++", unit="image")):
        image_debug_dir = DEBUG_OUTPUT_DIR / f"{idx:03d}_{image_path.stem}"
        debug = idx < DEBUG_FIRST_IMAGES
        hpp = LineSegmentation(
            debug=debug,
            page_yolo_model=page_yolo_model,
            use_warp_binary_by_local_angles=True,
            use_bijection_warp=False,
        )
        try:
            report = compare_one_image(
                image_path=image_path,
                labels=labels,
                hpp=hpp,
                page_yolo_model=page_yolo_model,
                dbnet_model=dbnet_model,
                dbnet_cfg=dbnet_cfg,
                device=device,
                debug=debug,
                output_dir=image_debug_dir,
            )
            image_reports.append(report)
        except Exception as exc:
            failed.append({
                "image_path": str(image_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    summary = {
        "num_requested": len(images),
        "num_success": len(image_reports),
        "num_failed": len(failed),
        "iou_threshold": IOU_THRESHOLD,
        "min_gt_score": MIN_GT_SCORE,
        "text_mask": "u_net",
        "hpp": summarize_method_reports(image_reports, "hpp"),
        "yolo_dbnetpp": summarize_method_reports(image_reports, "yolo_dbnetpp"),
    }
    report = {
        "summary": summary,
        "images": image_reports,
        "failed": failed,
    }

    report_path = DEBUG_OUTPUT_DIR / "compare_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed:
        print("Failed examples:")
        print(json.dumps(failed[:10], ensure_ascii=False, indent=2))
    print(f"Debug saved to: {DEBUG_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
