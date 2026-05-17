"""
Короткое описание:
    Запускает DBNet++ на одном исходном изображении и сохраняет визуализацию строк.
Вход:
    IMAGE_PATH: Path -- единственный путь, который нужно поменять для запуска.
Выход:
    debug_images/test_1_dbnetpp/dbnet_lines.jpg -- полигоны DBNet++ поверх изображения.
    debug_images/test_1_dbnetpp/gt_lines.jpg -- полигоны разметки поверх изображения.
    debug_images/test_1_dbnetpp/gt_vs_dbnet_lines.jpg -- разметка и ответ модели вместе.
    debug_images/test_1_dbnetpp/dbnet_lines.json -- координаты полигонов и score.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


# Единственный путь, который нужно менять для запуска.
IMAGE_PATH: Path = Path(
    "/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code/datasets/HWR200/hw_dataset/23/reuse1/ФотоСветлое/5.jpg"
)

# Корень проекта.
CODE_ROOT: Path = Path(__file__).resolve().parent

# Локальный репозиторий с архитектурой DBNet++ и postprocess.
HWR_REPO_ROOT: Path = CODE_ROOT / "handwriting-recognition"

# Checkpoint DBNet++.
DBNET_CKPT_PATH: Path = CODE_ROOT / "models" / "DBNet++" / "DBNet++.pt"

# Конфиг DBNet++.
DBNET_CFG_PATH: Path = HWR_REPO_ROOT / "config.yaml"

# Разметка HWR200 в PaddleOCR-style формате.
HWR200_LABELS_PATH: Path = CODE_ROOT / "datasets" / "HWR200" / "labels.txt"

# Корень изображений HWR200, относительно которого записаны пути в labels.txt.
HWR200_IMAGE_ROOT: Path = CODE_ROOT / "datasets" / "HWR200" / "hw_dataset"

# Папка debug-результатов.
DEBUG_OUTPUT_DIR: Path = CODE_ROOT / "debug_images" / "test_1_dbnetpp"

# Размер входа DBNet++. Изображение сжимается в 640 x 640 через resize longest side + padding.
DBNET_IMAGE_SIZE: int = 640

# Устройство инференса. Если cuda недоступна, код сам перейдет на cpu.
DEVICE: str = "cuda"

# Цвет полигона DBNet++ в BGR.
DBNET_POLYGON_COLOR: tuple[int, int, int] = (0, 0, 255)

# Цвет полигона GT-разметки в BGR.
GT_POLYGON_COLOR: tuple[int, int, int] = (0, 180, 0)

# Толщина линии полигона.
DBNET_POLYGON_THICKNESS: int = 3

# Толщина линии GT-разметки.
GT_POLYGON_THICKNESS: int = 2

# Рисовать score рядом с каждым полигоном.
DRAW_SCORES: bool = True


if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(HWR_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(HWR_REPO_ROOT))

from src.model import build_model
from src.postprocess import PostprocessConfig, decode_prob_map
from src.utils import preprocess_image_pil


def load_dbnet_model() -> tuple[Any, Any, torch.device]:
    """
    Короткое описание:
        Загружает DBNet++ из checkpoint и config.yaml.
    Вход:
        отсутствует.
    Выход:
        tuple[Any, Any, torch.device] -- модель, конфиг, устройство.
    """
    # Шаг 1: проверяем файлы.
    if not DBNET_CKPT_PATH.exists():
        raise FileNotFoundError(f"Не найден checkpoint DBNet++: {DBNET_CKPT_PATH}")
    if not DBNET_CFG_PATH.exists():
        raise FileNotFoundError(f"Не найден config DBNet++: {DBNET_CFG_PATH}")

    # Шаг 2: читаем конфиг и выбираем устройство.
    cfg = OmegaConf.load(DBNET_CFG_PATH)
    device = torch.device(DEVICE if DEVICE == "cpu" or torch.cuda.is_available() else "cpu")

    # Шаг 3: строим модель и загружаем веса.
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
) -> tuple[list[np.ndarray], list[float]]:
    """
    Короткое описание:
        Запускает DBNet++ на RGB-изображении и возвращает полигоны строк.
    Вход:
        image_rgb: Image.Image -- RGB-изображение.
        model: Any -- загруженная DBNet++ модель.
        cfg: Any -- конфиг DBNet++.
        device: torch.device -- устройство инференса.
    Выход:
        tuple[list[np.ndarray], list[float]] -- список полигонов и score.
    """
    # Шаг 1: сжимаем изображение в 640 x 640 через штатный resize + padding.
    tensor, meta = preprocess_image_pil(image_rgb, image_size=DBNET_IMAGE_SIZE)
    tensor = tensor.to(device)

    # Шаг 2: получаем probability map.
    output = model(tensor)
    prob = output["prob"][0, 0].float().cpu().numpy()

    # Шаг 3: декодируем probability map в полигоны.
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
) -> tuple[list[np.ndarray], list[float]]:
    """
    Короткое описание:
        Запускает DBNet++ напрямую на всем исходном изображении.
    Вход:
        image_bgr: np.ndarray -- исходное BGR-изображение.
        model: Any -- DBNet++ модель.
        cfg: Any -- DBNet++ конфиг.
        device: torch.device -- устройство инференса.
    Выход:
        tuple[list[np.ndarray], list[float]] -- полигоны и score.
    """
    # Шаг 1: конвертируем исходное изображение и запускаем DBNet++.
    image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    polygons, scores = predict_dbnet_polygons_on_image(image_rgb, model, cfg, device)
    return polygons, scores


def read_hwr200_labels() -> dict[str, list[dict[str, Any]]]:
    """
    Короткое описание:
        Читает HWR200 labels.txt в словарь относительный путь -> список строк.
    Вход:
        отсутствует.
    Выход:
        dict[str, list[dict[str, Any]]] -- разметка строк.
    """
    # Шаг 1: проверяем файл разметки.
    if not HWR200_LABELS_PATH.exists():
        raise FileNotFoundError(f"Не найден labels.txt: {HWR200_LABELS_PATH}")

    # Шаг 2: читаем PaddleOCR-style json по строкам.
    labels: dict[str, list[dict[str, Any]]] = {}
    with HWR200_LABELS_PATH.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            rel_path, payload = line.split("\t", 1)
            labels[rel_path] = json.loads(payload)
    return labels


def image_path_to_hwr200_rel(image_path: Path) -> str:
    """
    Короткое описание:
        Переводит абсолютный путь изображения к относительному пути внутри HWR200 labels.txt.
    Вход:
        image_path: Path -- путь к изображению.
    Выход:
        str -- относительный путь для labels.txt.
    """
    # Шаг 1: считаем путь относительно корня HWR200/hw_dataset.
    return str(image_path.resolve().relative_to(HWR200_IMAGE_ROOT.resolve())).replace("\\", "/")


def get_gt_polygons(image_path: Path) -> list[np.ndarray]:
    """
    Короткое описание:
        Возвращает полигоны GT-разметки для IMAGE_PATH.
    Вход:
        image_path: Path -- путь к изображению.
    Выход:
        list[np.ndarray] -- GT-полигоны строк.
    """
    # Шаг 1: читаем labels.txt и ищем текущую картинку.
    labels = read_hwr200_labels()
    rel_path = image_path_to_hwr200_rel(image_path)
    if rel_path not in labels:
        raise KeyError(f"В labels.txt нет разметки для {rel_path}")

    # Шаг 2: достаем points каждой строки.
    polygons: list[np.ndarray] = []
    for item in labels[rel_path]:
        if "points" not in item:
            continue
        polygons.append(np.asarray(item["points"], dtype=np.float32))
    return polygons


def draw_polygons_overlay(
    image_bgr: np.ndarray,
    polygons: list[np.ndarray],
    scores: Optional[list[float]],
    output_path: Path,
    color: tuple[int, int, int],
    title: str,
    thickness: int,
) -> None:
    """
    Короткое описание:
        Сохраняет визуализацию полигонов поверх изображения.
    Вход:
        image_bgr: np.ndarray -- исходное изображение.
        polygons: list[np.ndarray] -- полигоны.
        scores: Optional[list[float]] -- score каждого полигона или None.
        output_path: Path -- путь сохранения.
        color: tuple[int, int, int] -- BGR-цвет.
        title: str -- заголовок.
        thickness: int -- толщина линий.
    Выход:
        None
    """
    # Шаг 1: рисуем полигоны.
    visual = image_bgr.copy()
    for polygon_index, polygon in enumerate(polygons):
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            visual,
            [points],
            isClosed=True,
            color=color,
            thickness=thickness,
        )

        # Шаг 2: подписываем score около верхней точки полигона.
        if scores is not None and DRAW_SCORES and polygon_index < len(scores):
            flat_points = points.reshape(-1, 2)
            label_x = int(np.min(flat_points[:, 0]))
            label_y = int(max(20, np.min(flat_points[:, 1]) - 5))
            cv2.putText(
                visual,
                f"{scores[polygon_index]:.2f}",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    # Шаг 3: добавляем общий заголовок и сохраняем.
    cv2.putText(
        visual,
        title,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        color,
        3,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), visual)


def draw_gt_vs_dbnet_overlay(
    image_bgr: np.ndarray,
    gt_polygons: list[np.ndarray],
    dbnet_polygons: list[np.ndarray],
    output_path: Path,
) -> None:
    """
    Короткое описание:
        Сохраняет совместную визуализацию GT и DBNet++.
    Вход:
        image_bgr: np.ndarray -- исходное изображение.
        gt_polygons: list[np.ndarray] -- GT-полигоны.
        dbnet_polygons: list[np.ndarray] -- полигоны DBNet++.
        output_path: Path -- путь сохранения.
    Выход:
        None
    """
    # Шаг 1: рисуем GT зеленым.
    visual = image_bgr.copy()
    for polygon in gt_polygons:
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(visual, [points], isClosed=True, color=GT_POLYGON_COLOR, thickness=GT_POLYGON_THICKNESS)

    # Шаг 2: рисуем DBNet++ красным.
    for polygon in dbnet_polygons:
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(visual, [points], isClosed=True, color=DBNET_POLYGON_COLOR, thickness=DBNET_POLYGON_THICKNESS)

    # Шаг 3: подписываем легенду.
    cv2.putText(visual, f"GT green: {len(gt_polygons)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, GT_POLYGON_COLOR, 3, cv2.LINE_AA)
    cv2.putText(visual, f"DBNet red: {len(dbnet_polygons)}", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, DBNET_POLYGON_COLOR, 3, cv2.LINE_AA)
    cv2.imwrite(str(output_path), visual)


def save_prediction_json(
    image_path: Path,
    gt_polygons: list[np.ndarray],
    polygons: list[np.ndarray],
    scores: list[float],
    output_path: Path,
) -> None:
    """
    Короткое описание:
        Сохраняет координаты DBNet++ полигонов в json.
    Вход:
        image_path: Path -- путь исходного изображения.
        gt_polygons: list[np.ndarray] -- полигоны GT-разметки.
        polygons: list[np.ndarray] -- полигоны строк.
        scores: list[float] -- score каждого полигона.
        output_path: Path -- путь json-файла.
    Выход:
        None
    """
    # Шаг 1: собираем json-структуру.
    payload = {
        "image_path": str(image_path),
        "dbnet_image_size": DBNET_IMAGE_SIZE,
        "gt_count": int(len(gt_polygons)),
        "dbnet_count": int(len(polygons)),
        "gt": [
            {
                "points": np.asarray(polygon, dtype=float).round(2).tolist(),
            }
            for polygon in gt_polygons
        ],
        "detections": [
            {
                "score": float(scores[index]) if index < len(scores) else None,
                "points": np.asarray(polygon, dtype=float).round(2).tolist(),
            }
            for index, polygon in enumerate(polygons)
        ],
    }

    # Шаг 2: сохраняем json.
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def main() -> None:
    """
    Короткое описание:
        Запускает DBNet++ на IMAGE_PATH и сохраняет debug-визуализацию.
    Вход:
        None
    Выход:
        None
    """
    # Шаг 1: создаем debug-папку и читаем изображение.
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {IMAGE_PATH}")

    # Шаг 2: сохраняем ровно исходное изображение и читаем GT-разметку.
    cv2.imwrite(str(DEBUG_OUTPUT_DIR / "00_input.jpg"), image_bgr)
    gt_polygons = get_gt_polygons(IMAGE_PATH)

    # Шаг 3: грузим DBNet++ и запускаем предсказание на исходном изображении.
    model, cfg, device = load_dbnet_model()
    try:
        polygons, scores = predict_dbnet_polygons(image_bgr, model, cfg, device)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Шаг 4: сохраняем визуализацию и json.
    draw_polygons_overlay(
        image_bgr=image_bgr,
        polygons=polygons,
        scores=scores,
        output_path=DEBUG_OUTPUT_DIR / "dbnet_lines.jpg",
        color=DBNET_POLYGON_COLOR,
        title=f"DBNet++ lines: {len(polygons)}",
        thickness=DBNET_POLYGON_THICKNESS,
    )
    draw_polygons_overlay(
        image_bgr=image_bgr,
        polygons=gt_polygons,
        scores=None,
        output_path=DEBUG_OUTPUT_DIR / "gt_lines.jpg",
        color=GT_POLYGON_COLOR,
        title=f"GT lines: {len(gt_polygons)}",
        thickness=GT_POLYGON_THICKNESS,
    )
    draw_gt_vs_dbnet_overlay(
        image_bgr=image_bgr,
        gt_polygons=gt_polygons,
        dbnet_polygons=polygons,
        output_path=DEBUG_OUTPUT_DIR / "gt_vs_dbnet_lines.jpg",
    )
    save_prediction_json(
        image_path=IMAGE_PATH,
        gt_polygons=gt_polygons,
        polygons=polygons,
        scores=scores,
        output_path=DEBUG_OUTPUT_DIR / "dbnet_lines.json",
    )

    print(f"[OK] DBNet++ detections: {len(polygons)}")
    print(f"[OK] GT lines: {len(gt_polygons)}")
    print(f"[OK] Debug: {DEBUG_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
