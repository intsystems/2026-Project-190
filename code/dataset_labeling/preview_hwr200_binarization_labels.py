"""
Короткое описание:
    Строит preview псевдо-разметки HWR200 для обучения бинаризации.
Вход:
    Использует HWR200 images, labels.txt, clean split-файлы и YOLO-модель страницы.
Выход:
    Сохраняет debug-изображения с исходником, масками и бинаризацией.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HWR200_DIR = PROJECT_ROOT / "datasets" / "HWR200"
HWR200_IMAGES_DIR = HWR200_DIR / "hw_dataset"
HWR200_LABELS_TXT = HWR200_DIR / "labels.txt"
TRAIN_SPLIT_PATH = PROJECT_ROOT / "handwriting-recognition" / "splits" / "train_clean.txt"
VAL_SPLIT_PATH = PROJECT_ROOT / "handwriting-recognition" / "splits" / "val_clean.txt"
YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "yolo_detect_notebook" / "yolo_detect_notebook_1_(1-architecture).pt"
OUTPUT_DIR = PROJECT_ROOT / "debug_images" / "hwr200_binarization_preview"

PREVIEW_COUNT = 100  # Сколько примеров сохранить для ручной проверки.
YOLO_CONF = 0.25  # Минимальная уверенность YOLO для поиска страницы.
YOLO_IMGSZ = 640  # Размер входа YOLO. 640 заметно быстрее 1024 для preview.
YOLO_BATCH_SIZE = 4  # Размер батча YOLO. Если не хватает VRAM/RAM, поставить 1 или 2.
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"  # GPU сильно быстрее; при проблемах поставить "cpu".
LINE_POLYGON_PADDING = 8  # Расширение полигона строки перед локальной бинаризацией.
MIN_LINE_SCORE = 0.0  # Минимальный score строки из labels.txt.
MIN_COMPONENT_AREA = 6  # Маленький шум меньше этой площади удаляется.
LINE_LOCAL_OTSU_PARTS = 4  # На сколько вертикальных частей делим каждую строку для локального Otsu.
LOCAL_OTSU_MIN_STD = 2.0  # Если область почти однотонная, Otsu для нее не применяем.
MAX_DEBUG_IMAGE_WIDTH = 1800  # Ограничение ширины debug-монтажа.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def read_split_paths(split_paths: list[Path]) -> list[str]:
    """
    Короткое описание:
        Читает относительные пути изображений из split-файлов.
    Вход:
        split_paths (list[Path]): список split-файлов.
    Выход:
        list[str]: уникальные относительные пути изображений.
    """
    image_paths = []
    seen_paths = set()

    # Шаг 1: объединяем train_clean и val_clean, сохраняя порядок.
    for split_path in split_paths:
        with split_path.open("r", encoding="utf-8") as file:
            for line in file:
                rel_path = line.strip()
                if not rel_path or rel_path in seen_paths:
                    continue
                image_paths.append(rel_path)
                seen_paths.add(rel_path)
    return image_paths


def read_hwr200_line_labels(labels_path: Path) -> dict[str, list[dict]]:
    """
    Короткое описание:
        Загружает labels.txt HWR200 в словарь относительный путь -> полигоны строк.
    Вход:
        labels_path (Path): путь к labels.txt.
    Выход:
        dict[str, list[dict]]: разметка строк для каждого изображения.
    """
    labels = {}

    # Шаг 1: каждая строка содержит путь и JSON со списком line-полигонов.
    with labels_path.open("r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Read HWR200 labels"):
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            rel_path, payload = parts
            try:
                labels[rel_path] = json.loads(payload)
            except json.JSONDecodeError:
                continue
    return labels


def dilate_polygon(points: np.ndarray, padding_px: int, image_shape: tuple[int, int]) -> np.ndarray:
    """
    Короткое описание:
        Расширяет полигон строки через дилатацию бинарной маски.
    Вход:
        points (np.ndarray): точки полигона формы N x 2.
        padding_px (int): расширение в пикселях.
        image_shape (tuple[int, int]): размер изображения в формате (height, width).
    Выход:
        np.ndarray: расширенная маска полигона uint8.
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points.astype(np.int32)], 255)
    if padding_px <= 0:
        return mask

    # Шаг 1: расширяем область строки, чтобы не срезать верхние и нижние части букв.
    kernel_size = padding_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def remove_small_components(binary: np.ndarray) -> np.ndarray:
    """
    Короткое описание:
        Удаляет мелкий черный шум из бинарного изображения.
    Вход:
        binary (np.ndarray): бинарное изображение, где текст черный, фон белый.
    Выход:
        np.ndarray: очищенное бинарное изображение.
    """
    foreground = (binary < 128).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    cleaned = np.full_like(binary, 255)

    # Шаг 1: переносим только достаточно крупные компоненты текста.
    for component_idx in range(1, component_count):
        area = int(stats[component_idx, cv2.CC_STAT_AREA])
        if area >= MIN_COMPONENT_AREA:
            cleaned[labels == component_idx] = 0
    return cleaned


def build_yolo_page_mask(model: YOLO, image: np.ndarray) -> np.ndarray:
    """
    Короткое описание:
        Находит маску страницы через YOLO notebook detector.
    Вход:
        model (YOLO): загруженная YOLO-модель.
        image (np.ndarray): исходное изображение BGR.
    Выход:
        np.ndarray: маска страницы uint8, где страница 255.
    """
    height, width = image.shape[:2]
    page_mask = np.zeros((height, width), dtype=np.uint8)
    result = model.predict(image, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, device=YOLO_DEVICE, verbose=False)[0]

    # Шаг 1: если модель сегментационная, берем самую уверенную маску.
    if result.masks is not None and result.masks.data is not None and len(result.masks.data) > 0:
        scores = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.ones(len(result.masks.data))
        best_idx = int(np.argmax(scores))
        mask = result.masks.data[best_idx].detach().cpu().numpy()
        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        page_mask[mask > 0.5] = 255
        return page_mask

    # Шаг 2: если есть только detector boxes, используем самый уверенный bbox страницы.
    if result.boxes is not None and len(result.boxes) > 0:
        scores = result.boxes.conf.detach().cpu().numpy()
        best_idx = int(np.argmax(scores))
        x1, y1, x2, y2 = result.boxes.xyxy[best_idx].detach().cpu().numpy().astype(int)
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))
        page_mask[y1:y2, x1:x2] = 255
    return page_mask


def build_yolo_page_mask_from_result(result, image_shape: tuple[int, int]) -> np.ndarray:
    """
    Короткое описание:
        Строит маску страницы из уже полученного результата YOLO.
    Вход:
        result: результат Ultralytics для одного изображения.
        image_shape (tuple[int, int]): размер исходного изображения в формате (height, width).
    Выход:
        np.ndarray: маска страницы uint8, где страница 255.
    """
    height, width = image_shape
    page_mask = np.zeros((height, width), dtype=np.uint8)

    # Шаг 1: если модель сегментационная, используем самую уверенную маску.
    if result.masks is not None and result.masks.data is not None and len(result.masks.data) > 0:
        scores = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.ones(len(result.masks.data))
        best_idx = int(np.argmax(scores))
        mask = result.masks.data[best_idx].detach().cpu().numpy()
        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        page_mask[mask > 0.5] = 255
        return page_mask

    # Шаг 2: если есть только bbox, используем самый уверенный bbox.
    if result.boxes is not None and len(result.boxes) > 0:
        scores = result.boxes.conf.detach().cpu().numpy()
        best_idx = int(np.argmax(scores))
        x1, y1, x2, y2 = result.boxes.xyxy[best_idx].detach().cpu().numpy().astype(int)
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))
        page_mask[y1:y2, x1:x2] = 255
    return page_mask


def build_text_region_mask(line_labels: list[dict], image_shape: tuple[int, int], page_mask: np.ndarray) -> np.ndarray:
    """
    Короткое описание:
        Строит маску областей строк по labels.txt и ограничивает ее найденной страницей.
    Вход:
        line_labels (list[dict]): список строковых полигонов HWR200.
        image_shape (tuple[int, int]): размер изображения в формате (height, width).
        page_mask (np.ndarray): маска страницы от YOLO.
    Выход:
        np.ndarray: маска областей строк uint8.
    """
    text_region_mask = np.zeros(image_shape, dtype=np.uint8)

    # Шаг 1: переносим в маску только полигоны строк с достаточным score.
    for item in line_labels:
        score = float(item.get("score", 1.0))
        if score < MIN_LINE_SCORE:
            continue
        points = np.array(item.get("points", []), dtype=np.float32)
        if points.shape[0] < 3:
            continue
        text_region_mask = cv2.bitwise_or(
            text_region_mask,
            dilate_polygon(points, LINE_POLYGON_PADDING, image_shape),
        )

    # Шаг 2: если YOLO нашел страницу, запрещаем текст вне страницы.
    if np.any(page_mask):
        text_region_mask = cv2.bitwise_and(text_region_mask, page_mask)
    return text_region_mask


def get_valid_line_polygons(line_labels: list[dict]) -> list[np.ndarray]:
    """
    Короткое описание:
        Достает валидные полигоны строк из labels.txt с учетом score.
    Вход:
        line_labels (list[dict]): список строковых полигонов HWR200.
    Выход:
        list[np.ndarray]: список полигонов строк формы N x 2.
    """
    polygons = []

    # Шаг 1: оставляем только строки с достаточным score и минимум тремя точками.
    for item in line_labels:
        score = float(item.get("score", 1.0))
        if score < MIN_LINE_SCORE:
            continue
        points = np.array(item.get("points", []), dtype=np.float32)
        if points.shape[0] < 3:
            continue
        polygons.append(points)
    return polygons


def binarize_inside_text_regions(image: np.ndarray, text_region_mask: np.ndarray) -> np.ndarray:
    """
    Короткое описание:
        Делает псевдо-бинаризацию: Otsu внутри строк, все вне строк белое.
    Вход:
        image (np.ndarray): исходное изображение BGR.
        text_region_mask (np.ndarray): маска областей строк.
    Выход:
        np.ndarray: бинарное изображение uint8, где текст черный, фон белый.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    binary = np.full(gray.shape, 255, dtype=np.uint8)
    masked_values = gray[text_region_mask > 0]
    if masked_values.size == 0:
        return binary

    # Шаг 1: считаем общий Otsu только по пикселям внутри строк.
    threshold, _ = cv2.threshold(masked_values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_pixels = (gray < threshold) & (text_region_mask > 0)
    binary[text_pixels] = 0

    # Шаг 2: убираем мелкие шумовые компоненты.
    return remove_small_components(binary)


def binarize_by_local_line_otsu(
    image: np.ndarray,
    line_polygons: list[np.ndarray],
    page_mask: np.ndarray,
) -> np.ndarray:
    """
    Короткое описание:
        Делает локальный Otsu отдельно внутри каждого полигона строки.
    Вход:
        image (np.ndarray): исходное изображение BGR.
        line_polygons (list[np.ndarray]): список полигонов строк.
        page_mask (np.ndarray): маска страницы от YOLO.
    Выход:
        np.ndarray: бинарное изображение uint8, где текст черный, фон белый.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    binary = np.full(gray.shape, 255, dtype=np.uint8)

    # Шаг 1: для каждой строки считаем Otsu отдельно по нескольким вертикальным частям.
    for points in line_polygons:
        line_mask = dilate_polygon(points, LINE_POLYGON_PADDING, gray.shape)
        if np.any(page_mask):
            line_mask = cv2.bitwise_and(line_mask, page_mask)

        if not np.any(line_mask):
            continue

        x_coords = np.where(line_mask > 0)[1]
        x_min = int(x_coords.min())
        x_max = int(x_coords.max()) + 1
        line_width = max(1, x_max - x_min)

        # Шаг 2: режем строку на LINE_LOCAL_OTSU_PARTS частей по x и порожим каждую часть отдельно.
        for part_idx in range(LINE_LOCAL_OTSU_PARTS):
            part_x1 = x_min + int(round(part_idx * line_width / LINE_LOCAL_OTSU_PARTS))
            part_x2 = x_min + int(round((part_idx + 1) * line_width / LINE_LOCAL_OTSU_PARTS))
            if part_x2 <= part_x1:
                continue

            part_mask = np.zeros_like(line_mask)
            part_mask[:, part_x1:part_x2] = line_mask[:, part_x1:part_x2]
            values = gray[part_mask > 0]
            if values.size == 0 or float(np.std(values)) < LOCAL_OTSU_MIN_STD:
                continue

            threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary[(gray < threshold) & (part_mask > 0)] = 0

    # Шаг 3: убираем мелкий шум уже после объединения всех строк.
    return remove_small_components(binary)


def make_debug_grid(
    image: np.ndarray,
    page_mask: np.ndarray,
    text_region_mask: np.ndarray,
    binary: np.ndarray,
) -> np.ndarray:
    """
    Короткое описание:
        Собирает debug-монтаж из исходника, маски страницы, строк и бинаризации.
    Вход:
        image (np.ndarray): исходное изображение BGR.
        page_mask (np.ndarray): маска страницы.
        text_region_mask (np.ndarray): маска областей строк.
        binary (np.ndarray): результат псевдо-бинаризации.
    Выход:
        np.ndarray: BGR debug-монтаж.
    """
    page_vis = cv2.cvtColor(page_mask, cv2.COLOR_GRAY2BGR)
    text_vis = cv2.cvtColor(text_region_mask, cv2.COLOR_GRAY2BGR)
    binary_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Шаг 1: добавляем цветные подсказки к маскам.
    overlay = image.copy()
    overlay[text_region_mask > 0] = (0.65 * overlay[text_region_mask > 0] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[page_mask > 0] = (0.85 * overlay[page_mask > 0] + 0.15 * np.array([255, 0, 0])).astype(np.uint8)

    top = np.hstack([image, overlay])
    bottom = np.hstack([page_vis, text_vis, binary_vis])
    target_width = top.shape[1]
    bottom = cv2.resize(bottom, (target_width, int(bottom.shape[0] * target_width / bottom.shape[1])))
    grid = np.vstack([top, bottom])

    # Шаг 2: уменьшаем очень широкие preview, чтобы их удобно открывать.
    if grid.shape[1] > MAX_DEBUG_IMAGE_WIDTH:
        scale = MAX_DEBUG_IMAGE_WIDTH / grid.shape[1]
        grid = cv2.resize(grid, (MAX_DEBUG_IMAGE_WIDTH, int(grid.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return grid


def main() -> None:
    """
    Короткое описание:
        Сохраняет 100 preview-примеров псевдо-бинаризации HWR200.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найдена YOLO-модель: {YOLO_MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_rel_paths = read_split_paths([TRAIN_SPLIT_PATH, VAL_SPLIT_PATH])
    labels_by_path = read_hwr200_line_labels(HWR200_LABELS_TXT)
    model = YOLO(str(YOLO_MODEL_PATH))

    saved_count = 0
    skipped_count = 0
    batch_items = []

    print(f"YOLO device: {YOLO_DEVICE}, imgsz: {YOLO_IMGSZ}, batch: {YOLO_BATCH_SIZE}")

    def flush_batch() -> None:
        """
        Короткое описание:
            Прогоняет накопленный батч через YOLO и сохраняет preview-бинаризацию.
        Вход:
            отсутствует, использует batch_items из main.
        Выход:
            отсутствует.
        """
        nonlocal saved_count, skipped_count, batch_items
        if not batch_items:
            return

        # Шаг 1: YOLO работает батчем, чтобы не гонять модель по одному изображению.
        images = [item["image"] for item in batch_items]
        results = model.predict(
            images,
            conf=YOLO_CONF,
            imgsz=YOLO_IMGSZ,
            batch=YOLO_BATCH_SIZE,
            device=YOLO_DEVICE,
            verbose=False,
        )

        # Шаг 2: постобработку делаем последовательно, потому что она легкая относительно YOLO.
        for item, result in zip(batch_items, results):
            rel_path = item["rel_path"]
            image = item["image"]
            page_mask = build_yolo_page_mask_from_result(result, image.shape[:2])
            line_polygons = get_valid_line_polygons(labels_by_path[rel_path])
            text_region_mask = build_text_region_mask(labels_by_path[rel_path], image.shape[:2], page_mask)
            binary = binarize_by_local_line_otsu(image, line_polygons, page_mask)
            debug_grid = make_debug_grid(image, page_mask, text_region_mask, binary)

            safe_name = rel_path.replace("/", "__").replace(" ", "_")
            output_path = OUTPUT_DIR / f"{saved_count:03d}_{safe_name}.jpg"
            cv2.imwrite(str(output_path), debug_grid)
            saved_count += 1

            if saved_count >= PREVIEW_COUNT:
                break

        batch_items = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Шаг 1: идем по чистым split-файлам и сохраняем первые PREVIEW_COUNT удачных примеров.
    for rel_path in tqdm(image_rel_paths, desc="Preview HWR200 binarization"):
        if saved_count >= PREVIEW_COUNT:
            break
        if rel_path not in labels_by_path:
            skipped_count += 1
            continue
        if Path(rel_path).suffix not in IMAGE_EXTENSIONS:
            skipped_count += 1
            continue

        image_path = HWR200_IMAGES_DIR / rel_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped_count += 1
            continue

        batch_items.append({"rel_path": rel_path, "image": image})
        if len(batch_items) < YOLO_BATCH_SIZE:
            continue

        flush_batch()

    # Шаг 2: сохраняем хвост батча, если он остался.
    flush_batch()

    print(f"[OK] saved={saved_count}, skipped={skipped_count}, output={OUTPUT_DIR}")


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
