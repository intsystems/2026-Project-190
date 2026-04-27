from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# Корень проекта, от которого считаются все пути ниже.
PROJECT_ROOT: Path = Path("/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code")

# Исходная разметка строк HWR200. В каждой строке лежит путь и JSON со строками текста.
LABELS_TXT: Path = PROJECT_ROOT / "datasets/HWR200/labels.txt"

# Исходный корень изображений, относительно которого записаны пути в labels.txt.
SOURCE_IMAGES_ROOT: Path = PROJECT_ROOT / "datasets/HWR200/hw_dataset"

# Итоговый датасет YOLO segmentation для выделения страницы.
OUTPUT_DATASET_ROOT: Path = PROJECT_ROOT / "datasets/HWR200_page_seg_yolo"

# Папки стандартного YOLO-датасета.
OUTPUT_IMAGES_DIR: Path = OUTPUT_DATASET_ROOT / "images/train"
OUTPUT_LABELS_DIR: Path = OUTPUT_DATASET_ROOT / "labels/train"
OUTPUT_DEBUG_DIR: Path = OUTPUT_DATASET_ROOT / "debug_pages"

# JSON со списком успешно размеченных изображений.
OUTPUT_LABELED_JSON: Path = OUTPUT_DATASET_ROOT / "labeled_images.json"

# YAML-конфиг для обучения Ultralytics YOLO segmentation.
OUTPUT_DATA_YAML: Path = OUTPUT_DATASET_ROOT / "data.yaml"

# Единственный класс сегментации: страница, содержащая все строки.
YOLO_CLASS_ID: int = 0
YOLO_CLASS_NAME: str = "page"

# Сколько валидных изображений размечаем за один запуск.
MAX_IMAGES_TO_LABEL: int = 2000

# Сколько первых успешно размеченных изображений сохраняем с отладочной отрисовкой.
DEBUG_IMAGES_COUNT: int = 50

# Минимальное число строк, по которым можно строить страницу.
MIN_LINE_BOXES_COUNT: int = 1

# Минимальное число точек в одном боксе строки.
MIN_BOX_POINTS_COUNT: int = 4

# Толщина зеленой линии страницы на debug-картинках.
DEBUG_PAGE_LINE_WIDTH: int = 8

# Толщина синей линии исходных строк на debug-картинках.
DEBUG_LINE_WIDTH: int = 2

# Цвет минимального четырехугольника страницы в BGR.
DEBUG_PAGE_COLOR: Tuple[int, int, int] = (0, 255, 0)

# Цвет исходных строк в BGR.
DEBUG_LINE_COLOR: Tuple[int, int, int] = (255, 0, 0)


def prepare_output_dirs() -> None:
    """
    Короткое описание:
        Создает папки итогового YOLO-датасета.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    # Шаг 1: создаем все необходимые директории без удаления старых данных.
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def parse_label_line(raw_line: str) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """
    Короткое описание:
        Разбирает одну строку labels.txt в относительный путь и список боксов строк.
    Вход:
        raw_line (str): одна строка файла labels.txt.
    Выход:
        Optional[Tuple[str, List[Dict[str, Any]]]]: путь и список боксов или None при ошибке.
    """
    # Шаг 1: отделяем путь изображения от JSON-разметки.
    clean_line = raw_line.rstrip("\n")
    if not clean_line.strip() or "\t" not in clean_line:
        return None

    rel_path, payload = clean_line.split("\t", 1)
    try:
        boxes = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if not isinstance(boxes, list):
        return None
    return rel_path, boxes


def collect_line_points(boxes: List[Dict[str, Any]]) -> np.ndarray:
    """
    Короткое описание:
        Собирает все точки строк в один массив для построения общего четырехугольника.
    Вход:
        boxes (List[Dict[str, Any]]): список словарей строк из labels.txt.
    Выход:
        np.ndarray: массив точек формы N x 2 типа float32.
    """
    points: List[List[float]] = []

    # Шаг 1: берем только валидные боксы строк с достаточным числом точек.
    for box in boxes:
        raw_points = box.get("points")
        if not isinstance(raw_points, list) or len(raw_points) < MIN_BOX_POINTS_COUNT:
            continue

        for point in raw_points:
            if not isinstance(point, list) or len(point) != 2:
                continue
            points.append([float(point[0]), float(point[1])])

    return np.array(points, dtype=np.float32)


def build_min_page_quad(points: np.ndarray, image_width: int, image_height: int) -> Optional[np.ndarray]:
    """
    Короткое описание:
        Строит минимальный повернутый четырехугольник, содержащий все точки строк.
    Вход:
        points (np.ndarray): массив точек строк формы N x 2.
        image_width (int): ширина изображения в пикселях.
        image_height (int): высота изображения в пикселях.
    Выход:
        Optional[np.ndarray]: четыре точки страницы формы 4 x 2 или None при невалидных данных.
    """
    if points.shape[0] < MIN_BOX_POINTS_COUNT:
        return None

    # Шаг 1: строим минимальный прямоугольник по всем точкам строк.
    rect = cv2.minAreaRect(points)
    quad = cv2.boxPoints(rect).astype(np.float32)

    # Шаг 2: ограничиваем точки границами изображения.
    quad[:, 0] = np.clip(quad[:, 0], 0, image_width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, image_height - 1)

    area = cv2.contourArea(quad.astype(np.float32))
    if area <= 0.0:
        return None
    return quad


def polygon_to_yolo_seg_line(polygon: np.ndarray, image_width: int, image_height: int) -> str:
    """
    Короткое описание:
        Преобразует полигон страницы в строку YOLO segmentation.
    Вход:
        polygon (np.ndarray): полигон страницы формы N x 2 в пикселях.
        image_width (int): ширина изображения в пикселях.
        image_height (int): высота изображения в пикселях.
    Выход:
        str: строка YOLO-seg формата class x1 y1 x2 y2 ...
    """
    normalized = polygon.astype(np.float32).copy()
    normalized[:, 0] = normalized[:, 0] / float(image_width)
    normalized[:, 1] = normalized[:, 1] / float(image_height)
    coords = " ".join(f"{coord:.6f}" for coord in normalized.reshape(-1))
    return f"{YOLO_CLASS_ID} {coords}"


def make_output_image_name(index: int, source_path: Path) -> str:
    """
    Короткое описание:
        Создает стабильное имя изображения в итоговом датасете.
    Вход:
        index (int): порядковый номер валидного примера.
        source_path (Path): исходный путь изображения.
    Выход:
        str: имя файла изображения для папки images/train.
    """
    # Шаг 1: используем порядковый номер, чтобы избежать конфликтов одинаковых имен.
    suffix = source_path.suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        suffix = ".jpg"
    return f"hwr200_page_{index:06d}{suffix}"


def save_debug_image(image: np.ndarray, line_points: np.ndarray, page_quad: np.ndarray, output_path: Path) -> None:
    """
    Короткое описание:
        Сохраняет изображение с отрисовкой исходных точек строк и итоговой страницы.
    Вход:
        image (np.ndarray): исходное изображение BGR.
        line_points (np.ndarray): все точки строк формы N x 2.
        page_quad (np.ndarray): четырехугольник страницы формы 4 x 2.
        output_path (Path): путь сохранения debug-картинки.
    Выход:
        отсутствует.
    """
    debug_image = image.copy()

    # Шаг 1: отрисовываем выпуклую оболочку строк как ориентир исходной области.
    if line_points.shape[0] >= MIN_BOX_POINTS_COUNT:
        hull = cv2.convexHull(line_points.astype(np.float32)).astype(np.int32)
        cv2.polylines(debug_image, [hull], isClosed=True, color=DEBUG_LINE_COLOR, thickness=DEBUG_LINE_WIDTH)

    # Шаг 2: отрисовываем итоговый минимальный четырехугольник страницы.
    cv2.polylines(
        debug_image,
        [page_quad.astype(np.int32)],
        isClosed=True,
        color=DEBUG_PAGE_COLOR,
        thickness=DEBUG_PAGE_LINE_WIDTH,
    )
    cv2.imwrite(str(output_path), debug_image)


def write_data_yaml() -> None:
    """
    Короткое описание:
        Сохраняет YAML-конфиг датасета для YOLO segmentation.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    # Шаг 1: пишем минимальный конфиг с train-сплитом.
    yaml_text = (
        f"path: {OUTPUT_DATASET_ROOT}\n"
        "train: images/train\n"
        "val: images/train\n"
        "names:\n"
        f"  {YOLO_CLASS_ID}: {YOLO_CLASS_NAME}\n"
    )
    OUTPUT_DATA_YAML.write_text(yaml_text, encoding="utf-8")


def process_record(raw_line: str, output_index: int, debug_index: int) -> Optional[Dict[str, Any]]:
    """
    Короткое описание:
        Обрабатывает одну запись labels.txt и сохраняет изображение, YOLO-label и debug.
    Вход:
        raw_line (str): строка исходного labels.txt.
        output_index (int): номер успешно размеченного изображения.
        debug_index (int): номер debug-картинки среди успешных изображений.
    Выход:
        Optional[Dict[str, Any]]: метаданные сохраненного примера или None при пропуске.
    """
    parsed = parse_label_line(raw_line)
    if parsed is None:
        return None

    rel_path, boxes = parsed
    if ".ipynb_checkpoints" in rel_path or len(boxes) < MIN_LINE_BOXES_COUNT:
        return None

    source_image_path = SOURCE_IMAGES_ROOT / rel_path
    if not source_image_path.exists():
        return None

    image = cv2.imread(str(source_image_path))
    if image is None:
        return None

    image_height, image_width = image.shape[:2]
    line_points = collect_line_points(boxes)
    page_quad = build_min_page_quad(line_points, image_width=image_width, image_height=image_height)
    if page_quad is None:
        return None

    output_image_name = make_output_image_name(output_index, source_image_path)
    output_label_name = f"{Path(output_image_name).stem}.txt"
    output_debug_name = f"{Path(output_image_name).stem}_debug.jpg"

    output_image_path = OUTPUT_IMAGES_DIR / output_image_name
    output_label_path = OUTPUT_LABELS_DIR / output_label_name
    output_debug_path = OUTPUT_DEBUG_DIR / output_debug_name

    # Шаг 1: сохраняем копию изображения и YOLO-seg разметку страницы.
    shutil.copy2(source_image_path, output_image_path)
    output_label_path.write_text(
        polygon_to_yolo_seg_line(page_quad, image_width=image_width, image_height=image_height) + "\n",
        encoding="utf-8",
    )

    # Шаг 2: для первых примеров сохраняем отладочную визуализацию.
    saved_debug_path: Optional[str] = None
    if debug_index < DEBUG_IMAGES_COUNT:
        save_debug_image(image=image, line_points=line_points, page_quad=page_quad, output_path=output_debug_path)
        saved_debug_path = str(output_debug_path.relative_to(OUTPUT_DATASET_ROOT))

    return {
        "source_rel_path": rel_path,
        "source_image_path": str(source_image_path),
        "image": str(output_image_path.relative_to(OUTPUT_DATASET_ROOT)),
        "label": str(output_label_path.relative_to(OUTPUT_DATASET_ROOT)),
        "debug": saved_debug_path,
        "image_width": image_width,
        "image_height": image_height,
        "line_boxes_count": len(boxes),
        "page_polygon_pixels": page_quad.round(3).tolist(),
    }


def save_labeled_json(items: List[Dict[str, Any]]) -> None:
    """
    Короткое описание:
        Сохраняет JSON со списком изображений, которые были размечены.
    Вход:
        items (List[Dict[str, Any]]): метаданные успешно размеченных изображений.
    Выход:
        отсутствует.
    """
    payload = {
        "dataset_root": str(OUTPUT_DATASET_ROOT),
        "source_labels": str(LABELS_TXT),
        "source_images_root": str(SOURCE_IMAGES_ROOT),
        "max_images_to_label": MAX_IMAGES_TO_LABEL,
        "debug_images_count": DEBUG_IMAGES_COUNT,
        "class_id": YOLO_CLASS_ID,
        "class_name": YOLO_CLASS_NAME,
        "labeled_count": len(items),
        "items": items,
    }
    OUTPUT_LABELED_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """
    Короткое описание:
        Создает YOLO-seg датасет HWR200 для сегментации страниц по боксам строк.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    if not LABELS_TXT.exists():
        raise FileNotFoundError(f"Не найден файл разметки: {LABELS_TXT}")
    if not SOURCE_IMAGES_ROOT.exists():
        raise FileNotFoundError(f"Не найден корень изображений: {SOURCE_IMAGES_ROOT}")

    prepare_output_dirs()
    write_data_yaml()

    with LABELS_TXT.open("r", encoding="utf-8") as file:
        raw_lines = file.readlines()

    labeled_items: List[Dict[str, Any]] = []
    skipped_count = 0

    # Шаг 1: идем по labels.txt и набираем первые MAX_IMAGES_TO_LABEL валидных примеров.
    progress = tqdm(raw_lines, desc="Разметка HWR200 pages", unit="record")
    for raw_line in progress:
        if len(labeled_items) >= MAX_IMAGES_TO_LABEL:
            break

        item = process_record(
            raw_line=raw_line,
            output_index=len(labeled_items),
            debug_index=len(labeled_items),
        )
        if item is None:
            skipped_count += 1
            continue

        labeled_items.append(item)
        progress.set_postfix({"saved": len(labeled_items), "skipped": skipped_count})

    save_labeled_json(labeled_items)

    print(f"[OK] Размечено изображений: {len(labeled_items)}")
    print(f"[OK] Пропущено записей: {skipped_count}")
    print(f"[OK] Датасет сохранен: {OUTPUT_DATASET_ROOT}")
    print(f"[OK] JSON со списком: {OUTPUT_LABELED_JSON}")
    print(f"[OK] Debug-картинки: {OUTPUT_DEBUG_DIR}")


if __name__ == "__main__":
    main()
