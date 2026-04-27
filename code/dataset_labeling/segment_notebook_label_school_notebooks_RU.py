import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from school_notebooks_RU import CocoMaskGenerator


URLS: Dict[str, str] = {
    "images": "datasets/school_notebooks_RU/images_base",
    "train_data": "datasets/school_notebooks_RU/annotations_train.json",
    "val_data": "datasets/school_notebooks_RU/annotations_val.json",
    "test_data": "datasets/school_notebooks_RU/annotations_test.json",
}

OUTPUT_DIR: str = "datasets/school_notebooks_RU/images_segment_notebook"
NOTEBOOK_CATEGORY_ID: int = 3  # Категория paper в COCO-разметке.
YOLO_CLASS_ID: int = 0  # Единственный класс YOLO: notebook/page.
MIN_POLYGON_POINTS: int = 3  # Минимум точек для валидного полигона сегментации.


def annotation_to_polygons(annotation: dict) -> List[np.ndarray]:
    """
    Короткое описание:
        Преобразует COCO segmentation одной аннотации в список полигонов.
    Вход:
        annotation (dict): COCO-аннотация с полем segmentation.
    Выход:
        List[np.ndarray]: список полигонов формы N x 2.
    """
    polygons: List[np.ndarray] = []
    segmentation = annotation.get("segmentation")
    if not segmentation:
        return polygons

    # Шаг 1: берем каждый полигон из COCO segmentation.
    for polygon in segmentation:
        if len(polygon) < MIN_POLYGON_POINTS * 2:
            continue
        points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] >= MIN_POLYGON_POINTS:
            polygons.append(points)
    return polygons


def polygon_to_yolo_line(
    polygon: np.ndarray,
    image_width: int,
    image_height: int,
    class_id: int = YOLO_CLASS_ID,
) -> str:
    """
    Короткое описание:
        Преобразует полигон в строку YOLO-seg формата.
    Вход:
        polygon (np.ndarray): полигон формы N x 2 в пикселях.
        image_width (int): ширина изображения.
        image_height (int): высота изображения.
        class_id (int): номер класса YOLO.
    Выход:
        str: строка YOLO-seg вида class x1 y1 x2 y2 ...
    """
    normalized = polygon.copy().astype(np.float32)
    normalized[:, 0] = np.clip(normalized[:, 0], 0, image_width - 1) / float(image_width)
    normalized[:, 1] = np.clip(normalized[:, 1], 0, image_height - 1) / float(image_height)

    # Шаг 1: разворачиваем точки в плоский список координат.
    coords = " ".join(f"{value:.6f}" for value in normalized.reshape(-1))
    return f"{class_id} {coords}"


def process_one_image(generator: CocoMaskGenerator, image_id: int, file_name: str) -> int:
    """
    Короткое описание:
        Создает YOLO-seg txt-разметку страниц для одного изображения.
    Вход:
        generator (CocoMaskGenerator): объект для чтения COCO-аннотаций.
        image_id (int): идентификатор изображения в COCO.
        file_name (str): имя изображения.
    Выход:
        int: количество сохраненных полигонов страниц.
    """
    image_info = generator.image_info.get(image_id, {})
    image_width = int(image_info.get("width", 0))
    image_height = int(image_info.get("height", 0))
    if image_width <= 0 or image_height <= 0:
        image_height, image_width = generator.get_image_shape(image_id)

    yolo_lines: List[str] = []
    annotations = generator.get_annotations(image_id, NOTEBOOK_CATEGORY_ID)

    # Шаг 1: переводим все paper-полигоны в YOLO-seg.
    for annotation in annotations:
        for polygon in annotation_to_polygons(annotation):
            yolo_lines.append(polygon_to_yolo_line(polygon, image_width, image_height))

    # Шаг 2: сохраняем txt. Пустой файл означает, что страниц для изображения нет.
    output_path = Path(OUTPUT_DIR) / f"{Path(file_name).stem}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        if yolo_lines:
            file.write("\n".join(yolo_lines) + "\n")

    return len(yolo_lines)


def process_split(split_key: str) -> None:
    """
    Короткое описание:
        Создает YOLO-seg разметку страниц для одного COCO-сплита.
    Вход:
        split_key (str): ключ сплита в URLS.
    Выход:
        отсутствует.
    """
    generator = CocoMaskGenerator(URLS[split_key])
    image_items = sorted(generator.image_info.items(), key=lambda item: item[0])
    total_polygons = 0

    # Шаг 1: последовательно обрабатываем изображения, чтобы не копить данные в RAM.
    for image_id, info in tqdm(image_items, desc=f"Segment {split_key}", unit="image"):
        file_name = info.get("file_name")
        if not file_name:
            continue
        total_polygons += process_one_image(generator, image_id, file_name)

    print(f"[OK] {split_key}: сохранено полигонов страниц {total_polygons}")


def main() -> None:
    """
    Короткое описание:
        Создает YOLO-seg разметку тетрадных страниц для train/val/test.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Шаг 1: обрабатываем все COCO-сплиты.
    for split_key in ("train_data", "val_data", "test_data"):
        process_split(split_key)


if __name__ == "__main__":
    main()
