import os
import sys
import tempfile
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
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

OUTPUT_DIR: str = "datasets/school_notebooks_RU/images_detect_lines"
LINE_CATEGORY_ID: int = 4
SOURCE_CATEGORY_IDS: List[int] = [0, 1, 2]
YOLO_CLASS_ID: int = 0
PIXEL_PADDING: int = 25
DEBUG: bool = False
DEBUG_SHOW_COUNT: int = 0
TMP_FILES_DIR: Path = Path("tmp_files/detect_lines_yolo")
TMP_DEBUG_DIR: Path = TMP_FILES_DIR / "debug_preview"
TEMP_FILE_COUNTER = itertools.count()


def pad_polygon_pixels(
    polygon: np.ndarray,
    image_shape: Tuple[int, int],
    padding_px: int,
    temp_dir: Path,
) -> np.ndarray:
    """
    Короткое описание:
        Расширяет полигон в пикселях через бинарную маску и дилатацию.
    Вход:
        polygon (np.ndarray): Полигон формы (N, 2), где N >= 3.
        image_shape (Tuple[int, int]): Размер изображения (height, width).
        padding_px (int): Радиус пиксельного расширения.
    Выход:
        np.ndarray: Расширенный полигон формы (M, 2).
    """
    height, width = image_shape
    if padding_px <= 0:
        return polygon.astype(np.int32)

    file_id = next(TEMP_FILE_COUNTER)
    mask_path = temp_dir / f"pad_mask_{file_id}.dat"
    padded_path = temp_dir / f"pad_dilated_{file_id}.dat"

    try:
        mask = np.memmap(str(mask_path), dtype=np.uint8, mode="w+", shape=(height, width))
        padded_mask = np.memmap(str(padded_path), dtype=np.uint8, mode="w+", shape=(height, width))
        mask[:] = 0
        padded_mask[:] = 0

        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
        kernel_size = 2 * int(padding_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cv2.dilate(mask, kernel, dst=padded_mask, iterations=1)

        contours, _ = cv2.findContours(np.asarray(padded_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return polygon.astype(np.int32)
        largest = max(contours, key=cv2.contourArea)
        return largest.reshape(-1, 2).astype(np.int32)
    finally:
        try:
            del mask
            del padded_mask
        except Exception:
            pass
        if mask_path.exists():
            mask_path.unlink()
        if padded_path.exists():
            padded_path.unlink()


def polygon_intersection_area(poly_a: np.ndarray, poly_b: np.ndarray, temp_dir: Path) -> float:
    """
    Короткое описание:
        Считает площадь пересечения двух полигонов в пикселях.
    Вход:
        poly_a (np.ndarray): Первый полигон формы (N, 2).
        poly_b (np.ndarray): Второй полигон формы (M, 2).
    Выход:
        float: Площадь пересечения.
    """
    all_pts = np.vstack([poly_a, poly_b]).astype(np.int32)
    x, y, w, h = cv2.boundingRect(all_pts)
    if w <= 0 or h <= 0:
        return 0.0

    file_id = next(TEMP_FILE_COUNTER)
    mask_a_path = temp_dir / f"inter_a_{file_id}.dat"
    mask_b_path = temp_dir / f"inter_b_{file_id}.dat"

    try:
        mask_a = np.memmap(str(mask_a_path), dtype=np.uint8, mode="w+", shape=(h, w))
        mask_b = np.memmap(str(mask_b_path), dtype=np.uint8, mode="w+", shape=(h, w))
        mask_a[:] = 0
        mask_b[:] = 0

        cv2.fillPoly(mask_a, [poly_a - np.array([x, y])], 1)
        cv2.fillPoly(mask_b, [poly_b - np.array([x, y])], 1)
        cv2.bitwise_and(mask_a, mask_b, dst=mask_a)
        return float(cv2.countNonZero(np.asarray(mask_a)))
    finally:
        try:
            del mask_a
            del mask_b
        except Exception:
            pass
        if mask_a_path.exists():
            mask_a_path.unlink()
        if mask_b_path.exists():
            mask_b_path.unlink()


def assign_sources_to_lines(
    line_polygons: List[np.ndarray],
    source_polygons: List[np.ndarray],
    temp_dir: Path,
) -> Dict[int, List[np.ndarray]]:
    """
    Короткое описание:
        Назначает каждый source-полигон строке с максимальной площадью пересечения.
    Вход:
        line_polygons (List[np.ndarray]): Список полигонов строк с паддингом.
        source_polygons (List[np.ndarray]): Список исходных полигонов категорий источников.
    Выход:
        Dict[int, List[np.ndarray]]: Индекс строки -> список назначенных source-полигонов.
    """
    assignments: Dict[int, List[np.ndarray]] = {idx: [] for idx in range(len(line_polygons))}
    for source_poly in source_polygons:
        best_idx = -1
        best_area = 0.0
        for line_idx, line_poly in enumerate(line_polygons):
            inter_area = polygon_intersection_area(source_poly, line_poly, temp_dir)
            if inter_area > best_area:
                best_area = inter_area
                best_idx = line_idx
        if best_idx >= 0 and best_area > 0.0:
            assignments[best_idx].append(source_poly)
    return assignments


def extract_segments_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """
    Короткое описание:
        Извлекает внешние контуры из бинарной маски.
    Вход:
        mask (np.ndarray): Бинарная маска 0/255.
    Выход:
        List[np.ndarray]: Список контуров формы (K, 2).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments: List[np.ndarray] = []
    for cnt in contours:
        points = cnt.reshape(-1, 2)
        if points.shape[0] >= 3:
            segments.append(points.astype(np.int32))
    return segments


def to_yolo_seg_line(polygon: np.ndarray, img_w: int, img_h: int, class_id: int) -> str:
    """
    Короткое описание:
        Преобразует полигон в строку формата YOLO-seg.
    Вход:
        polygon (np.ndarray): Полигон формы (N, 2).
        img_w (int): Ширина изображения.
        img_h (int): Высота изображения.
        class_id (int): Идентификатор класса.
    Выход:
        str: Строка формата YOLO-seg.
    """
    coords: List[str] = []
    for x, y in polygon:
        coords.append(f"{float(x) / float(img_w):.6f}")
        coords.append(f"{float(y) / float(img_h):.6f}")
    return f"{class_id} " + " ".join(coords)


def build_source_polygons(generator: CocoMaskGenerator, image_id: int) -> List[np.ndarray]:
    """
    Короткое описание:
        Собирает полигоны источников из нескольких категорий.
    Вход:
        generator (CocoMaskGenerator): Генератор аннотаций COCO.
        image_id (int): Идентификатор изображения.
    Выход:
        List[np.ndarray]: Список source-полигонов.
    """
    source_polygons: List[np.ndarray] = []
    for cat_id in SOURCE_CATEGORY_IDS:
        source_polygons.extend(generator.get_polygons_by_category(image_id, cat_id))
    return source_polygons


def process_one_image(
    generator: CocoMaskGenerator,
    image_id: int,
    file_name: str,
    collect_debug: bool,
) -> Tuple[bool, Optional[str]]:
    """
    Короткое описание:
        Обрабатывает одно изображение: строит YOLO-seg разметку и сохраняет .txt.
    Вход:
        generator (CocoMaskGenerator): Генератор аннотаций COCO.
        image_id (int): Идентификатор изображения.
        file_name (str): Имя файла изображения.
        collect_debug (bool): Флаг, нужно ли вернуть отладочную визуализацию.
    Выход:
        Tuple[bool, np.ndarray]: Флаг наличия debug-картинки и сама debug-картинка.
    """
    with tempfile.TemporaryDirectory(dir=str(TMP_FILES_DIR), prefix=f"{Path(file_name).stem}_") as tmp_name:
        temp_dir = Path(tmp_name)
        image_path = os.path.join(URLS["images"], file_name)
        image = cv2.imread(image_path)
        if image is None:
            return False, None

        img_h, img_w = image.shape[:2]
        line_polygons = generator.get_polygons_by_category(image_id, LINE_CATEGORY_ID)
        if not line_polygons:
            out_txt = os.path.join(OUTPUT_DIR, f"{Path(file_name).stem}.txt")
            with open(out_txt, "w", encoding="utf-8"):
                pass
            return False, None

        padded_lines = [
            pad_polygon_pixels(poly, (img_h, img_w), PIXEL_PADDING, temp_dir)
            for poly in line_polygons
        ]
        source_polygons = build_source_polygons(generator, image_id)
        assignments = assign_sources_to_lines(padded_lines, source_polygons, temp_dir)

        yolo_lines: List[str] = []
        debug_image = image.copy() if collect_debug else None
        has_debug = False

        for line_idx, padded_line in enumerate(padded_lines):
            file_id = next(TEMP_FILE_COUNTER)
            union_path = temp_dir / f"union_mask_{file_id}.dat"
            try:
                union_mask = np.memmap(str(union_path), dtype=np.uint8, mode="w+", shape=(img_h, img_w))
                union_mask[:] = 0
                cv2.fillPoly(union_mask, [padded_line], 255)
                for source_poly in assignments.get(line_idx, []):
                    cv2.fillPoly(union_mask, [source_poly.astype(np.int32)], 255)

                segments = extract_segments_from_mask(np.asarray(union_mask))
                for segment in segments:
                    yolo_lines.append(to_yolo_seg_line(segment, img_w, img_h, YOLO_CLASS_ID))
                    if collect_debug and debug_image is not None:
                        cv2.polylines(debug_image, [segment], True, (0, 255, 0), 2)
                        has_debug = True
            finally:
                try:
                    del union_mask
                except Exception:
                    pass
                if union_path.exists():
                    union_path.unlink()

        out_txt = os.path.join(OUTPUT_DIR, f"{Path(file_name).stem}.txt")
        with open(out_txt, "w", encoding="utf-8") as fp:
            if yolo_lines:
                fp.write("\n".join(yolo_lines) + "\n")

        if collect_debug and has_debug and debug_image is not None:
            debug_path = TMP_DEBUG_DIR / f"{Path(file_name).stem}_debug.png"
            cv2.imwrite(str(debug_path), debug_image)
            return True, str(debug_path)

        return False, None


def process_split(split_key: str, debug: bool) -> None:
    """
    Короткое описание:
        Генерирует YOLO-seg разметку для одного сплита.
    Вход:
        split_key (str): Ключ сплита в словаре URLS.
        debug (bool): Флаг отрисовки примеров.
    Выход:
        None.
    """
    generator = CocoMaskGenerator(URLS[split_key])
    debug_images: List[Tuple[str, str]] = []
    image_items = sorted(generator.image_info.items(), key=lambda item: item[0])

    if debug:
        head_items = image_items[:DEBUG_SHOW_COUNT]
        tail_items = image_items[DEBUG_SHOW_COUNT:]

        for image_id, info in tqdm(head_items, desc=f"Label {split_key} debug head", unit="image"):
            file_name = info.get("file_name")
            if not file_name:
                continue
            has_debug, dbg_path = process_one_image(generator, image_id, file_name, collect_debug=True)
            if has_debug:
                debug_images.append((file_name, str(dbg_path)))

        for name, dbg_path in debug_images:
            dbg = cv2.imread(dbg_path)
            if dbg is None:
                continue
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.imshow(name, dbg)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(name)
            if key == ord("q"):
                break

        for image_id, info in tqdm(tail_items, desc=f"Label {split_key} tail", unit="image"):
            file_name = info.get("file_name")
            if not file_name:
                continue
            process_one_image(generator, image_id, file_name, collect_debug=False)
    else:
        for image_id, info in tqdm(image_items, desc=f"Label {split_key}", unit="image"):
            file_name = info.get("file_name")
            if not file_name:
                continue
            process_one_image(generator, image_id, file_name, collect_debug=False)


def main() -> None:
    """
    Короткое описание:
        Создает YOLO-seg разметку строк для train/val/test.
    Вход:
        None.
    Выход:
        None.
    """
    TMP_FILES_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split_key in ("train_data", "val_data", "test_data"):
        process_split(split_key, DEBUG)


if __name__ == "__main__":
    main()
