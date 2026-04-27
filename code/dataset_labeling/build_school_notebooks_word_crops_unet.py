from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT: Path = Path("/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from school_notebooks_RU import CocoMaskGenerator
from u_net_binarization import binarize_image_with_loaded_model, load_unet_model


# Корень исходного датасета с изображениями и COCO-разметкой.
SOURCE_DATASET_ROOT: Path = PROJECT_ROOT / "datasets/school_notebooks_RU"
SOURCE_IMAGES_DIR: Path = SOURCE_DATASET_ROOT / "images_base"
ANNOTATION_PATHS: Tuple[Path, ...] = (
    SOURCE_DATASET_ROOT / "annotations_train.json",
    SOURCE_DATASET_ROOT / "annotations_val.json",
    SOURCE_DATASET_ROOT / "annotations_test.json",
)

# Итоговый датасет бинаризованных вырезок слов.
OUTPUT_DATASET_ROOT: Path = PROJECT_ROOT / "datasets/school_notebooks_words_binary_unet_100"
OUTPUT_IMAGES_DIR: Path = OUTPUT_DATASET_ROOT / "images"
OUTPUT_DEBUG_DIR: Path = OUTPUT_DATASET_ROOT / "debug"
OUTPUT_DEBUG_OVERLAY_DIR: Path = OUTPUT_DEBUG_DIR / "overlays"
OUTPUT_DEBUG_CROP_DIR: Path = OUTPUT_DEBUG_DIR / "crops"
OUTPUT_MANIFEST_PATH: Path = OUTPUT_DATASET_ROOT / "manifest.json"
OUTPUT_LABELS_JSONL_PATH: Path = OUTPUT_DATASET_ROOT / "labels.jsonl"

# Путь к весам U-Net для бинаризации.
UNET_MODEL_PATH: Path = PROJECT_ROOT / "models/u_net/unet_binarization_3_(6-architecture).pth"

# Категории слов и коротких текстовых объектов в school_notebooks_RU.
WORD_CATEGORY_IDS: Tuple[int, ...] = (0, 1, 2)
WORD_CATEGORY_NAMES: Dict[int, str] = {
    0: "pupil_text",
    1: "pupil_comment",
    2: "teacher_comment",
}

# Сколько слов нужно сохранить в этом запуске.
MAX_WORD_CROPS: int = 1000

# Сколько первых примеров сохраняем в debug.
DEBUG_CROPS_COUNT: int = 30
DEBUG_OVERLAY_IMAGES_COUNT: int = 10

# Паддинг вокруг bbox слова в пикселях.
CROP_PADDING_PX: int = 12

# Минимальный размер bbox слова до паддинга.
MIN_WORD_WIDTH_PX: int = 8
MIN_WORD_HEIGHT_PX: int = 8

# Размер входа U-Net. Чем больше размер, тем лучше детализация и выше расход памяти.
UNET_TARGET_SIZE: Tuple[int, int] = (3000, 3000)

# Порог U-Net после sigmoid.
UNET_THRESHOLD: float = 0.5

# Цвета debug-разметки в BGR.
DEBUG_WORD_COLOR: Tuple[int, int, int] = (0, 255, 0)
DEBUG_TEXT_COLOR: Tuple[int, int, int] = (0, 0, 255)

# Число колонок в debug-листе вырезанных слов.
CONTACT_SHEET_COLUMNS: int = 5
CONTACT_SHEET_CELL_SIZE: Tuple[int, int] = (220, 120)


def prepare_output_dirs() -> None:
    """
    Короткое описание:
        Пересоздает папку итогового датасета слов.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    # Шаг 1: удаляем только папку результата этого скрипта.
    if OUTPUT_DATASET_ROOT.exists():
        shutil.rmtree(OUTPUT_DATASET_ROOT)

    # Шаг 2: создаем структуру датасета и debug-папки.
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DEBUG_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DEBUG_CROP_DIR.mkdir(parents=True, exist_ok=True)


def segmentation_to_polygon(annotation: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Короткое описание:
        Преобразует COCO segmentation аннотации в полигон.
    Вход:
        annotation (Dict[str, Any]): COCO-аннотация слова.
    Выход:
        Optional[np.ndarray]: полигон формы N x 2 или None.
    """
    segmentation = annotation.get("segmentation")
    if not segmentation or not segmentation[0]:
        return None

    # Шаг 1: берем первый полигон, потому в этом датасете текстовые элементы размечены одним контуром.
    points = np.array(segmentation[0], dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
        return None
    return points


def clipped_bbox_from_polygon(
    polygon: np.ndarray,
    image_width: int,
    image_height: int,
    padding_px: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Короткое описание:
        Строит bbox полигона с паддингом и ограничением границами изображения.
    Вход:
        polygon (np.ndarray): полигон слова формы N x 2.
        image_width (int): ширина изображения.
        image_height (int): высота изображения.
        padding_px (int): паддинг вокруг bbox.
    Выход:
        Optional[Tuple[int, int, int, int]]: bbox x1, y1, x2, y2 или None.
    """
    x, y, width, height = cv2.boundingRect(polygon.astype(np.int32))
    if width < MIN_WORD_WIDTH_PX or height < MIN_WORD_HEIGHT_PX:
        return None

    # Шаг 1: расширяем bbox и не выходим за границы изображения.
    x1 = max(0, x - padding_px)
    y1 = max(0, y - padding_px)
    x2 = min(image_width, x + width + padding_px)
    y2 = min(image_height, y + height + padding_px)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_binary_word(binary_image: np.ndarray, polygon: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Короткое описание:
        Вырезает бинарное слово и делает фон вне полигона белым.
    Вход:
        binary_image (np.ndarray): бинаризованная страница, 0 текст и 255 фон.
        polygon (np.ndarray): полигон слова в координатах страницы.
        bbox (Tuple[int, int, int, int]): bbox слова x1, y1, x2, y2.
    Выход:
        np.ndarray: бинарный crop слова.
    """
    x1, y1, x2, y2 = bbox
    crop = binary_image[y1:y2, x1:x2].copy()
    crop = np.where(crop < 128, 0, 255).astype(np.uint8)

    # Шаг 1: создаем локальную маску полигона и стираем все, что лежит вне слова.
    local_polygon = (polygon - np.array([x1, y1], dtype=np.float32)).astype(np.int32)
    polygon_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [local_polygon], 255)
    crop[polygon_mask == 0] = 255
    return crop


def save_debug_crop(
    crop: np.ndarray,
    original_image: np.ndarray,
    polygon: np.ndarray,
    bbox: Tuple[int, int, int, int],
    output_stem: str,
) -> None:
    """
    Короткое описание:
        Сохраняет debug-картинки для одного вырезанного слова.
    Вход:
        crop (np.ndarray): бинарная вырезка слова.
        original_image (np.ndarray): исходное BGR-изображение страницы.
        polygon (np.ndarray): полигон слова в координатах страницы.
        bbox (Tuple[int, int, int, int]): bbox слова x1, y1, x2, y2.
        output_stem (str): имя примера без расширения.
    Выход:
        отсутствует.
    """
    x1, y1, x2, y2 = bbox
    source_crop = original_image[y1:y2, x1:x2].copy()
    local_polygon = (polygon - np.array([x1, y1], dtype=np.float32)).astype(np.int32)
    cv2.polylines(source_crop, [local_polygon], True, DEBUG_WORD_COLOR, 2)

    # Шаг 1: сохраняем рядом исходный фрагмент с полигоном и бинарный результат.
    cv2.imwrite(str(OUTPUT_DEBUG_CROP_DIR / f"{output_stem}_source.jpg"), source_crop)
    cv2.imwrite(str(OUTPUT_DEBUG_CROP_DIR / f"{output_stem}_binary.png"), crop)


def save_debug_overlay(
    image: np.ndarray,
    polygons: List[np.ndarray],
    image_name: str,
    overlay_index: int,
) -> None:
    """
    Короткое описание:
        Сохраняет страницу с наложенными полигонами слов, которые были из нее вырезаны.
    Вход:
        image (np.ndarray): исходное BGR-изображение.
        polygons (List[np.ndarray]): список полигонов слов.
        image_name (str): имя исходного изображения.
        overlay_index (int): номер debug-страницы.
    Выход:
        отсутствует.
    """
    debug_image = image.copy()

    # Шаг 1: рисуем все сохраненные слова на исходной странице.
    for word_index, polygon in enumerate(polygons):
        cv2.polylines(debug_image, [polygon.astype(np.int32)], True, DEBUG_WORD_COLOR, 3)
        point = polygon.astype(np.int32)[0]
        cv2.putText(
            debug_image,
            str(word_index),
            (int(point[0]), int(point[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            DEBUG_TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

    safe_name = Path(image_name).stem.replace("/", "_")
    output_path = OUTPUT_DEBUG_OVERLAY_DIR / f"overlay_{overlay_index:03d}_{safe_name}.jpg"
    cv2.imwrite(str(output_path), debug_image)


def build_contact_sheet(crop_paths: List[Path]) -> None:
    """
    Короткое описание:
        Собирает общий debug-лист первых бинарных вырезок слов.
    Вход:
        crop_paths (List[Path]): пути к бинарным crop-файлам.
    Выход:
        отсутствует.
    """
    if not crop_paths:
        return

    cell_width, cell_height = CONTACT_SHEET_CELL_SIZE
    columns = CONTACT_SHEET_COLUMNS
    rows = int(np.ceil(len(crop_paths) / float(columns)))
    sheet = np.full((rows * cell_height, columns * cell_width), 255, dtype=np.uint8)

    # Шаг 1: вписываем каждую вырезку в фиксированную ячейку без изменения пропорций.
    for idx, crop_path in enumerate(crop_paths):
        crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
        if crop is None:
            continue

        height, width = crop.shape[:2]
        scale = min((cell_width - 20) / max(width, 1), (cell_height - 20) / max(height, 1), 1.0)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)

        row = idx // columns
        col = idx % columns
        y0 = row * cell_height + (cell_height - resized_height) // 2
        x0 = col * cell_width + (cell_width - resized_width) // 2
        sheet[y0:y0 + resized_height, x0:x0 + resized_width] = resized

    cv2.imwrite(str(OUTPUT_DEBUG_DIR / "contact_sheet_binary_words.png"), sheet)


def collect_word_annotations(generator: CocoMaskGenerator, image_id: int) -> List[Dict[str, Any]]:
    """
    Короткое описание:
        Собирает аннотации слов для одного изображения.
    Вход:
        generator (CocoMaskGenerator): COCO-генератор текущего сплита.
        image_id (int): идентификатор изображения.
    Выход:
        List[Dict[str, Any]]: список аннотаций категорий WORD_CATEGORY_IDS.
    """
    annotations: List[Dict[str, Any]] = []

    # Шаг 1: объединяем ученический текст, ученические комментарии и комментарии учителя.
    for category_id in WORD_CATEGORY_IDS:
        annotations.extend(generator.get_annotations(image_id, category_id))
    return annotations


def process_one_image(
    generator: CocoMaskGenerator,
    split_name: str,
    image_id: int,
    file_name: str,
    model: Any,
    device: Any,
    saved_count: int,
    debug_crop_paths: List[Path],
    overlay_count: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Короткое описание:
        Бинаризует одно изображение U-Net и вырезает из него слова до лимита датасета.
    Вход:
        generator (CocoMaskGenerator): COCO-генератор текущего сплита.
        split_name (str): имя сплита train/val/test.
        image_id (int): идентификатор изображения.
        file_name (str): имя файла изображения.
        model (Any): загруженная U-Net модель.
        device (Any): torch device для U-Net.
        saved_count (int): сколько слов уже сохранено до этого изображения.
        debug_crop_paths (List[Path]): список путей первых debug-crop.
        overlay_count (int): сколько overlay-страниц уже сохранено.
    Выход:
        Tuple[List[Dict[str, Any]], int]: metadata сохраненных слов и новый overlay_count.
    """
    image_path = SOURCE_IMAGES_DIR / file_name
    image = cv2.imread(str(image_path))
    if image is None:
        return [], overlay_count

    annotations = collect_word_annotations(generator, image_id)
    if not annotations:
        return [], overlay_count

    # Шаг 1: бинаризуем всю страницу один раз.
    binary_image = binarize_image_with_loaded_model(
        image=image,
        model=model,
        device=device,
        target_size=UNET_TARGET_SIZE,
        threshold=UNET_THRESHOLD,
        debug=False,
    )

    image_height, image_width = binary_image.shape[:2]
    saved_items: List[Dict[str, Any]] = []
    saved_polygons_for_overlay: List[np.ndarray] = []

    # Шаг 2: вырезаем слова по полигонам до общего лимита.
    for annotation in annotations:
        if saved_count + len(saved_items) >= MAX_WORD_CROPS:
            break

        polygon = segmentation_to_polygon(annotation)
        if polygon is None:
            continue

        bbox = clipped_bbox_from_polygon(
            polygon=polygon,
            image_width=image_width,
            image_height=image_height,
            padding_px=CROP_PADDING_PX,
        )
        if bbox is None:
            continue

        crop = crop_binary_word(binary_image=binary_image, polygon=polygon, bbox=bbox)
        output_index = saved_count + len(saved_items)
        output_stem = f"word_{output_index:06d}"
        output_path = OUTPUT_IMAGES_DIR / f"{output_stem}.png"
        cv2.imwrite(str(output_path), crop)

        category_id = int(annotation.get("category_id", -1))
        attributes = annotation.get("attributes") or {}
        translation = attributes.get("translation", "")

        if output_index < DEBUG_CROPS_COUNT:
            save_debug_crop(
                crop=crop,
                original_image=image,
                polygon=polygon,
                bbox=bbox,
                output_stem=output_stem,
            )
            debug_crop_paths.append(output_path)

        saved_polygons_for_overlay.append(polygon)
        saved_items.append(
            {
                "image": str(output_path.relative_to(OUTPUT_DATASET_ROOT)),
                "source_split": split_name,
                "source_image": str(image_path),
                "source_file_name": file_name,
                "source_image_id": int(image_id),
                "category_id": category_id,
                "category_name": WORD_CATEGORY_NAMES.get(category_id, "unknown"),
                "translation": str(translation),
                "bbox_xyxy": list(map(int, bbox)),
                "polygon": polygon.round(3).tolist(),
            }
        )

    # Шаг 3: сохраняем overlay для первых страниц, из которых реально взяли слова.
    if saved_polygons_for_overlay and overlay_count < DEBUG_OVERLAY_IMAGES_COUNT:
        save_debug_overlay(
            image=image,
            polygons=saved_polygons_for_overlay,
            image_name=file_name,
            overlay_index=overlay_count,
        )
        overlay_count += 1

    return saved_items, overlay_count


def write_outputs(items: List[Dict[str, Any]]) -> None:
    """
    Короткое описание:
        Сохраняет manifest.json и labels.jsonl итогового датасета.
    Вход:
        items (List[Dict[str, Any]]): список metadata вырезанных слов.
    Выход:
        отсутствует.
    """
    manifest = {
        "dataset_root": str(OUTPUT_DATASET_ROOT),
        "source_dataset_root": str(SOURCE_DATASET_ROOT),
        "source_images_dir": str(SOURCE_IMAGES_DIR),
        "annotation_paths": [str(path) for path in ANNOTATION_PATHS],
        "unet_model_path": str(UNET_MODEL_PATH),
        "max_word_crops": MAX_WORD_CROPS,
        "word_category_ids": list(WORD_CATEGORY_IDS),
        "word_category_names": WORD_CATEGORY_NAMES,
        "count": len(items),
        "items": items,
    }
    OUTPUT_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Шаг 1: jsonl удобен для обучения OCR/классификаторов, где одна строка равна одному crop.
    with OUTPUT_LABELS_JSONL_PATH.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    """
    Короткое описание:
        Создает датасет из 100 бинаризованных U-Net вырезок слов school_notebooks_RU.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    if not SOURCE_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка изображений: {SOURCE_IMAGES_DIR}")
    if not UNET_MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найдены веса U-Net: {UNET_MODEL_PATH}")

    prepare_output_dirs()
    model, device = load_unet_model(model_path=str(UNET_MODEL_PATH))

    items: List[Dict[str, Any]] = []
    debug_crop_paths: List[Path] = []
    overlay_count = 0

    try:
        # Шаг 1: идем по train/val/test, пока не наберем MAX_WORD_CROPS слов.
        for annotation_path in ANNOTATION_PATHS:
            if len(items) >= MAX_WORD_CROPS:
                break
            if not annotation_path.exists():
                continue

            split_name = annotation_path.stem.replace("annotations_", "")
            generator = CocoMaskGenerator(str(annotation_path))
            image_records = sorted(generator.image_info.items(), key=lambda pair: pair[0])

            progress = tqdm(image_records, desc=f"Words {split_name}", unit="image")
            for image_id, image_info in progress:
                if len(items) >= MAX_WORD_CROPS:
                    break

                file_name = image_info.get("file_name")
                if not file_name:
                    continue

                new_items, overlay_count = process_one_image(
                    generator=generator,
                    split_name=split_name,
                    image_id=int(image_id),
                    file_name=str(file_name),
                    model=model,
                    device=device,
                    saved_count=len(items),
                    debug_crop_paths=debug_crop_paths,
                    overlay_count=overlay_count,
                )
                items.extend(new_items)
                progress.set_postfix({"saved": len(items)})

        build_contact_sheet(debug_crop_paths)
        write_outputs(items)
    finally:
        del model

    print(f"[OK] Сохранено слов: {len(items)}")
    print(f"[OK] Датасет: {OUTPUT_DATASET_ROOT}")
    print(f"[OK] Manifest: {OUTPUT_MANIFEST_PATH}")
    print(f"[OK] Debug: {OUTPUT_DEBUG_DIR}")


if __name__ == "__main__":
    main()
