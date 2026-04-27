from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


# Корень проекта, от которого считаются все пути датасетов.
PROJECT_ROOT: Path = Path("/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code")

# Первый исходный YOLO-seg датасет страниц, построенный по HWR200.
HWR200_DATASET_ROOT: Path = PROJECT_ROOT / "datasets/HWR200_page_seg_yolo"
HWR200_IMAGES_DIR: Path = HWR200_DATASET_ROOT / "images/train"
HWR200_LABELS_DIR: Path = HWR200_DATASET_ROOT / "labels/train"

# Второй исходный датасет: изображения лежат отдельно от YOLO-seg txt-разметки.
SCHOOL_IMAGES_DIR: Path = PROJECT_ROOT / "datasets/school_notebooks_RU/images_base"
SCHOOL_LABELS_DIR: Path = PROJECT_ROOT / "datasets/school_notebooks_RU/images_segment_notebook"

# Итоговый объединенный датасет для notebook-а с background augmentation.
OUTPUT_DATASET_ROOT: Path = PROJECT_ROOT / "datasets/hwr200_school_page_seg_yolo"
OUTPUT_IMAGES_DIR: Path = OUTPUT_DATASET_ROOT / "images"
OUTPUT_LABELS_DIR: Path = OUTPUT_DATASET_ROOT / "labels"
OUTPUT_MANIFEST_PATH: Path = OUTPUT_DATASET_ROOT / "manifest.json"
OUTPUT_DATA_YAML_PATH: Path = OUTPUT_DATASET_ROOT / "data.yaml"
OUTPUT_ZIP_PATH: Path = PROJECT_ROOT / "datasets/hwr200_school_page_seg_yolo.zip"

# Единственный класс YOLO segmentation.
YOLO_CLASS_ID: int = 0
YOLO_CLASS_NAME: str = "page"

# Расширения изображений, которые считаются валидными.
IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
    ".TIF",
    ".TIFF",
)


def prepare_output_dataset() -> None:
    """
    Короткое описание:
        Пересоздает папку итогового датасета и удаляет старый zip этого датасета.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    # Шаг 1: удаляем только папку и zip, которые полностью принадлежат этому скрипту.
    if OUTPUT_DATASET_ROOT.exists():
        shutil.rmtree(OUTPUT_DATASET_ROOT)
    if OUTPUT_ZIP_PATH.exists():
        OUTPUT_ZIP_PATH.unlink()

    # Шаг 2: создаем чистую структуру плоского датасета images/labels.
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)


def validate_label_file(label_path: Path) -> bool:
    """
    Короткое описание:
        Проверяет, что файл YOLO-seg содержит хотя бы одну валидную строку.
    Вход:
        label_path (Path): путь к txt-файлу разметки.
    Выход:
        bool: True, если файл можно копировать в датасет.
    """
    if not label_path.exists() or not label_path.is_file():
        return False

    # Шаг 1: проверяем минимальный формат class x1 y1 x2 y2 x3 y3.
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 7 and len(parts[1:]) % 2 == 0:
            return True
    return False


def build_image_index(images_dir: Path) -> Dict[str, Path]:
    """
    Короткое описание:
        Строит индекс изображений по stem, чтобы связать school txt и image.
    Вход:
        images_dir (Path): папка с исходными изображениями.
    Выход:
        Dict[str, Path]: словарь stem изображения -> путь к изображению.
    """
    image_index: Dict[str, Path] = {}

    # Шаг 1: берем первое изображение для каждого stem, дубли с тем же stem пропускаем.
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix not in IMAGE_EXTENSIONS:
            continue
        image_index.setdefault(image_path.stem, image_path)
    return image_index


def copy_pair(image_path: Path, label_path: Path, output_stem: str, source_name: str) -> Dict[str, str]:
    """
    Короткое описание:
        Копирует одну пару image/label в итоговый датасет с новым уникальным именем.
    Вход:
        image_path (Path): исходный путь изображения.
        label_path (Path): исходный путь YOLO-seg txt-разметки.
        output_stem (str): новое имя без расширения.
        source_name (str): имя источника для manifest.json.
    Выход:
        Dict[str, str]: запись manifest для скопированной пары.
    """
    output_image_name = f"{output_stem}{image_path.suffix.lower()}"
    output_label_name = f"{output_stem}.txt"
    output_image_path = OUTPUT_IMAGES_DIR / output_image_name
    output_label_path = OUTPUT_LABELS_DIR / output_label_name

    # Шаг 1: копируем данные без изменения координат, потому размеры изображений не меняются.
    shutil.copy2(image_path, output_image_path)
    shutil.copy2(label_path, output_label_path)

    return {
        "source": source_name,
        "source_image": str(image_path),
        "source_label": str(label_path),
        "image": str(output_image_path.relative_to(OUTPUT_DATASET_ROOT)),
        "label": str(output_label_path.relative_to(OUTPUT_DATASET_ROOT)),
    }


def copy_hwr200_pairs(start_index: int) -> Tuple[List[Dict[str, str]], int, int]:
    """
    Короткое описание:
        Копирует пары image/label из HWR200_page_seg_yolo.
    Вход:
        start_index (int): начальный номер для имен итоговых файлов.
    Выход:
        Tuple[List[Dict[str, str]], int, int]: manifest-записи, следующий индекс, число пропусков.
    """
    items: List[Dict[str, str]] = []
    skipped_count = 0
    output_index = start_index

    # Шаг 1: идем по label-файлам, чтобы в датасет попали только размеченные изображения.
    label_paths = sorted(HWR200_LABELS_DIR.glob("*.txt"))
    for label_path in tqdm(label_paths, desc="Copy HWR200", unit="pair"):
        if not validate_label_file(label_path):
            skipped_count += 1
            continue

        candidates = sorted(HWR200_IMAGES_DIR.glob(f"{label_path.stem}.*"))
        image_path = next((path for path in candidates if path.suffix in IMAGE_EXTENSIONS), None)
        if image_path is None:
            skipped_count += 1
            continue

        items.append(
            copy_pair(
                image_path=image_path,
                label_path=label_path,
                output_stem=f"hwr200_{output_index:06d}",
                source_name="HWR200_page_seg_yolo",
            )
        )
        output_index += 1

    return items, output_index, skipped_count


def copy_school_pairs(start_index: int) -> Tuple[List[Dict[str, str]], int, int]:
    """
    Короткое описание:
        Копирует пары image/label из school_notebooks_RU.
    Вход:
        start_index (int): начальный номер для имен итоговых файлов.
    Выход:
        Tuple[List[Dict[str, str]], int, int]: manifest-записи, следующий индекс, число пропусков.
    """
    items: List[Dict[str, str]] = []
    skipped_count = 0
    output_index = start_index
    image_index = build_image_index(SCHOOL_IMAGES_DIR)

    # Шаг 1: связываем каждый txt с одноименным изображением из images_base.
    label_paths = sorted(SCHOOL_LABELS_DIR.glob("*.txt"))
    for label_path in tqdm(label_paths, desc="Copy school", unit="pair"):
        if not validate_label_file(label_path):
            skipped_count += 1
            continue

        image_path = image_index.get(label_path.stem)
        if image_path is None:
            skipped_count += 1
            continue

        items.append(
            copy_pair(
                image_path=image_path,
                label_path=label_path,
                output_stem=f"school_{output_index:06d}",
                source_name="school_notebooks_RU",
            )
        )
        output_index += 1

    return items, output_index, skipped_count


def write_data_yaml() -> None:
    """
    Короткое описание:
        Сохраняет YAML-конфиг YOLO для плоского датасета.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    # Шаг 1: оставляем train/val на images, потому notebook дальше сам делает разбиение.
    yaml_text = (
        f"path: {OUTPUT_DATASET_ROOT}\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        f"  {YOLO_CLASS_ID}: {YOLO_CLASS_NAME}\n"
    )
    OUTPUT_DATA_YAML_PATH.write_text(yaml_text, encoding="utf-8")


def write_manifest(items: List[Dict[str, str]], skipped_by_source: Dict[str, int]) -> None:
    """
    Короткое описание:
        Сохраняет JSON с полным списком скопированных пар.
    Вход:
        items (List[Dict[str, str]]): записи обо всех парах image/label.
        skipped_by_source (Dict[str, int]): число пропущенных записей по источникам.
    Выход:
        отсутствует.
    """
    payload = {
        "dataset_root": str(OUTPUT_DATASET_ROOT),
        "zip_path": str(OUTPUT_ZIP_PATH),
        "class_id": YOLO_CLASS_ID,
        "class_name": YOLO_CLASS_NAME,
        "total_count": len(items),
        "skipped_by_source": skipped_by_source,
        "sources": {
            "HWR200_page_seg_yolo": {
                "images": str(HWR200_IMAGES_DIR),
                "labels": str(HWR200_LABELS_DIR),
            },
            "school_notebooks_RU": {
                "images": str(SCHOOL_IMAGES_DIR),
                "labels": str(SCHOOL_LABELS_DIR),
            },
        },
        "items": items,
    }
    OUTPUT_MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def zip_dataset() -> None:
    """
    Короткое описание:
        Упаковывает объединенный датасет в zip-файл.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    files = sorted(path for path in OUTPUT_DATASET_ROOT.rglob("*") if path.is_file())

    # Шаг 1: кладем в архив папку верхнего уровня, чтобы после распаковки было понятное имя датасета.
    with zipfile.ZipFile(OUTPUT_ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in tqdm(files, desc="Zip dataset", unit="file"):
            archive_name = Path(OUTPUT_DATASET_ROOT.name) / path.relative_to(OUTPUT_DATASET_ROOT)
            zip_file.write(path, archive_name)


def main() -> None:
    """
    Короткое описание:
        Объединяет HWR200_page_seg_yolo и school_notebooks_RU в один YOLO-seg датасет.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    prepare_output_dataset()

    # Шаг 1: копируем оба источника в один плоский датасет с уникальными именами.
    hwr_items, next_index, hwr_skipped = copy_hwr200_pairs(start_index=0)
    school_items, _, school_skipped = copy_school_pairs(start_index=0)
    items = hwr_items + school_items

    # Шаг 2: сохраняем служебные файлы и архив.
    write_data_yaml()
    write_manifest(
        items=items,
        skipped_by_source={
            "HWR200_page_seg_yolo": hwr_skipped,
            "school_notebooks_RU": school_skipped,
        },
    )
    zip_dataset()

    print(f"[OK] HWR200 pairs: {len(hwr_items)}")
    print(f"[OK] School pairs: {len(school_items)}")
    print(f"[OK] Total pairs: {len(items)}")
    print(f"[OK] Dataset: {OUTPUT_DATASET_ROOT}")
    print(f"[OK] Zip: {OUTPUT_ZIP_PATH}")


if __name__ == "__main__":
    main()
