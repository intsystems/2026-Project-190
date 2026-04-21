from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from shutil import copy2
from typing import Dict, Iterable, List, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Основные пути и категории.
SOURCE_HW_DATASET = Path(
    "/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code/datasets/HWR200/hw_dataset"
)
TARGET_IMAGES_ROOT = Path(
    "/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code/datasets/HWR200/images"
)
CATEGORY_NAMES: Tuple[str, ...] = ("Сканы", "ФотоСветлое", "ФотоТемное", "Тесты")
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> ArgumentParser:
    """
    Короткое описание:
        Формирует и возвращает аргументы командной строки.
    Вход:
        Нет.
    Выход:
        ArgumentParser - распарсенные аргументы для запуска скрипта.
    """
    parser = ArgumentParser(
        description="Копирует фото и тексты HWR200 в единую структуру images/ с переименованием.")
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_HW_DATASET,
        help="Путь к исходной папке hw_dataset.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=TARGET_IMAGES_ROOT,
        help="Путь к целевой папке images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать, что будет скопировано, без фактического копирования.",
    )
    return parser.parse_args()


def collect_writer_dirs(source_root: Path) -> List[Path]:
    """
    Короткое описание:
        Находит папки авторов первого уровня в hw_dataset.
    Вход:
        source_root: Path - путь к исходной папке hw_dataset.
    Выход:
        List[Path] - отсортированный список папок авторов.
    """
    writer_dirs = [path for path in source_root.iterdir() if path.is_dir()]
    writer_dirs.sort(key=lambda path: path.name)
    return writer_dirs


def ensure_target_dirs(target_root: Path, dry_run: bool) -> None:
    """
    Короткое описание:
        Создает целевые подпапки категорий в images.
    Вход:
        target_root: Path - корень для выходных данных.
        dry_run: bool - если True, папки не создаются.
    Выход:
        None.
    """
    if dry_run:
        return

    for category_name in CATEGORY_NAMES:
        (target_root / category_name).mkdir(parents=True, exist_ok=True)


def safe_copy(src_path: Path, dst_path: Path, dry_run: bool) -> None:
    """
    Короткое описание:
        Копирует файл с защитой от перезаписи через суффикс _dupN.
    Вход:
        src_path: Path - исходный файл.
        dst_path: Path - целевой файл.
        dry_run: bool - если True, копирование не выполняется.
    Выход:
        None.
    """
    final_dst_path = dst_path
    duplicate_index = 1

    while final_dst_path.exists():
        final_dst_path = dst_path.with_name(
            f"{dst_path.stem}_dup{duplicate_index}{dst_path.suffix}")
        duplicate_index += 1

    if not dry_run:
        copy2(src_path, final_dst_path)


def iter_doc_text_files(writer_dir: Path) -> Iterable[Path]:
    """
    Короткое описание:
        Перебирает текстовые файлы документов автора.
    Вход:
        writer_dir: Path - папка конкретного автора в hw_dataset.
    Выход:
        Iterable[Path] - итератор по файлам *.txt.
    """
    for text_path in sorted(writer_dir.glob("*.txt")):
        if text_path.is_file():
            yield text_path


def copy_text_and_images_for_document(
    writer_id: str,
    doc_text_path: Path,
    writer_dir: Path,
    target_root: Path,
    dry_run: bool,
) -> Dict[str, int]:
    """
    Короткое описание:
        Копирует текст и изображения одного документа в целевую структуру.
    Вход:
        writer_id: str - идентификатор автора (имя папки).
        doc_text_path: Path - путь к файлу текста документа.
        writer_dir: Path - путь к папке автора.
        target_root: Path - корень папки images.
        dry_run: bool - режим без фактического копирования.
    Выход:
        Dict[str, int] - статистика копирования по одному документу.
    """
    doc_name = doc_text_path.stem
    base_name = f"{writer_id}_{doc_name}"

    stats: Dict[str, int] = {
        "texts": 0,
        "scan_images": 0,
        "light_images": 0,
        "dark_images": 0,
        "test_images": 0,
    }

    # Копирование текста в папку Тесты с именем writer_doc.txt
    target_text_path = target_root / "Тесты" / f"{base_name}.txt"
    safe_copy(doc_text_path, target_text_path, dry_run=dry_run)
    stats["texts"] += 1

    document_dir = writer_dir / doc_name
    if not document_dir.exists() or not document_dir.is_dir():
        return stats

    category_to_stat_key: Dict[str, str] = {
        "Сканы": "scan_images",
        "ФотоСветлое": "light_images",
        "ФотоТемное": "dark_images",
        "Тесты": "test_images",
    }

    for category_name in CATEGORY_NAMES:
        source_category_dir = document_dir / category_name
        if not source_category_dir.exists() or not source_category_dir.is_dir():
            continue

        for image_path in sorted(source_category_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            target_image_name = f"{base_name}_{image_path.name}"
            target_image_path = target_root / category_name / target_image_name
            safe_copy(image_path, target_image_path, dry_run=dry_run)
            stats[category_to_stat_key[category_name]] += 1

    return stats


def main() -> None:
    """
    Короткое описание:
        Копирует HWR200 в новую структуру images с нужным форматом имен файлов.
    Вход:
        Нет.
    Выход:
        None - печатает итоговую статистику после обработки.
    """
    args = parse_args()
    source_root: Path = args.source
    target_root: Path = args.target
    dry_run: bool = args.dry_run

    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"Исходная папка не найдена: {source_root}")

    ensure_target_dirs(target_root=target_root, dry_run=dry_run)

    total_stats: Dict[str, int] = {
        "writers": 0,
        "documents": 0,
        "texts": 0,
        "scan_images": 0,
        "light_images": 0,
        "dark_images": 0,
        "test_images": 0,
    }

    writer_dirs = collect_writer_dirs(source_root)

    for writer_dir in tqdm(writer_dirs, desc="Авторы", unit="author"):
        writer_id = writer_dir.name
        text_files = list(iter_doc_text_files(writer_dir))

        if not text_files:
            continue

        total_stats["writers"] += 1

        for doc_text_path in tqdm(text_files, desc=f"Документы {writer_id}", unit="doc", leave=False):
            doc_stats = copy_text_and_images_for_document(
                writer_id=writer_id,
                doc_text_path=doc_text_path,
                writer_dir=writer_dir,
                target_root=target_root,
                dry_run=dry_run,
            )

            total_stats["documents"] += 1
            for key in ("texts", "scan_images", "light_images", "dark_images", "test_images"):
                total_stats[key] += doc_stats[key]

    print("\nИтоговая статистика:")
    print(f"  Авторов обработано: {total_stats['writers']}")
    print(f"  Документов обработано: {total_stats['documents']}")
    print(f"  Текстов скопировано: {total_stats['texts']}")
    print(f"  Сканов скопировано: {total_stats['scan_images']}")
    print(f"  ФотоСветлое скопировано: {total_stats['light_images']}")
    print(f"  ФотоТемное скопировано: {total_stats['dark_images']}")
    print(f"  Тестовых изображений скопировано: {total_stats['test_images']}")
    print(f"  Режим dry-run: {dry_run}")
    print(f"  Папка результата: {target_root}")


if __name__ == "__main__":
    main()
