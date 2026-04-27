from pathlib import Path
import json
import colorsys
from PIL import Image, ImageDraw, ImageFont


DATASET_ROOT = Path("/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code/datasets/HWR200")
IMAGE_ROOT = DATASET_ROOT / "hw_dataset"
LABELS_TXT = DATASET_ROOT / "labels.txt"
OUTPUT_DIR = Path("/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code/debug_images/hwr200_labels")
TARGET_IMAGE_NAME = "/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code/datasets/HWR200/images/ФотоТемное/53_reuse2_2.jpg"  # Имя конкретного изображения или относительный путь из labels.txt.
LINE_WIDTH = 3  # Толщина линий рамок.
FONT_SIZE = 42  # Размер подписи score.


def _color_for(i: int, n: int) -> tuple[int, int, int]:
    h = i / max(n, 1)
    r, g, b = colorsys.hsv_to_rgb(h, 0.9, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def _load_font(size: int = 42):
    return ImageFont.load_default(size=size)


def render_boxes(
    image_path: Path | str,
    boxes: list[dict],
    line_width: int = 3,
    font_size: int = 42,
) -> Image.Image:
    """Draw polygons + scores on image, return the PIL.Image (not saved)."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)

    for i, b in enumerate(boxes):
        pts = [(float(x), float(y)) for x, y in b["points"]]
        color = _color_for(i, len(boxes))
        draw.polygon(pts, outline=color, width=line_width)

        score = b.get("score")
        if score is None:
            continue
        label = f"{score:.2f}"
        x, y = pts[0]
        tx, ty = x, max(0, y - font_size - 4)
        bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((tx, ty), label, fill="black", font=font)

    return img


def show_from_labels(
    labels_txt: Path | str,
    dataset_root: Path | str,
    idx: int | None = None,
    rel_path: str | None = None,
    line_width: int = 3,
    font_size: int = 42,
) -> Image.Image:
    """
    Отрисовать одну картинку из labels.txt. Вернуть PIL.Image.

    Args:
        labels_txt:   путь к labels.txt
        dataset_root: корень датасета (относительно него пути в labels.txt)
        idx:          номер строки в labels.txt (0-based). Используется если rel_path=None.
        rel_path:     относительный путь из labels.txt — альтернатива idx.
    """
    labels_txt = Path(labels_txt)
    dataset_root = Path(dataset_root)

    with open(labels_txt, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    target = None
    if rel_path is not None:
        for line in lines:
            rel, payload = line.split("\t", 1)
            if rel == rel_path:
                target = (rel, payload)
                break
        if target is None:
            raise ValueError(f"rel_path {rel_path!r} not found in {labels_txt}")
    else:
        if idx is None:
            raise ValueError("Provide either idx or rel_path")
        if not (0 <= idx < len(lines)):
            raise IndexError(f"idx {idx} out of range [0, {len(lines)})")
        target = lines[idx].split("\t", 1)

    rel, payload = target
    boxes = json.loads(payload)
    print(f"{rel}  ({len(boxes)} boxes)")

    return render_boxes(
        dataset_root / rel,
        boxes,
        line_width=line_width,
        font_size=font_size,
    )


def make_label_candidates(image_name: str) -> list[str]:
    """
    Короткое описание:
        Строит возможные относительные пути labels.txt из полного пути, имени файла или flat-имени HWR200.
    Вход:
        image_name (str): путь к изображению, относительный путь или имя вида 53_reuse2_2.jpg.
    Выход:
        list[str]: список возможных относительных путей внутри hw_dataset.
    """
    # Шаг 1: нормализуем строку и убираем случайный завершающий обратный слеш из консоли.
    clean_name = image_name.strip().rstrip("\\").replace("\\", "/")
    target_path = Path(clean_name)
    candidates: list[str] = []

    # Шаг 2: если путь уже внутри hw_dataset, переводим его в относительный путь labels.txt.
    try:
        candidates.append(str(target_path.resolve().relative_to(IMAGE_ROOT.resolve())).replace("\\", "/"))
    except ValueError:
        pass

    # Шаг 3: если путь внутри старой папки images, восстанавливаем структуру hw_dataset.
    images_root = DATASET_ROOT / "images"
    try:
        rel_to_images = target_path.resolve().relative_to(images_root.resolve())
        if len(rel_to_images.parts) >= 2:
            photo_mode = rel_to_images.parts[0]
            stem_parts = Path(rel_to_images.parts[-1]).stem.split("_")
            if len(stem_parts) >= 3:
                doc_id = stem_parts[0]
                page_id = stem_parts[-1]
                variant = "_".join(stem_parts[1:-1])
                candidates.append(f"{doc_id}/{variant}/{photo_mode}/{page_id}.jpg")
    except ValueError:
        pass

    # Шаг 4: если дано просто flat-имя, пробуем восстановить путь для всех режимов съемки.
    stem_parts = target_path.stem.split("_")
    if len(stem_parts) >= 3:
        doc_id = stem_parts[0]
        page_id = stem_parts[-1]
        variant = "_".join(stem_parts[1:-1])
        for photo_mode in ("ФотоТемное", "ФотоСветлое", "Сканы"):
            candidates.append(f"{doc_id}/{variant}/{photo_mode}/{page_id}.jpg")

    # Шаг 5: добавляем исходную строку как есть для точного относительного пути из labels.txt.
    candidates.append(clean_name)
    return list(dict.fromkeys(candidates))


def find_label_record(labels_txt: Path, image_name: str) -> tuple[int, str]:
    """
    Короткое описание:
        Находит одну строку labels.txt по относительному пути, имени файла или stem изображения.
    Вход:
        labels_txt (Path): путь к labels.txt.
        image_name (str): относительный путь, имя файла или stem изображения.
    Выход:
        tuple[int, str]: индекс строки в labels.txt и относительный путь изображения.
    """
    # Шаг 1: читаем все непустые строки разметки.
    with open(labels_txt, encoding="utf-8") as file:
        lines = [line.rstrip("\n") for line in file if line.strip()]

    # Шаг 2: ищем сначала точное совпадение с относительным путем.
    candidates = make_label_candidates(image_name)
    label_index_by_rel = {}
    loose_matches = []
    target_path = Path(image_name.strip().rstrip("\\").replace("\\", "/"))
    target_name = target_path.name
    target_stem = target_path.stem

    for idx, line in enumerate(lines):
        rel_path = line.split("\t", 1)[0]
        label_index_by_rel[rel_path] = idx
        rel_obj = Path(rel_path)
        if rel_obj.name == target_name or rel_obj.stem == target_stem:
            loose_matches.append((idx, rel_path))

    for candidate in candidates:
        if candidate in label_index_by_rel:
            return label_index_by_rel[candidate], candidate

    # Шаг 3: если нашли несколько файлов с одинаковым именем, просим указать относительный путь.
    if len(loose_matches) == 1:
        return loose_matches[0]
    if len(loose_matches) > 1:
        examples = "\n".join(rel_path for _, rel_path in loose_matches[:10])
        raise ValueError(
            "Найдено несколько изображений с таким именем. "
            f"Укажи TARGET_IMAGE_NAME как относительный путь из labels.txt.\n{examples}"
        )

    raise FileNotFoundError(f"Не найдено изображение в labels.txt: {image_name}")


def main() -> None:
    """
    Короткое описание:
        Сохраняет визуализацию разметки HWR200 для одного заданного изображения.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    if not LABELS_TXT.exists():
        raise FileNotFoundError(f"Не найден файл разметки: {LABELS_TXT}")
    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(f"Не найден корень изображений: {IMAGE_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Шаг 1: находим ровно одну строку разметки для указанного изображения.
    idx, rel_path = find_label_record(LABELS_TXT, TARGET_IMAGE_NAME)
    image_path = IMAGE_ROOT / rel_path
    if ".ipynb_checkpoints" in rel_path or not image_path.exists():
        raise FileNotFoundError(f"Изображение из labels.txt не найдено или служебное: {image_path}")

    # Шаг 2: рисуем и сохраняем только один результат.
    image = show_from_labels(
        labels_txt=LABELS_TXT,
        dataset_root=IMAGE_ROOT,
        idx=idx,
        line_width=LINE_WIDTH,
        font_size=FONT_SIZE,
    )
    safe_name = rel_path.replace("/", "_").replace("\\", "_")
    output_path = OUTPUT_DIR / f"hwr200_labels_{safe_name}"
    image.save(output_path)
    print(f"[OK] {output_path}")


if __name__ == "__main__":
    main()
