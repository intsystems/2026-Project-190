"""
Короткое описание:
    Генерирует синтетические страницы с текстовыми строками и подробной разметкой строк.
Вход:
    Слова берутся из бинарных word-crops school_notebooks_words_binary_unet_100.
Выход:
    Сохраняет изображения страниц, маски, JSON-разметку и debug-визуализации.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORD_CROPS_DATASET_DIR = PROJECT_ROOT / "datasets" / "school_notebooks_words_binary_unet_100"
WORD_CROPS_IMAGES_DIR = WORD_CROPS_DATASET_DIR / "images"
WORD_CROPS_LABELS_PATH = WORD_CROPS_DATASET_DIR / "labels.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "synthetic_text_pages"
IMAGES_DIR = OUTPUT_DIR / "images"
MASKS_DIR = OUTPUT_DIR / "masks"
ANNOTATIONS_DIR = OUTPUT_DIR / "annotations"
DEBUG_DIR = OUTPUT_DIR / "debug"

SYNTHETIC_PAGE_COUNT = 1000  # Сколько синтетических страниц сгенерировать.
DEBUG_SAVE_COUNT = 20  # Сколько первых debug-оверлеев сохранить для быстрой проверки качества.
RANDOM_SEED = 42  # Фиксирует генерацию для воспроизводимого debug.
PAGE_SIZE = (1400, 1900)  # Размер страницы в формате (width, height).
PAGE_BACKGROUND_VALUE = 255  # Цвет белого фона страницы.
TEXT_BASE_COLORS_BGR = ((25, 25, 25), (35, 35, 170), (120, 35, 35))  # Черный, красный, синий в BGR.
TEXT_WORD_COLOR_JITTER = 16  # Небольшое изменение оттенка от слова к слову.
WORD_TARGET_HEIGHT_RANGE = (24, 38)  # Высота реального word-crop после масштабирования.
WORD_SCALE_JITTER_RANGE = (0.88, 1.12)  # Небольшой шум масштаба слова внутри строки.
MIN_WORD_BLACK_PIXELS = 16  # Минимум черных пикселей, чтобы crop не был пустым.
MAX_WORD_TRIM_PADDING = 3  # Сколько белого поля оставлять вокруг слова после обрезки.
LINE_COUNT_RANGE = (33, 64)  # Диапазон числа строк на странице.
MIN_PLACED_LINES = 20  # Если строк меньше, страница считается неудачной и генерируется заново.
PAGE_GENERATION_ATTEMPTS = 8  # Число попыток пересобрать страницу, чтобы не получить белый лист.
TEXT_BOX_MARGIN_X_PERCENT_RANGE = (0.01, 0.05)  # Белые поля слева и справа.
TEXT_BOX_MARGIN_Y_PERCENT_RANGE = (0.01, 0.05)  # Белые поля сверху и снизу.
LINE_ANGLE_RANGE = (-3.0, 3.0)  # Небольшой наклон каждой строки в градусах.
LINE_JITTER_Y_FACTOR = 0.08  # Вертикальный шум позиции строки относительно шага строк.
WORD_GAP_RANGE = (8, 20)  # Диапазон расстояний между словами.
MAX_WORD_OVERLAP_RATIO = 0.05  # Максимальная доля пересечения нового слова со старыми.
MAX_LINE_PLACE_ATTEMPTS = 30  # Число попыток поставить строку без сильного пересечения слов.
GLOBAL_PARABOLA_STRENGTH_RANGE = (-0.00008, 0.00008)  # Слабое искривление всей страницы по параболе.
EDGE_BEND_PROBABILITY = 0.5  # Вероятность вместо параболы сделать загиб одного края страницы.
EDGE_BEND_WIDTH_PERCENT_RANGE = (0.08, 0.22)  # Какая доля ширины участвует в загибе края.
EDGE_BEND_STRENGTH_RANGE = (-45.0, 45.0)  # Максимальный вертикальный сдвиг края в пикселях.
EDGE_BEND_POWER_RANGE = (1.4, 2.8)  # Чем больше степень, тем локальнее загиб у самого края.
MIN_WORDS_IN_LINE = 7  # Минимум слов в синтетической строке.
MAX_WORDS_IN_LINE = 18  # Максимум слов в синтетической строке.
LINE_TARGET_WIDTH_FRACTION_RANGE = (0.35, 0.98)  # Доля ширины текстового блока для случайной длины строки.
LINE_WIDTH_TOLERANCE_PERCENT = 8.0  # Допустимое отклонение собранной строки от целевой длины.
LINE_ASSEMBLY_ATTEMPTS = 60  # Число попыток собрать строку около target_width.
FIRST_LINE_TARGET_WIDTH_FRACTION_RANGE = (0.82, 0.98)  # Первая строка должна быть достаточно длинной.
FIRST_LINE_X_START_LUFT_PERCENT = 1.0  # Люфт старта первой строки от начала рамки, % ширины блока.
NEXT_LINE_LENGTH_CHANGE_PERCENT = 5.0  # Насколько % следующая строка может быть короче или длиннее предыдущей.
NEXT_LINE_ANGLE_CHANGE_PERCENT = 5.0  # Насколько % может измениться угол относительно предыдущей строки.
NEXT_LINE_X_START_LUFT_PERCENT = 4.0  # Люфт начала строки по x относительно первой строки, % ширины блока.
NEXT_LINE_X_END_LUFT_PERCENT = 6.0  # Люфт конца строки хранится для debug, старт строки он больше не двигает.
NEXT_LINE_Y_LUFT_PERCENT = 8.0  # Люфт строки по y относительно регулярного шага, % высоты шага.
LINE_OUTLIER_PROBABILITY = 0  # Вероятность отключить одно из правил преемственности для текущей строки.
LINE_MASK_POLYGON_EPS_FACTOR = 0.0015  # Упрощение плотных контуров текста для JSON.
LINE_POLYGON_DILATE_SIZE = 3  # Небольшое расширение маски перед построением плотных полигонов.
DEBUG_LINE_COLOR = (0, 0, 255)  # Цвет прямоугольника строки в debug BGR.
DEBUG_POLYGON_COLOR = (0, 180, 0)  # Цвет многоугольника строки в debug BGR.


def is_outlier() -> bool:
    """
    Короткое описание:
        Случайно решает, отключать ли одно локальное правило преемственности строки.
    Вход:
        отсутствует.
    Выход:
        bool: True, если правило нужно отключить для текущей строки.
    """
    return random.random() < LINE_OUTLIER_PROBABILITY


def sample_next_value(
    previous_value: float | None,
    global_range: tuple[float, float],
    change_percent: float,
) -> float:
    """
    Короткое описание:
        Семплирует значение около предыдущего или из общего диапазона при выбросе.
    Вход:
        previous_value (float | None): значение предыдущей строки.
        global_range (tuple[float, float]): общий допустимый диапазон.
        change_percent (float): допустимое изменение в процентах.
    Выход:
        float: новое значение.
    """
    # Шаг 1: для первой строки или выброса берем значение из полного диапазона.
    if previous_value is None or is_outlier():
        return random.uniform(*global_range)

    # Шаг 2: для обычной строки берем значение рядом с предыдущим.
    delta = abs(previous_value) * change_percent / 100.0
    if delta < 1e-6:
        delta = max(abs(global_range[1] - global_range[0]) * change_percent / 100.0, 1e-6)
    value = random.uniform(previous_value - delta, previous_value + delta)
    return float(np.clip(value, global_range[0], global_range[1]))


def sample_next_length_fraction(previous_fraction: float | None) -> float:
    """
    Короткое описание:
        Выбирает долю ширины следующей строки с учетом длины предыдущей строки.
    Вход:
        previous_fraction (float | None): доля ширины предыдущей строки.
    Выход:
        float: новая доля ширины текстового блока.
    """
    return sample_next_value(
        previous_fraction,
        LINE_TARGET_WIDTH_FRACTION_RANGE,
        NEXT_LINE_LENGTH_CHANGE_PERCENT,
    )


def sample_next_x(
    previous_x: int | None,
    global_min: int,
    global_max: int,
    luft_px: int,
) -> int:
    """
    Короткое описание:
        Выбирает x-координату рядом с предыдущей или свободно при выбросе.
    Вход:
        previous_x (int | None): x-координата предыдущей строки.
        global_min (int): минимальная допустимая координата.
        global_max (int): максимальная допустимая координата.
        luft_px (int): допустимый люфт в пикселях.
    Выход:
        int: новая x-координата.
    """
    # Шаг 1: если предыдущей строки нет или сработал выброс, берем x из полного диапазона.
    if previous_x is None or is_outlier():
        return random.randint(global_min, max(global_min, global_max))

    # Шаг 2: иначе держимся рядом с предыдущей координатой.
    x_min = max(global_min, previous_x - luft_px)
    x_max = min(global_max, previous_x + luft_px)
    return random.randint(x_min, max(x_min, x_max))


def jitter_text_color(base_color_bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Короткое описание:
        Слегка меняет оттенок базового цвета текста для отдельного слова.
    Вход:
        base_color_bgr (tuple[int, int, int]): базовый цвет страницы в BGR.
    Выход:
        tuple[int, int, int]: цвет слова в RGB для PIL.
    """
    # Шаг 1: добавляем небольшой шум к каждому каналу и переводим BGR в RGB.
    bgr = np.array(base_color_bgr, dtype=np.int32)
    jitter = np.random.randint(-TEXT_WORD_COLOR_JITTER, TEXT_WORD_COLOR_JITTER + 1, size=3)
    bgr = np.clip(bgr + jitter, 0, 255).astype(np.uint8)
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def load_word_labels(labels_path: Path) -> dict[str, str]:
    """
    Короткое описание:
        Загружает подписи word-crops из labels.jsonl.
    Вход:
        labels_path (Path): путь к labels.jsonl датасета слов.
    Выход:
        dict[str, str]: словарь имя файла crop -> текст слова.
    """
    labels: dict[str, str] = {}
    if not labels_path.exists():
        return labels

    # Шаг 1: читаем jsonl построчно, чтобы не зависеть от размера файла.
    with labels_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_name = Path(str(item.get("image", ""))).name
            translation = str(item.get("translation", "")).strip()
            if image_name and translation:
                labels[image_name] = translation
    return labels


def trim_binary_word(binary_word: np.ndarray) -> np.ndarray | None:
    """
    Короткое описание:
        Обрезает лишние белые поля вокруг бинарного слова.
    Вход:
        binary_word (np.ndarray): бинарное изображение слова, 0 текст и 255 фон.
    Выход:
        np.ndarray | None: обрезанное слово или None, если текст не найден.
    """
    binary = np.where(binary_word < 128, 0, 255).astype(np.uint8)
    ys, xs = np.where(binary < 128)
    if len(xs) < MIN_WORD_BLACK_PIXELS:
        return None

    # Шаг 1: оставляем небольшой белый кант, чтобы слова не липли к краю crop.
    x1 = max(0, int(xs.min()) - MAX_WORD_TRIM_PADDING)
    y1 = max(0, int(ys.min()) - MAX_WORD_TRIM_PADDING)
    x2 = min(binary.shape[1], int(xs.max()) + MAX_WORD_TRIM_PADDING + 1)
    y2 = min(binary.shape[0], int(ys.max()) + MAX_WORD_TRIM_PADDING + 1)
    return binary[y1:y2, x1:x2]


def load_word_crop_pool(images_dir: Path, labels_path: Path) -> list[dict]:
    """
    Короткое описание:
        Загружает реальные бинарные word-crops для генерации строк.
    Вход:
        images_dir (Path): папка images датасета слов.
        labels_path (Path): путь к labels.jsonl с подписями.
    Выход:
        list[dict]: список слов с изображением, маской и текстом.
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"Не найдена папка word-crops: {images_dir}")

    labels = load_word_labels(labels_path)
    word_pool = []

    # Шаг 1: читаем PNG-слова и сразу обрезаем лишний белый фон.
    for image_path in tqdm(sorted(images_dir.glob("*.png")), desc="Read word crops"):
        binary = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if binary is None:
            continue
        trimmed = trim_binary_word(binary)
        if trimmed is None:
            continue
        word_pool.append({
            "text": labels.get(image_path.name, image_path.stem),
            "image": trimmed,
            "source": str(image_path),
            "width": int(trimmed.shape[1]),
            "height": int(trimmed.shape[0]),
        })

    if not word_pool:
        raise RuntimeError(f"Не удалось загрузить ни одного word-crop из {images_dir}")
    return word_pool


def resize_word_crop(word_image: np.ndarray, target_height: int) -> np.ndarray:
    """
    Короткое описание:
        Масштабирует word-crop к заданной высоте с сохранением пропорций.
    Вход:
        word_image (np.ndarray): бинарный crop слова.
        target_height (int): целевая высота.
    Выход:
        np.ndarray: масштабированное бинарное слово.
    """
    height, width = word_image.shape[:2]
    scale = max(1.0, float(target_height)) / max(1.0, float(height))
    target_width = max(1, int(round(width * scale)))
    resized = cv2.resize(word_image, (target_width, int(target_height)), interpolation=cv2.INTER_NEAREST)
    return np.where(resized < 128, 0, 255).astype(np.uint8)


def sample_word_line(
    word_pool: list[dict],
    target_width: int,
    target_height: int,
) -> list[dict]:
    """
    Короткое описание:
        Подбирает реальные word-crops так, чтобы ширина строки была около target_width.
    Вход:
        word_pool (list[dict]): список загруженных word-crops.
        target_width (int): желаемая ширина строки.
        target_height (int): базовая высота слов в строке.
    Выход:
        list[dict]: список слов с уже масштабированными изображениями.
    """
    best_words: list[dict] = []
    best_error = float("inf")
    tolerance = max(1.0, target_width * LINE_WIDTH_TOLERANCE_PERCENT / 100.0)

    # Шаг 1: несколько раз собираем строку и выбираем вариант с лучшей близостью к target_width.
    for _ in range(LINE_ASSEMBLY_ATTEMPTS):
        selected_words: list[dict] = []
        current_width = 40
        word_count_limit = random.randint(MIN_WORDS_IN_LINE, MAX_WORDS_IN_LINE)

        for word_idx in range(word_count_limit):
            source_word = random.choice(word_pool)
            height_jitter = random.uniform(*WORD_SCALE_JITTER_RANGE)
            word_height = max(8, int(round(target_height * height_jitter)))
            resized_word = resize_word_crop(source_word["image"], word_height)
            gap = random.randint(*WORD_GAP_RANGE) if selected_words else 0
            candidate_width = current_width + gap + resized_word.shape[1]

            # Шаг 2: после минимума слов не вылезаем далеко за целевую длину строки.
            if (
                len(selected_words) >= MIN_WORDS_IN_LINE
                and candidate_width > target_width + tolerance
            ):
                break

            selected_words.append({
                "text": source_word["text"],
                "image": resized_word,
                "gap_before": int(gap),
                "source": source_word["source"],
            })
            current_width = candidate_width

            if len(selected_words) >= MIN_WORDS_IN_LINE and current_width >= target_width - tolerance:
                break

        if len(selected_words) < MIN_WORDS_IN_LINE:
            continue

        error = abs(current_width - target_width)
        if error < best_error:
            best_error = error
            best_words = selected_words
        if error <= tolerance:
            return selected_words

    return best_words


def render_line_layer(
    line_words: list[dict],
    base_color_bgr: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Короткое описание:
        Собирает строку из реальных бинарных word-crops и word-mask на слое.
    Вход:
        line_words (list[dict]): список слов с масштабированными бинарными crop.
        base_color_bgr (tuple[int, int, int]): базовый цвет текста страницы в BGR.
    Выход:
        tuple[np.ndarray, np.ndarray, list[dict]]: BGR-слой, маска строки и bbox слов.
    """
    if not line_words:
        return (
            np.full((1, 1, 3), 255, dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
            [],
        )

    word_heights = [int(item["image"].shape[0]) for item in line_words]
    word_widths = [int(item["image"].shape[1]) for item in line_words]
    width = sum(word_widths) + sum(int(item["gap_before"]) for item in line_words) + 40
    height = max(word_heights) + 50

    layer = np.full((height, width, 3), 255, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    word_boxes = []
    x = 20
    baseline_y = 20 + max(word_heights)

    # Шаг 1: вклеиваем реальные бинарные слова отдельно, чтобы сохранить естественную форму букв.
    for word_idx, word_item in enumerate(line_words):
        x += int(word_item["gap_before"])
        word_image = word_item["image"]
        word_height, word_width = word_image.shape[:2]
        y = baseline_y - word_height
        word_mask = word_image < 128
        color_rgb = jitter_text_color(base_color_bgr)
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        layer_roi = layer[y:y + word_height, x:x + word_width]
        mask_roi = mask[y:y + word_height, x:x + word_width]
        layer_roi[word_mask] = color_bgr
        mask_roi[word_mask] = 255
        word_boxes.append({
            "word": str(word_item["text"]),
            "bbox": [int(x), int(y), int(x + word_width), int(y + word_height)],
            "source": str(word_item["source"]),
        })
        x += word_width

    return layer, mask, word_boxes


def remap_page_and_masks(
    page: np.ndarray,
    line_masks: list[np.ndarray],
    shifts: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Короткое описание:
        Применяет одинаковый вертикальный remap к странице и всем маскам строк.
    Вход:
        page (np.ndarray): изображение страницы BGR.
        line_masks (list[np.ndarray]): маски строк в координатах страницы.
        shifts (np.ndarray): вертикальный сдвиг для каждого x.
    Выход:
        tuple[np.ndarray, list[np.ndarray]]: искривленная страница и маски строк.
    """
    height, width = page.shape[:2]
    map_x = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    map_y = np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width))

    # Шаг 1: cv2.remap использует обратную карту, поэтому вычитаем желаемый сдвиг.
    map_y = map_y - shifts.astype(np.float32)[None, :]
    warped_page = cv2.remap(page, map_x, map_y, cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    # Шаг 2: вместо remap каждой строки отдельно кодируем все строки в одну label-mask.
    label_mask = np.zeros((height, width), dtype=np.uint16)
    for line_idx, mask in enumerate(line_masks, start=1):
        label_mask[mask > 0] = line_idx
    warped_label_mask = cv2.remap(label_mask, map_x, map_y, cv2.INTER_NEAREST, borderValue=0)
    warped_line_masks = [
        ((warped_label_mask == line_idx) * 255).astype(np.uint8)
        for line_idx in range(1, len(line_masks) + 1)
    ]
    return warped_page, warped_line_masks


def warp_page_parabola(
    page: np.ndarray,
    line_masks: list[np.ndarray],
    strength: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Короткое описание:
        Слабо искривляет всю страницу и все маски строк одним преобразованием.
    Вход:
        page (np.ndarray): изображение страницы BGR.
        line_masks (list[np.ndarray]): список масок строк в координатах страницы.
        strength (float): коэффициент параболы.
    Выход:
        tuple[np.ndarray, list[np.ndarray]]: искривленная страница и маски строк.
    """
    _, width = page.shape[:2]
    x_coords = np.arange(width, dtype=np.float32)
    center_x = (width - 1) / 2.0
    shifts = strength * (x_coords - center_x) ** 2
    return remap_page_and_masks(page, line_masks, shifts)


def warp_page_edge_bend(
    page: np.ndarray,
    line_masks: list[np.ndarray],
    side: str,
    width_percent: float,
    strength: float,
    power: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Короткое описание:
        Делает страницу почти прямой, но слегка загибает левый или правый край.
    Вход:
        page (np.ndarray): изображение страницы BGR.
        line_masks (list[np.ndarray]): маски строк в координатах страницы.
        side (str): какой край гнуть, "left" или "right".
        width_percent (float): доля ширины страницы, где действует загиб.
        strength (float): максимальный вертикальный сдвиг края.
        power (float): степень затухания загиба от края к центру.
    Выход:
        tuple[np.ndarray, list[np.ndarray]]: искривленная страница и маски строк.
    """
    _, width = page.shape[:2]
    edge_width = max(1.0, width * width_percent)
    x_coords = np.arange(width, dtype=np.float32)

    # Шаг 1: строим вес загиба: максимум у края, почти ноль на остальной странице.
    if side == "left":
        distance_from_edge = x_coords
    else:
        distance_from_edge = (width - 1) - x_coords
    edge_weight = np.clip(1.0 - distance_from_edge / edge_width, 0.0, 1.0) ** power
    shifts = strength * edge_weight
    return remap_page_and_masks(page, line_masks, shifts)


def warp_page_final(
    page: np.ndarray,
    line_masks: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """
    Короткое описание:
        Выбирает финальное искривление страницы и применяет его к странице и маскам.
    Вход:
        page (np.ndarray): изображение страницы BGR.
        line_masks (list[np.ndarray]): маски строк в координатах страницы.
    Выход:
        tuple[np.ndarray, list[np.ndarray], dict]: изображение, маски и параметры warp.
    """
    # Шаг 1: иногда делаем локальный загиб края, иногда обычную слабую параболу.
    if random.random() < EDGE_BEND_PROBABILITY:
        side = random.choice(["left", "right"])
        width_percent = random.uniform(*EDGE_BEND_WIDTH_PERCENT_RANGE)
        strength = random.uniform(*EDGE_BEND_STRENGTH_RANGE)
        power = random.uniform(*EDGE_BEND_POWER_RANGE)
        warped_page, warped_masks = warp_page_edge_bend(
            page,
            line_masks,
            side=side,
            width_percent=width_percent,
            strength=strength,
            power=power,
        )
        return warped_page, warped_masks, {
            "type": "edge_bend",
            "side": side,
            "width_percent": float(width_percent),
            "strength": float(strength),
            "power": float(power),
        }

    strength = random.uniform(*GLOBAL_PARABOLA_STRENGTH_RANGE)
    warped_page, warped_masks = warp_page_parabola(page, line_masks, strength)
    return warped_page, warped_masks, {
        "type": "parabola",
        "strength": float(strength),
    }


def rotate_layer(
    layer: np.ndarray,
    mask: np.ndarray,
    angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Короткое описание:
        Поворачивает слой строки и возвращает affine-матрицу.
    Вход:
        layer (np.ndarray): BGR-слой строки.
        mask (np.ndarray): маска строки.
        angle (float): угол поворота в градусах.
    Выход:
        tuple[np.ndarray, np.ndarray, np.ndarray]: повернутые layer, mask и матрица 2 x 3.
    """
    height, width = mask.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    rotated_layer = cv2.warpAffine(layer, matrix, (new_width, new_height), borderValue=(255, 255, 255))
    rotated_mask = cv2.warpAffine(mask, matrix, (new_width, new_height), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated_layer, rotated_mask, matrix


def paste_line(
    page: np.ndarray,
    page_mask: np.ndarray,
    line_layer: np.ndarray,
    line_mask: np.ndarray,
    x: int,
    y: int,
) -> None:
    """
    Короткое описание:
        Вклеивает строку на страницу и обновляет общую маску текста.
    Вход:
        page (np.ndarray): страница BGR, изменяется на месте.
        page_mask (np.ndarray): общая маска текста, изменяется на месте.
        line_layer (np.ndarray): BGR-слой строки.
        line_mask (np.ndarray): маска строки.
        x (int): координата x вставки.
        y (int): координата y вставки.
    Выход:
        отсутствует.
    """
    height, width = line_mask.shape
    roi = page[y:y + height, x:x + width]
    roi_mask = page_mask[y:y + height, x:x + width]
    text_pixels = line_mask > 0
    roi[text_pixels] = line_layer[text_pixels]
    roi_mask[text_pixels] = 255


def overlap_ratio(candidate_mask: np.ndarray, used_mask: np.ndarray, x: int, y: int) -> float:
    """
    Короткое описание:
        Считает долю пересечения кандидата с уже занятыми пикселями.
    Вход:
        candidate_mask (np.ndarray): маска нового слова или строки.
        used_mask (np.ndarray): общая маска уже размещенного текста.
        x (int): координата x вставки.
        y (int): координата y вставки.
    Выход:
        float: доля пересечения от площади кандидата.
    """
    height, width = candidate_mask.shape
    page_height, page_width = used_mask.shape
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(page_width, x + width)
    y2 = min(page_height, y + height)
    if x1 >= x2 or y1 >= y2:
        return 1.0

    # Шаг 1: сравниваем только общую область, чтобы не падать при выходе за границу.
    mask_x1 = x1 - x
    mask_y1 = y1 - y
    mask_x2 = mask_x1 + (x2 - x1)
    mask_y2 = mask_y1 + (y2 - y1)
    used_roi = used_mask[y1:y2, x1:x2]
    candidate_roi = candidate_mask[mask_y1:mask_y2, mask_x1:mask_x2]
    candidate_pixels = candidate_roi > 0
    candidate_area = int(candidate_pixels.sum())
    if candidate_area == 0:
        return 0.0
    intersection = int(np.logical_and(candidate_pixels, used_roi > 0).sum())
    return intersection / candidate_area


def line_annotation(line_mask: np.ndarray, text: str) -> dict | None:
    """
    Короткое описание:
        Строит minAreaRect и плотные контуры текста для строки.
    Вход:
        line_mask (np.ndarray): маска строки в координатах страницы.
        text (str): текст строки.
    Выход:
        dict | None: JSON-разметка строки или None для пустой маски.
    """
    points_y, points_x = np.where(line_mask > 0)
    if len(points_x) == 0:
        return None

    points = np.column_stack([points_x, points_y]).astype(np.float32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect).astype(np.float32)

    # Шаг 1: работаем только с bbox строки, чтобы не гонять findContours по всей странице.
    x_min = max(0, int(points_x.min()) - LINE_POLYGON_DILATE_SIZE - 2)
    y_min = max(0, int(points_y.min()) - LINE_POLYGON_DILATE_SIZE - 2)
    x_max = min(line_mask.shape[1], int(points_x.max()) + LINE_POLYGON_DILATE_SIZE + 3)
    y_max = min(line_mask.shape[0], int(points_y.max()) + LINE_POLYGON_DILATE_SIZE + 3)

    # Шаг 2: немного расширяем буквы, чтобы контур шел вокруг текста, а не по внутренним пикселям антиалиасинга.
    polygon_mask = line_mask[y_min:y_max, x_min:x_max].copy()
    if LINE_POLYGON_DILATE_SIZE > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (LINE_POLYGON_DILATE_SIZE, LINE_POLYGON_DILATE_SIZE),
        )
        polygon_mask = cv2.dilate(polygon_mask, kernel, iterations=1)

    # Шаг 3: строим плотные контуры по компонентам текста, а не по большому bbox строки.
    contours, _ = cv2.findContours(polygon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 2:
            continue
        epsilon = LINE_MASK_POLYGON_EPS_FACTOR * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(polygon) >= 3:
            polygon = polygon.astype(np.float32)
            polygon[:, 0] += x_min
            polygon[:, 1] += y_min
            polygons.append(polygon.round(2).tolist())

    # Шаг 4: оставляем также общий convex hull для совместимости с прежним форматом.
    hull = cv2.convexHull(points).reshape(-1, 2).astype(np.float32)
    return {
        "text": text,
        "rotated_rect": {
            "center": [float(rect[0][0]), float(rect[0][1])],
            "size": [float(rect[1][0]), float(rect[1][1])],
            "angle": float(rect[2]),
            "points": box.round(2).tolist(),
        },
        "polygon": hull.round(2).tolist(),
        "polygons": polygons,
    }


def generate_page_once(page_idx: int, word_pool: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Короткое описание:
        Делает одну попытку генерации синтетической страницы.
    Вход:
        page_idx (int): номер страницы.
        word_pool (list[dict]): реальные бинарные word-crops.
    Выход:
        tuple[np.ndarray, np.ndarray, dict]: изображение страницы, маска текста и JSON-разметка.
    """
    page_width, page_height = PAGE_SIZE
    page = np.full((page_height, page_width, 3), PAGE_BACKGROUND_VALUE, dtype=np.uint8)
    used_mask = np.zeros((page_height, page_width), dtype=np.uint8)
    page_base_color_bgr = random.choice(TEXT_BASE_COLORS_BGR)
    margin_x = int(page_width * random.uniform(*TEXT_BOX_MARGIN_X_PERCENT_RANGE))
    margin_y = int(page_height * random.uniform(*TEXT_BOX_MARGIN_Y_PERCENT_RANGE))
    text_box = [margin_x, margin_y, page_width - margin_x, page_height - margin_y]
    line_count = random.randint(*LINE_COUNT_RANGE)
    line_step = (text_box[3] - text_box[1]) / max(line_count, 1)
    line_items = []
    text_box_width = text_box[2] - text_box[0]
    previous_length_fraction = None
    previous_angle = None
    first_x_start = None
    previous_y = None

    # Шаг 1: строки идут сверху вниз и достаточно плотно, но каждая имеет свой небольшой наклон.
    for line_idx in range(line_count):
        placed = False
        for _ in range(MAX_LINE_PLACE_ATTEMPTS):
            if first_x_start is None:
                length_fraction = random.uniform(*FIRST_LINE_TARGET_WIDTH_FRACTION_RANGE)
            else:
                length_fraction = sample_next_length_fraction(previous_length_fraction)
            target_width = int(text_box_width * length_fraction)

            # Шаг 2: высоту слова держим ниже шага строк, чтобы строки не слипались.
            max_line_word_height = max(8, int(line_step * 0.78))
            word_height_high = min(WORD_TARGET_HEIGHT_RANGE[1], max_line_word_height)
            word_height_low = min(WORD_TARGET_HEIGHT_RANGE[0], word_height_high)
            target_word_height = random.randint(word_height_low, word_height_high)
            line_words = sample_word_line(word_pool, target_width, target_word_height)
            if not line_words:
                continue
            line_text = " ".join(str(item["text"]) for item in line_words)
            layer, mask, word_boxes = render_line_layer(line_words, page_base_color_bgr)
            angle = sample_next_value(previous_angle, LINE_ANGLE_RANGE, NEXT_LINE_ANGLE_CHANGE_PERCENT)
            layer, mask, _ = rotate_layer(layer, mask, angle)
            line_height, line_width = mask.shape
            if line_width >= text_box[2] - text_box[0] or line_height >= line_step * 2.5:
                continue

            regular_y = int(text_box[1] + line_idx * line_step)
            y_luft_px = max(1, int(line_step * NEXT_LINE_Y_LUFT_PERCENT / 100.0))
            if previous_y is None or is_outlier():
                y_base = int(regular_y + random.uniform(-LINE_JITTER_Y_FACTOR, LINE_JITTER_Y_FACTOR) * line_step)
            else:
                y_base = int(previous_y + line_step + random.randint(-y_luft_px, y_luft_px))

            x_start_luft_px = max(1, int(text_box_width * NEXT_LINE_X_START_LUFT_PERCENT / 100.0))
            x_min = text_box[0]
            x_max = max(text_box[0], text_box[2] - line_width)
            y_min = max(text_box[1], y_base - y_luft_px)
            y_max = min(text_box[3] - line_height, y_base + y_luft_px)
            if y_max < y_min:
                continue

            if first_x_start is None:
                first_line_luft_px = max(1, int(text_box_width * FIRST_LINE_X_START_LUFT_PERCENT / 100.0))
                x = random.randint(text_box[0], min(x_max, text_box[0] + first_line_luft_px))
            else:
                x = sample_next_x(first_x_start, x_min, x_max, x_start_luft_px)

            if x < 0 or y_min < 0 or x + line_width > page_width:
                continue
            y = random.randint(y_min, y_max)
            if overlap_ratio(mask, used_mask, x, y) > MAX_WORD_OVERLAP_RATIO:
                continue

            paste_line(page, used_mask, layer, mask, x, y)
            line_page_mask = np.zeros((page_height, page_width), dtype=np.uint8)
            line_page_mask[y:y + line_height, x:x + line_width][mask > 0] = 255
            line_items.append({
                "text": line_text,
                "words": word_boxes,
                "mask": line_page_mask,
            })
            previous_length_fraction = line_width / max(1.0, float(text_box_width))
            previous_angle = float(angle)
            if first_x_start is None:
                first_x_start = int(x)
            previous_y = int(y)
            placed = True
            break

        if not placed:
            continue

    # Шаг 2: после размещения всех строк применяем одно финальное искривление ко всей странице.
    page, warped_line_masks, warp_info = warp_page_final(page, [item["mask"] for item in line_items])
    used_mask = np.zeros((page_height, page_width), dtype=np.uint8)
    annotations = []
    for item, warped_mask in zip(line_items, warped_line_masks):
        used_mask = cv2.bitwise_or(used_mask, warped_mask)
        annotation = line_annotation(warped_mask, item["text"])
        if annotation is not None:
            annotations.append(annotation)

    return page, used_mask, {
        "page_index": int(page_idx),
        "page_size": [int(page_width), int(page_height)],
        "text_box": text_box,
        "word_crops_dataset": str(WORD_CROPS_DATASET_DIR),
        "base_color_bgr": list(page_base_color_bgr),
        "warp": warp_info,
        "line_count": len(annotations),
        "lines": annotations,
    }


def generate_page(page_idx: int, word_pool: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Короткое описание:
        Генерирует страницу и повторяет попытку, если получился почти пустой лист.
    Вход:
        page_idx (int): номер страницы.
        word_pool (list[dict]): реальные бинарные word-crops.
    Выход:
        tuple[np.ndarray, np.ndarray, dict]: изображение страницы, маска текста и JSON-разметка.
    """
    best_result = None
    best_line_count = -1

    # Шаг 1: несколько раз пробуем собрать страницу, чтобы не сохранять белые листы.
    for attempt_idx in range(PAGE_GENERATION_ATTEMPTS):
        page, mask, annotation = generate_page_once(page_idx, word_pool)
        line_count = int(annotation["line_count"])
        annotation["generation_attempt"] = int(attempt_idx)
        if line_count > best_line_count:
            best_result = (page, mask, annotation)
            best_line_count = line_count
        if line_count >= MIN_PLACED_LINES:
            return page, mask, annotation

    if best_result is None:
        raise RuntimeError("Не удалось сгенерировать синтетическую страницу")
    return best_result


def draw_debug(page: np.ndarray, annotation: dict) -> np.ndarray:
    """
    Короткое описание:
        Рисует разметку строк поверх синтетической страницы.
    Вход:
        page (np.ndarray): изображение страницы BGR.
        annotation (dict): JSON-разметка страницы.
    Выход:
        np.ndarray: debug-изображение BGR.
    """
    debug = page.copy()
    x1, y1, x2, y2 = annotation["text_box"]
    cv2.rectangle(debug, (x1, y1), (x2, y2), (220, 220, 220), 2)

    # Шаг 1: рисуем rotated_rect красным, общий hull желтым и плотные контуры зеленым.
    for line_idx, line in enumerate(annotation["lines"]):
        rect_points = np.array(line["rotated_rect"]["points"], dtype=np.int32)
        polygon = np.array(line["polygon"], dtype=np.int32)
        cv2.polylines(debug, [rect_points], True, DEBUG_LINE_COLOR, 2)
        cv2.polylines(debug, [polygon], True, (0, 180, 180), 1)
        for tight_polygon in line.get("polygons", []):
            tight_polygon = np.array(tight_polygon, dtype=np.int32)
            cv2.polylines(debug, [tight_polygon], True, DEBUG_POLYGON_COLOR, 1)
        center = tuple(np.mean(rect_points, axis=0).astype(int))
        cv2.putText(debug, str(line_idx), center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return debug


def main() -> None:
    """
    Короткое описание:
        Генерирует SYNTHETIC_PAGE_COUNT синтетических страниц и сохраняет debug.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Шаг 1: очищаем только папку синтетического датасета, чтобы не смешивать старый и новый прогон.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for directory in [IMAGES_DIR, MASKS_DIR, ANNOTATIONS_DIR, DEBUG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    word_pool = load_word_crop_pool(WORD_CROPS_IMAGES_DIR, WORD_CROPS_LABELS_PATH)
    print(f"[OK] word-crops loaded: {len(word_pool)}")

    # Шаг 2: генерируем страницы, маски, JSON и debug-оверлеи.
    for page_idx in tqdm(range(SYNTHETIC_PAGE_COUNT), desc="Generate synthetic text pages"):
        page, mask, annotation = generate_page(page_idx, word_pool)
        stem = f"synthetic_text_page_{page_idx:04d}"
        cv2.imwrite(str(IMAGES_DIR / f"{stem}.png"), page)
        cv2.imwrite(str(MASKS_DIR / f"{stem}_mask.png"), mask)
        (ANNOTATIONS_DIR / f"{stem}.json").write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if page_idx < DEBUG_SAVE_COUNT:
            cv2.imwrite(str(DEBUG_DIR / f"{stem}_debug.png"), draw_debug(page, annotation))

    print(f"[OK] saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
