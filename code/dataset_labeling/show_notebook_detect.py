from pathlib import Path
from typing import List

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "school_notebooks_RU"
IMAGES_DIR = DATASET_DIR / "images_base"
LABELS_DIR = DATASET_DIR / "images_detect_notebook"
OUTPUT_DIR = PROJECT_ROOT / "debug_images" / "show_notebook"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")
SHOW_COUNT = 5  # Количество примеров, которые сохраняем при запуске.
LINE_WIDTH = 4  # Толщина контура страницы.
FILL_ALPHA = 0.25  # Прозрачность зеленой заливки полигона.
SHOW_WINDOW = False  # Показывать изображения через cv2.imshow.


def find_image_path(stem: str) -> Path:
    """
    Короткое описание:
        Находит изображение по имени файла разметки без расширения.
    Вход:
        stem (str): имя файла без расширения.
    Выход:
        Path: путь к найденному изображению.
    """
    # Шаг 1: проверяем все допустимые расширения изображения.
    for extension in IMAGE_EXTENSIONS:
        image_path = IMAGES_DIR / f"{stem}{extension}"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"Не найдено изображение для {stem}")


def read_polygons(label_path: Path, image_width: int, image_height: int) -> List[np.ndarray]:
    """
    Короткое описание:
        Читает YOLO-seg разметку страниц и переводит нормированные точки в пиксели.
    Вход:
        label_path (Path): путь к txt-файлу разметки.
        image_width (int): ширина изображения.
        image_height (int): высота изображения.
    Выход:
        List[np.ndarray]: список полигонов формы N x 2.
    """
    polygons: List[np.ndarray] = []
    # Шаг 1: читаем строки формата class x1 y1 x2 y2 ...
    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()
            if len(values) < 7:
                continue

            # Шаг 2: переводим координаты из [0, 1] в пиксели изображения.
            coords = np.array(values[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] = np.clip(coords[:, 0] * image_width, 0, image_width - 1)
            coords[:, 1] = np.clip(coords[:, 1] * image_height, 0, image_height - 1)
            polygons.append(coords.astype(np.int32))
    return polygons


def draw_polygons(image: np.ndarray, polygons: List[np.ndarray]) -> np.ndarray:
    """
    Короткое описание:
        Рисует полигоны страниц поверх изображения.
    Вход:
        image (np.ndarray): исходное изображение BGR.
        polygons (List[np.ndarray]): список полигонов формы N x 2.
    Выход:
        np.ndarray: изображение с отрисованной разметкой.
    """
    result = image.copy()
    overlay = image.copy()

    # Шаг 1: рисуем заливку и контур каждой найденной страницы.
    for polygon in polygons:
        cv2.fillPoly(overlay, [polygon], (0, 255, 0))
        cv2.polylines(result, [polygon], isClosed=True, color=(0, 0, 255), thickness=LINE_WIDTH)

    # Шаг 2: смешиваем заливку с исходным изображением.
    cv2.addWeighted(overlay, FILL_ALPHA, result, 1.0 - FILL_ALPHA, 0, dst=result)
    return result


def render_label(label_path: Path, output_path: Path) -> None:
    """
    Короткое описание:
        Загружает одну разметку, рисует ее на соответствующем изображении и сохраняет результат.
    Вход:
        label_path (Path): путь к txt-файлу разметки.
        output_path (Path): путь для сохранения визуализации.
    Выход:
        отсутствует.
    """
    # Шаг 1: находим изображение и читаем его.
    image_path = find_image_path(label_path.stem)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    # Шаг 2: читаем полигоны, рисуем и сохраняем результат.
    image_height, image_width = image.shape[:2]
    polygons = read_polygons(label_path, image_width, image_height)
    result = draw_polygons(image, polygons)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    print(f"[OK] {label_path.name}: {len(polygons)} polygons -> {output_path}")

    # Шаг 3: при необходимости показываем окно для ручной проверки.
    if SHOW_WINDOW:
        cv2.namedWindow("notebook polygons", cv2.WINDOW_NORMAL)
        cv2.imshow("notebook polygons", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> None:
    """
    Короткое описание:
        Сохраняет несколько визуализаций разметки images_detect_notebook.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка разметки: {LABELS_DIR}")
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка изображений: {IMAGES_DIR}")

    # Шаг 1: выбираем первые txt-файлы, для которых есть соответствующее изображение.
    saved_count = 0
    for label_path in sorted(LABELS_DIR.glob("*.txt")):
        try:
            output_path = OUTPUT_DIR / f"{label_path.stem}.jpg"
            render_label(label_path, output_path)
        except FileNotFoundError:
            continue

        saved_count += 1
        if saved_count >= SHOW_COUNT:
            break

    if saved_count == 0:
        raise RuntimeError("Не найдено ни одной пары изображение-разметка")


if __name__ == "__main__":
    main()
