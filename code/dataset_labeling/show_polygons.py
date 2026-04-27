from pathlib import Path
from typing import List

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "school_notebooks_RU"
IMAGES_DIR = DATASET_DIR / "images_base"
LABELS_DIR = DATASET_DIR / "images_detect_lines"
OUTPUT_PATH = PROJECT_ROOT / "debug_images" / "show_polygons.jpg"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")
LABEL_NAME = "ru_hw2022_6_IMG_7175"
LABEL_PATH = None
SHOW_WINDOW = True


def find_image_path(stem: str) -> Path:
    """
    Короткое описание:
        находит изображение по имени файла разметки без расширения.
    Вход:
        stem: str -- имя файла без расширения.
    Выход:
        Path -- путь к найденному изображению.
    """
    for extension in IMAGE_EXTENSIONS:
        image_path = IMAGES_DIR / f"{stem}{extension}"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"Не найдено изображение для {stem}")


def read_polygons(label_path: Path, image_width: int, image_height: int) -> List[np.ndarray]:
    """
    Короткое описание:
        читает YOLO-seg разметку и переводит нормированные точки в пиксели.
    Вход:
        label_path: Path -- путь к txt-файлу разметки.
        image_width: int -- ширина изображения.
        image_height: int -- высота изображения.
    Выход:
        List[np.ndarray] -- список полигонов формы N x 2.
    """
    polygons: List[np.ndarray] = []
    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()
            if len(values) < 7:
                continue

            coords = np.array(values[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] = np.clip(coords[:, 0] * image_width, 0, image_width - 1)
            coords[:, 1] = np.clip(coords[:, 1] * image_height, 0, image_height - 1)
            polygons.append(coords.astype(np.int32))
    return polygons


def draw_polygons(image: np.ndarray, polygons: List[np.ndarray]) -> np.ndarray:
    """
    Короткое описание:
        рисует полигоны поверх изображения.
    Вход:
        image: np.ndarray -- исходное изображение.
        polygons: List[np.ndarray] -- список полигонов формы N x 2.
    Выход:
        np.ndarray -- изображение с отрисованными полигонами.
    """
    result = image.copy()
    overlay = image.copy()

    for polygon in polygons:
        cv2.fillPoly(overlay, [polygon], (0, 255, 0))
        cv2.polylines(result, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)

    cv2.addWeighted(overlay, 0.25, result, 0.75, 0, dst=result)
    return result


def resolve_label_path() -> Path:
    """
    Короткое описание:
        выбирает txt-файл разметки по гиперпараметрам.
    Вход:
        None
    Выход:
        Path -- путь к txt-файлу разметки.
    """
    if LABEL_PATH is not None:
        return Path(LABEL_PATH)

    if LABEL_NAME is not None:
        label_name = LABEL_NAME if LABEL_NAME.endswith(".txt") else f"{LABEL_NAME}.txt"
        return LABELS_DIR / label_name

    label_paths = sorted(LABELS_DIR.glob("*.txt"))
    if not label_paths:
        raise FileNotFoundError(f"В {LABELS_DIR} нет txt-файлов")
    return label_paths[0]


def main() -> None:
    """
    Короткое описание:
        загружает одну разметку, одно изображение и показывает полигоны.
    Вход:
        None
    Выход:
        None
    """
    label_path = resolve_label_path()
    image_path = find_image_path(label_path.stem)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    image_height, image_width = image.shape[:2]
    polygons = read_polygons(label_path, image_width, image_height)
    result = draw_polygons(image, polygons)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), result)

    if SHOW_WINDOW:
        cv2.namedWindow("polygons", cv2.WINDOW_NORMAL)
        cv2.imshow("polygons", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
