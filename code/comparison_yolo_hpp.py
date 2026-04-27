import gc
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
from skimage.morphology import skeletonize
from tqdm import tqdm
from ultralytics import YOLO

import grade_hpp
from hpp_method import LineSegmentation
from u_net_binarization import load_unet_model, binarize_image_with_loaded_model


# Корень проекта: от него строятся все остальные пути.
PROJECT_ROOT = Path(__file__).resolve().parent

# Папка с исходными изображениями тетрадей.
IMAGES_DIR = PROJECT_ROOT / "datasets" / "school_notebooks_RU" / "images_base"

# Папка с polygon-labels для строк.
LABELS_DIR = PROJECT_ROOT / "datasets" / "school_notebooks_RU" / "images_detect_lines"

# Куда сохраняется итоговый json с метриками.
OUTPUT_JSON_PATH = PROJECT_ROOT / "comparison_yolo_hpp_results.json"

# Куда сохраняются debug-картинки сравнения.
DEBUG_DIR = PROJECT_ROOT / "debug_images" / "comparison_yolo_hpp"

# Вес YOLO-модели, которая ищет строки.
YOLO_LINES_MODEL_PATH = PROJECT_ROOT / "models" / "yolo_detect_notebook" / "yolo_detect_lines_1_(2-architecture).pt"

# Вес U-Net модели для бинаризации рукописного текста.
UNET_MODEL_PATH = PROJECT_ROOT / "models" / "u_net" / "unet_binarization_3_(6-architecture).pth"

# Допустимые расширения изображений при поиске пары к label-файлу.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")

# Максимум изображений, если LABEL_NAMES пустой.
MAX_IMAGES = 20

# Ручной список label-файлов для обработки; если [], берутся первые MAX_IMAGES.
LABEL_NAMES = ["1_11.txt", "1_19.txt", "8_109.txt", "2786.txt", "2796.txt", "1_18.txt", "1_23.txt", "2910.txt", "2812.txt", "2013.txt", "2015.txt", "2016.txt", "2884.txt"]

# Включает сохранение debug-визуализаций.
DEBUG = True

# Сколько первых строк каждого типа сохранять в debug.
DEBUG_MAX_LINES = 5

# Сколько лучших matched-пар сохранять в debug после венгерского алгоритма.
DEBUG_MAX_MATCHED_PAIRS = 10

# Сколько первых вызовов PCA подробно сохранять по шагам.
PCA_DEBUG_MAX_SAVES = 0

# Минимальная уверенность YOLO для предсказанной строки.
YOLO_CONF_THRESHOLD = 0.4

# Вес PCA IoU в score для венгерского сопоставления строк.
MATCH_PCA_IOU_WEIGHT = 0.5

# Вес сравнения матриц попарных расстояний в score для венгерского сопоставления строк.
MATCH_PAIRWISE_WEIGHT = 0.25

# Вес Chamfer score в score для венгерского сопоставления строк.
MATCH_CHAMFER_WEIGHT = 0.25

# Размер холста H x W для PCA-выровненной маски строки.
PCA_CANVAS_SIZE = (96, 384)

# Сколько точек брать для сравнения матриц попарных расстояний.
PAIRWISE_SAMPLE_SIZE = 64

# Максимальный горизонтальный зазор между компонентами слова в vertical HPP.
VERTICAL_WORD_GAP = 6

# Минимальная ширина слова после разбиения vertical HPP.
MIN_WORD_WIDTH = 3

# Минимальное число черных пикселей, чтобы маска считалась непустой строкой.
MIN_TEXT_PIXELS = 5

# Максимальный сдвиг class-matrix при честном сравнении по текстовым пикселям.
MAX_MASK_ALIGNMENT_SHIFT = 200

# Счетчик уже сохраненных PCA-debug примеров.
PCA_DEBUG_COUNTER = 0


def bool_mask_to_debug_image(mask: np.ndarray, min_height: int = 120) -> np.ndarray:
    """
    Короткое описание:
        переводит булеву маску в удобное для просмотра изображение.
    Вход:
        mask: np.ndarray -- булева маска.
        min_height: int -- минимальная высота debug-изображения.
    Выход:
        np.ndarray -- uint8 изображение для сохранения.
    """
    # Шаг 1: переводим True-пиксели в черный цвет.
    image = np.where(mask, 0, 255).astype(np.uint8)

    # Шаг 2: увеличиваем маленькие маски без сглаживания.
    if image.shape[0] < min_height:
        scale = max(1, int(np.ceil(min_height / image.shape[0])))
        image = cv2.resize(
            image,
            (image.shape[1] * scale, image.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    return image


def save_first_masks_debug(masks: List[np.ndarray], image_stem: str, name: str) -> None:
    """
    Короткое описание:
        сохраняет первые маски метода для быстрой визуальной проверки.
    Вход:
        masks: List[np.ndarray] -- список масок строк.
        image_stem: str -- имя изображения без расширения.
        name: str -- имя метода или target.
    Выход:
        None
    """
    if not DEBUG or not masks:
        return

    # Шаг 1: создаем директорию debug.
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # Шаг 2: сохраняем только первые DEBUG_MAX_LINES строк.
    for line_idx, mask in enumerate(masks[:DEBUG_MAX_LINES]):
        debug_image = bool_mask_to_debug_image(mask)
        cv2.imwrite(str(DEBUG_DIR / f"{image_stem}_{name}_line_{line_idx:03d}.jpg"), debug_image)


def save_matched_pair_debug(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    pred_aligned: np.ndarray,
    target_aligned: np.ndarray,
    debug_prefix: str,
    matched_idx: int,
) -> None:
    """
    Короткое описание:
        сохраняет сопоставленную пару raw, PCA и skeleton.
    Вход:
        pred_mask: np.ndarray -- исходная маска предсказания.
        target_mask: np.ndarray -- исходная target-маска.
        pred_aligned: np.ndarray -- PCA-выровненная маска предсказания.
        target_aligned: np.ndarray -- PCA-выровненная target-маска.
        debug_prefix: str -- префикс имени debug-файла.
        matched_idx: int -- номер пары в debug-выводе.
    Выход:
        None
    """
    if not DEBUG:
        return

    # Шаг 1: готовим raw, PCA и skeleton-представления пары.
    raw_pred = bool_mask_to_debug_image(pred_mask)
    raw_target = bool_mask_to_debug_image(target_mask)
    pca_pred = bool_mask_to_debug_image(pred_aligned)
    pca_target = bool_mask_to_debug_image(target_aligned)
    skeleton_pred = bool_mask_to_debug_image(skeletonize(pred_aligned))
    skeleton_target = bool_mask_to_debug_image(skeletonize(target_aligned))

    # Шаг 2: приводим все панели к одной высоте.
    panels = [raw_pred, raw_target, pca_pred, pca_target, skeleton_pred, skeleton_target]
    max_height = max(panel.shape[0] for panel in panels)
    resized_panels = []
    for panel in panels:
        if panel.shape[0] != max_height:
            new_width = max(1, int(panel.shape[1] * max_height / panel.shape[0]))
            panel = cv2.resize(panel, (new_width, max_height), interpolation=cv2.INTER_NEAREST)
        resized_panels.append(cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR))

    # Шаг 3: сохраняем одну горизонтальную картинку для пары.
    separator = np.full((max_height, 8, 3), 230, dtype=np.uint8)
    collage = resized_panels[0]
    for panel in resized_panels[1:]:
        collage = np.hstack([collage, separator, panel])

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DEBUG_DIR / f"{debug_prefix}_matched_pair_{matched_idx:03d}.jpg"), collage)


def render_points_debug(points_xy: np.ndarray,
                        vectors: List[Tuple[np.ndarray, np.ndarray]] = None,
                        canvas_size: Tuple[int, int] = (360, 520)) -> np.ndarray:
    """
    Короткое описание:
        рисует облако точек и векторы на debug-холсте.
    Вход:
        points_xy: np.ndarray -- точки формы N x 2 в координатах x, y.
        vectors: List[Tuple[np.ndarray, np.ndarray]] -- пары origin, vector.
        canvas_size: Tuple[int, int] -- размер холста H x W.
    Выход:
        np.ndarray -- BGR debug-изображение.
    """
    # Шаг 1: создаем белый холст и проверяем, что точки не пустые.
    canvas_h, canvas_w = canvas_size
    image = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    if len(points_xy) == 0:
        return image

    # Шаг 2: нормируем координаты точек в границы холста.
    min_xy = points_xy.min(axis=0)
    max_xy = points_xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    scale = min((canvas_w - 40) / span[0], (canvas_h - 40) / span[1])

    def to_canvas(point_xy: np.ndarray) -> Tuple[int, int]:
        shifted = (point_xy - min_xy) * scale + np.array([20.0, 20.0])
        x = int(np.clip(round(float(shifted[0])), 0, canvas_w - 1))
        y = int(np.clip(round(float(shifted[1])), 0, canvas_h - 1))
        return x, y

    # Шаг 3: рисуем точки.
    for point_xy in points_xy:
        cv2.circle(image, to_canvas(point_xy), 1, (0, 0, 0), -1)

    # Шаг 4: рисуем переданные векторы поверх облака.
    if vectors:
        colors = [(0, 0, 255), (0, 160, 0), (255, 0, 0)]
        for idx, (origin, vector) in enumerate(vectors):
            color = colors[idx % len(colors)]
            start = to_canvas(origin)
            end = to_canvas(origin + vector)
            cv2.arrowedLine(image, start, end, color, 2, tipLength=0.2)
            cv2.circle(image, start, 4, color, -1)
    return image


def save_pca_debug(mask: np.ndarray,
                   points_xy: np.ndarray,
                   centered: np.ndarray,
                   aligned: np.ndarray,
                   scaled: np.ndarray,
                   result: np.ndarray,
                   mean_xy: np.ndarray,
                   main_axis: np.ndarray,
                   second_axis: np.ndarray,
                   scale: float,
                   offset_x: float,
                   offset_y: float) -> None:
    """
    Короткое описание:
        сохраняет подробную визуализацию этапов PCA-выравнивания.
    Вход:
        mask: np.ndarray -- исходная булева маска.
        points_xy: np.ndarray -- исходные точки x, y.
        centered: np.ndarray -- центрированные точки.
        aligned: np.ndarray -- точки после поворота PCA.
        scaled: np.ndarray -- точки после масштабирования.
        result: np.ndarray -- итоговая булева маска.
        mean_xy: np.ndarray -- центр масс точек.
        main_axis: np.ndarray -- главный PCA-вектор.
        second_axis: np.ndarray -- второй PCA-вектор.
        scale: float -- коэффициент масштабирования.
        offset_x: float -- сдвиг по x на холсте.
        offset_y: float -- сдвиг по y на холсте.
    Выход:
        None
    """
    global PCA_DEBUG_COUNTER

    if not DEBUG or PCA_DEBUG_COUNTER >= PCA_DEBUG_MAX_SAVES:
        return

    # Шаг 1: создаем отдельный префикс для одного вызова PCA.
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    debug_prefix = DEBUG_DIR / f"pca_{PCA_DEBUG_COUNTER:03d}"
    PCA_DEBUG_COUNTER += 1

    # Шаг 2: сохраняем исходную маску.
    cv2.imwrite(
        str(debug_prefix.with_name(f"{debug_prefix.name}_01_input_mask.jpg")),
        bool_mask_to_debug_image(mask),
    )

    # Шаг 3: сохраняем исходные точки с главным и вторым PCA-вектором.
    original_vectors = [
        (mean_xy, main_axis * 80.0),
        (mean_xy, second_axis * 80.0),
    ]
    original_debug = render_points_debug(points_xy, original_vectors)
    cv2.imwrite(str(debug_prefix.with_name(f"{debug_prefix.name}_02_original_vectors.jpg")), original_debug)

    # Шаг 4: сохраняем центрированные точки и оси PCA из начала координат.
    centered_vectors = [
        (np.array([0.0, 0.0], dtype=np.float32), main_axis * 80.0),
        (np.array([0.0, 0.0], dtype=np.float32), second_axis * 80.0),
    ]
    centered_debug = render_points_debug(centered, centered_vectors)
    cv2.imwrite(str(debug_prefix.with_name(f"{debug_prefix.name}_03_centered_vectors.jpg")), centered_debug)

    # Шаг 5: сохраняем точки после поворота PCA.
    aligned_debug = render_points_debug(aligned)
    cv2.imwrite(str(debug_prefix.with_name(f"{debug_prefix.name}_04_rotated_points.jpg")), aligned_debug)

    # Шаг 6: сохраняем масштабированные точки перед финальным округлением.
    canvas_points = scaled + np.array([offset_x, offset_y], dtype=np.float32)
    canvas_debug = render_points_debug(canvas_points, canvas_size=(PCA_CANVAS_SIZE[0] * 3, PCA_CANVAS_SIZE[1] * 3))
    cv2.imwrite(str(debug_prefix.with_name(f"{debug_prefix.name}_05_canvas_points.jpg")), canvas_debug)

    # Шаг 7: сохраняем итоговую PCA-маску.
    cv2.imwrite(
        str(debug_prefix.with_name(f"{debug_prefix.name}_06_final_mask.jpg")),
        bool_mask_to_debug_image(result),
    )

    # Шаг 8: сохраняем численные параметры PCA.
    info = {
        "mean_xy": mean_xy.astype(float).tolist(),
        "main_axis_xy": main_axis.astype(float).tolist(),
        "second_axis_xy": second_axis.astype(float).tolist(),
        "scale": float(scale),
        "offset_x": float(offset_x),
        "offset_y": float(offset_y),
        "input_shape": list(mask.shape),
        "output_shape": list(result.shape),
        "points_count": int(len(points_xy)),
    }
    debug_prefix.with_name(f"{debug_prefix.name}_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def find_image_path(stem: str) -> Path:
    """
    Короткое описание:
        находит изображение по имени txt-файла разметки.
    Вход:
        stem: str -- имя файла без расширения.
    Выход:
        Path -- путь к изображению.
    """
    # Шаг 1: перебираем допустимые расширения изображения.
    for extension in IMAGE_EXTENSIONS:
        image_path = IMAGES_DIR / f"{stem}{extension}"
        if image_path.exists():
            return image_path

    # Шаг 2: явно падаем, если соответствующая картинка не найдена.
    raise FileNotFoundError(f"Не найдено изображение для {stem}")


def get_label_paths() -> List[Path]:
    """
    Короткое описание:
        получает список файлов разметки из ручного списка или из директории.
    Вход:
        None
    Выход:
        List[Path] -- список путей к txt-файлам разметки.
    """
    # Шаг 1: если задан ручной список, используем только его.
    if LABEL_NAMES:
        return [
            LABELS_DIR / (label_name if label_name.endswith(".txt") else f"{label_name}.txt")
            for label_name in LABEL_NAMES
        ]

    # Шаг 2: иначе берем первые MAX_IMAGES файлов из директории.
    return sorted(LABELS_DIR.glob("*.txt"))[:MAX_IMAGES]


def read_yolo_polygons(label_path: Path, image_width: int, image_height: int) -> List[np.ndarray]:
    """
    Короткое описание:
        читает YOLO-seg полигоны и переводит нормированные координаты в пиксели.
    Вход:
        label_path: Path -- путь к txt-разметке.
        image_width: int -- ширина изображения.
        image_height: int -- высота изображения.
    Выход:
        List[np.ndarray] -- список полигонов формы N x 2.
    """
    polygons: List[np.ndarray] = []

    # Шаг 1: читаем txt построчно, не загружая лишние файлы.
    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()
            if len(values) < 7:
                continue

            # Шаг 2: переводим нормированные координаты YOLO-seg в пиксели.
            coords = np.array(values[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] = np.clip(coords[:, 0] * image_width, 0, image_width - 1)
            coords[:, 1] = np.clip(coords[:, 1] * image_height, 0, image_height - 1)
            polygons.append(coords.astype(np.int32))
    return polygons


def crop_bool_mask(mask: np.ndarray) -> np.ndarray:
    """
    Короткое описание:
        вырезает минимальный прямоугольник вокруг True-пикселей.
    Вход:
        mask: np.ndarray -- булева маска.
    Выход:
        np.ndarray -- обрезанная булева маска.
    """
    # Шаг 1: находим границы всех True-пикселей.
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((1, 1), dtype=bool)

    # Шаг 2: возвращаем только минимальный прямоугольник вокруг объекта.
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def masks_from_polygons(polygons: List[np.ndarray], binary: np.ndarray) -> List[np.ndarray]:
    """
    Короткое описание:
        строит target-маски строк как черные пиксели внутри полигонов.
    Вход:
        polygons: List[np.ndarray] -- полигоны строк в пикселях.
        binary: np.ndarray -- бинарное изображение, где текст равен 0.
    Выход:
        List[np.ndarray] -- список булевых масок строк.
    """
    masks: List[np.ndarray] = []
    black_pixels = binary == 0

    # Шаг 1: для каждого полигона создаем отдельную маску области.
    for polygon in polygons:
        polygon_mask = np.zeros(binary.shape[:2], dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [polygon], 255)

        # Шаг 2: target -- только черные пиксели текста внутри полигона.
        line_mask = np.logical_and(polygon_mask == 255, black_pixels)
        if int(line_mask.sum()) >= MIN_TEXT_PIXELS:
            masks.append(crop_bool_mask(line_mask))
    return masks


def masks_from_yolo_model(image: np.ndarray, binary: np.ndarray, model: YOLO) -> List[np.ndarray]:
    """
    Короткое описание:
        получает строки YOLO-моделью и оставляет черные пиксели внутри предсказанных масок.
    Вход:
        image: np.ndarray -- исходное изображение BGR.
        binary: np.ndarray -- бинарное изображение, где текст равен 0.
        model: YOLO -- загруженная YOLO-seg модель строк.
    Выход:
        List[np.ndarray] -- список булевых масок строк.
    """
    # Шаг 1: запускаем YOLO-seg модель строк на одном изображении.
    results = model(image, conf=YOLO_CONF_THRESHOLD, verbose=False)
    if not results or results[0].masks is None:
        return []

    masks: List[np.ndarray] = []
    image_height, image_width = binary.shape[:2]
    black_pixels = binary == 0

    # Шаг 2: каждую YOLO-маску приводим к размеру исходного изображения.
    for mask in results[0].masks.data.cpu().numpy():
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        mask_big = cv2.resize(mask_uint8, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

        # Шаг 3: оставляем только черные пиксели текста внутри предсказанной строки.
        line_mask = np.logical_and(mask_big == 255, black_pixels)
        if int(line_mask.sum()) >= MIN_TEXT_PIXELS:
            masks.append(crop_bool_mask(line_mask))
    return masks


def masks_from_hpp(image_path: Path) -> List[np.ndarray]:
    """
    Короткое описание:
        получает строки методом HPP и переводит множества пикселей в булевые маски.
    Вход:
        image_path: Path -- путь к изображению.
    Выход:
        List[np.ndarray] -- список булевых масок строк.
    """
    # Шаг 1: запускаем HPP-сегментацию строк.
    segmenter = LineSegmentation(debug=False)
    hpp_result = segmenter.segment_lines(image_path=str(image_path))
    if isinstance(hpp_result, tuple):
        line_pixels = hpp_result[0]
    else:
        line_pixels = hpp_result

    masks: List[np.ndarray] = []

    # Шаг 2: каждое множество пикселей переводим в компактную булеву маску.
    for pixels in line_pixels:
        if len(pixels) < MIN_TEXT_PIXELS:
            continue

        coords = np.array(list(pixels), dtype=np.int32)
        xs = coords[:, 0]
        ys = coords[:, 1]
        mask = np.zeros((ys.max() - ys.min() + 1, xs.max() - xs.min() + 1), dtype=bool)
        mask[ys - ys.min(), xs - xs.min()] = True
        masks.append(mask)
    return masks


def class_matrix_from_yolo_model(image: np.ndarray, binary: np.ndarray, model: YOLO) -> np.ndarray:
    """
    Короткое описание:
        строит class-matrix строк по YOLO-lines: 0 фон, 1 первая строка, 2 вторая и т.д.
    Вход:
        image: np.ndarray -- исходное изображение BGR.
        binary: np.ndarray -- бинарное изображение, где текст равен 0.
        model: YOLO -- загруженная YOLO-seg модель строк.
    Выход:
        np.ndarray -- матрица классов размера исходного изображения.
    """
    class_matrix = np.zeros(binary.shape[:2], dtype=np.int32)
    results = model(image, conf=YOLO_CONF_THRESHOLD, verbose=False)
    if not results or results[0].masks is None:
        return class_matrix

    image_height, image_width = binary.shape[:2]
    black_pixels = binary == 0
    line_items = []
    for mask in results[0].masks.data.cpu().numpy():
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        mask_big = cv2.resize(mask_uint8, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        line_mask = np.logical_and(mask_big == 255, black_pixels)
        if int(line_mask.sum()) < MIN_TEXT_PIXELS:
            continue
        ys = np.where(line_mask)[0]
        line_items.append((float(np.mean(ys)), line_mask))

    # Сортировка сверху вниз нужна, чтобы номера классов были стабильнее.
    line_items.sort(key=lambda item: item[0])
    for class_idx, (_, line_mask) in enumerate(line_items, start=1):
        class_matrix[line_mask] = class_idx
    return class_matrix


def class_matrix_from_hpp(image_path: Path) -> np.ndarray:
    """
    Короткое описание:
        получает full-size class-matrix строк методом HPP.
    Вход:
        image_path: Path -- путь к изображению.
    Выход:
        np.ndarray -- матрица классов в координатах исходного изображения.
    """
    segmenter = LineSegmentation(debug=False)
    hpp_result = segmenter.segment_lines(image_path=str(image_path), return_class_matrix=True)
    if len(hpp_result) < 3 or hpp_result[2] is None:
        image = cv2.imread(str(image_path))
        if image is None:
            return np.zeros((1, 1), dtype=np.int32)
        return np.zeros(image.shape[:2], dtype=np.int32)
    return hpp_result[2]


def text_iou_and_dice(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Короткое описание:
        считает IoU и Dice только по факту текста class > 0, без учета номера строки.
    Вход:
        pred: np.ndarray -- предсказанная class-matrix.
        target: np.ndarray -- target class-matrix.
    Выход:
        Dict[str, float] -- text_iou и text_dice.
    """
    pred_text = pred > 0
    target_text = target > 0
    intersection = int(np.sum(np.logical_and(pred_text, target_text)))
    union = int(np.sum(np.logical_or(pred_text, target_text)))
    denominator = int(np.sum(pred_text) + np.sum(target_text))
    return {
        "text_iou": float(intersection / union) if union > 0 else 1.0,
        "text_dice": float(2.0 * intersection / denominator) if denominator > 0 else 1.0,
    }


def evaluate_class_matrix(pred_matrix: np.ndarray,
                          target_matrix: np.ndarray) -> Dict[str, object]:
    """
    Короткое описание:
        считает честные метрики class-matrix по аналогии с grade_hpp/experiment_find_line_regions.
    Вход:
        pred_matrix: np.ndarray -- предсказанная class-matrix.
        target_matrix: np.ndarray -- target class-matrix.
    Выход:
        Dict[str, object] -- metrics и per_class.
    """
    pred_metric, target_metric = grade_hpp.prepare_pair_for_metrics(pred_matrix, target_matrix)
    pred_metric, alignment_metrics = grade_hpp.align_pred_by_text_intersection(
        pred_metric,
        target_metric,
        max_shift=MAX_MASK_ALIGNMENT_SHIFT,
    )

    metrics = {
        "cross_entropy": grade_hpp.deterministic_cross_entropy(pred_metric, target_metric, text_only=False),
        "cross_entropy_text_only": grade_hpp.deterministic_cross_entropy(pred_metric, target_metric, text_only=True),
        "pixel_accuracy": float(np.mean(pred_metric == target_metric)),
        "text_pixel_accuracy": float(np.mean(pred_metric[target_metric > 0] == target_metric[target_metric > 0]))
        if int(np.sum(target_metric > 0)) > 0 else 0.0,
    }
    metrics.update(text_iou_and_dice(pred_metric, target_metric))
    metrics.update(alignment_metrics)
    metrics.update(grade_hpp.line_detection_metrics(pred_metric, target_metric))

    return {
        "metrics": metrics,
        "per_class": grade_hpp.per_class_metrics(pred_metric, target_metric),
    }


def save_class_matrix_debug(stem: str,
                            method_name: str,
                            pred_matrix: np.ndarray,
                            target_matrix: np.ndarray) -> None:
    """
    Короткое описание:
        сохраняет debug class-matrix для нового сравнения.
    Вход:
        stem: str -- имя изображения.
        method_name: str -- hpp или yolo.
        pred_matrix: np.ndarray -- предсказанная матрица.
        target_matrix: np.ndarray -- target матрица.
    Выход:
        None
    """
    if not DEBUG:
        return
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    pred_metric, target_metric = grade_hpp.prepare_pair_for_metrics(pred_matrix, target_matrix)
    pred_metric, _ = grade_hpp.align_pred_by_text_intersection(
        pred_metric,
        target_metric,
        max_shift=MAX_MASK_ALIGNMENT_SHIFT,
    )
    cv2.imwrite(
        str(DEBUG_DIR / f"{stem}_{method_name}_pred_classes.png"),
        grade_hpp.class_matrix_to_color(pred_metric),
    )
    cv2.imwrite(
        str(DEBUG_DIR / f"{stem}_target_classes.png"),
        grade_hpp.class_matrix_to_color(target_metric),
    )


def pca_align_mask(mask: np.ndarray, canvas_size: Tuple[int, int] = PCA_CANVAS_SIZE) -> np.ndarray:
    """
    Короткое описание:
        выравнивает маску PCA и переносит ее на фиксированный холст.
    Вход:
        mask: np.ndarray -- булева маска.
        canvas_size: Tuple[int, int] -- размер холста H x W.
    Выход:
        np.ndarray -- булева маска на фиксированном холсте.
    """
    # Шаг 1: получаем координаты пикселей маски.
    points_yx = np.argwhere(mask)
    canvas_h, canvas_w = canvas_size
    result = np.zeros((canvas_h, canvas_w), dtype=bool)
    if len(points_yx) == 0:
        return result

    # Шаг 2: центрируем точки и поворачиваем их по главной PCA-оси.
    points_xy = points_yx[:, ::-1].astype(np.float32)
    mean_xy = points_xy.mean(axis=0)
    centered = points_xy - mean_xy
    if len(points_xy) >= 2:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        main_axis = vt[0].astype(np.float32)
        second_axis = vt[1].astype(np.float32)
        aligned = centered @ vt.T
    else:
        main_axis = np.array([1.0, 0.0], dtype=np.float32)
        second_axis = np.array([0.0, 1.0], dtype=np.float32)
        aligned = centered

    # Шаг 3: масштабируем результат на фиксированный холст.
    min_xy = aligned.min(axis=0)
    max_xy = aligned.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    scale = min((canvas_w - 4) / span[0], (canvas_h - 4) / span[1])
    scaled = (aligned - min_xy) * scale
    offset_x = (canvas_w - scaled[:, 0].max()) / 2.0
    offset_y = (canvas_h - scaled[:, 1].max()) / 2.0
    xs = np.clip(np.round(scaled[:, 0] + offset_x).astype(np.int32), 0, canvas_w - 1)
    ys = np.clip(np.round(scaled[:, 1] + offset_y).astype(np.int32), 0, canvas_h - 1)

    # Шаг 4: переносим точки на итоговую маску.
    result[ys, xs] = True
    save_pca_debug(
        mask,
        points_xy,
        centered,
        aligned,
        scaled,
        result,
        mean_xy,
        main_axis,
        second_axis,
        scale,
        offset_x,
        offset_y,
    )
    return result


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Короткое описание:
        считает IoU двух булевых масок одинакового размера.
    Вход:
        mask_a: np.ndarray -- первая маска.
        mask_b: np.ndarray -- вторая маска.
    Выход:
        float -- IoU.
    """
    # Шаг 1: считаем пересечение и объединение.
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    # Шаг 2: защищаемся от пустого union.
    return float(intersection / union) if union > 0 else 0.0


def vertical_hpp_words(mask: np.ndarray) -> List[np.ndarray]:
    """
    Короткое описание:
        делит строку на слова по вертикальному HPP.
    Вход:
        mask: np.ndarray -- булева маска строки.
    Выход:
        List[np.ndarray] -- список булевых масок слов.
    """
    # Шаг 1: строим вертикальный профиль строки.
    projection = mask.sum(axis=0)
    text_cols = projection > 0
    words: List[np.ndarray] = []
    start = None
    last_text = None

    # Шаг 2: режем строку на слова по длинным пустым промежуткам.
    for idx, is_text in enumerate(text_cols):
        if is_text and start is None:
            start = idx
        if is_text:
            last_text = idx
        if start is not None and last_text is not None and idx - last_text > VERTICAL_WORD_GAP:
            end = last_text
            if end - start + 1 >= MIN_WORD_WIDTH:
                words.append(crop_bool_mask(mask[:, start:end + 1]))
            start = None
            last_text = None

    # Шаг 3: добавляем последнее слово, если оно дошло до конца строки.
    if start is not None and last_text is not None and last_text - start + 1 >= MIN_WORD_WIDTH:
        words.append(crop_bool_mask(mask[:, start:last_text + 1]))
    return words


def skeleton_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Короткое описание:
        считает IoU скелетов двух PCA-выровненных масок.
    Вход:
        mask_a: np.ndarray -- первая маска.
        mask_b: np.ndarray -- вторая маска.
    Выход:
        float -- IoU скелетов.
    """
    # Шаг 1: строим скелеты двух масок.
    skeleton_a = skeletonize(mask_a)
    skeleton_b = skeletonize(mask_b)

    # Шаг 2: сравниваем скелеты через IoU.
    return iou(skeleton_a, skeleton_b)


def pairwise_distance_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Короткое описание:
        сравнивает отсортированные матрицы попарных расстояний точек масок.
    Вход:
        mask_a: np.ndarray -- первая PCA-выровненная маска.
        mask_b: np.ndarray -- вторая PCA-выровненная маска.
    Выход:
        float -- оценка от 0 до 1, где 1 лучше.
    """
    # Шаг 1: берем координаты точек двух масок.
    points_a = np.argwhere(mask_a)
    points_b = np.argwhere(mask_b)
    if len(points_a) < 2 or len(points_b) < 2:
        return 0.0

    # Шаг 2: ограничиваем число точек, чтобы не раздувать RAM.
    points_a = points_a[np.linspace(0, len(points_a) - 1, min(PAIRWISE_SAMPLE_SIZE, len(points_a))).astype(int)]
    points_b = points_b[np.linspace(0, len(points_b) - 1, min(PAIRWISE_SAMPLE_SIZE, len(points_b))).astype(int)]

    # Шаг 3: считаем и сортируем попарные расстояния внутри каждой маски.
    distances_a = np.sort(pdist(points_a).astype(np.float32))
    distances_b = np.sort(pdist(points_b).astype(np.float32))
    size = min(len(distances_a), len(distances_b))
    if size == 0:
        return 0.0

    # Шаг 4: нормализуем расстояния и переводим ошибку в score.
    distances_a = distances_a[:size] / (distances_a[:size].max() + 1e-6)
    distances_b = distances_b[:size] / (distances_b[:size].max() + 1e-6)
    return float(1.0 / (1.0 + np.mean(np.abs(distances_a - distances_b))))


def chamfer_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Короткое описание:
        считает симметричную Chamfer-оценку между точками масок.
    Вход:
        mask_a: np.ndarray -- первая PCA-выровненная маска.
        mask_b: np.ndarray -- вторая PCA-выровненная маска.
    Выход:
        float -- оценка от 0 до 1, где 1 лучше.
    """
    # Шаг 1: берем координаты точек двух масок.
    points_a = np.argwhere(mask_a)
    points_b = np.argwhere(mask_b)
    if len(points_a) == 0 or len(points_b) == 0:
        return 0.0

    # Шаг 2: ограничиваем число точек для экономии памяти.
    points_a = points_a[np.linspace(0, len(points_a) - 1, min(PAIRWISE_SAMPLE_SIZE, len(points_a))).astype(int)]
    points_b = points_b[np.linspace(0, len(points_b) - 1, min(PAIRWISE_SAMPLE_SIZE, len(points_b))).astype(int)]

    # Шаг 3: считаем симметричное среднее расстояние до ближайших точек.
    distances = cdist(points_a, points_b)
    chamfer = float(distances.min(axis=1).mean() + distances.min(axis=0).mean()) / 2.0
    diagonal = float(np.linalg.norm(PCA_CANVAS_SIZE))

    # Шаг 4: переводим расстояние в score.
    return float(1.0 / (1.0 + chamfer / diagonal))


def word_skeleton_score(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """
    Короткое описание:
        сравнивает слова после vertical HPP, PCA-выравнивания и скелетизации.
    Вход:
        pred_mask: np.ndarray -- маска предсказанной строки.
        target_mask: np.ndarray -- маска целевой строки.
    Выход:
        float -- средняя оценка соответствия слов.
    """
    # Шаг 1: режем предсказанную и целевую строки на слова vertical HPP.
    pred_words = vertical_hpp_words(pred_mask)
    target_words = vertical_hpp_words(target_mask)
    if not pred_words or not target_words:
        return 0.0

    # Шаг 2: строим матрицу skeleton IoU для всех пар слов.
    scores = np.zeros((len(pred_words), len(target_words)), dtype=np.float32)
    for i, pred_word in enumerate(pred_words):
        pred_aligned = pca_align_mask(pred_word)
        for j, target_word in enumerate(target_words):
            target_aligned = pca_align_mask(target_word)
            scores[i, j] = skeleton_iou(pred_aligned, target_aligned)

    # Шаг 3: сопоставляем слова венгерским алгоритмом и усредняем score.
    row_indices, col_indices = linear_sum_assignment(-scores)
    return float(scores[row_indices, col_indices].sum() / max(len(pred_words), len(target_words)))


def match_and_score(pred_masks: List[np.ndarray],
                    target_masks: List[np.ndarray],
                    debug_prefix: str = "") -> Dict[str, float]:
    """
    Короткое описание:
        сопоставляет предсказанные и целевые строки и считает метрики качества.
    Вход:
        pred_masks: List[np.ndarray] -- маски предсказанных строк.
        target_masks: List[np.ndarray] -- маски целевых строк.
        debug_prefix: str -- префикс имени debug-файла.
    Выход:
        Dict[str, float] -- набор метрик.
    """
    # Шаг 1: инициализируем метрики с учетом количества строк.
    metrics = {
        "pred_count": float(len(pred_masks)),
        "target_count": float(len(target_masks)),
        "line_count_abs_error": float(abs(len(pred_masks) - len(target_masks))),
        "pca_iou": 0.0,
        "word_skeleton_iou": 0.0,
        "pairwise_distance_score": 0.0,
        "chamfer_score": 0.0,
        "matched_fraction": 0.0,
    }
    if not pred_masks or not target_masks:
        return metrics

    # Шаг 2: PCA-выравниваем строки на общий холст.
    pred_aligned = [pca_align_mask(mask) for mask in pred_masks]
    target_aligned = [pca_align_mask(mask) for mask in target_masks]

    # Шаг 3: строим матрицы score между всеми парами строк.
    pca_iou_matrix = np.zeros((len(pred_masks), len(target_masks)), dtype=np.float32)
    pairwise_matrix = np.zeros((len(pred_masks), len(target_masks)), dtype=np.float32)
    chamfer_matrix = np.zeros((len(pred_masks), len(target_masks)), dtype=np.float32)
    for i, pred_mask in enumerate(pred_aligned):
        for j, target_mask in enumerate(target_aligned):
            pca_iou_matrix[i, j] = iou(pred_mask, target_mask)
            pairwise_matrix[i, j] = pairwise_distance_score(pred_mask, target_mask)
            chamfer_matrix[i, j] = chamfer_score(pred_mask, target_mask)

    # Шаг 4: собираем общий score и выбираем лучшие пары венгерским алгоритмом.
    score_matrix = (
        MATCH_PCA_IOU_WEIGHT * pca_iou_matrix
        + MATCH_PAIRWISE_WEIGHT * pairwise_matrix
        + MATCH_CHAMFER_WEIGHT * chamfer_matrix
    )

    row_indices, col_indices = linear_sum_assignment(-score_matrix)
    matched_count = len(row_indices)
    if matched_count == 0:
        return metrics

    # Шаг 5: считаем все метрики для найденных пар строк.
    pca_ious = []
    word_scores = []
    distance_scores = []
    chamfer_scores = []
    for pred_idx, target_idx in zip(row_indices, col_indices):
        pca_ious.append(float(pca_iou_matrix[pred_idx, target_idx]))
        word_scores.append(word_skeleton_score(pred_masks[pred_idx], target_masks[target_idx]))
        distance_scores.append(float(pairwise_matrix[pred_idx, target_idx]))
        chamfer_scores.append(float(chamfer_matrix[pred_idx, target_idx]))

    # Шаг 5.5: если включен debug, сохраняем лучшие сопоставленные пары по общему score.
    if DEBUG and debug_prefix:
        matched_scores = score_matrix[row_indices, col_indices]
        best_pair_indices = np.argsort(-matched_scores)[:DEBUG_MAX_MATCHED_PAIRS]
        for debug_idx, pair_idx in enumerate(best_pair_indices):
            first_pred_idx = int(row_indices[pair_idx])
            first_target_idx = int(col_indices[pair_idx])
            save_matched_pair_debug(
                pred_masks[first_pred_idx],
                target_masks[first_target_idx],
                pred_aligned[first_pred_idx],
                target_aligned[first_target_idx],
                debug_prefix,
                debug_idx,
            )

    # Шаг 6: усредняем метрики и добавляем долю сопоставленных строк.
    metrics["pca_iou"] = float(np.mean(pca_ious))
    metrics["word_skeleton_iou"] = float(np.mean(word_scores))
    metrics["pairwise_distance_score"] = float(np.mean(distance_scores))
    metrics["chamfer_score"] = float(np.mean(chamfer_scores))
    metrics["matched_fraction"] = float(matched_count / max(len(pred_masks), len(target_masks)))
    return metrics


def evaluate_image(label_path: Path, yolo_model: YOLO, unet_model, unet_device) -> Dict[str, Dict[str, float]]:
    """
    Короткое описание:
        считает target-маски, HPP-ответ, YOLO-ответ и метрики для одного изображения.
    Вход:
        label_path: Path -- путь к txt-разметке строк.
        yolo_model: YOLO -- загруженная модель строк.
        unet_model: Any -- загруженная U-Net модель.
        unet_device: Any -- устройство U-Net.
    Выход:
        Dict[str, Dict[str, float]] -- метрики двух методов.
    """
    # Шаг 1: находим и читаем исходное изображение.
    image_path = find_image_path(label_path.stem)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    # Шаг 2: бинаризуем изображение через уже загруженную U-Net.
    binary = binarize_image_with_loaded_model(image, unet_model, unet_device)
    image_height, image_width = binary.shape[:2]

    # Шаг 3: строим target class-matrix из YOLO-seg разметки и черных пикселей.
    polygons = read_yolo_polygons(label_path, image_width, image_height)
    target_matrix = grade_hpp.build_target_class_matrix(polygons, binary)

    # Шаг 4: получаем предсказания YOLO-lines и HPP как full-size class-matrix.
    yolo_matrix = class_matrix_from_yolo_model(image, binary, yolo_model)
    hpp_matrix = class_matrix_from_hpp(image_path)
    if DEBUG:
        save_class_matrix_debug(label_path.stem, "yolo", yolo_matrix, target_matrix)
        save_class_matrix_debug(label_path.stem, "hpp", hpp_matrix, target_matrix)

    # Шаг 5: считаем честные class-matrix метрики двух методов относительно target.
    hpp_score = evaluate_class_matrix(hpp_matrix, target_matrix)
    yolo_score = evaluate_class_matrix(yolo_matrix, target_matrix)
    result = {
        "target_shape": [int(target_matrix.shape[0]), int(target_matrix.shape[1])],
        "hpp_shape": [int(hpp_matrix.shape[0]), int(hpp_matrix.shape[1])],
        "yolo_shape": [int(yolo_matrix.shape[0]), int(yolo_matrix.shape[1])],
        "hpp": hpp_score,
        "yolo": yolo_score,
    }

    # Шаг 6: очищаем тяжелые объекты после изображения.
    del image, binary, polygons, target_matrix, yolo_matrix, hpp_matrix
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def average_method_metrics(results: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """
    Короткое описание:
        усредняет новые metrics отдельно для HPP и YOLO.
    Вход:
        results: Dict[str, Dict] -- результаты по изображениям.
    Выход:
        Dict[str, Dict[str, float]] -- средние метрики методов.
    """
    averaged: Dict[str, Dict[str, float]] = {}
    for method_name in ("hpp", "yolo"):
        metric_values: Dict[str, List[float]] = {}
        for result in results.values():
            if method_name not in result or "metrics" not in result[method_name]:
                continue
            for key, value in result[method_name]["metrics"].items():
                metric_values.setdefault(key, []).append(float(value))
        averaged[method_name] = {
            key: float(np.mean(values))
            for key, values in metric_values.items()
            if values
        }
    return averaged


def main() -> None:
    """
    Короткое описание:
        запускает сравнение YOLO и HPP на размеченных изображениях и сохраняет JSON.
    Вход:
        None
    Выход:
        None
    """
    # Шаг 1: собираем список файлов разметки и загружаем модели один раз.
    label_paths = get_label_paths()
    yolo_model = YOLO(str(YOLO_LINES_MODEL_PATH))
    unet_model, unet_device = load_unet_model(str(UNET_MODEL_PATH))

    # Шаг 2: обрабатываем изображения по одному, не держа датасет в памяти.
    results: Dict[str, Dict] = {}
    for label_path in tqdm(label_paths, desc="comparison_yolo_hpp"):
        try:
            results[label_path.stem] = evaluate_image(label_path, yolo_model, unet_model, unet_device)
        except Exception as exc:
            results[label_path.stem] = {"error": str(exc)}

    # Шаг 3: сохраняем итоговые метрики в JSON.
    output = {
        "label_names": LABEL_NAMES,
        "average_metrics": average_method_metrics(results),
        "results": results,
    }
    OUTPUT_JSON_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
