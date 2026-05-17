import json
import gc
import heapq
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from tqdm import tqdm


# Корень проекта code. Файл лежит в experiments/experiment_1_compare_paper_hough.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing import correct_perspective

# Папка для всех debug-артефактов нового автономного метода.
DEBUG_IMAGES_DIR = str(PROJECT_ROOT / "debug_images" / "experiment_1_compare_paper_hough" / "louloudis_exact")

# Путь к изображению для запуска файла напрямую. Если пусто, скрипт только сообщает, как его настроить.
INPUT_IMAGE_PATH = '/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code/datasets/HWR200/hw_dataset/165/reuse0/ФотоСветлое/2.jpg'

# Сохранять подробные debug-изображения и json-отчет при прямом запуске.
DEBUG = True

# В экспериментальной версии включаем внешний поиск страницы: сначала выделяем лист,
# затем применяем метод статьи к найденной странице.
USE_YOLO_PAGE_SEGMENTATION = True

# YOLO 3_2 segmentation модель страницы тетради.
YOLO_PAGE_SEGMENTATION_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolo_segment_notebook"
    / "yolo_segment_notebook_3_(2-architecture).pt"
)

# Минимальная уверенность YOLO для маски страницы. Держим низко и выбираем лучший
# кандидат по confidence, чтобы не отрезать страницу ранним порогом.
YOLO_PAGE_CONF = 0.05

# Размер входа YOLO. Для модели yolo_segment_notebook_3_(2-architecture)
# на HWR200 стабильнее работает 640, как в ручном test.py.
YOLO_PAGE_IMGSZ = 640

# Устройство YOLO. None означает, что Ultralytics сам выберет доступное устройство.
YOLO_PAGE_DEVICE = None

# U-Net модель бинаризации.
UNET_BINARIZATION_MODEL_PATH = PROJECT_ROOT / "models" / "u_net" / "unet_binarization_3_(6-architecture).pth"

# Размер входа U-Net, на котором обучалась модель.
UNET_TARGET_SIZE = (3000, 3000)

# Порог sigmoid для U-Net.
UNET_THRESHOLD = 0.5

# Устройство U-Net. None означает auto: cuda если доступна, иначе cpu.
UNET_DEVICE = None

# Вариант my: Subset 2 перед Hough делится через HPP + energy + A*, а не skeleton.
USE_PRE_HOUGH_SUBSET2_ENERGY_SPLIT = True

# Вариант my: первый bin гистограммы высот связных компонент отсекается как шумовой.
USE_ROBUST_AH_SKIP_FIRST_BIN = True

# Вариант my: центр диапазона theta оценивается глобально по максимуму дисперсии HPP.
USE_DYNAMIC_HOUGH_THETA_CENTER = True

# Отклонение theta вокруг глобально оцененного угла.
DYNAMIC_HOUGH_THETA_DEVIATION_DEG = 5.0

# Шаг перебора углов при оценке глобального наклона по HPP.
GLOBAL_SKEW_ANGLE_MIN_DEG = -30.0
GLOBAL_SKEW_ANGLE_MAX_DEG = 30.0
GLOBAL_SKEW_ANGLE_STEPS = 121

# Параметры HPP/energy/A* для локального деления Subset 2.
SUBSET2_HPP_SMOOTH_KERNEL_FACTOR = 0.5

# Сохранять отдельные файлы вида *_my_subset2_hpp_*.json/png для каждой Subset 2 компоненты.
# По умолчанию выключено: иначе debug быстро превращается в сотни мелких файлов.
SAVE_SUBSET2_HPP_DEBUG_FILES = False

# Сохранять отдельные картинки A*/energy split для каждой Subset 2 компоненты.
# По умолчанию выключено по той же причине: для одной страницы таких компонент могут быть сотни.
SAVE_SUBSET2_ENERGY_SPLIT_DEBUG_FILES = False

# Компактный overview split Subset 2: исходная компонента, corrected crop,
# HPP-регионы, A* швы и итоговые части. Включено для ручного debug.
SAVE_SUBSET2_SPLIT_OVERVIEW_DEBUG_FILES = True
MAX_SUBSET2_SPLIT_OVERVIEW_DEBUG_FILES = 80

SUBSET2_HPP_MIN_REGION_HEIGHT_FACTOR = 0.25
SUBSET2_TEXT_PIXEL_COST = 10.0
SUBSET2_GRADIENT_COST = 0.5
SUBSET2_BLOCK_REGION_TOP_PADDING = 1
SUBSET2_BLOCK_REGION_BOTTOM_PADDING = 2


# Минимальная площадь компоненты. В статье Louloudis et al. 2008 после бинаризации
# сказано извлечь connected components; дополнительный порог площади не задан.
MIN_COMPONENT_AREA = 1

# Минимальная высота компоненты для AH. В статье отдельный порог не задан.
MIN_COMPONENT_HEIGHT_FOR_AH = 1

# Размер bin гистограммы AH: один пиксель.
AH_HISTOGRAM_BIN_WIDTH = 1

# По статье средняя ширина символа AW принимается равной AH.
AW_EQUALS_AH = True

# Границы Subset 1 из Louloudis et al. 2008.
SUBSET1_MIN_HEIGHT_FACTOR = 0.5
SUBSET1_MAX_HEIGHT_FACTOR = 3.0
SUBSET1_MIN_WIDTH_FACTOR = 0.5

# Граница Subset 2 из Louloudis et al. 2008.
SUBSET2_MIN_HEIGHT_FACTOR = 3.0

# Границы Subset 3 из Louloudis et al. 2008.
SUBSET3_MAX_HEIGHT_FACTOR = 3.0
SUBSET3_SMALL_HEIGHT_FACTOR = 0.5
SUBSET3_NARROW_WIDTH_FACTOR = 0.5

# Диапазон Hough theta из статьи: 85..95 градусов с разрешением 1 градус.
HOUGH_THETA_MIN_DEG = 85
HOUGH_THETA_MAX_DEG = 95
HOUGH_THETA_STEP_DEG = 1

# Разрешение rho из статьи: 0.2 * AH.
HOUGH_RHO_STEP_AH_FACTOR = 0.2

# Окрестность rho-пика из статьи: ячейки pi - 5 ... pi + 5 при фиксированном theta.
HOUGH_RHO_NEIGHBOR_CELLS = 5

# Порог остановки n1 из статьи, в экспериментах n1 = 5.
HOUGH_MIN_VOTES_N1 = 5

# Порог вторичной проверки n2 из статьи, в экспериментах n2 = 9.
HOUGH_SECONDARY_VOTES_N2 = 9

# Максимальное отклонение вторичной линии от доминирующего угла, когда votes < n2.
HOUGH_DOMINANT_ANGLE_TOLERANCE_DEG = 2.0

# Компонента относится к линии, если не меньше трети ее block-точек попали в область линии.
HOUGH_COMPONENT_MIN_BLOCK_FRACTION = 0.3

# Порог создания новых строк из статьи: расстояние до ближайшей строки должно быть около 0.9 * Ad.
LINE_DISTANCE_FACTOR = 0.5

# Допуск для формулы dis ~= 0.9 * Ad: в статье задано приближенное равенство,
# но не раскрыт численный интервал "close".
NEW_LINE_DISTANCE_TOLERANCE_FACTOR = 0.25

# Вертикальный допуск группировки новых строк относительно AH.
# В статье сказано "a grouping technique", но точная формула группировки не раскрыта.
NEW_LINE_GROUPING_AH_FACTOR = 0.8

# Размер окрестности удаления junction point из статьи: 3 x 3.
JUNCTION_REMOVAL_NEIGHBORHOOD = 3

# Зона split для Subset 2 по формуле статьи hc / 2 < y < 3 * hc / 2 с обрезкой по высоте компоненты.
SUBSET2_ZONE_TOP_FACTOR = 0.5
SUBSET2_ZONE_BOTTOM_FACTOR = 1.5

# Цвета debug-разметки в BGR.
DEBUG_COLOR_SUBSET1 = (0, 180, 0)
DEBUG_COLOR_SUBSET2 = (0, 0, 220)
DEBUG_COLOR_SUBSET3 = (220, 0, 0)
DEBUG_COLOR_LINE = (0, 0, 255)
DEBUG_COLOR_POINT = (0, 180, 255)


class LouloudisTextLineDetector:
    """
    Подробное описание:
        Автономно воспроизводит метод Louloudis et al. 2008 для обнаружения строк
        рукописного текста. Бинаризация выполняется через U-Net, далее идут
        оценка AH, разбиение компонент
        на три подмножества, block-based Hough, создание пропущенных строк и
        разделение вертикально соединенных компонент Subset 2.
    """

    def __init__(
        self,
        image: np.ndarray,
        debug: bool = False,
        debug_output_dir: str = DEBUG_IMAGES_DIR,
        use_tqdm: bool = True,
        unet_model: Any = None,
        unet_device: Any = None,
    ):
        """
        Короткое описание:
            Инициализирует детектор строк.
        Вход:
            image: np.ndarray -- исходное изображение BGR, RGB или grayscale.
            debug: bool -- сохранять debug-артефакты.
            debug_output_dir: str -- папка для debug-файлов.
            use_tqdm: bool -- показывать tqdm на тяжелых циклах.
            unet_model: Any -- заранее загруженная U-Net модель или None.
            unet_device: Any -- устройство заранее загруженной U-Net модели.
        Выход:
            None
        """
        self.image = image.copy()
        self.debug = bool(debug)
        self.debug_output_dir = str(debug_output_dir)
        self.use_tqdm = bool(use_tqdm)
        self.unet_model = unet_model
        self.unet_device = unet_device
        self.debug_counter = 0

        self.gray = self._to_gray(self.image)
        self.filtered_gray: Optional[np.ndarray] = None
        self.rough_foreground: Optional[np.ndarray] = None
        self.background_surface: Optional[np.ndarray] = None
        self.binary: Optional[np.ndarray] = None
        self.components: List[Dict[str, Any]] = []
        self.subset1: List[Dict[str, Any]] = []
        self.subset2: List[Dict[str, Any]] = []
        self.subset3: List[Dict[str, Any]] = []
        self.lines: List[Dict[str, Any]] = []
        self.my_subset2_split_debug: List[Dict[str, Any]] = []
        self.my_hough_debug: List[Dict[str, Any]] = []
        self.my_component_size_histogram_debug: Dict[str, Any] = {}
        self.average_character_height = 0.0
        self.average_character_width = 0.0

        if self.debug:
            os.makedirs(self.debug_output_dir, exist_ok=True)

    def detect(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Короткое описание:
            Запускает полный pipeline и возвращает class-matrix строк.
        Вход:
            None
        Выход:
            Tuple[np.ndarray, List[Dict[str, Any]]] -- матрица классов H x W и список строк.
        """
        # Шаг 1: бинаризуем изображение через U-Net.
        self.binary = self.binarize_and_enhance()

        # Шаг 2: извлекаем связные компоненты и оцениваем AH/AW.
        self.components = self.extract_connected_components(self.binary)
        self.average_character_height = self.estimate_average_character_height(self.components)
        self.average_character_width = self.average_character_height

        # Шаг 3: делим компоненты на Subset 1, Subset 2, Subset 3.
        self.partition_connected_components()

        # Шаг 4: в варианте my заранее делим Subset 2 и добавляем части в Subset 1.
        if USE_PRE_HOUGH_SUBSET2_ENERGY_SPLIT:
            self.split_subset2_by_energy_before_hough()

        # Шаг 5: ищем первичные строки block-based Hough по Subset 1.
        self.lines = self.block_based_hough_transform()

        # Шаг 6: исправляем ложное дробление.
        self.merge_close_text_lines()
        # В варианте my этап создания новых строк отключен:
        # неприсвоенные компоненты далее присваиваются ближайшим найденным строкам.

        # Шаг 7: присваиваем Subset 3. Subset 2 уже обработан до Hough.
        self.assign_remaining_components_to_lines()
        if not USE_PRE_HOUGH_SUBSET2_ENERGY_SPLIT:
            self.split_subset2_and_assign_to_lines()

        # Шаг 8: строим итоговую class-matrix и debug.
        class_matrix = self.build_class_matrix()
        if self.debug:
            self.save_debug_summary(class_matrix)
        return class_matrix, self.lines

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        """
        Короткое описание:
            Преобразует изображение к grayscale uint8.
        Вход:
            image: np.ndarray -- исходное изображение.
        Выход:
            np.ndarray -- grayscale изображение.
        """
        # Шаг 1: обрабатываем grayscale и цветные изображения.
        if len(image.shape) == 2:
            gray = image.copy()
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Шаг 2: приводим тип к uint8.
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray

    def _save_debug_image(self, image: np.ndarray, name: str) -> None:
        """
        Короткое описание:
            Сохраняет debug-изображение с порядковым номером.
        Вход:
            image: np.ndarray -- изображение для сохранения.
            name: str -- смысловое имя файла.
        Выход:
            None
        """
        if not self.debug:
            return
        os.makedirs(self.debug_output_dir, exist_ok=True)
        output_path = os.path.join(self.debug_output_dir, f"{self.debug_counter:03d}_{name}.png")
        cv2.imwrite(output_path, image)
        self.debug_counter += 1

    def _binary_to_debug(self, binary: np.ndarray) -> np.ndarray:
        """
        Короткое описание:
            Переводит внутреннюю бинарную маску 0/1 в картинку 255/0 для просмотра.
        Вход:
            binary: np.ndarray -- маска, где текст равен 1.
        Выход:
            np.ndarray -- изображение, где текст черный, фон белый.
        """
        return np.where(binary > 0, 0, 255).astype(np.uint8)

    def binarize_and_enhance(self) -> np.ndarray:
        """
        Короткое описание:
            Бинаризует изображение только через U-Net.
        Вход:
            None
        Выход:
            np.ndarray -- бинарная маска 0/1, где текст равен 1.
        """
        return self.binarize_with_unet()

    def binarize_with_unet(self) -> np.ndarray:
        """
        Короткое описание:
            Бинаризует изображение только U-Net без дополнительного post-processing.
        Вход:
            None
        Выход:
            np.ndarray -- бинарная маска 0/1, где текст равен 1.
        """
        # Шаг 1: проверяем веса и лениво импортируем U-Net.
        if not UNET_BINARIZATION_MODEL_PATH.exists():
            raise FileNotFoundError(f"Не найдена U-Net модель бинаризации: {UNET_BINARIZATION_MODEL_PATH}")
        try:
            from u_net_binarization import binarize_image_with_loaded_model, load_unet_model
        except ImportError as error:
            raise ImportError("Не удалось импортировать u_net_binarization.py из корня проекта") from error

        # Шаг 2: используем переданную модель или загружаем локально для одного изображения.
        owns_model = self.unet_model is None
        if owns_model:
            model, device = load_unet_model(
                model_path=str(UNET_BINARIZATION_MODEL_PATH),
                device=UNET_DEVICE,
            )
        else:
            model = self.unet_model
            device = self.unet_device

        try:
            unet_mask = binarize_image_with_loaded_model(
                image=self.image,
                model=model,
                device=device,
                target_size=UNET_TARGET_SIZE,
                threshold=UNET_THRESHOLD,
                debug=False,
            )
        finally:
            if owns_model:
                del model
                gc.collect()

        # Шаг 3: U-Net возвращает 0 для текста и 255 для фона, переводим во внутренний формат 1/0.
        if len(unet_mask.shape) == 3:
            unet_mask = cv2.cvtColor(unet_mask, cv2.COLOR_BGR2GRAY)
        binary = (unet_mask < 128).astype(np.uint8)
        self._save_debug_image(self._binary_to_debug(binary), "01_unet_binary")

        return binary.astype(np.uint8)

    def extract_connected_components(self, binary: np.ndarray) -> List[Dict[str, Any]]:
        """
        Короткое описание:
            Извлекает связные компоненты текста из бинарной маски.
        Вход:
            binary: np.ndarray -- маска 0/1, где текст равен 1.
        Выход:
            List[Dict[str, Any]] -- список компонент с bbox, mask, centroid, area.
        """
        # Шаг 1: строим 8-связную разметку компонент.
        labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        components: List[Dict[str, Any]] = []

        # Шаг 2: сохраняем компоненты без фона.
        for label in range(1, labels_count):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < MIN_COMPONENT_AREA:
                continue
            component_mask = (labels[y:y + height, x:x + width] == label).astype(np.uint8)
            components.append(
                {
                    "id": len(components),
                    "bbox": (x, y, width, height),
                    "mask": component_mask,
                    "centroid": (float(centroids[label][0]), float(centroids[label][1])),
                    "area": area,
                    "assigned": False,
                }
            )

        if self.debug:
            self._save_components_debug(components, "06_connected_components")
        return components

    def estimate_average_character_height_from_binary(self, binary: np.ndarray) -> float:
        """
        Короткое описание:
            Оценивает среднюю высоту символа как среднее высот без первого bin.
        Вход:
            binary: np.ndarray -- бинарная маска 0/1.
        Выход:
            float -- средняя высота символа.
        """
        # Шаг 1: извлекаем высоты компонент.
        labels_count, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        heights = [
            int(stats[label, cv2.CC_STAT_HEIGHT])
            for label in range(1, labels_count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= MIN_COMPONENT_AREA
            and int(stats[label, cv2.CC_STAT_HEIGHT]) >= MIN_COMPONENT_HEIGHT_FOR_AH
        ]

        # Шаг 2: удаляем первый bin гистограммы и считаем weighted mean высот.
        if len(heights) == 0:
            return 10.0
        values, counts = np.unique(np.asarray(heights, dtype=np.int32), return_counts=True)
        if USE_ROBUST_AH_SKIP_FIRST_BIN and len(values) > 1:
            values = values[1:]
            counts = counts[1:]
        return float(np.sum(values.astype(np.float64) * counts.astype(np.float64)) / max(1.0, float(np.sum(counts))))

    def estimate_average_character_height(self, components: List[Dict[str, Any]]) -> float:
        """
        Короткое описание:
            Оценивает AH как среднее bounding-box heights компонент без первого bin.
        Вход:
            components: List[Dict[str, Any]] -- список связных компонент.
        Выход:
            float -- средняя высота символа AH.
        """
        # Шаг 1: собираем высоты и ширины компонент.
        heights = [
            int(component["bbox"][3])
            for component in components
            if int(component["bbox"][3]) >= MIN_COMPONENT_HEIGHT_FOR_AH
        ]
        widths = [
            int(component["bbox"][2])
            for component in components
            if int(component["bbox"][3]) >= MIN_COMPONENT_HEIGHT_FOR_AH
        ]
        if len(heights) == 0:
            return 10.0

        # Шаг 2: строим гистограмму с bin width = 1, удаляем первый bin и считаем weighted mean.
        height_values_all, height_counts_all = np.unique(np.asarray(heights, dtype=np.int32), return_counts=True)
        width_values_all, width_counts_all = np.unique(np.asarray(widths, dtype=np.int32), return_counts=True)
        height_values_used = height_values_all
        height_counts_used = height_counts_all
        skipped_first_height_bin = False
        skipped_height_bin = None
        if USE_ROBUST_AH_SKIP_FIRST_BIN and len(height_values_all) > 1:
            skipped_first_height_bin = True
            skipped_height_bin = {
                "height": int(height_values_all[0]),
                "count": int(height_counts_all[0]),
            }
            height_values_used = height_values_all[1:]
            height_counts_used = height_counts_all[1:]
        average_height = float(
            np.sum(height_values_used.astype(np.float64) * height_counts_used.astype(np.float64))
            / max(1.0, float(np.sum(height_counts_used)))
        )

        # Шаг 3: сохраняем debug-гистограмму высот и ширин.
        self.my_component_size_histogram_debug = {
            "average_character_height": average_height,
            "average_character_width": average_height if AW_EQUALS_AH else None,
            "bin_width_px": int(AH_HISTOGRAM_BIN_WIDTH),
            "robust_skip_first_height_bin": bool(USE_ROBUST_AH_SKIP_FIRST_BIN),
            "skipped_first_height_bin": bool(skipped_first_height_bin),
            "skipped_height_bin": skipped_height_bin,
            "height_bins_total": int(len(height_values_all)),
            "height_bins_used_for_ah": int(len(height_values_used)),
            "width_bins_total": int(len(width_values_all)),
            "width_bins_used_for_aw": 0 if AW_EQUALS_AH else int(len(width_values_all)),
            "height_histogram_all": {
                "values": [int(value) for value in height_values_all.tolist()],
                "counts": [int(value) for value in height_counts_all.tolist()],
            },
            "height_histogram_used_for_ah": {
                "values": [int(value) for value in height_values_used.tolist()],
                "counts": [int(value) for value in height_counts_used.tolist()],
            },
            "width_histogram_all": {
                "values": [int(value) for value in width_values_all.tolist()],
                "counts": [int(value) for value in width_counts_all.tolist()],
            },
            "height_count_used_for_mean": int(np.sum(height_counts_used)),
            "note": "В my-версии AH считается как weighted mean высот connected components после удаления первого bin; AW принимается равной AH.",
        }
        if self.debug:
            path = os.path.join(self.debug_output_dir, f"{self.debug_counter:03d}_07_component_size_histograms.json")
            with open(path, "w", encoding="utf-8") as file:
                json.dump(self.my_component_size_histogram_debug, file, indent=2, ensure_ascii=False)
            self.debug_counter += 1
            histogram_image = self.render_component_size_histograms(
                height_values_all,
                height_counts_all,
                height_values_used,
                height_counts_used,
                width_values_all,
                width_counts_all,
                average_height,
                skipped_height_bin,
            )
            self._save_debug_image(histogram_image, "07_component_size_histograms")
        return average_height

    def render_component_size_histograms(
        self,
        height_values_all: np.ndarray,
        height_counts_all: np.ndarray,
        height_values_used: np.ndarray,
        height_counts_used: np.ndarray,
        width_values_all: np.ndarray,
        width_counts_all: np.ndarray,
        average_height: float,
        skipped_height_bin: Optional[Dict[str, int]],
    ) -> np.ndarray:
        """
        Короткое описание:
            Рисует debug-картинку гистограмм высот и ширин connected components.
        Вход:
            height_values_all: np.ndarray -- все bin высот.
            height_counts_all: np.ndarray -- счетчики всех bin высот.
            height_values_used: np.ndarray -- bin высот после робастного отсечения.
            height_counts_used: np.ndarray -- счетчики используемых bin высот.
            width_values_all: np.ndarray -- все bin ширин.
            width_counts_all: np.ndarray -- счетчики всех bin ширин.
            average_height: float -- средний AH после удаления первого bin.
            skipped_height_bin: Optional[Dict[str, int]] -- отсеченный первый bin.
        Выход:
            np.ndarray -- BGR debug-изображение.
        """
        canvas = np.full((620, 1100, 3), 255, dtype=np.uint8)
        self.draw_histogram_panel(
            canvas,
            (40, 70, 500, 250),
            height_values_all,
            height_counts_all,
            "Height histogram: all bins",
            highlight_value=None,
            skipped_value=skipped_height_bin["height"] if skipped_height_bin is not None else None,
        )
        self.draw_histogram_panel(
            canvas,
            (560, 70, 1020, 250),
            height_values_used,
            height_counts_used,
            "Height histogram: used for AH",
            highlight_value=None,
            skipped_value=None,
        )
        self.draw_histogram_panel(
            canvas,
            (40, 350, 500, 530),
            width_values_all,
            width_counts_all,
            "Width histogram: debug only",
            highlight_value=None,
            skipped_value=None,
        )
        info_lines = [
            f"AH mean = {average_height:.1f}px; AW = AH",
            f"height bins total: {len(height_values_all)}",
            f"height bins used: {len(height_values_used)}",
            f"height components used: {int(np.sum(height_counts_used))}",
            f"width bins total: {len(width_values_all)}",
            f"skip first height bin: {bool(skipped_height_bin)}",
        ]
        if skipped_height_bin is not None:
            info_lines.append(f"skipped: h={skipped_height_bin['height']} count={skipped_height_bin['count']}")
        for index, line in enumerate(info_lines):
            cv2.putText(canvas, line, (560, 360 + 32 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 40), 2)
        return canvas

    def draw_histogram_panel(
        self,
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        values: np.ndarray,
        counts: np.ndarray,
        title: str,
        highlight_value: Optional[int],
        skipped_value: Optional[int],
    ) -> None:
        """
        Короткое описание:
            Рисует один panel гистограммы на canvas.
        Вход:
            canvas: np.ndarray -- BGR изображение.
            rect: Tuple[int, int, int, int] -- область panel.
            values: np.ndarray -- значения bin.
            counts: np.ndarray -- счетчики bin.
            title: str -- заголовок.
            highlight_value: Optional[int] -- выделяемый bin.
            skipped_value: Optional[int] -- отсеченный bin.
        Выход:
            None
        """
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (220, 220, 220), 1)
        cv2.putText(canvas, title, (x1, y1 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
        if len(values) == 0:
            return
        max_count = max(1, int(np.max(counts)))
        panel_width = max(1, x2 - x1 - 20)
        panel_height = max(1, y2 - y1 - 30)
        bar_width = max(1, int(panel_width / max(1, len(values))))
        max_visible_bins = min(len(values), max(1, int(panel_width / bar_width)))
        for index in range(max_visible_bins):
            value = int(values[index])
            count = int(counts[index])
            left = x1 + 10 + index * bar_width
            right = min(x2 - 10, left + bar_width - 1)
            bar_height = int(round((count / max_count) * panel_height))
            color = (90, 120, 210)
            if highlight_value is not None and value == highlight_value:
                color = (0, 170, 0)
            if skipped_value is not None and value == skipped_value:
                color = (0, 0, 220)
            cv2.rectangle(canvas, (left, y2 - 20 - bar_height), (right, y2 - 20), color, -1)
        cv2.putText(canvas, f"bins={len(values)} max_count={max_count}", (x1 + 8, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1)

    def partition_connected_components(self) -> None:
        """
        Короткое описание:
            Делит connected components на Subset 1, Subset 2 и Subset 3 по формулам статьи.
        Вход:
            None
        Выход:
            None
        """
        # Шаг 1: готовим пороги через AH и AW.
        ah = float(self.average_character_height)
        aw = float(self.average_character_width)
        self.subset1 = []
        self.subset2 = []
        self.subset3 = []

        # Шаг 2: применяем три формулы из статьи.
        for component in self.components:
            _, _, width, height = component["bbox"]
            in_subset1 = (
                SUBSET1_MIN_HEIGHT_FACTOR * ah <= height < SUBSET1_MAX_HEIGHT_FACTOR * ah
                and SUBSET1_MIN_WIDTH_FACTOR * aw <= width
            )
            in_subset2 = height >= SUBSET2_MIN_HEIGHT_FACTOR * ah
            in_subset3 = (
                (height < SUBSET3_MAX_HEIGHT_FACTOR * ah and width < SUBSET3_NARROW_WIDTH_FACTOR * aw)
                or (height < SUBSET3_SMALL_HEIGHT_FACTOR * ah and SUBSET3_NARROW_WIDTH_FACTOR * aw < width)
            )

            if in_subset1:
                component["subset"] = 1
                self.subset1.append(component)
            elif in_subset2:
                component["subset"] = 2
                self.subset2.append(component)
            elif in_subset3:
                component["subset"] = 3
                self.subset3.append(component)
            else:
                component["subset"] = 3
                self.subset3.append(component)

        if self.debug:
            self._save_partition_debug()

    def _save_components_debug(self, components: List[Dict[str, Any]], name: str) -> None:
        """
        Короткое описание:
            Сохраняет bbox связных компонент.
        Вход:
            components: List[Dict[str, Any]] -- компоненты для отрисовки.
            name: str -- имя debug-файла.
        Выход:
            None
        """
        # Шаг 1: готовим цветное изображение.
        debug_image = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)

        # Шаг 2: рисуем bbox.
        for component in components:
            x, y, width, height = component["bbox"]
            cv2.rectangle(debug_image, (x, y), (x + width, y + height), (0, 180, 0), 1)
        self._save_debug_image(debug_image, name)

    def _save_partition_debug(self) -> None:
        """
        Короткое описание:
            Сохраняет цветную визуализацию Subset 1, Subset 2, Subset 3.
        Вход:
            None
        Выход:
            None
        """
        # Шаг 1: готовим изображение.
        debug_image = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)

        # Шаг 2: рисуем три подмножества разными цветами.
        for component in self.subset1:
            x, y, width, height = component["bbox"]
            cv2.rectangle(debug_image, (x, y), (x + width, y + height), DEBUG_COLOR_SUBSET1, 2)
        for component in self.subset2:
            x, y, width, height = component["bbox"]
            cv2.rectangle(debug_image, (x, y), (x + width, y + height), DEBUG_COLOR_SUBSET2, 2)
        for component in self.subset3:
            x, y, width, height = component["bbox"]
            cv2.rectangle(debug_image, (x, y), (x + width, y + height), DEBUG_COLOR_SUBSET3, 2)
        self._save_debug_image(debug_image, "08_partition_subsets")

    def split_subset2_by_energy_before_hough(self) -> None:
        """
        Короткое описание:
            Делит Subset 2 до голосования Hough через HPP + energy + A* и добавляет части в Subset 1.
        Вход:
            None
        Выход:
            None
        """
        # Шаг 1: обрабатываем большие компоненты и переносим результат в Subset 1.
        if len(self.subset2) == 0:
            return
        if self.debug:
            self._save_partition_debug()
        new_subset1_components: List[Dict[str, Any]] = []
        for component_index, component in enumerate(self.subset2):
            parts = self.split_subset2_component_by_energy(component, component_index)
            self.my_subset2_split_debug.append(
                {
                    "component_index": int(component_index),
                    "source_bbox": [int(value) for value in component["bbox"]],
                    "source_area": int(component.get("area", 0)),
                    "parts_count": int(len(parts)),
                    "parts_bboxes": [[int(value) for value in part["bbox"]] for part in parts],
                    "parts_areas": [int(part.get("area", 0)) for part in parts],
                }
            )
            for part in parts:
                part["subset"] = 1
                part["assigned"] = False
                new_subset1_components.append(part)
        self.subset1.extend(new_subset1_components)
        self.subset2 = []
        if self.debug:
            self._save_partition_debug()

    def split_subset2_component_by_energy(self, component: Dict[str, Any], component_index: int) -> List[Dict[str, Any]]:
        """
        Короткое описание:
            Делит одну большую компоненту через локальное выравнивание, HPP, energy и A*.
        Вход:
            component: Dict[str, Any] -- компонента Subset 2.
            component_index: int -- индекс компоненты для debug.
        Выход:
            List[Dict[str, Any]] -- части компоненты в исходных координатах.
        """
        # Шаг 1: готовим бинарный crop компоненты: текст 0, фон 255.
        source_x, source_y, width, height = component["bbox"]
        if width < 3 or height < 3:
            return [component]
        local_text_mask = component["mask"].astype(np.uint8)
        local_binary = np.where(local_text_mask > 0, 0, 255).astype(np.uint8)

        # Шаг 2: локально выравниваем компоненту через correct_perspective.
        try:
            _, corrected_binary, _, affine_matrix = correct_perspective(
                local_binary,
                debug=False,
                return_matrix=True,
            )
        except Exception:
            return [component]
        corrected_binary = np.where(corrected_binary < 128, 0, 255).astype(np.uint8)

        # Шаг 3: ищем выраженные HPP-регионы. Если нескольких пиков нет, считаем компоненту единой.
        line_regions, hpp_debug = self.find_subset2_hpp_regions(corrected_binary, return_debug=True)
        if len(line_regions) < 2:
            if self.debug:
                self.save_subset2_hpp_debug(component_index, corrected_binary, hpp_debug, line_regions)
                self.save_subset2_split_overview_debug(
                    component=component,
                    local_binary=local_binary,
                    corrected_binary=corrected_binary,
                    line_regions=line_regions,
                    seams=[],
                    corrected_part_masks=[],
                    restored_part_masks=[],
                    component_index=component_index,
                    status="not_split_less_than_two_hpp_regions",
                    hpp_debug=hpp_debug,
                )
            return [component]

        # Шаг 4: строим energy и A* швы между соседними HPP-регионами.
        energy = self.compute_subset2_energy(corrected_binary)
        blocked_mask = self.build_subset2_blocked_mask(corrected_binary.shape, line_regions)
        seams = []
        for region_index in range(len(line_regions) - 1):
            _, end_prev = line_regions[region_index]
            start_next, _ = line_regions[region_index + 1]
            start_y = int(round((end_prev + start_next) / 2.0))
            seam = self.find_subset2_a_star_seam(energy, blocked_mask, start_y)
            if len(seam) == corrected_binary.shape[1]:
                seams.append(seam)
        if len(seams) == 0:
            if self.debug:
                self.save_subset2_split_overview_debug(
                    component=component,
                    local_binary=local_binary,
                    corrected_binary=corrected_binary,
                    line_regions=line_regions,
                    seams=[],
                    corrected_part_masks=[],
                    restored_part_masks=[],
                    component_index=component_index,
                    status="not_split_no_a_star_seams",
                    hpp_debug=hpp_debug,
                )
            return [component]
        seams = sorted(seams, key=lambda item: float(np.mean(item)))

        # Шаг 5: по швам строим классы в выровненном crop и возвращаем их в исходный bbox.
        corrected_parts = self.split_corrected_binary_by_seams(corrected_binary, seams)
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)
        original_parts = []
        restored_part_masks = []
        for part_mask in corrected_parts:
            restored_mask = cv2.warpAffine(
                part_mask.astype(np.uint8),
                inverse_matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            restored_mask = ((restored_mask > 0) & (local_text_mask > 0)).astype(np.uint8)
            restored_part_masks.append(restored_mask)
            part = self.component_from_local_mask(restored_mask, component)
            if part is not None:
                original_parts.append(part)

        if self.debug:
            self.save_subset2_hpp_debug(component_index, corrected_binary, hpp_debug, line_regions)
            self.save_subset2_energy_split_debug(component, corrected_binary, line_regions, seams, original_parts, component_index)
            self.save_subset2_split_overview_debug(
                component=component,
                local_binary=local_binary,
                corrected_binary=corrected_binary,
                line_regions=line_regions,
                seams=seams,
                corrected_part_masks=corrected_parts,
                restored_part_masks=restored_part_masks,
                component_index=component_index,
                status="split" if len(original_parts) >= 2 else "not_split_less_than_two_restored_parts",
                hpp_debug=hpp_debug,
            )
        return original_parts if len(original_parts) >= 2 else [component]

    def find_subset2_hpp_regions(self, binary: np.ndarray, return_debug: bool = False) -> Any:
        """
        Короткое описание:
            Находит ярко выраженные HPP-пики внутри Subset 2 компоненты.
        Вход:
            binary: np.ndarray -- бинарный crop, текст 0, фон 255.
        Выход:
            Any -- интервалы строк-пиков или пара (regions, debug).
        """
        # Шаг 1: строим и сглаживаем HPP.
        hpp = np.sum(binary == 0, axis=1).astype(np.float32)
        if float(np.max(hpp)) <= 0.0:
            empty_debug = {"hpp": [], "smoothed_hpp": [], "threshold": 0.0, "max_value": 0.0}
            return ([], empty_debug) if return_debug else []
        kernel_size = max(3, int(round(SUBSET2_HPP_SMOOTH_KERNEL_FACTOR * max(1.0, self.average_character_height))))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        smoothed = np.convolve(hpp, kernel, mode="same")

        # Шаг 2: ищем локальные пики по адаптивному порогу среднего уровня (без доли от max).
        max_value = float(np.max(smoothed))
        threshold = float(np.mean(smoothed))
        peak_mask = smoothed >= threshold
        regions = []
        row = 0
        min_height = max(1, int(round(SUBSET2_HPP_MIN_REGION_HEIGHT_FACTOR * max(1.0, self.average_character_height))))
        while row < len(peak_mask):
            if not peak_mask[row]:
                row += 1
                continue
            start = row
            while row + 1 < len(peak_mask) and peak_mask[row + 1]:
                row += 1
            end = row
            if end - start + 1 >= min_height:
                regions.append((int(start), int(end)))
            row += 1
        hpp_debug = {
            "hpp": hpp.astype(float).tolist(),
            "smoothed_hpp": smoothed.astype(float).tolist(),
            "threshold": float(threshold),
            "max_value": float(max_value),
            "kernel_size": int(kernel_size),
            "min_region_height": int(min_height),
        }
        return (regions, hpp_debug) if return_debug else regions

    def save_subset2_hpp_debug(
        self,
        component_index: int,
        corrected_binary: np.ndarray,
        hpp_debug: Dict[str, Any],
        line_regions: List[Tuple[int, int]],
    ) -> None:
        """
        Короткое описание:
            Сохраняет подробный debug HPP-пиков для Subset 2 компоненты.
        Вход:
            component_index: int -- индекс компоненты.
            corrected_binary: np.ndarray -- выровненный бинарный crop.
            hpp_debug: Dict[str, Any] -- численные HPP-данные.
            line_regions: List[Tuple[int, int]] -- найденные пики.
        Выход:
            None
        """
        if not self.debug or not SAVE_SUBSET2_HPP_DEBUG_FILES:
            return
        # Шаг 1: сохраняем картинку с найденными HPP-пиками.
        vis = cv2.cvtColor(corrected_binary, cv2.COLOR_GRAY2BGR)
        for start, end in line_regions:
            cv2.rectangle(vis, (0, int(start)), (vis.shape[1] - 1, int(end)), (0, 180, 0), 1)
        self._save_debug_image(vis, f"my_subset2_hpp_regions_{component_index:03d}")

        # Шаг 2: сохраняем json с профилем, адаптивным порогом и регионами.
        path = os.path.join(self.debug_output_dir, f"{self.debug_counter:03d}_my_subset2_hpp_{component_index:03d}.json")
        payload = dict(hpp_debug)
        payload["line_regions"] = [[int(start), int(end)] for start, end in line_regions]
        payload["component_index"] = int(component_index)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        self.debug_counter += 1

    def compute_subset2_energy(self, binary: np.ndarray) -> np.ndarray:
        """
        Короткое описание:
            Строит energy matrix для A*: текст дорогой, границы текста тоже дорогие.
        Вход:
            binary: np.ndarray -- бинарный crop, текст 0, фон 255.
        Выход:
            np.ndarray -- energy matrix.
        """
        # Шаг 1: текстовые пиксели и границы получают повышенную цену.
        energy = (255 - binary).astype(np.float32) * SUBSET2_TEXT_PIXEL_COST
        grad_x = cv2.Sobel(binary, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(binary, cv2.CV_32F, 0, 1, ksize=3)
        energy += SUBSET2_GRADIENT_COST * np.sqrt(grad_x * grad_x + grad_y * grad_y)
        return np.nan_to_num(energy, nan=0.0, posinf=1e9, neginf=0.0)

    def build_subset2_blocked_mask(self, shape: Tuple[int, int], line_regions: List[Tuple[int, int]]) -> np.ndarray:
        """
        Короткое описание:
            Запрещает A* проходить через HPP-регионы строк.
        Вход:
            shape: Tuple[int, int] -- размер H, W.
            line_regions: List[Tuple[int, int]] -- HPP-регионы.
        Выход:
            np.ndarray -- bool-маска запрета.
        """
        # Шаг 1: расширяем найденные регионы на небольшой padding.
        height, _ = shape
        blocked = np.zeros(shape, dtype=bool)
        for start, end in line_regions:
            y0 = max(0, int(start) - SUBSET2_BLOCK_REGION_TOP_PADDING)
            y1 = min(height - 1, int(end) + SUBSET2_BLOCK_REGION_BOTTOM_PADDING)
            blocked[y0:y1 + 1, :] = True
        return blocked

    def find_subset2_a_star_seam(self, energy: np.ndarray, blocked_mask: np.ndarray, start_y: int) -> List[int]:
        """
        Короткое описание:
            Находит горизонтальный шов A* между двумя HPP-регионами.
        Вход:
            energy: np.ndarray -- energy matrix.
            blocked_mask: np.ndarray -- True означает запрет.
            start_y: int -- стартовая строка.
        Выход:
            List[int] -- y-координата шва для каждого x.
        """
        # Шаг 1: инициализируем A* слева направо.
        height, width = energy.shape
        if blocked_mask.shape != energy.shape or not (0 <= start_y < height):
            return []
        if blocked_mask[start_y, 0]:
            free_rows = np.where(~blocked_mask[:, 0])[0]
            if len(free_rows) == 0:
                return []
            start_y = int(free_rows[np.argmin(np.abs(free_rows - start_y))])

        start = (start_y, 0)
        goal_x = width - 1
        open_set = [(float(energy[start_y, 0]), start_y, 0)]
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        g_score: Dict[Tuple[int, int], float] = {start: float(energy[start_y, 0])}
        visited = set()

        # Шаг 2: разрешены только ходы вправо-вверх, вправо, вправо-вниз.
        while open_set:
            _, y, x = heapq.heappop(open_set)
            current = (y, x)
            if current in visited:
                continue
            visited.add(current)
            if x == goal_x:
                seam = [0] * width
                while current is not None:
                    cy, cx = current
                    seam[cx] = cy
                    current = came_from.get(current)
                return seam
            nx = x + 1
            if nx >= width:
                continue
            for dy in (-1, 0, 1):
                ny = y + dy
                if not (0 <= ny < height) or blocked_mask[ny, nx]:
                    continue
                neighbor = (ny, nx)
                tentative = g_score[current] + float(energy[ny, nx])
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    heuristic = float(goal_x - nx)
                    heapq.heappush(open_set, (tentative + heuristic, ny, nx))
        return []

    def split_corrected_binary_by_seams(self, binary: np.ndarray, seams: List[List[int]]) -> List[np.ndarray]:
        """
        Короткое описание:
            Делит выровненный бинарный crop на части между швами.
        Вход:
            binary: np.ndarray -- бинарный crop, текст 0, фон 255.
            seams: List[List[int]] -- A* швы.
        Выход:
            List[np.ndarray] -- маски частей, текст 1.
        """
        # Шаг 1: добавляем верхнюю и нижнюю границы как виртуальные швы.
        height, width = binary.shape
        all_seams = [[0] * width] + seams + [[height - 1] * width]
        parts = []
        text_mask = binary == 0
        for seam_index in range(len(all_seams) - 1):
            upper = all_seams[seam_index]
            lower = all_seams[seam_index + 1]
            part = np.zeros_like(binary, dtype=np.uint8)
            for x in range(width):
                y0 = min(upper[x], lower[x])
                y1 = max(upper[x], lower[x])
                if y1 > y0 + 1:
                    part[y0 + 1:y1, x] = text_mask[y0 + 1:y1, x].astype(np.uint8)
            if np.any(part > 0):
                parts.append(part)
        return parts

    def save_subset2_energy_split_debug(
        self,
        component: Dict[str, Any],
        corrected_binary: np.ndarray,
        line_regions: List[Tuple[int, int]],
        seams: List[List[int]],
        parts: List[Dict[str, Any]],
        component_index: int,
    ) -> None:
        """
        Короткое описание:
            Сохраняет debug локального деления Subset 2.
        Вход:
            component: Dict[str, Any] -- исходная компонента.
            corrected_binary: np.ndarray -- выровненный crop.
            line_regions: List[Tuple[int, int]] -- HPP-регионы.
            seams: List[List[int]] -- найденные швы.
            parts: List[Dict[str, Any]] -- итоговые части.
            component_index: int -- индекс компоненты.
        Выход:
            None
        """
        if not self.debug or not SAVE_SUBSET2_ENERGY_SPLIT_DEBUG_FILES:
            return
        # Шаг 1: рисуем HPP-регионы и A* швы.
        vis = cv2.cvtColor(corrected_binary, cv2.COLOR_GRAY2BGR)
        for start, end in line_regions:
            cv2.rectangle(vis, (0, int(start)), (vis.shape[1] - 1, int(end)), (255, 0, 0), 1)
        for seam in seams:
            for x in range(1, len(seam)):
                cv2.line(vis, (x - 1, int(seam[x - 1])), (x, int(seam[x])), (0, 0, 255), 1)
        self._save_debug_image(vis, f"my_subset2_energy_split_{component_index:03d}")

    def save_subset2_split_overview_debug(
        self,
        component: Dict[str, Any],
        local_binary: np.ndarray,
        corrected_binary: np.ndarray,
        line_regions: List[Tuple[int, int]],
        seams: List[List[int]],
        corrected_part_masks: List[np.ndarray],
        restored_part_masks: List[np.ndarray],
        component_index: int,
        status: str,
        hpp_debug: Dict[str, Any],
    ) -> None:
        """
        Короткое описание:
            Сохраняет простой debug: вырезанная компонента и красная линия seam.
        """
        if not self.debug or not SAVE_SUBSET2_SPLIT_OVERVIEW_DEBUG_FILES:
            return
        if component_index >= MAX_SUBSET2_SPLIT_OVERVIEW_DEBUG_FILES:
            return

        seam_vis = cv2.cvtColor(corrected_binary, cv2.COLOR_GRAY2BGR)
        for seam in seams:
            for x in range(1, len(seam)):
                cv2.line(seam_vis, (x - 1, int(seam[x - 1])), (x, int(seam[x])), (0, 0, 255), 1)
        self._save_debug_image(seam_vis, f"my_subset2_component_red_seam_{component_index:03d}_{status}")


    def compute_block_gravity_centers(self, component: Dict[str, Any]) -> List[Tuple[float, float]]:
        """
        Короткое описание:
            Делит компоненту на блоки ширины AW и считает center of gravity текста в каждом блоке.
        Вход:
            component: Dict[str, Any] -- связная компонента.
        Выход:
            List[Tuple[float, float]] -- центры тяжести блоков в координатах изображения.
        """
        # Шаг 1: готовим bbox и маску.
        x, y, width, _ = component["bbox"]
        mask = component["mask"]
        block_width = max(1, int(round(self.average_character_width)))
        centers: List[Tuple[float, float]] = []

        # Шаг 2: идем блоками фиксированной ширины AW, последний блок может быть меньше.
        for start_col in range(0, width, block_width):
            end_col = min(width, start_col + block_width)
            block = mask[:, start_col:end_col]
            rows, cols = np.where(block > 0)
            if len(rows) == 0:
                continue
            center_x = x + start_col + float(np.mean(cols))
            center_y = y + float(np.mean(rows))
            centers.append((center_x, center_y))
        return centers

    def block_based_hough_transform(self) -> List[Dict[str, Any]]:
        """
        Короткое описание:
            Выполняет block-based Hough transform по Subset 1.
        Вход:
            None
        Выход:
            List[Dict[str, Any]] -- найденные строки с rho, theta и компонентами.
        """
        if len(self.subset1) == 0:
            return []

        # Шаг 1: строим block-точки для всех компонент Subset 1.
        all_points: List[Tuple[float, float]] = []
        point_to_component: List[int] = []
        component_to_points: Dict[int, List[int]] = {}
        for component_index, component in enumerate(self.subset1):
            centers = self.compute_block_gravity_centers(component)
            component["block_centers"] = centers
            component_to_points[component_index] = []
            for center in centers:
                point_index = len(all_points)
                all_points.append(center)
                point_to_component.append(component_index)
                component_to_points[component_index].append(point_index)

        if self.debug:
            self._save_hough_points_debug(all_points)

        # Шаг 2: задаем дискретизацию Hough-пространства.
        height, width = self.binary.shape
        if USE_DYNAMIC_HOUGH_THETA_CENTER:
            theta_center = self.estimate_dynamic_hough_theta_center()
            theta_min = theta_center - DYNAMIC_HOUGH_THETA_DEVIATION_DEG
            theta_max = theta_center + DYNAMIC_HOUGH_THETA_DEVIATION_DEG
            theta_values = np.arange(theta_min, theta_max + HOUGH_THETA_STEP_DEG, HOUGH_THETA_STEP_DEG, dtype=np.float32)
        else:
            theta_values = np.arange(HOUGH_THETA_MIN_DEG, HOUGH_THETA_MAX_DEG + 1, HOUGH_THETA_STEP_DEG, dtype=np.float32)
        rho_step = max(1.0, HOUGH_RHO_STEP_AH_FACTOR * float(self.average_character_height))
        max_rho = float(np.hypot(width, height))
        rho_min = -max_rho
        rho_count = int(np.ceil((2.0 * max_rho) / rho_step)) + 1

        # Шаг 3: итеративно извлекаем самые сильные линии.
        available_points = set(range(len(all_points)))
        assigned_components = set()
        lines: List[Dict[str, Any]] = []

        progress = tqdm(
            desc="Louloudis Hough",
            disable=not self.use_tqdm,
            leave=False,
            total=None,
        )
        iteration_index = 0
        while True:
            iteration_index += 1
            progress.update(1)
            if len(available_points) < HOUGH_MIN_VOTES_N1:
                self.my_hough_debug.append(
                    {
                        "iteration": int(iteration_index),
                        "stop_reason": "available_points_less_than_n1",
                        "available_points": int(len(available_points)),
                    }
                )
                break

            accumulator: Dict[Tuple[int, int], List[int]] = {}
            for point_index in available_points:
                x, y = all_points[point_index]
                for theta_index, theta in enumerate(theta_values):
                    theta_rad = np.deg2rad(float(theta))
                    rho = x * np.cos(theta_rad) + y * np.sin(theta_rad)
                    rho_index = int(round((rho - rho_min) / rho_step))
                    if 0 <= rho_index < rho_count:
                        accumulator.setdefault((rho_index, theta_index), []).append(point_index)

            if len(accumulator) == 0:
                self.my_hough_debug.append(
                    {
                        "iteration": int(iteration_index),
                        "stop_reason": "empty_accumulator",
                        "available_points": int(len(available_points)),
                    }
                )
                break

            best_cell, best_points = max(accumulator.items(), key=lambda item: len(item[1]))
            best_votes = len(best_points)
            if best_votes < HOUGH_MIN_VOTES_N1:
                self.my_hough_debug.append(
                    {
                        "iteration": int(iteration_index),
                        "stop_reason": "best_votes_less_than_n1",
                        "available_points": int(len(available_points)),
                        "best_votes": int(best_votes),
                    }
                )
                break

            rho_index, theta_index = best_cell
            theta = float(theta_values[theta_index])
            rejected_by_angle = False
            if best_votes < HOUGH_SECONDARY_VOTES_N2 and len(lines) > 0:
                dominant_theta = float(np.median([line["theta"] for line in lines]))
                if abs(theta - dominant_theta) > HOUGH_DOMINANT_ANGLE_TOLERANCE_DEG:
                    rejected_by_angle = True
                    self.my_hough_debug.append(
                        {
                            "iteration": int(iteration_index),
                            "stop_reason": None,
                            "action": "reject_by_dominant_angle",
                            "available_points": int(len(available_points)),
                            "best_votes": int(best_votes),
                            "rho_index": int(rho_index),
                            "theta": float(theta),
                            "dominant_theta": float(dominant_theta),
                        }
                    )
                    for point_index in best_points:
                        available_points.discard(point_index)
                    continue

            # Шаг 4: собираем область pi - 5 ... pi + 5 при том же theta.
            neighbor_points = set()
            for neighbor_rho_index in range(rho_index - HOUGH_RHO_NEIGHBOR_CELLS, rho_index + HOUGH_RHO_NEIGHBOR_CELLS + 1):
                neighbor_points.update(accumulator.get((neighbor_rho_index, theta_index), []))

            # Шаг 5: компоненту подтверждаем, если в область попала минимум половина ее block-точек.
            component_votes: Dict[int, int] = {}
            for point_index in neighbor_points:
                component_index = point_to_component[point_index]
                component_votes[component_index] = component_votes.get(component_index, 0) + 1

            line_component_indices: List[int] = []
            for component_index, vote_count in component_votes.items():
                if component_index in assigned_components:
                    continue
                total_blocks = max(1, len(component_to_points.get(component_index, [])))
                if vote_count >= HOUGH_COMPONENT_MIN_BLOCK_FRACTION * total_blocks:
                    line_component_indices.append(component_index)

            if len(line_component_indices) == 0:
                self.my_hough_debug.append(
                    {
                        "iteration": int(iteration_index),
                        "stop_reason": None,
                        "action": "discard_neighbor_points_no_components",
                        "available_points": int(len(available_points)),
                        "best_votes": int(best_votes),
                        "neighbor_points": int(len(neighbor_points)),
                        "rho_index": int(rho_index),
                        "theta": float(theta),
                    }
                )
                for point_index in neighbor_points:
                    available_points.discard(point_index)
                continue

            rho = rho_min + rho_index * rho_step
            self.my_hough_debug.append(
                {
                    "iteration": int(iteration_index),
                    "stop_reason": None,
                    "action": "accept_line",
                    "available_points_before": int(len(available_points)),
                    "best_votes": int(best_votes),
                    "neighbor_points": int(len(neighbor_points)),
                    "rho_index": int(rho_index),
                    "rho": float(rho),
                    "theta": float(theta),
                    "components_count": int(len(line_component_indices)),
                    "component_indices": [int(index) for index in line_component_indices],
                    "component_bboxes": [[int(value) for value in self.subset1[index]["bbox"]] for index in line_component_indices],
                    "rejected_by_angle": bool(rejected_by_angle),
                }
            )
            line = {
                "rho": float(rho),
                "theta": theta,
                "subset1_indices": line_component_indices,
                "components": [self.subset1[index] for index in line_component_indices],
                "source": "hough",
            }
            lines.append(line)

            # Шаг 6: удаляем из Hough все голоса назначенных компонент.
            for component_index in line_component_indices:
                assigned_components.add(component_index)
                self.subset1[component_index]["assigned"] = True
                for point_index in component_to_points.get(component_index, []):
                    available_points.discard(point_index)
        progress.close()

        # Шаг 7: сортируем строки сверху вниз и сохраняем debug.
        lines = self.sort_lines_top_down(lines)
        if self.debug:
            self._save_lines_debug(lines, "10_hough_lines")
        return lines

    def estimate_dynamic_hough_theta_center(self) -> float:
        """
        Короткое описание:
            Оценивает центр theta по глобальному наклону строк через максимум дисперсии HPP.
        Вход:
            None
        Выход:
            float -- центр theta для Hough-пространства.
        """
        # Шаг 1: ищем угол поворота, при котором HPP имеет максимальную дисперсию.
        if self.binary is None:
            return 90.0
        text_mask = (self.binary > 0).astype(np.uint8)
        height, width = text_mask.shape
        center = (width // 2, height // 2)
        angles = np.linspace(GLOBAL_SKEW_ANGLE_MIN_DEG, GLOBAL_SKEW_ANGLE_MAX_DEG, GLOBAL_SKEW_ANGLE_STEPS)
        best_angle = 0.0
        best_score = -1.0
        for angle in angles:
            matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
            rotated = cv2.warpAffine(
                text_mask,
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            score = float(np.var(np.sum(rotated > 0, axis=1)))
            if score > best_score:
                best_score = score
                best_angle = float(angle)

        # Шаг 2: correct-поворот примерно равен -skew, а нормаль строки theta = 90 + skew.
        theta_center = 90.0 - best_angle
        if self.debug:
            path = os.path.join(self.debug_output_dir, f"{self.debug_counter:03d}_my_dynamic_theta.json")
            with open(path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "best_hpp_rotation_angle_deg": float(best_angle),
                        "theta_center_deg": float(theta_center),
                        "theta_min_deg": float(theta_center - DYNAMIC_HOUGH_THETA_DEVIATION_DEG),
                        "theta_max_deg": float(theta_center + DYNAMIC_HOUGH_THETA_DEVIATION_DEG),
                    },
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
            self.debug_counter += 1
        return float(theta_center)

    def _save_hough_points_debug(self, points: List[Tuple[float, float]]) -> None:
        """
        Короткое описание:
            Сохраняет центры блоков, которые голосуют в Hough.
        Вход:
            points: List[Tuple[float, float]] -- точки голосования.
        Выход:
            None
        """
        # Шаг 1: готовим изображение.
        debug_image = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)

        # Шаг 2: рисуем block gravity centers.
        for x, y in points:
            cv2.circle(debug_image, (int(round(x)), int(round(y))), 2, DEBUG_COLOR_POINT, -1)
        self._save_debug_image(debug_image, "09_hough_block_centers")

    def y_on_line(self, line: Dict[str, Any], x: float) -> float:
        """
        Короткое описание:
            Вычисляет y линии в точке x из polar-параметров Hough.
        Вход:
            line: Dict[str, Any] -- строка с rho и theta.
            x: float -- координата x.
        Выход:
            float -- координата y.
        """
        theta_rad = np.deg2rad(float(line["theta"]))
        sin_theta = float(np.sin(theta_rad))
        if abs(sin_theta) < 1e-6:
            return 0.0
        return (float(line["rho"]) - x * float(np.cos(theta_rad))) / sin_theta

    def sort_lines_top_down(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Короткое описание:
            Сортирует строки сверху вниз по пересечению с серединой изображения.
        Вход:
            lines: List[Dict[str, Any]] -- список строк.
        Выход:
            List[Dict[str, Any]] -- отсортированный список строк.
        """
        # Шаг 1: считаем y_mid для каждой строки.
        middle_x = self.gray.shape[1] / 2.0
        for line in lines:
            line["y_mid"] = float(self.y_on_line(line, middle_x))

        # Шаг 2: сортируем по y_mid.
        return sorted(lines, key=lambda item: float(item["y_mid"]))

    def merge_close_text_lines(self) -> None:
        """
        Короткое описание:
            Сливает соседние строки, если расстояние между ними меньше среднего Ad.
        Вход:
            None
        Выход:
            None
        """
        if len(self.lines) < 2:
            return

        # Шаг 1: сортируем и считаем среднюю дистанцию Ad.
        self.lines = self.sort_lines_top_down(self.lines)
        y_values = [float(line["y_mid"]) for line in self.lines]
        distances = [abs(y_values[index + 1] - y_values[index]) for index in range(len(y_values) - 1)]
        average_distance = float(np.mean(distances)) if len(distances) > 0 else float(self.average_character_height)

        # Шаг 2: последовательно сливаем соседние строки с d < Ad.
        merged_lines: List[Dict[str, Any]] = []
        index = 0
        while index < len(self.lines):
            current_line = self.lines[index]
            merged_components = list(current_line.get("components", []))
            merged_indices = list(current_line.get("subset1_indices", []))
            next_index = index + 1

            while next_index < len(self.lines):
                distance = abs(float(self.lines[next_index]["y_mid"]) - float(self.lines[next_index - 1]["y_mid"]))
                if distance >= average_distance * LINE_DISTANCE_FACTOR:
                    break
                merged_components.extend(self.lines[next_index].get("components", []))
                merged_indices.extend(self.lines[next_index].get("subset1_indices", []))
                next_index += 1

            current_line["components"] = merged_components
            current_line["subset1_indices"] = merged_indices
            current_line["source"] = "hough_merged" if next_index > index + 1 else current_line.get("source", "hough")
            merged_lines.append(current_line)
            index = next_index

        self.lines = self.sort_lines_top_down(merged_lines)
        if self.debug:
            self._save_lines_debug(self.lines, "11_merged_lines")

    def create_new_text_lines(self) -> None:
        """
        Короткое описание:
            Создает строки из Subset 1 компонент, которые не были найдены Hough.
        Вход:
            None
        Выход:
            None
        """
        if len(self.lines) == 0:
            return

        # Шаг 1: считаем среднее расстояние между уже найденными строками.
        self.lines = self.sort_lines_top_down(self.lines)
        y_values = [float(line["y_mid"]) for line in self.lines]
        if len(y_values) < 2:
            return
        average_distance = float(np.mean([abs(y_values[index + 1] - y_values[index]) for index in range(len(y_values) - 1)]))

        target_distance = LINE_DISTANCE_FACTOR * average_distance
        tolerance = NEW_LINE_DISTANCE_TOLERANCE_FACTOR * average_distance

        # Шаг 2: ищем компоненты-кандидаты по расстоянию до ближайшей существующей строки.
        candidates: List[Dict[str, Any]] = []
        for component in self.subset1:
            if bool(component.get("assigned", False)):
                continue
            centers = self.compute_block_gravity_centers(component)
            if len(centers) == 0:
                continue
            candidate_blocks = 0
            for x, y in centers:
                min_distance = min(abs(y - self.y_on_line(line, x)) for line in self.lines)
                if abs(min_distance - target_distance) <= tolerance:
                    candidate_blocks += 1
            if candidate_blocks >= HOUGH_COMPONENT_MIN_BLOCK_FRACTION * len(centers):
                candidates.append(component)

        if len(candidates) == 0:
            return

        # Шаг 3: группируем кандидаты по вертикальной координате.
        candidates = sorted(candidates, key=lambda component: float(component["centroid"][1]))
        groups: List[List[Dict[str, Any]]] = []
        current_group: List[Dict[str, Any]] = []
        for component in candidates:
            if len(current_group) == 0:
                current_group.append(component)
                continue
            current_y = float(component["centroid"][1])
            group_y = float(np.mean([item["centroid"][1] for item in current_group]))
            if abs(current_y - group_y) <= NEW_LINE_GROUPING_AH_FACTOR * float(self.average_character_height):
                current_group.append(component)
            else:
                groups.append(current_group)
                current_group = [component]
        if len(current_group) > 0:
            groups.append(current_group)

        # Шаг 4: берем известное направление (угол) и оцениваем только rho.
        reference_theta = self.get_reference_theta_for_new_lines()

        # Шаг 5: для каждой группы оцениваем линию при фиксированном направлении.
        for group in groups:
            line = self.fit_line_from_components(group, fixed_theta=reference_theta)
            line["components"] = group
            line["source"] = "created"
            self.lines.append(line)
            for component in group:
                component["assigned"] = True

        self.lines = self.sort_lines_top_down(self.lines)
        if self.debug:
            self._save_lines_debug(self.lines, "12_created_lines")

    def get_reference_theta_for_new_lines(self) -> float:
        """
        Короткое описание:
            Возвращает опорный угол направления для новых линий.
        Вход:
            None
        Выход:
            float -- угол theta в градусах.
        """
        if USE_DYNAMIC_HOUGH_THETA_CENTER:
            return float(self.estimate_dynamic_hough_theta_center())
        if len(self.lines) == 0:
            return 90.0
        return float(np.median([float(line["theta"]) for line in self.lines]))

    def fit_line_from_components(self, components: List[Dict[str, Any]], fixed_theta: Optional[float] = None) -> Dict[str, Any]:
        """
        Короткое описание:
            Оценивает polar-параметры строки по центрам компонент.
        Вход:
            components: List[Dict[str, Any]] -- компоненты одной строки.
        Выход:
            Dict[str, Any] -- строка с rho и theta.
        """
        # Шаг 1: собираем точки центров блоков.
        points: List[Tuple[float, float]] = []
        for component in components:
            points.extend(self.compute_block_gravity_centers(component))
        if len(points) == 0:
            points = [tuple(component["centroid"]) for component in components]

        # Шаг 2: если угол известен, оцениваем только rho.
        points_array = np.asarray(points, dtype=np.float32)
        x_values = points_array[:, 0]
        y_values = points_array[:, 1]
        if fixed_theta is not None:
            theta = float(fixed_theta) % 180.0
            theta_rad = np.deg2rad(theta)
            rho = float(np.median(points_array[:, 0] * np.cos(theta_rad) + points_array[:, 1] * np.sin(theta_rad)))
            return {"rho": rho, "theta": theta}

        if len(points_array) >= 2 and np.std(x_values) > 1e-6:
            slope, intercept = np.polyfit(x_values, y_values, deg=1)
        else:
            slope = 0.0
            intercept = float(np.mean(y_values))

        # Шаг 3: переводим y = ax + b в x cos(theta) + y sin(theta) = rho.
        theta = float(np.rad2deg(np.arctan2(1.0, -float(slope))))
        if theta < 0.0:
            theta += 180.0
        rho = float(intercept * np.sin(np.deg2rad(theta)))
        return {"rho": rho, "theta": theta}

    def assign_remaining_components_to_lines(self) -> None:
        """
        Короткое описание:
            Присваивает Subset 3 и неприсвоенные Subset 1 ближайшим строкам.
        Вход:
            None
        Выход:
            None
        """
        if len(self.lines) == 0:
            return

        # Шаг 1: готовим контейнеры итоговых компонент.
        for line in self.lines:
            line.setdefault("components", [])
            line.setdefault("subset3_components", [])
            line.setdefault("split_components", [])

        # Шаг 2: присваиваем маленькие компоненты и оставшиеся обычные компоненты ближайшей строке.
        remaining_components = list(self.subset3) + [component for component in self.subset1 if not bool(component.get("assigned", False))]
        for component in remaining_components:
            best_line = self.find_closest_line(component)
            if best_line is None:
                continue
            if component.get("subset") == 3:
                best_line["subset3_components"].append(component)
            else:
                best_line["components"].append(component)
                component["assigned"] = True

    def find_closest_line(self, component: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Короткое описание:
            Находит ближайшую строку к компоненте по вертикальному расстоянию до centroid.
        Вход:
            component: Dict[str, Any] -- связная компонента.
        Выход:
            Optional[Dict[str, Any]] -- ближайшая строка или None.
        """
        if len(self.lines) == 0:
            return None
        x, y = component["centroid"]
        best_line = None
        best_distance = float("inf")
        for line in self.lines:
            distance = abs(float(y) - self.y_on_line(line, float(x)))
            if distance < best_distance:
                best_distance = distance
                best_line = line
        return best_line

    def split_subset2_and_assign_to_lines(self) -> None:
        """
        Короткое описание:
            Делит вертикально соединенные компоненты Subset 2 и назначает части строкам.
        Вход:
            None
        Выход:
            None
        """
        if len(self.lines) == 0:
            return

        # Шаг 1: обрабатываем каждую большую компоненту.
        for component_index, component in enumerate(self.subset2):
            split_parts = self.split_vertically_connected_component(component)
            if self.debug:
                self._save_split_debug(component, split_parts, component_index)

            # Шаг 2: назначаем каждую часть ближайшей строке.
            for part in split_parts:
                best_line = self.find_closest_line(part)
                if best_line is not None:
                    best_line.setdefault("split_components", []).append(part)

    def split_vertically_connected_component(self, component: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Короткое описание:
            Разделяет одну Subset 2 компоненту по skeleton и junction points.
        Вход:
            component: Dict[str, Any] -- большая связная компонента.
        Выход:
            List[Dict[str, Any]] -- список разделенных частей.
        """
        # Шаг 1: строим skeleton и junction points.
        mask = component["mask"].astype(np.uint8)
        height, width = mask.shape
        if height < 3 or width < 3:
            return [component]

        skeleton = skeletonize(mask > 0).astype(np.uint8)
        junctions = self.find_junction_points(skeleton)

        # Шаг 2: удаляем 3 x 3 окрестность junction points внутри зоны Z.
        zone_top = int(round(SUBSET2_ZONE_TOP_FACTOR * height))
        zone_bottom = int(round(SUBSET2_ZONE_BOTTOM_FACTOR * height))
        zone_top = max(0, min(height - 1, zone_top))
        zone_bottom = max(zone_top + 1, min(height, zone_bottom))
        modified_skeleton = skeleton.copy()
        removed_any = False

        radius = JUNCTION_REMOVAL_NEIGHBORHOOD // 2
        for x, y in junctions:
            if zone_top <= y < zone_bottom:
                y0 = max(0, y - radius)
                y1 = min(height, y + radius + 1)
                x0 = max(0, x - radius)
                x1 = min(width, x + radius + 1)
                modified_skeleton[y0:y1, x0:x1] = 0
                removed_any = True

        # Шаг 3: если junction point нет, удаляем skeleton pixels в середине зоны Z.
        if not removed_any:
            middle_y = (zone_top + zone_bottom) // 2
            modified_skeleton[max(0, middle_y - radius):min(height, middle_y + radius + 1), :] = 0

        # Шаг 4: извлекаем компоненты skeleton и помечаем самую верхнюю.
        labels_count, labels = cv2.connectedComponents(modified_skeleton.astype(np.uint8), connectivity=8)
        skeleton_components: List[Tuple[int, float]] = []
        for label in range(1, labels_count):
            ys, _ = np.where(labels == label)
            if len(ys) > 0:
                skeleton_components.append((label, float(np.min(ys))))
        if len(skeleton_components) < 2:
            return [component]
        upper_label = min(skeleton_components, key=lambda item: item[1])[0]

        flagged_skeleton = (labels == upper_label).astype(np.uint8)
        nonflagged_skeleton = ((labels > 0) & (labels != upper_label)).astype(np.uint8)
        if int(np.sum(flagged_skeleton)) == 0 or int(np.sum(nonflagged_skeleton)) == 0:
            return [component]

        # Шаг 5: каждый foreground-пиксель относим к ближайшему flagged или non-flagged skeleton pixel.
        distance_to_flagged = ndimage.distance_transform_edt(flagged_skeleton == 0)
        distance_to_nonflagged = ndimage.distance_transform_edt(nonflagged_skeleton == 0)
        upper_mask = ((mask > 0) & (distance_to_flagged <= distance_to_nonflagged)).astype(np.uint8)
        lower_mask = ((mask > 0) & (distance_to_flagged > distance_to_nonflagged)).astype(np.uint8)

        # Шаг 6: превращаем маски частей обратно в компоненты.
        parts = []
        for part_mask in [upper_mask, lower_mask]:
            part = self.component_from_local_mask(part_mask, component)
            if part is not None:
                parts.append(part)
        return parts if len(parts) >= 2 else [component]

    def find_junction_points(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """
        Короткое описание:
            Находит junction points как skeleton-пиксели с тремя и более соседями.
        Вход:
            skeleton: np.ndarray -- skeleton-маска 0/1.
        Выход:
            List[Tuple[int, int]] -- координаты junction points в формате x, y.
        """
        # Шаг 1: считаем число 8-соседей без центрального пикселя.
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT) - skeleton

        # Шаг 2: junction point имеет минимум три соседних skeleton-пикселя.
        ys, xs = np.where((skeleton > 0) & (neighbor_count >= 3))
        return [(int(x), int(y)) for x, y in zip(xs, ys)]

    def component_from_local_mask(self, local_mask: np.ndarray, source_component: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Короткое описание:
            Создает компоненту из локальной маски части исходной компоненты.
        Вход:
            local_mask: np.ndarray -- локальная маска 0/1.
            source_component: Dict[str, Any] -- исходная компонента.
        Выход:
            Optional[Dict[str, Any]] -- новая компонента или None.
        """
        # Шаг 1: ищем bbox непустой маски.
        rows, cols = np.where(local_mask > 0)
        if len(rows) == 0:
            return None
        source_x, source_y, _, _ = source_component["bbox"]
        y0 = int(np.min(rows))
        y1 = int(np.max(rows)) + 1
        x0 = int(np.min(cols))
        x1 = int(np.max(cols)) + 1
        cropped = local_mask[y0:y1, x0:x1].astype(np.uint8)

        # Шаг 2: считаем абсолютные параметры bbox и centroid.
        absolute_x = source_x + x0
        absolute_y = source_y + y0
        width = x1 - x0
        height = y1 - y0
        centroid_x = absolute_x + float(np.mean(cols - x0))
        centroid_y = absolute_y + float(np.mean(rows - y0))
        return {
            "id": None,
            "bbox": (absolute_x, absolute_y, width, height),
            "mask": cropped,
            "centroid": (centroid_x, centroid_y),
            "area": int(np.sum(cropped)),
            "subset": 2,
            "assigned": True,
        }

    def build_class_matrix(self) -> np.ndarray:
        """
        Короткое описание:
            Строит итоговую class-matrix: 0 фон, 1..N строки текста.
        Вход:
            None
        Выход:
            np.ndarray -- class-matrix int32 размера исходного изображения.
        """
        # Шаг 1: готовим пустую матрицу.
        class_matrix = np.zeros(self.gray.shape[:2], dtype=np.int32)
        self.lines = self.sort_lines_top_down(self.lines)

        # Шаг 2: для каждой линии строим ориентированный прямоугольник по направлению Hough.
        height, width = class_matrix.shape
        for line_index, line in enumerate(self.lines, start=1):
            polygon = self.build_line_polygon_from_segmentation(line, width, height)
            band_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(band_mask, [polygon.reshape(-1, 1, 2)], 1)
            class_matrix[band_mask > 0] = int(line_index)

        if self.debug:
            self._save_class_matrix_debug(class_matrix)
        return class_matrix

    def build_line_polygon_from_segmentation(self, line: Dict[str, Any], image_width: int, image_height: int) -> np.ndarray:
        """
        Короткое описание:
            Строит ориентированный прямоугольник линии по направлению Hough.
            Высота прямоугольника берется как диапазон, покрывающий 90% пикселей
            сегментации этой строки по нормали к линии.
        Вход:
            line: Dict[str, Any] -- строка с theta/rho и компонентами.
            image_width: int -- ширина изображения.
            image_height: int -- высота изображения.
        Выход:
            np.ndarray -- 4 вершины прямоугольника int32.
        """
        # Шаг 1: собираем пиксели сегментации строки.
        points = self.collect_line_points(line)
        if points.shape[0] < 3:
            # Fallback: узкая полоса вокруг линии.
            fallback_half_height = max(1.0, float(self.average_character_height) * 0.2)
            y_left = float(self.y_on_line(line, 0.0))
            y_right = float(self.y_on_line(line, float(image_width - 1)))
            fallback = np.array(
                [
                    [0.0, y_left - fallback_half_height],
                    [float(image_width - 1), y_right - fallback_half_height],
                    [float(image_width - 1), y_right + fallback_half_height],
                    [0.0, y_left + fallback_half_height],
                ],
                dtype=np.float32,
            )
            fallback[:, 0] = np.clip(fallback[:, 0], 0, image_width - 1)
            fallback[:, 1] = np.clip(fallback[:, 1], 0, image_height - 1)
            return np.round(fallback).astype(np.int32)

        # Шаг 2: строим ориентированный minAreaRect по пикселям строки.
        rect = cv2.minAreaRect(points.astype(np.float32))
        corners = cv2.boxPoints(rect).astype(np.float32)
        corners[:, 0] = np.clip(corners[:, 0], 0, image_width - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, image_height - 1)
        return np.round(corners).astype(np.int32)

    def collect_line_points(self, line: Dict[str, Any]) -> np.ndarray:
        """
        Короткое описание:
            Собирает абсолютные координаты пикселей сегментации, назначенных строке.
        Вход:
            line: Dict[str, Any] -- строка.
        Выход:
            np.ndarray -- массив Nx2 (x, y).
        """
        points_list: List[np.ndarray] = []
        all_components: List[Dict[str, Any]] = []
        all_components.extend(line.get("components", []))
        all_components.extend(line.get("subset3_components", []))
        all_components.extend(line.get("split_components", []))
        for component in all_components:
            x, y, _, _ = component["bbox"]
            mask = component["mask"] > 0
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            absolute_points = np.column_stack([xs.astype(np.float32) + float(x), ys.astype(np.float32) + float(y)])
            points_list.append(absolute_points)
        if len(points_list) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.vstack(points_list).astype(np.float32)

    def _save_lines_debug(self, lines: List[Dict[str, Any]], name: str) -> None:
        """
        Короткое описание:
            Сохраняет debug-визуализацию линий и их компонент.
        Вход:
            lines: List[Dict[str, Any]] -- строки для отрисовки.
            name: str -- имя debug-файла.
        Выход:
            None
        """
        # Шаг 1: готовим изображение.
        debug_image = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)

        # Шаг 2: рисуем линии и номера.
        for line_index, line in enumerate(lines, start=1):
            self.draw_hough_line(debug_image, line, DEBUG_COLOR_LINE, 2)
            y_mid = int(round(float(line.get("y_mid", self.y_on_line(line, self.gray.shape[1] / 2.0)))))
            cv2.putText(debug_image, str(line_index), (5, max(15, y_mid)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DEBUG_COLOR_LINE, 2)
            for component in line.get("components", []):
                x, y, width, height = component["bbox"]
                cv2.rectangle(debug_image, (x, y), (x + width, y + height), (0, 160, 0), 1)
        self._save_debug_image(debug_image, name)

    def draw_hough_line(self, image: np.ndarray, line: Dict[str, Any], color: Tuple[int, int, int], thickness: int) -> None:
        """
        Короткое описание:
            Рисует Hough-линию на изображении.
        Вход:
            image: np.ndarray -- изображение для рисования.
            line: Dict[str, Any] -- строка с rho и theta.
            color: Tuple[int, int, int] -- BGR-цвет.
            thickness: int -- толщина линии.
        Выход:
            None
        """
        # Шаг 1: вычисляем точки пересечения с левым и правым краем.
        height, width = image.shape[:2]
        y_left = int(round(self.y_on_line(line, 0.0)))
        y_right = int(round(self.y_on_line(line, float(width - 1))))

        # Шаг 2: рисуем отрезок с обрезкой координат.
        y_left = int(np.clip(y_left, -height, 2 * height))
        y_right = int(np.clip(y_right, -height, 2 * height))
        cv2.line(image, (0, y_left), (width - 1, y_right), color, thickness)

    def _save_split_debug(self, component: Dict[str, Any], parts: List[Dict[str, Any]], component_index: int) -> None:
        """
        Короткое описание:
            Сохраняет debug разделения Subset 2 компоненты.
        Вход:
            component: Dict[str, Any] -- исходная компонента.
            parts: List[Dict[str, Any]] -- разделенные части.
            component_index: int -- индекс компоненты в Subset 2.
        Выход:
            None
        """
        # Шаг 1: готовим локальное изображение компоненты.
        x, y, width, height = component["bbox"]
        debug_image = np.full((height, width, 3), 255, dtype=np.uint8)
        debug_image[component["mask"] > 0] = (80, 80, 80)

        # Шаг 2: накладываем части разными цветами.
        colors = [(0, 180, 0), (0, 0, 220), (220, 0, 0), (0, 180, 220)]
        for part_index, part in enumerate(parts):
            part_x, part_y, part_width, part_height = part["bbox"]
            local_x = part_x - x
            local_y = part_y - y
            color = colors[part_index % len(colors)]
            region = debug_image[local_y:local_y + part_height, local_x:local_x + part_width]
            region[part["mask"] > 0] = color

        self._save_debug_image(debug_image, f"13_subset2_split_{component_index:03d}")

    def _save_class_matrix_debug(self, class_matrix: np.ndarray) -> None:
        """
        Короткое описание:
            Сохраняет итоговую class-matrix цветной картинкой.
        Вход:
            class_matrix: np.ndarray -- матрица классов строк.
        Выход:
            None
        """
        # Шаг 1: создаем стабильную палитру.
        colored = np.full((*class_matrix.shape, 3), 255, dtype=np.uint8)
        rng = np.random.default_rng(12345)
        max_class = int(np.max(class_matrix))
        colors = rng.integers(40, 230, size=(max_class + 1, 3), dtype=np.uint8)

        # Шаг 2: раскрашиваем классы строк.
        for class_index in range(1, max_class + 1):
            colored[class_matrix == class_index] = colors[class_index].tolist()
        self._save_debug_image(colored, "14_final_class_matrix")

    def save_debug_summary(self, class_matrix: np.ndarray) -> None:
        """
        Короткое описание:
            Сохраняет json-отчет с параметрами и числом найденных объектов.
        Вход:
            class_matrix: np.ndarray -- итоговая class-matrix.
        Выход:
            None
        """
        # Шаг 1: собираем численные итоги.
        summary = {
            "average_character_height": float(self.average_character_height),
            "average_character_width": float(self.average_character_width),
            "components": len(self.components),
            "subset1": len(self.subset1),
            "subset2": len(self.subset2),
            "subset3": len(self.subset3),
            "lines": len(self.lines),
            "class_matrix_classes": int(np.max(class_matrix)),
            "my_subset2_split_debug": self.my_subset2_split_debug,
            "my_hough_debug": self.my_hough_debug,
            "my_component_size_histogram_debug": self.my_component_size_histogram_debug,
            "save_subset2_hpp_debug_files": bool(SAVE_SUBSET2_HPP_DEBUG_FILES),
            "save_subset2_energy_split_debug_files": bool(SAVE_SUBSET2_ENERGY_SPLIT_DEBUG_FILES),
            "hough_theta_min_deg": HOUGH_THETA_MIN_DEG,
            "hough_theta_max_deg": HOUGH_THETA_MAX_DEG,
            "hough_rho_step_ah_factor": HOUGH_RHO_STEP_AH_FACTOR,
            "hough_min_votes_n1": HOUGH_MIN_VOTES_N1,
            "hough_secondary_votes_n2": HOUGH_SECONDARY_VOTES_N2,
        }

        # Шаг 2: сохраняем json.
        path = os.path.join(self.debug_output_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)


def run_on_image(
    image_path: str,
    debug: bool = DEBUG,
    debug_output_dir: str = DEBUG_IMAGES_DIR,
    yolo_model: Any = None,
    unet_model: Any = None,
    unet_device: Any = None,
    return_page_info: bool = False,
    use_tqdm: bool = True,
) -> Any:
    """
    Короткое описание:
        Загружает изображение и запускает LouloudisTextLineDetector.
    Вход:
        image_path: str -- путь к исходному изображению.
        debug: bool -- сохранять debug-артефакты.
        debug_output_dir: str -- папка debug.
        yolo_model: Any -- заранее загруженная YOLO модель или None.
        unet_model: Any -- заранее загруженная U-Net модель или None.
        unet_device: Any -- устройство U-Net.
        return_page_info: bool -- вернуть информацию о crop страницы.
        use_tqdm: bool -- показывать tqdm на тяжелых циклах.
    Выход:
        Any -- class-matrix и список строк, опционально page_info.
    """
    # Шаг 1: читаем изображение.
    run_start_time = time.perf_counter()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    # Шаг 2: при необходимости выделяем страницу YOLO-сегментацией.
    if USE_YOLO_PAGE_SEGMENTATION:
        page_result = extract_largest_page_with_yolo(
            image=image,
            debug=debug,
            debug_output_dir=debug_output_dir,
            yolo_model=yolo_model,
        )
        if page_result is not None:
            image = page_result["page_image"]
        else:
            page_result = {
                "page_image": image,
                "page_mask": np.ones(image.shape[:2], dtype=np.uint8) * 255,
                "bbox": {"x": 0, "y": 0, "w": int(image.shape[1]), "h": int(image.shape[0])},
                "confidence": None,
            }
    else:
        page_result = {
            "page_image": image,
            "page_mask": np.ones(image.shape[:2], dtype=np.uint8) * 255,
            "bbox": {"x": 0, "y": 0, "w": int(image.shape[1]), "h": int(image.shape[0])},
            "confidence": None,
        }

    # Шаг 3: запускаем детектор.
    detector = LouloudisTextLineDetector(
        image=image,
        debug=debug,
        debug_output_dir=debug_output_dir,
        use_tqdm=use_tqdm,
        unet_model=unet_model,
        unet_device=unet_device,
    )
    detector_start_time = time.perf_counter()
    class_matrix, lines = detector.detect()
    detector_runtime_sec = time.perf_counter() - detector_start_time
    page_result["timing_seconds"] = {
        "total_run_on_image": float(time.perf_counter() - run_start_time),
        "detector": float(detector_runtime_sec),
    }
    if return_page_info:
        return class_matrix, lines, page_result
    return class_matrix, lines


def extract_largest_page_with_yolo(
    image: np.ndarray,
    debug: bool = DEBUG,
    debug_output_dir: str = DEBUG_IMAGES_DIR,
    yolo_model: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Короткое описание:
        Находит страницу через YOLO 3_2 segmentation и возвращает crop страницы на белом фоне.
    Вход:
        image: np.ndarray -- исходное BGR-изображение.
        debug: bool -- сохранять debug-артефакты YOLO.
        debug_output_dir: str -- папка для debug-файлов.
        yolo_model: Any -- заранее загруженная YOLO модель или None.
    Выход:
        Optional[Dict[str, Any]] -- словарь с page_image, page_mask, bbox или None.
    """
    # Шаг 1: проверяем наличие весов и лениво импортируем YOLO.
    if not YOLO_PAGE_SEGMENTATION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найдена YOLO-модель страницы: {YOLO_PAGE_SEGMENTATION_MODEL_PATH}")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError("Для YOLO-сегментации установи ultralytics в окружение /home/sasha/Documents/venv") from error

    # Шаг 2: запускаем модель страницы.
    owns_model = yolo_model is None
    model = YOLO(str(YOLO_PAGE_SEGMENTATION_MODEL_PATH)) if owns_model else yolo_model
    result = model.predict(
        image,
        conf=YOLO_PAGE_CONF,
        imgsz=YOLO_PAGE_IMGSZ,
        device=YOLO_PAGE_DEVICE,
        verbose=False,
    )[0]

    if debug:
        os.makedirs(debug_output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_output_dir, "yolo_00_input.jpg"), image)
        cv2.imwrite(os.path.join(debug_output_dir, "yolo_01_result.jpg"), result.plot())

    # Шаг 3: если масок нет, возвращаем None и метод продолжит работу по исходному изображению.
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        if debug:
            with open(os.path.join(debug_output_dir, "yolo_warning.txt"), "w", encoding="utf-8") as file:
                file.write("YOLO не нашел маску страницы. Метод запущен по исходному изображению.\n")
        return None

    # Шаг 4: выбираем маску страницы с максимальной confidence.
    image_height, image_width = image.shape[:2]
    masks = result.masks.data.cpu().numpy()
    best_mask = None
    best_area = -1
    best_confidence = -1.0
    candidates: List[Dict[str, Any]] = []
    confidences = []
    if result.boxes is not None and result.boxes.conf is not None:
        confidences = [float(value) for value in result.boxes.conf.detach().cpu().numpy().tolist()]

    for mask_index, mask in enumerate(masks):
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        mask_full = cv2.resize(mask_binary, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        area = int(np.sum(mask_full > 0))
        confidence = confidences[mask_index] if mask_index < len(confidences) else 0.0
        candidates.append(
            {
                "index": int(mask_index),
                "confidence": float(confidence),
                "mask_area": int(area),
            }
        )
        if confidence > best_confidence or (abs(confidence - best_confidence) < 1e-9 and area > best_area):
            best_confidence = float(confidence)
            best_area = area
            best_mask = mask_full

    if best_mask is None or best_area <= 0:
        return None

    # Шаг 5: берем bbox маски и кладем все вне страницы на белый фон.
    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    page_roi = image[y:y + height, x:x + width]
    mask_roi = best_mask[y:y + height, x:x + width]
    white_page = np.full_like(page_roi, 255)
    page_image = np.where(mask_roi[:, :, None] > 0, page_roi, white_page)

    # Шаг 6: сохраняем debug crop и маску.
    if debug:
        cv2.imwrite(os.path.join(debug_output_dir, "yolo_02_page_mask.png"), best_mask)
        cv2.imwrite(os.path.join(debug_output_dir, "yolo_03_page_crop.jpg"), page_image)
        info = {
            "model_path": str(YOLO_PAGE_SEGMENTATION_MODEL_PATH),
            "conf": YOLO_PAGE_CONF,
            "imgsz": YOLO_PAGE_IMGSZ,
            "bbox": {"x": int(x), "y": int(y), "w": int(width), "h": int(height)},
            "mask_area": int(best_area),
            "selected_confidence": float(best_confidence),
            "selection_rule": "max_confidence_then_area",
            "candidates": candidates,
        }
        with open(os.path.join(debug_output_dir, "yolo_page_info.json"), "w", encoding="utf-8") as file:
            json.dump(info, file, indent=2, ensure_ascii=False)

    # Шаг 7: освобождаем модель после одного изображения, если загрузили ее здесь.
    if owns_model:
        del model
        gc.collect()

    return {
        "page_image": page_image,
        "page_mask": mask_roi,
        "bbox": {"x": int(x), "y": int(y), "w": int(width), "h": int(height)},
        "confidence": float(best_confidence),
        "candidates": candidates,
    }


def main() -> None:
    """
    Короткое описание:
        Точка входа для прямого запуска скрипта.
    Вход:
        None
    Выход:
        None
    """
    # Шаг 1: проверяем, задан ли путь к изображению.
    if INPUT_IMAGE_PATH == "":
        print("Задай INPUT_IMAGE_PATH в louloudis_text_line_detection_exact.py и запусти файл снова.")
        return

    # Шаг 2: запускаем метод и сохраняем class-matrix.
    class_matrix, lines, page_info = run_on_image(
        INPUT_IMAGE_PATH,
        debug=DEBUG,
        debug_output_dir=DEBUG_IMAGES_DIR,
        return_page_info=True,
    )
    output_dir = Path(DEBUG_IMAGES_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(output_dir / "class_matrix.npy"), class_matrix)

    # Шаг 3: при одиночном запуске сохраняем target из labels.txt, если это HWR200 путь.
    save_target_overlay_for_input(INPUT_IMAGE_PATH, DEBUG_IMAGES_DIR)

    print(f"[OK] Найдено строк: {len(lines)}")
    print(f"[OK] timing={page_info.get('timing_seconds')}")
    print(f"[OK] Debug: {DEBUG_IMAGES_DIR}")


def save_target_overlay_for_input(input_image_path: str, debug_output_dir: str) -> None:
    """
    Короткое описание:
        Сохраняет target-оверлей GT полигонов из HWR200 labels.txt для входного изображения.
    Вход:
        input_image_path: str -- путь к входному изображению.
        debug_output_dir: str -- папка debug.
    Выход:
        None
    """
    # Шаг 1: проверяем принадлежность изображения HWR200 датасету.
    image_path = Path(input_image_path).resolve()
    dataset_root = (PROJECT_ROOT / "datasets" / "HWR200" / "hw_dataset").resolve()
    labels_path = PROJECT_ROOT / "datasets" / "HWR200" / "labels.txt"
    if not labels_path.exists():
        return
    try:
        relative_path = image_path.relative_to(dataset_root).as_posix()
    except ValueError:
        return

    # Шаг 2: загружаем GT-полигоны для текущего relative path.
    gt_polygons: List[np.ndarray] = []
    with open(labels_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2 or parts[0] != relative_path:
                continue
            objects = json.loads(parts[1])
            for item in objects:
                points = item.get("points", [])
                if len(points) >= 3:
                    gt_polygons.append(np.asarray(points, dtype=np.float32).reshape(-1, 2))
            break
    if len(gt_polygons) == 0:
        return

    # Шаг 3: рисуем target поверх исходного изображения и сохраняем.
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    overlay = image.copy()
    for polygon in gt_polygons:
        pts = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 200, 0), thickness=2, lineType=cv2.LINE_AA)
    output_dir = Path(debug_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "target_00_original.jpg"), image)
    cv2.imwrite(str(output_dir / "target_01_gt_polygons.jpg"), overlay)
    info = {
        "relative_path": relative_path,
        "gt_count": int(len(gt_polygons)),
        "labels_path": str(labels_path),
    }
    with open(output_dir / "target_info.json", "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
