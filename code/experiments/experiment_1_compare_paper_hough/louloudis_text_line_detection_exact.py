import json
import gc
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

# Папка для всех debug-артефактов нового автономного метода.
DEBUG_IMAGES_DIR = str(PROJECT_ROOT / "debug_images" / "experiment_1_compare_paper_hough" / "louloudis_exact")

# Путь к изображению для запуска файла напрямую. Если пусто, скрипт только сообщает, как его настроить.
INPUT_IMAGE_PATH = '/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code/datasets/HWR200/hw_dataset/1/reuse5/ФотоТемное/3.jpg'

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

# Минимальная площадь компоненты. В статье Louloudis et al. 2008 после бинаризации
# сказано извлечь connected components; дополнительный порог площади не задан.
MIN_COMPONENT_AREA = 1

# Минимальная высота компоненты для AH. В статье отдельный порог не задан.
MIN_COMPONENT_HEIGHT_FOR_AH = 1

# Размер bin гистограммы AH: один пиксель. Так максимум гистограммы буквально соответствует высоте компоненты.
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

# Компонента относится к линии, если не меньше половины ее block-точек попали в область линии.
HOUGH_COMPONENT_MIN_BLOCK_FRACTION = 0.5

# Порог создания новых строк из статьи: расстояние до ближайшей строки должно быть около 0.9 * Ad.
NEW_LINE_DISTANCE_FACTOR = 0.9

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

        # Шаг 4: ищем первичные строки block-based Hough по Subset 1.
        self.lines = self.block_based_hough_transform()

        # Шаг 5: исправляем ложное дробление и добавляем строки, которые Hough не нашел.
        self.merge_close_text_lines()
        self.create_new_text_lines()

        # Шаг 6: присваиваем Subset 3 и разделяем Subset 2.
        self.assign_remaining_components_to_lines()
        self.split_subset2_and_assign_to_lines()

        # Шаг 7: строим итоговую class-matrix и debug.
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
            Оценивает среднюю высоту символа как максимум гистограммы высот компонент.
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

        # Шаг 2: берем пик гистограммы высот.
        if len(heights) == 0:
            return 10.0
        values, counts = np.unique(np.asarray(heights, dtype=np.int32), return_counts=True)
        return float(values[int(np.argmax(counts))])

    def estimate_average_character_height(self, components: List[Dict[str, Any]]) -> float:
        """
        Короткое описание:
            Оценивает AH по максимуму гистограммы bounding-box heights компонент.
        Вход:
            components: List[Dict[str, Any]] -- список связных компонент.
        Выход:
            float -- средняя высота символа AH.
        """
        # Шаг 1: собираем высоты компонент.
        heights = [
            int(component["bbox"][3])
            for component in components
            if int(component["bbox"][3]) >= MIN_COMPONENT_HEIGHT_FOR_AH
        ]
        if len(heights) == 0:
            return 10.0

        # Шаг 2: строим гистограмму с bin width = 1 и берем максимальный bin.
        values, counts = np.unique(np.asarray(heights, dtype=np.int32), return_counts=True)
        peak_height = float(values[int(np.argmax(counts))])

        # Шаг 3: сохраняем debug-гистограмму в json-friendly формате.
        if self.debug:
            histogram_data = {
                "average_character_height": peak_height,
                "heights": [int(value) for value in values.tolist()],
                "counts": [int(value) for value in counts.tolist()],
            }
            path = os.path.join(self.debug_output_dir, f"{self.debug_counter:03d}_07_ah_histogram.json")
            with open(path, "w", encoding="utf-8") as file:
                json.dump(histogram_data, file, indent=2, ensure_ascii=False)
            self.debug_counter += 1
        return peak_height

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
        while True:
            progress.update(1)
            if len(available_points) < HOUGH_MIN_VOTES_N1:
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
                break

            best_cell, best_points = max(accumulator.items(), key=lambda item: len(item[1]))
            best_votes = len(best_points)
            if best_votes < HOUGH_MIN_VOTES_N1:
                break

            rho_index, theta_index = best_cell
            theta = float(theta_values[theta_index])
            if best_votes < HOUGH_SECONDARY_VOTES_N2 and len(lines) > 0:
                dominant_theta = float(np.median([line["theta"] for line in lines]))
                if abs(theta - dominant_theta) > HOUGH_DOMINANT_ANGLE_TOLERANCE_DEG:
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
                for point_index in neighbor_points:
                    available_points.discard(point_index)
                continue

            rho = rho_min + rho_index * rho_step
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
                if distance >= average_distance:
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

        target_distance = NEW_LINE_DISTANCE_FACTOR * average_distance
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

        # Шаг 4: выбираем опорный угол (направление) из уже найденных Hough-линий.
        reference_theta = self.get_reference_theta_for_new_lines()

        # Шаг 5: для каждой группы оцениваем только rho при фиксированном направлении.
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
        if len(self.lines) == 0:
            return 90.0
        return float(np.median([float(line["theta"]) for line in self.lines]))

    def fit_line_from_components(self, components: List[Dict[str, Any],], fixed_theta: Optional[float] = None) -> Dict[str, Any]:
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

        # Шаг 2: подгоняем y = ax + b, если угол не фиксирован.
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
    print(f"[OK] Найдено строк: {len(lines)}")
    print(f"[OK] timing={page_info.get('timing_seconds')}")
    print(f"[OK] Debug: {DEBUG_IMAGES_DIR}")


if __name__ == "__main__":
    main()
