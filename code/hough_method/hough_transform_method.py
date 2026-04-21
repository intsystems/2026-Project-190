import numpy as np
import cv2
from skimage import measure
from skimage.morphology import skeletonize
from scipy import ndimage
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
import os
from u_net_binarization import binarize_image
import matplotlib.pyplot as plt
from post_processing import crop_line_rectangle
from processing import extract_pages_with_yolo
import joblib
import colorsys
import scipy.signal
import time
import json

# Палитра из 30 хорошо различимых цветов (RGB)
BASE_COLORS = [
    (255, 0, 0),    # красный
    (0, 255, 0),    # зелёный
    (0, 0, 255),    # синий
    (255, 255, 0),  # жёлтый
    (255, 0, 255),  # пурпурный
    (0, 255, 255),  # циан
    (128, 0, 0),    # тёмно-красный
    (0, 128, 0),    # тёмно-зелёный
    (0, 0, 128),    # тёмно-синий
    (128, 128, 0),  # оливковый
    (128, 0, 128),  # фиолетовый
    (0, 128, 128),  # бирюзовый
    (255, 128, 0),  # оранжевый
    (255, 0, 128),  # розовый
    (128, 255, 0),  # салатовый
    (0, 255, 128),  # мятный
    (128, 0, 255),  # лавандовый
    (0, 128, 255),  # голубой
    (255, 128, 128),# светло-красный
    (128, 255, 128),# светло-зелёный
    (128, 128, 255),# светло-синий
    (255, 255, 128),# светло-жёлтый
    (255, 128, 255),# светло-пурпурный
    (128, 255, 255),# светло-циан
    (192, 0, 0),    # тёмно-красный
    (0, 192, 0),    # тёмно-зелёный
    (0, 0, 192),    # тёмно-синий
    (192, 192, 0),  # тёмно-оливковый
    (192, 0, 192),  # тёмно-фиолетовый
    (0, 192, 192),  # тёмно-бирюзовый
]

class TextLineDetector:
    """
    Класс для обнаружения строк текста в рукописных документах.
    Реализует алгоритм из статьи:
    G. Louloudis et al., "Text line detection in handwritten documents",
    Pattern Recognition 41 (2008) 3758--3772.

    Параметры:
        image (np.ndarray): исходное изображение (RGB или оттенки серого)
        params (dict, optional): переопределение параметров алгоритма
        debug (bool): включить сохранение промежуточных изображений
        debug_output_dir (str): директория для сохранения отладочных изображений

    Атрибуты:
        image: исходное изображение
        gray: изображение в оттенках серого
        binary: бинарное изображение (текст – белый, фон – чёрный)
        components: список связных компонент (словари с bbox, mask, centroid, label)
        AH, AW: средняя высота/ширина символа
        subset1, subset2, subset3: разбитые компоненты
        lines: найденные линии (список словарей с параметрами)
    """

    def __init__(self, image: np.ndarray,
                 params: Optional[Dict[str, Any]] = None,
                 debug: bool = False,
                 debug_output_dir: str = "debug_images"):
        """
        Инициализация детектора.

        На вход:
            image (np.ndarray): изображение в формате RGB (3 канала) или оттенки серого.
            params (dict): словарь с параметрами (см. _set_default_params).
            debug (bool): режим отладки – сохранять промежуточные изображения.
            debug_output_dir (str): папка для сохранения отладочных изображений.

        Выход:
            None
        """
        self.image = image
        self.params = params or {}
        self._set_default_params()

        if len(self.image.shape) == 3:
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = self.image

        self.binary = None
        self.components = []
        self.AH = None
        self.AW = None
        self.subset1 = []
        self.subset2 = []
        self.subset3 = []
        self.lines = []

        self.debug = debug
        self.debug_output_dir = debug_output_dir
        if self.debug:
            os.makedirs(self.debug_output_dir, exist_ok=True)
            self.debug_counter = 0

    def _set_default_params(self):
        """
        Устанавливает значения параметров по умолчанию, если они не заданы в params.

        Параметры:
            binarization_method: метод бинаризации ('otsu', 'adaptive', 'u_net')
            hough_theta_range: диапазон углов для преобразования Хафа (в градусах)
            hough_rho_step_factor: шаг по ро (в долях от AH)
            hough_max_votes_threshold: порог голосов для основной линии
            hough_secondary_threshold: порог для дополнительных линий
            hough_angle_tolerance: допуск по углу для вторичных линий
            hough_neighborhood_radius: радиус окрестности для объединения голосов
            merge_distance_factor: коэффициент для слияния линий (по расстоянию)
            subset1_height_bounds: границы высоты для subset1 (доли от AH)
            subset1_width_factor: минимальная ширина для subset1 (доли от AW)
            subset2_height_factor: минимальная высота для subset2 (доли от AH)
            subset3_height_factor: максимальная высота для subset3 (доли от AH)
            subset3_width_factor: максимальная ширина для subset3 (доли от AW)
            hough_small_dataset_threshold: порог для адаптации порогов при малом количестве компонент
            hough_large_dataset_threshold: порог для адаптации порогов при большом количестве компонент
            hough_min_max_votes: минимальное значение для hough_max_votes_threshold после адаптации
            hough_min_secondary_votes: минимальное значение для hough_secondary_threshold после адаптации
            hough_max_max_votes: максимальное значение для hough_max_votes_threshold после адаптации
            hough_max_secondary_votes: максимальное значение для hough_secondary_threshold после адаптации
            new_line_lower_factor: нижний множитель среднего расстояния для определения кандидата в новую строку
            new_line_upper_factor: верхний множитель среднего расстояния для определения кандидата
            new_line_vertical_grouping_factor: множитель AH для группировки компонент по вертикали
            angle_filter_threshold: порог отклонения угла для отбрасывания строк (в градусах)
            min_components_for_skew: минимальное количество компонент для оценки наклона
            number_component_blocks_that_voted_for_the_line: минимальное число блоков компоненты, проголосовавших за линию, для её подтверждения
        """
        defaults = {
            'binarization_method': 'u_net',
            'hough_theta_range': (-5, 5),
            'hough_rho_step_factor': 0.15,
            'hough_max_votes_threshold': 5,
            'hough_secondary_threshold': 10,
            'hough_angle_tolerance': 2,
            'hough_neighborhood_radius': 5,
            'merge_distance_factor': 1.0,
            'subset1_height_bounds': (0.5, 3.0),
            'subset1_width_factor': 0.5,
            'subset2_height_factor': 3.0,
            'subset3_height_factor': 0.5,
            'subset3_width_factor': 0.5,
            'hough_small_dataset_threshold': 50,
            'hough_large_dataset_threshold': 200,
            'hough_min_max_votes': 3,
            'hough_min_secondary_votes': 5,
            'hough_max_max_votes': 10,
            'hough_max_secondary_votes': 15,
            'new_line_lower_factor': 0.7,
            'new_line_upper_factor': 1.1,
            'new_line_vertical_grouping_factor': 0.8,
            'angle_filter_threshold': 1,
            'min_components_for_skew': 5,
            'number_component_blocks_that_voted_for_the_line': 3
        }
        for key, default in defaults.items():
            if key not in self.params:
                self.params[key] = default

    def _save_debug_image(self, img, name):
        """
        Сохраняет изображение в отладочную директорию, если включён debug.

        На вход:
            img (np.ndarray): изображение для сохранения
            name (str): имя файла (без расширения)
        """
        if not self.debug:
            return
        fname = os.path.join(self.debug_output_dir, f"{self.debug_counter:03d}_{name}.png")
        cv2.imwrite(fname, img)
        self.debug_counter += 1

    def binarize(self) -> np.ndarray:
        """
        Выполняет бинаризацию изображения.

        Вход:
            Использует self.gray и self.params['binarization_method'].

        Выход:
            binary (np.ndarray): бинарное изображение (0 – фон, 255 – текст).
        """
        if self.params['binarization_method'] == 'otsu':
            _, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif self.params['binarization_method'] == 'adaptive':
            binary = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
        elif self.params['binarization_method'] == 'u_net':
            binary = 255 - binarize_image(self.image)
        elif self.params['binarization_method'] == 'is_binary': # уже бинарное изображение
            binary = 255 - self.image
        else:
            raise ValueError(f"Неподдерживаемый метод бинаризации: {self.params['binarization_method']}")
        self.binary = binary
        if self.debug:
            self._save_debug_image(self.binary, "binarized")
        return self.binary

    def extract_connected_components(self) -> List[Dict[str, Any]]:
        """
        Извлекает связные компоненты из бинарного изображения.

        Выход:
            components (list): список словарей, каждый содержит:
                'bbox': (x, y, w, h) – прямоугольник компоненты,
                'mask': 2D-маска компоненты,
                'centroid': (cx, cy) – центр масс,
                'label': метка компоненты.
        """
        labeled = measure.label(self.binary, connectivity=2)
        props = measure.regionprops(labeled, intensity_image=self.binary)
        components = []
        for prop in props:
            y, x, y2, x2 = prop.bbox
            w = x2 - x
            h = y2 - y
            centroid = prop.centroid
            centroid_xy = (centroid[1], centroid[0])
            components.append({
                'bbox': (x, y, w, h),
                'mask': prop.image,
                'centroid': centroid_xy,
                'label': prop.label
            })
        self.components = components
        if self.debug:
            debug_img = self.image.copy() if len(self.image.shape) == 3 else cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            for comp in self.components:
                x, y, w, h = comp['bbox']
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            self._save_debug_image(debug_img, "components")
        return components

    def estimate_average_character_height(self) -> Tuple[float, float]:
        """
        Оценивает среднюю высоту символа AH и среднюю ширину AW по гистограмме высот компонент.

        Выход:
            (AH, AW) (float, float): средняя высота и ширина символа.
        """
        if not self.components:
            raise ValueError("Нет связных компонент для оценки высоты.")
        heights = [comp['bbox'][3] for comp in self.components if comp['bbox'][3] > 4] # надо что-то решать с выбросами!!!!! Как-то понимать когда их стоит убирать, а когда нет
        hist, bin_edges = np.histogram(heights, bins=len(heights)//10)
        hist, bin_edges = hist[1:], bin_edges[1:]

        peak_idx = None
        if len(hist) != 0:
            peak_idx = np.argmax(hist)
        else:
            heights = [comp['bbox'][3] for comp in self.components]
            hist, bin_edges = np.histogram(heights, bins=len(heights)//10)
            hist, bin_edges = hist[1:], bin_edges[1:]
            peak_idx = np.argmax(hist)

        AH = (bin_edges[peak_idx] + bin_edges[peak_idx+1]) / 2.0
        AW = AH
        self.AH = AH
        self.AW = AW
        if self.debug:
            plt.figure(figsize=(8,4))
            plt.hist(heights, bins=len(heights)//10, edgecolor='black', alpha=0.7)
            plt.axvline(self.AH, color='red', linestyle='--', label=f'AH = {self.AH:.1f}')
            plt.xlabel('Высота компоненты (пиксели)')
            plt.ylabel('Частота')
            plt.title('Гистограмма высот связных компонент')
            plt.legend()
            plt.grid(True, alpha=0.3)
            hist_path = os.path.join(self.debug_output_dir, f"{self.debug_counter:03d}_height_histogram.png")
            plt.savefig(hist_path)
            plt.close()
            plt.show()
            self.debug_counter += 1
        return AH, AW

    def partition_components(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Разбивает компоненты на три подмножества согласно критериям из статьи:
            subset1: обычные символы (высота в [0.5*AH, 3*AH] и ширина >= 0.5*AW)
            subset2: большие компоненты (высота >= 3*AH)
            subset3: маленькие или узкие компоненты (высота < 0.5*AH и ширина > 0.5*AW,
                     или высота < 3*AH и ширина < 0.5*AW)

        Выход:
            (subset1, subset2, subset3) (list, list, list) – три списка компонент.
        """
        AH, AW = self.AH, self.AW
        subset1, subset2, subset3 = [], [], []


        h_min = self.params['subset1_height_bounds'][0] * AH
        h_max = self.params['subset1_height_bounds'][1] * AH
        w_min = self.params['subset1_width_factor'] * AW

        for comp in self.components:
            x, y, w, h = comp['bbox']
            if (h_min <= h < h_max) and (w >= w_min):
                subset1.append(comp)
            elif h >= self.params['subset2_height_factor'] * AH:
                subset2.append(comp)
            elif (h < self.params['subset2_height_factor'] * AH and w < self.params['subset3_width_factor'] * AW) or \
                (h < self.params['subset3_height_factor'] * AH and w > self.params['subset3_width_factor'] * AW):
                subset3.append(comp)
            else:
                subset1.append(comp)


        self.subset1 = subset1
        self.subset2 = subset2
        self.subset3 = subset3
        if self.debug:
            debug_img = self.image.copy() if len(self.image.shape) == 3 else cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            for comp in subset1:
                x, y, w, h = comp['bbox']
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self._save_debug_image(debug_img, "partition_subset1")
            debug_img = self.image.copy() if len(self.image.shape) == 3 else cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            for comp in subset2:
                x, y, w, h = comp['bbox']
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self._save_debug_image(debug_img, "partition_subset2")
            debug_img = self.image.copy() if len(self.image.shape) == 3 else cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            for comp in subset3:
                x, y, w, h = comp['bbox']
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self._save_debug_image(debug_img, "partition_subset3")
        return subset1, subset2, subset3

    def _compute_gravity_centers(self, comp: Dict, AW: float) -> List[Tuple[float, float]]:
        """
        Вычисляет центры тяжести для блоков компоненты (разбивает по ширине на AW-блоки).

        Вход:
            comp (dict): компонента с ключами 'bbox', 'mask'
            AW (float): средняя ширина символа

        Выход:
            centers (list): список (x, y) центров блоков.
        """
        x, y, w, h = comp['bbox']
        mask = comp['mask']
        nb = int(np.ceil(w / AW))
        block_width = w / nb
        centers = []
        for i in range(nb):
            start_col = int(i * block_width)
            end_col = int((i+1) * block_width)
            if end_col > w:
                end_col = w
            block_mask = mask[:, start_col:end_col]
            if np.sum(block_mask) == 0:
                continue
            indices = np.argwhere(block_mask)
            if len(indices) == 0:
                continue
            mean_row = np.mean(indices[:, 0])
            mean_col = np.mean(indices[:, 1]) + start_col
            cx = x + mean_col
            cy = y + mean_row
            centers.append((cx, cy))
        return centers

    def _hough_vote(self, points_with_idx, theta_range, rho_step):
        """
        Выполняет голосование в пространстве Хафа для заданных точек.

        Вход:
            points_with_idx: список кортежей (idx, x, y)
            theta_range: (min, max) углов в градусах
            rho_step: шаг по ρ

        Выход:
            acc: словарь {(rho_idx, theta_idx): list(point_indices)}
            rho_bins: массив бинов ρ
            theta_bins: массив бинов θ
        """
        theta_min, theta_max = theta_range
        theta_res = 1
        theta_bins = np.arange(theta_min, theta_max + theta_res, theta_res)
        h, w = self.binary.shape
        max_rho = np.sqrt(w**2 + h**2)
        rho_bins = np.arange(-max_rho, max_rho + rho_step, rho_step)
        acc = defaultdict(list)
        for idx, x, y in points_with_idx:
            for idx_t, theta in enumerate(theta_bins):
                theta_rad = np.deg2rad(theta)
                rho = x * np.cos(theta_rad) + y * np.sin(theta_rad)
                idx_r = int(np.round((rho - rho_bins[0]) / rho_step))
                if 0 <= idx_r < len(rho_bins):
                    acc[(idx_r, idx_t)].append(idx)
        return acc, rho_bins, theta_bins

    def _adjust_hough_thresholds(self, num_components):
        """
        Адаптирует пороги Хафа в зависимости от количества компонент.

        Вход:
            num_components (int): количество компонент в subset1.
        """
        small_thresh = self.params['hough_small_dataset_threshold']
        large_thresh = self.params['hough_large_dataset_threshold']
        min_max_votes = self.params['hough_min_max_votes']
        min_sec_votes = self.params['hough_min_secondary_votes']
        max_max_votes = self.params['hough_max_max_votes']
        max_sec_votes = self.params['hough_max_secondary_votes']

        if num_components < small_thresh:
            self.params['hough_max_votes_threshold'] = max(min_max_votes, self.params['hough_max_votes_threshold'] // 2)
            self.params['hough_secondary_threshold'] = max(min_sec_votes, self.params['hough_secondary_threshold'] // 2)
        elif num_components > large_thresh:
            self.params['hough_max_votes_threshold'] = min(max_max_votes, self.params['hough_max_votes_threshold'] + 2)
            self.params['hough_secondary_threshold'] = min(max_sec_votes, self.params['hough_secondary_threshold'] + 2)

    def _estimate_skew(self):
        """
        Оценивает наклон текста по центроидам компонент subset1.

        Выход:
            angle_deg (float): угол наклона в градусах.
        """
        if len(self.subset1) < self.params['min_components_for_skew']:
            return 0
        pts = [comp['centroid'] for comp in self.subset1]
        pts = np.array(pts)
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        main_axis = eigvecs[:, np.argmax(eigvals)]
        angle = np.arctan2(main_axis[1], main_axis[0])
        angle_deg = np.rad2deg(angle)

        if angle_deg < 0:
            angle_deg += 180
        return angle_deg

    def block_based_hough(self) -> List[Dict]:
        """
        Основной метод для обнаружения линий на основе блочного преобразования Хафа.

        Выход:
            lines (list): список найденных линий, каждая – словарь с ключами:
                'rho': параметр ρ,
                'theta': угол в градусах,
                'components': индексы компонент subset1, принадлежащих линии,
                'point_indices': индексы точек, давших голоса.
        """
        if not self.subset1:
            return []
        self._adjust_hough_thresholds(len(self.subset1))
        skew_est = self._estimate_skew()
        self.params['hough_theta_range'] = (skew_est + self.params['hough_theta_range'][0], skew_est + self.params['hough_theta_range'][1])

        all_points = []
        point_to_comp = []
        comp_points = {}
        for comp_idx, comp in enumerate(self.subset1):
            centers = self._compute_gravity_centers(comp, self.AW)
            for pt in centers:
                point_idx = len(all_points)
                all_points.append(pt)
                point_to_comp.append(comp_idx)
                comp_points.setdefault(comp_idx, []).append(point_idx)

        if self.debug:
            debug_img = self.image.copy() if len(self.image.shape) == 3 else cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            for comp_idx, comp in enumerate(self.subset1):
                x, y, w, h = comp['bbox']
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                centers = self._compute_gravity_centers(comp, self.AW)
                for cx, cy in centers:
                    cv2.circle(debug_img, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            self._save_debug_image(debug_img, "hough_blocks")

        theta_range = self.params['hough_theta_range']
        rho_step = self.params['hough_rho_step_factor'] * self.AH
        available_points = set(range(len(all_points)))
        lines = []
        comp_assigned = set()
        max_iter = 1000
        iter_count = 0

        # Итеративно извлекаем доминирующие линии из множества ещё не использованных точек.
        # На каждой итерации:
        # 1) строим аккумулятор Хафа по оставшимся точкам,
        # 2) берём самый сильный пик,
        # 3) расширяем его по соседству по rho,
        # 4) подтверждаем компоненты по числу голосов,
        # 5) удаляем присвоенные точки, чтобы следующая итерация искала уже другую линию.
        while iter_count < max_iter:
            iter_count += 1
            # Если осталось слишком мало точек, сильную линию уже не собрать.
            if len(available_points) < self.params['hough_max_votes_threshold']:
                if self.debug:
                    print(f"[DEBUG] Hough stopped: only {len(available_points)} points left")
                break
            # Голосуем в Хаф-пространстве только по ещё не использованным точкам.
            points_with_idx = [(idx, all_points[idx][0], all_points[idx][1]) for idx in available_points]
            acc, rho_bins, theta_bins = self._hough_vote(points_with_idx, theta_range, rho_step)
            if not acc:
                break
            # Берём ячейку аккумулятора с максимальным числом голосов.
            max_cell = max(acc.items(), key=lambda kv: len(kv[1]))
            (rho_idx, theta_idx), point_indices = max_cell
            num_votes = len(point_indices)
            # Пик слишком слабый: дальнейший поиск линий прекращаем.
            if num_votes < self.params['hough_max_votes_threshold']:
                break
            theta = theta_bins[theta_idx]
            rho_center = rho_bins[rho_idx]
            # Для вторичных линий контролируем, чтобы их угол не слишком расходился
            # с уже найденным доминирующим направлением.
            if num_votes < self.params['hough_secondary_threshold'] and lines:
                dominant_theta = np.median([line['theta'] for line in lines])
                if abs(theta - dominant_theta) > self.params['hough_angle_tolerance']:
                    for pt_idx in point_indices:
                        available_points.discard(pt_idx)
                    continue
            # Расширяем пик по окрестности rho, чтобы собрать стабильный набор соседних точек.
            radius = self.params['hough_neighborhood_radius']
            neighbor_points = set()
            rho_low = rho_center - radius
            rho_high = rho_center + radius
            for idx_r in range(len(rho_bins)):
                if rho_low <= rho_bins[idx_r] <= rho_high:
                    cell_key = (idx_r, theta_idx)
                    if cell_key in acc:
                        neighbor_points.update(acc[cell_key])
            comp_votes = defaultdict(int)
            for pt_idx in neighbor_points:
                comp_idx = point_to_comp[pt_idx]
                comp_votes[comp_idx] += 1
            assigned_comps = []
            # Компонента считается принадлежащей линии, если проголосовала хотя бы треть её блоков.
            for comp_idx, vote_count in comp_votes.items():
                total_blocks = len(comp_points[comp_idx])
                if vote_count >= total_blocks / self.params['number_component_blocks_that_voted_for_the_line']:
                    if comp_idx not in comp_assigned:
                        assigned_comps.append(comp_idx)
                        comp_assigned.add(comp_idx)
            # Если ни одна компонента не подтверждена, убираем эти точки и пробуем найти другой пик.
            if not assigned_comps:
                for pt_idx in neighbor_points:
                    available_points.discard(pt_idx)
                continue
            # Фиксируем найденную линию и привязанные к ней компоненты.
            line = {
                'rho': rho_center,
                'theta': theta,
                'components': assigned_comps,
                'point_indices': list(neighbor_points)
            }
            lines.append(line)
            # Удаляем точки уже назначенных компонент из дальнейшего голосования.
            for comp_idx in assigned_comps:
                for pt_idx in comp_points[comp_idx]:
                    available_points.discard(pt_idx)

        if self.debug and iter_count >= max_iter:
            print("[DEBUG] Hough iteration limit reached")
        self.lines = lines
        if self.debug:
            debug_img = self.image.copy() if len(self.image.shape) == 3 else cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            for line in lines:
                self._draw_line(debug_img, line, (0, 0, 255), thickness=3)
            self._save_debug_image(debug_img, "hough_lines")
            print(f"[DEBUG] Hough found {len(lines)} lines")
        return lines

    def _draw_line(self, img, line, color, thickness=2):
        """
        Рисует линию, заданную параметрами rho, theta, на изображении img.

        Вход:
            img (np.ndarray): изображение (RGB или серое)
            line (dict): словарь с ключами 'rho', 'theta'
            color (tuple): цвет (BGR или RGB)
            thickness (int): толщина линии
        """
        h, w = img.shape[:2]
        rho = line['rho']
        theta = line['theta']
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        points = []
        for x in [0, w-1]:
            if abs(b) > 1e-6:
                y = (rho - a * x) / b
                if 0 <= y < h:
                    points.append((int(x), int(y)))
        for y in [0, h-1]:
            if abs(a) > 1e-6:
                x = (rho - b * y) / a
                if 0 <= x < w:
                    points.append((int(x), int(y)))
        if len(points) >= 2:
            cv2.line(img, points[0], points[1], color, thickness)

    def postprocess_merge_lines(self):
        """
        Сливает близкие линии (по вертикали) в одну на основе среднего расстояния между ними.
        """
        # Если линий меньше двух, сливать нечего.
        if len(self.lines) < 2:
            return

        # mid_x нужен для приведения каждой линии к единой "вертикальной" координате y_mid
        # (координата линии в середине изображения по X).
        h, w = self.binary.shape
        mid_x = w / 2.0

        # Для каждой линии считаем y_mid из (rho, theta) в полярной форме.
        # Это позволяет сравнивать линии между собой по вертикальному положению.
        # avg_theta = np.mean([line['theta'] for line in self.lines])
        for line in self.lines:
            theta_rad = np.deg2rad(line['theta'])
            sin_theta = np.sin(theta_rad)
            cos_theta = np.cos(theta_rad)
            if abs(sin_theta) < 1e-6:
                # Защита от почти горизонтальной/вырожденной конфигурации.
                line['y_mid'] = 0
            else:
                line['y_mid'] = (line['rho'] - mid_x * cos_theta) / sin_theta

        # Сортируем линии сверху вниз (по y_mid), чтобы проверять слияние соседних.
        self.lines.sort(key=lambda l: l['y_mid'])

        # Оцениваем "типичный" межстрочный интервал как среднее соседних расстояний.
        # Он используется как адаптивная база для порога слияния.
        distances = []
        for i in range(len(self.lines)-1):
            d = abs(self.lines[i+1]['y_mid'] - self.lines[i]['y_mid'])
            distances.append(d)

        # Если расстояния не посчитались, используем среднюю высоту символа AH.
        avg_dist = np.mean(distances) if distances else self.AH

        # Линейный проход по отсортированным линиям:
        # текущая линия "втягивает" следующие, если они достаточно близки по y_mid.
        merged = []
        i = 0
        while i < len(self.lines):
            curr_line = self.lines[i]
            # Компоненты итоговой линии начинаются с компонент текущей.
            comps = curr_line['components'][:]
            j = i + 1

            # Критерий слияния:
            # расстояние по y_mid < avg_dist * merge_distance_factor.
            # Чем больше merge_distance_factor, тем агрессивнее объединение.
            while j < len(self.lines) and abs(self.lines[j]['y_mid'] - curr_line['y_mid']) < avg_dist * self.params['merge_distance_factor']:
                comps.extend(self.lines[j]['components'])
                j += 1

            # Параметры rho/theta/y_mid берём от "якорной" (первой) линии группы.
            merged_line = {
                'rho': curr_line['rho'],
                'theta': curr_line['theta'],
                'components': comps,
                'y_mid': curr_line['y_mid']
            }
            merged.append(merged_line)
            # Переходим к следующей ещё не слитой линии.
            i = j

        # Обновляем список линий после слияния.
        self.lines = merged
        if self.debug:
            # В debug-режиме сохраняем визуализацию итоговых слитых линий.
            debug_img = self.image.copy() if len(self.image.shape) == 3 else cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            for line in self.lines:
                self._draw_line(debug_img, line, (0, 0, 255), thickness=3)
            self._save_debug_image(debug_img, "merged_lines")
            print(f"[DEBUG] After merging: {len(self.lines)} lines")

    def postprocess_assign_subset3(self):
        """
        Присваивает компоненты subset3 ближайшим строкам на основе вертикального расстояния.
        """
        if not self.lines:
            return
        h, w = self.binary.shape
        mid_x = w / 2.0
        for line in self.lines:
            if 'y_mid' not in line:
                theta_rad = np.deg2rad(line['theta'])
                sin_theta = np.sin(theta_rad)
                cos_theta = np.cos(theta_rad)
                if abs(sin_theta) < 1e-6:
                    line['y_mid'] = 0
                else:
                    line['y_mid'] = (line['rho'] - mid_x * cos_theta) / sin_theta
        for comp in self.subset3:
            cx, cy = comp['centroid']
            min_dist = float('inf')
            best_line = None
            for line in self.lines:
                theta_rad = np.deg2rad(line['theta'])
                sin_theta = np.sin(theta_rad)
                if abs(sin_theta) < 1e-6:
                    continue
                y_line = (line['rho'] - cx * np.cos(theta_rad)) / sin_theta
                dist = abs(cy - y_line)
                if dist < min_dist:
                    min_dist = dist
                    best_line = line
            if best_line is not None:
                best_line.setdefault('components_subset3', []).append(comp)

    def _skeletonize(self, mask):
        """
        Вычисляет скелет бинарной маски.

        Вход:
            mask (np.ndarray): бинарная маска (0/1)

        Выход:
            skeleton (np.ndarray): скелет (0/1)
        """
        return skeletonize(mask).astype(np.uint8)

    def _find_junctions(self, skeleton):
        """
        Находит узлы (точки пересечения) на скелете.

        Вход:
            skeleton (np.ndarray): бинарный скелет (0/1)

        Выход:
            junctions (list): список (x, y) координат узлов.
        """
        kernel = np.ones((3,3), dtype=np.uint8)
        neighbor_count = ndimage.convolve(skeleton, kernel, mode='constant', cval=0)
        junctions = np.argwhere((skeleton == 1) & (neighbor_count >= 4))
        return [(c, r) for r, c in junctions]

    def _remove_junctions_in_zone(self, skeleton, junctions, zone_y_range, neigh_size):
        """
        Удаляет узлы в заданной вертикальной зоне, зануляя окрестность.

        Вход:
            skeleton (np.ndarray): скелет
            junctions (list): список узлов
            zone_y_range (tuple): (y_min, y_max) – границы зоны (относительные координаты в маске)
            neigh_size (int): размер квадратной окрестности

        Выход:
            skeleton_mod (np.ndarray): модифицированный скелет
        """
        skel = skeleton.copy()
        y_min, y_max = zone_y_range
        for x, y in junctions:
            if y_min <= y <= y_max:
                for dy in range(-neigh_size//2, neigh_size//2+1):
                    for dx in range(-neigh_size//2, neigh_size//2+1):
                        ny = y + dy
                        nx = x + dx
                        if 0 <= nx < skel.shape[1] and 0 <= ny < skel.shape[0]:
                            skel[ny, nx] = 0
        return skel

    def _split_vertically_connected(self, comp):
        """
        Делит компоненту (subset2), слепленную по вертикали,
        на части с помощью горизонтального профиля (axis=1).
        """
        mask = (comp['mask'] > 0).astype(np.uint8)
        x0, y0, _, _ = comp['bbox']
        h, w = mask.shape

        if h < 8 or w < 3 or np.sum(mask) == 0:
            return [comp]

        ah = self.AH if self.AH is not None else max(8.0, h / 3.0)
        min_line_height = max(3, int(round(0.35 * ah)))
        min_gap = max(2, int(round(0.18 * ah)))
        min_region_height = max(3, int(round(0.30 * ah)))

        profile = np.sum(mask, axis=1).astype(np.float32)
        if len(profile) >= 7:
            win = max(7, int(round(0.45 * ah)))
            if win % 2 == 0:
                win += 1
            win = min(win, len(profile) if len(profile) % 2 == 1 else len(profile) - 1)
            profile_smooth = scipy.signal.savgol_filter(profile, win, 2) if win >= 5 else profile.copy()
        else:
            profile_smooth = profile.copy()

        p_min = float(np.min(profile_smooth))
        p_max = float(np.max(profile_smooth))
        if p_max - p_min < 1e-6:
            return [comp]

        norm_hpp = (profile_smooth - p_min) / (p_max - p_min)

        text_rows = norm_hpp > 0.22
        if not np.any(text_rows):
            return [comp]

        regions = []
        in_region = False
        start = 0
        for i, is_text in enumerate(text_rows):
            if is_text and not in_region:
                start = i
                in_region = True
            elif not is_text and in_region:
                regions.append((start, i - 1))
                in_region = False
        if in_region:
            regions.append((start, len(text_rows) - 1))

        merged_regions = []
        merge_gap = max(2, int(round(0.15 * ah)))
        for reg in regions:
            if not merged_regions:
                merged_regions.append(reg)
                continue
            prev_start, prev_end = merged_regions[-1]
            cur_start, cur_end = reg
            if cur_start - prev_end <= merge_gap:
                merged_regions[-1] = (prev_start, cur_end)
            else:
                merged_regions.append(reg)
        regions = [(s, e) for s, e in merged_regions if e - s + 1 >= min_region_height]

        if len(regions) <= 1:
            return [comp]

        start_points = []
        for i in range(len(regions) - 1):
            _, end_prev = regions[i]
            start_next, _ = regions[i + 1]
            if start_next - end_prev - 1 < min_gap:
                continue
            start_points.append((end_prev + start_next) // 2)

        if not start_points:
            return [comp]

        distance_to_text = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 3)
        distance_to_text = distance_to_text / (np.max(distance_to_text) + 1e-6)

        energy = mask.astype(np.float32) * 1000.0
        energy += (1.0 - distance_to_text) * 25.0

        row_penalty = np.abs(np.arange(h, dtype=np.float32)[:, None] - np.array(start_points, dtype=np.float32)[None, :])

        def _extract_seam(energy_map, start_y):
            cost = np.full((h, w), np.inf, dtype=np.float32)
            parent = np.full((h, w), -1, dtype=np.int32)
            cost[:, 0] = energy_map[:, 0] + np.abs(np.arange(h, dtype=np.float32) - start_y) * 3.0

            for x in range(1, w):
                prev = cost[:, x - 1]
                for y in range(h):
                    y0 = max(0, y - 1)
                    y1 = min(h, y + 2)
                    local_prev = prev[y0:y1]
                    best_off = int(np.argmin(local_prev))
                    best_y = y0 + best_off
                    smooth_penalty = abs(y - best_y) * 2.0
                    cost[y, x] = energy_map[y, x] + local_prev[best_off] + smooth_penalty
                    parent[y, x] = best_y

            end_y = int(np.argmin(cost[:, -1]))
            seam = np.zeros(w, dtype=np.int32)
            seam[-1] = end_y
            for x in range(w - 1, 0, -1):
                seam[x - 1] = parent[seam[x], x]
            return seam

        seams = []
        for seam_idx, start_y in enumerate(start_points):
            seam_energy = energy + row_penalty[:, [seam_idx]] * 1.5
            seam = _extract_seam(seam_energy, start_y)
            if np.mean(mask[seam, np.arange(w)]) > 0.35:
                continue
            seams.append(seam)

        if not seams:
            return [comp]

        seams = sorted(seams, key=lambda s: float(np.mean(s)))
        filtered_seams = []
        for seam in seams:
            if not filtered_seams:
                filtered_seams.append(seam)
                continue
            if np.mean(np.abs(seam - filtered_seams[-1])) >= min_line_height:
                filtered_seams.append(seam)
        seams = filtered_seams

        if not seams:
            return [comp]

        seams_full = [np.zeros(w, dtype=np.int32)] + seams + [np.full(w, h - 1, dtype=np.int32)]
        parts = []
        total_area = max(1, int(np.sum(mask)))

        for i in range(len(seams_full) - 1):
            upper = seams_full[i]
            lower = seams_full[i + 1]
            part_mask = np.zeros_like(mask, dtype=np.uint8)

            for x in range(w):
                y_top = int(min(upper[x], lower[x]))
                y_bottom = int(max(upper[x], lower[x]))
                if y_bottom - y_top <= 1:
                    continue
                col = mask[y_top + 1:y_bottom, x]
                if col.size:
                    part_mask[y_top + 1:y_bottom, x] = col

            if np.sum(part_mask) == 0:
                continue

            rows, cols = np.where(part_mask > 0)
            if len(rows) == 0:
                continue

            part_h = int(np.max(rows) - np.min(rows) + 1)
            if part_h < min_line_height and np.sum(part_mask) < 0.1 * total_area:
                continue

            r0, r1 = int(np.min(rows)), int(np.max(rows))
            c0, c1 = int(np.min(cols)), int(np.max(cols))
            cropped = part_mask[r0:r1 + 1, c0:c1 + 1]

            x = x0 + c0
            y = y0 + r0
            w_part = c1 - c0 + 1
            h_part = r1 - r0 + 1

            parts.append({
                'bbox': (x, y, w_part, h_part),
                'mask': cropped,
                'centroid': (x + w_part / 2, y + h_part / 2),
                'label': None
            })

        # if self.debug:
        #     debug_img = np.full((h, w, 3), 255, dtype=np.uint8)
        #     debug_img[mask > 0] = (50, 50, 50)

        #     overlay = debug_img.copy()
        #     for idx, part in enumerate(parts):
        #         px, py, pw, ph = part['bbox']
        #         local_x = px - x0
        #         local_y = py - y0
        #         color = BASE_COLORS[idx % len(BASE_COLORS)]
        #         overlay[local_y:local_y + ph, local_x:local_x + pw][part['mask'] > 0] = color
        #     debug_img = cv2.addWeighted(debug_img, 0.45, overlay, 0.55, 0)

        #     for seam in seams:
        #         debug_img[seam, np.arange(w)] = (0, 0, 255)
        #     self._save_debug_image(debug_img, "split_vertically_connected")

        #     plt.figure(figsize=(7, 4))
        #     plt.plot(profile, label='hpp', alpha=0.4)
        #     plt.plot(profile_smooth, label='hpp_smooth', linewidth=2)
        #     for start_row, end_row in regions:
        #         plt.axvspan(start_row, end_row, color='green', alpha=0.15)
        #     for start_y in start_points:
        #         plt.axvline(start_y, color='red', linestyle='--', alpha=0.7)
        #     plt.title('split_vertically_connected hpp')
        #     plt.xlabel('row')
        #     plt.ylabel('ink pixels')
        #     plt.legend()
        #     plt.grid(True, alpha=0.2)
        #     hist_path = os.path.join(self.debug_output_dir, f"{self.debug_counter:03d}_split_vertically_connected_profile.png")
        #     plt.savefig(hist_path)
        #     plt.close()
        #     self.debug_counter += 1

        return parts if len(parts) >= 2 else [comp]

    def _closest_lines_to_the_mask(self, x, y, w, h):
        """Выбираем ближайщею к маске линию и вторую ближайщею"""
        cx = x + w/2.0
        cy = y + h/2.0
        min_dist = float('inf')
        best_line = None
        next_line = None

        for line in self.lines:
            theta_rad = np.deg2rad(line['theta'])
            sin_theta = np.sin(theta_rad)
            cos_theta = np.cos(theta_rad)
            if abs(sin_theta) < 1e-6:
                continue
            y_line = (line['rho'] - cx * cos_theta) / sin_theta
            dist = abs(cy - y_line)
            if dist < min_dist:
                min_dist = dist

                next_line = best_line
                best_line = line

        return best_line, next_line

    def postprocess_split_subset2_add_to_subset1(self):
        """
        Делит компоненты subset2 (вертикально слепленные строки)
        и добавляет валидные части в subset1 для Hough.
        """

        new_parts = []

        for idx, comp in enumerate(self.subset2):
            parts = self._split_vertically_connected(comp)

            valid_parts = []

            for part in parts:
                _, _, w, h = part['bbox']

                # фильтрация мусора
                # if h < 0.3 * self.AH:
                #     continue
                # if w < 0.2 * self.AW:
                #     continue

                valid_parts.append(part)
                new_parts.append(part)

            # Debug визуализация
            # if self.debug:
            #     debug_img = self.image.copy() if len(self.image.shape) == 3 else cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

            #     x, y, w, h = comp['bbox']
            #     cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 255, 0), 2)

            #     overlay = debug_img.copy()

            #     for i, part in enumerate(parts):
            #         px, py, pw, ph = part['bbox']
            #         mask = part['mask']

            #         color = BASE_COLORS[i % len(BASE_COLORS)]

            #         region = overlay[py:py+ph, px:px+pw]
            #         region[mask > 0] = color

            #     debug_img = cv2.addWeighted(debug_img, 0.6, overlay, 0.4, 0)
            #     self._save_debug_image(debug_img, f"split_subset2_{idx}")

        # добавляем в subset1
        self.subset1.extend(new_parts)


    def _assign_all_components_to_lines(self):
        """
        Собирает все компоненты (subset1, subset3) и присваивает их
        ближайшим строкам по вертикальному расстоянию. Результат сохраняется в поле
        'all_components' каждой линии.
        """
        if not self.lines:
            return
        h, w = self.binary.shape
        mid_x = w / 2.0
        for line in self.lines:
            if 'y_mid' not in line:
                theta_rad = np.deg2rad(line['theta'])
                sin_theta = np.sin(theta_rad)
                cos_theta = np.cos(theta_rad)
                if abs(sin_theta) < 1e-6:
                    line['y_mid'] = 0
                else:
                    line['y_mid'] = (line['rho'] - mid_x * cos_theta) / sin_theta

        for line in self.lines:
            line['all_components'] = []

        # Subset1
        for comp in self.subset1:
            x, y, w, h = comp['bbox']
            best_line, next_line = self._closest_lines_to_the_mask(x, y, w, h)
            if best_line is not None:
                best_line['all_components'].append(('subset1', x, y, comp['mask']))
        # Subset3
        for comp in self.subset3:
            x, y, w, h = comp['bbox']
            best_line, next_line = self._closest_lines_to_the_mask(x, y, w, h)
            if best_line is not None:
                best_line['all_components'].append(('subset3', x, y, comp['mask']))

    def _create_colored_segmentation(self) -> np.ndarray:
        """
        Создаёт цветное изображение, где каждая строка текста закрашена своим цветом.
        Использует палитру из 30 хорошо различимых цветов.
        Если строк больше 30, цвета циклически повторяются с небольшим сдвигом оттенка,
        чтобы сохранить различимость.

        Выход:
            result (np.ndarray): цветное изображение сегментации (фон белый, текст цветной)
            line_masks (list): список масок для каждой линии (каждая маска – список координат (y,x))
        """
        if not self.lines:
            return self.image.copy() if len(self.image.shape) == 3 else cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR), []


        h, w = self.binary.shape
        output = np.zeros((h, w, 3), dtype=np.uint8)

        n = len(self.lines)
        colors = []
        for i in range(n):
            if i < 30:
                colors.append(BASE_COLORS[i])
            else:
                base = BASE_COLORS[i % 30]
                # Преобразуем RGB -> HSV через colorsys (нормализованные 0..1)
                r, g, b = base[0]/255.0, base[1]/255.0, base[2]/255.0
                p, s, v = colorsys.rgb_to_hsv(r, g, b)   # h в диапазоне 0..1
                # OpenCV Hue = h * 180 (так как OpenCV хранит 0..180)
                h_opencv = p * 180
                shift = (i // 30) * 12 % 180
                new_h_opencv = (h_opencv + shift) % 180
                new_h = new_h_opencv / 180.0
                # Обратно в RGB
                new_r, new_g, new_b = colorsys.hsv_to_rgb(new_h, s, v)
                colors.append((int(new_r*255), int(new_g*255), int(new_b*255)))

        line_masks = []  # формируем маски для каждой линии
        for idx, line in enumerate(self.lines):
            color = colors[idx]
            line_masks.append([])
            for _, x, y, mask in line.get('all_components', []):
                rows, cols = np.where(mask)
                output[y + rows, x + cols] = color
                for yy, xx in zip(y + rows, x + cols):
                    line_masks[idx].append((xx, yy))

        # Фон – белый, текст – цветной
        bg = np.ones((h, w, 3), dtype=np.uint8) * 255
        mask_color = (output.sum(axis=2) > 0)
        result = bg
        result[mask_color] = output[mask_color]
        return result, line_masks

    def postprocess_create_new_lines(self):
        """
        Создаёт новые строки из оставшихся компонент Subset1, которые не были
        обнаружены на этапе Hough. Реализует этап "Creation of New Text Lines"
        из статьи Louloudis et al.
        """
        if not self.lines:
            return

        h, w = self.binary.shape
        mid_x = w / 2.0

        # 1. Вычисляем y_mid для существующих линий, если ещё не вычислены
        for line in self.lines:
            if 'y_mid' not in line:
                theta_rad = np.deg2rad(line['theta'])
                sin_theta = np.sin(theta_rad)
                cos_theta = np.cos(theta_rad)
                if abs(sin_theta) < 1e-6:
                    line['y_mid'] = 0
                else:
                    line['y_mid'] = (line['rho'] - mid_x * cos_theta) / sin_theta

        # 2. Сортируем линии по y_mid и вычисляем среднее расстояние между строками Ad
        self.lines.sort(key=lambda l: l['y_mid'])
        distances = []
        for i in range(len(self.lines)-1):
            d = abs(self.lines[i+1]['y_mid'] - self.lines[i]['y_mid'])
            distances.append(d)
        if not distances:
            return
        Ad = np.mean(distances)

        # 3. Определяем множество индексов компонент Subset1, уже присвоенных линиям
        assigned_indices = set()
        for line in self.lines:
            if 'components' in line:
                assigned_indices.update(line['components'])

        # 4. Собираем оставшиеся компоненты Subset1
        remaining = [(idx, comp) for idx, comp in enumerate(self.subset1) if idx not in assigned_indices]
        if not remaining:
            return

        # 5. Для каждой оставшейся компоненты проверяем, образует ли она новую строку
        candidates_for_new_lines = []  # список (idx, comp, list_of_candidate_centers)
        for idx, comp in remaining:
            centers = self._compute_gravity_centers(comp, self.AW)  # список (cx, cy)
            if not centers:
                continue
            # Для каждого центра определяем, является ли он кандидатом
            candidate_centers = []
            for cx, cy in centers:
                # Находим ближайшую существующую линию
                min_dist = float('inf')
                for line in self.lines:
                    theta_rad = np.deg2rad(line['theta'])
                    sin_theta = np.sin(theta_rad)
                    cos_theta = np.cos(theta_rad)
                    if abs(sin_theta) < 1e-6:
                        continue
                    y_line = (line['rho'] - cx * cos_theta) / sin_theta
                    dist = abs(cy - y_line)
                    if dist < min_dist:
                        min_dist = dist
                # Условие кандидата: расстояние до ближайшей линии близко к среднему Ad
                lower_factor = self.params['new_line_lower_factor']
                upper_factor = self.params['new_line_upper_factor']
                if lower_factor * Ad <= min_dist <= upper_factor * Ad:
                    candidate_centers.append((cx, cy))
            # Компонента образует новую строку, если более половины её блоков — кандидаты
            if len(candidate_centers) >= len(centers) / 2:
                candidates_for_new_lines.append((idx, comp, candidate_centers))

        if not candidates_for_new_lines:
            return

        # 6. Группируем компоненты в новые строки по вертикальной близости
        # Сортируем по среднему y кандидатов (или по центроиду компоненты)
        comp_data = []
        for idx, comp, centers in candidates_for_new_lines:
            avg_y = np.mean([cy for (_, cy) in centers])
            comp_data.append((idx, comp, centers, avg_y))
        comp_data.sort(key=lambda x: x[3])  # сортировка по avg_y

        new_lines_groups = []
        current_group = []
        for item in comp_data:
            if not current_group:
                current_group.append(item)
            else:
                last_avg_y = current_group[-1][3]
                if abs(item[3] - last_avg_y) < self.params['new_line_vertical_grouping_factor'] * self.AH:
                    current_group.append(item)
                else:
                    new_lines_groups.append(current_group)
                    current_group = [item]
        if current_group:
            new_lines_groups.append(current_group)

        # 7. Для каждой группы создаём новую линию
        # Получаем медианный угол существующих линий (для параллельности)
        existing_angles = [line['theta'] for line in self.lines]
        median_theta = np.median(existing_angles) if existing_angles else 90.0

        for group in new_lines_groups:
            all_points = []
            for item in group:
                centers = item[2]
                all_points.extend(centers)
            if len(all_points) < 2:
                avg_x = np.mean([p[0] for p in all_points])
                avg_y = np.mean([p[1] for p in all_points])
                theta_rad = np.deg2rad(median_theta)
                rho = avg_x * np.cos(theta_rad) + avg_y * np.sin(theta_rad)
                theta_deg = median_theta   # <-- добавлено
            else:
                pts = np.array(all_points)
                mean = np.mean(pts, axis=0)
                centered = pts - mean
                cov = np.cov(centered.T)
                eigvals, eigvecs = np.linalg.eig(cov)
                main_dir = eigvecs[:, np.argmax(eigvals)]
                theta_rad = np.arctan2(main_dir[1], main_dir[0])
                theta_deg = np.rad2deg(theta_rad)
                if theta_deg < 0:
                    theta_deg += 180
                if abs(theta_deg - 90) > 30:
                    theta_deg = median_theta
                rho = np.mean([x * np.cos(np.deg2rad(theta_deg)) + y * np.sin(np.deg2rad(theta_deg))
                            for x, y in all_points])
            new_line = {
                'rho': rho,
                'theta': theta_deg,
                'components': [item[0] for item in group],
                'y_mid': None
            }
            self.lines.append(new_line)

        if self.debug:
            debug_img = self.image.copy() if len(self.image.shape) == 3 else cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            for line in self.lines:
                self._draw_line(debug_img, line, (0, 0, 255), thickness=3)
            self._save_debug_image(debug_img, "new_lines")
            print(f"[DEBUG] After after new lines: {len(self.lines)} lines")

        if self.debug:
            print(f"[DEBUG] postprocess_create_new_lines: added {len(new_lines_groups)} new lines")

    def postprocess_filtering_lines_skew(self):
        if len(self.lines) > 1:
            median_theta = np.median([line['theta'] for line in self.lines])
            filtered = []
            for line in self.lines:
                if abs(line['theta'] - median_theta) <= self.params['angle_filter_threshold']:
                    filtered.append(line)
                else:
                    if self.debug:
                        print(f"[DEBUG] Отброшена линия с углом {line['theta']:.1f}° (отклонение {abs(line['theta'] - median_theta):.1f}°)")
            self.lines = filtered
        if self.debug:
            # В debug-режиме сохраняем визуализацию итоговых слитых линий.
            debug_img = self.image.copy() if len(self.image.shape) == 3 else cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            for line in self.lines:
                self._draw_line(debug_img, line, (0, 0, 255), thickness=3)
            self._save_debug_image(debug_img, "filtering_lines")


    def detect_text_lines(self) -> List[Dict]:
        """
        Основной метод детекции строк текста.
        Выполняет последовательно все шаги алгоритма:
            - бинаризация
            - извлечение связных компонент
            - оценка средней высоты символа
            - разбиение компонент на подмножества
            - блочное преобразование Хафа для обнаружения линий
            - постобработка: слияние линий, присвоение subset3 и split_parts
            - создание цветной сегментации
            - вырезание строк и сохранение в папку input/lines

        Выход:
            lines (list): список словарей с параметрами найденных строк.
        """
        self.binarize()
        self.extract_connected_components()
        self.estimate_average_character_height()
        self.partition_components() #
        self.postprocess_split_subset2_add_to_subset1()
        self.block_based_hough()
        self.postprocess_filtering_lines_skew()
        if self.debug:
            # Визуализация неприкреплённых subset1-компонент сразу после Hough.
            # Компоненты: зелёный, линии Хафа: красный.
            debug_img = self.image.copy() if len(self.image.shape) == 3 else cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

            assigned_indices = set()
            for line in self.lines:
                if 'components' in line:
                    assigned_indices.update(line['components'])

            for idx, comp in enumerate(self.subset1):
                if idx in assigned_indices:
                    continue
                x, y, w, h = comp['bbox']
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

            for line in self.lines:
                self._draw_line(debug_img, line, (0, 0, 255), thickness=3)

            self._save_debug_image(debug_img, "hough_unassigned_subset1")
        self.postprocess_merge_lines()
        self.postprocess_create_new_lines()
        #self.postprocess_assign_subset3()
        self._assign_all_components_to_lines()

        seg_img, line_masks = self._create_colored_segmentation()
        if self.debug:
            self._save_debug_image(seg_img, "final_segmentation")
            print(f"[DEBUG] Final number of lines: {len(self.lines)}")

        line_crops = []

        for i in range(len(line_masks)):
            H, W = self.image.shape[:2]
            white_image = np.ones((H, W, 3), dtype=np.uint8) * 255
            for x, y in line_masks[i]:
                white_image[y, x] = (0, 0, 0)

            # Вырезаем и выпрямляем
            crop = crop_line_rectangle(white_image, line_masks[i], debug=False, padding=0)
            line_crops.append(crop)

        if self.debug:
            save_dir = "output/lines"
            os.makedirs(save_dir, exist_ok=True)
            for idx, crop in enumerate(line_crops):
                filename = os.path.join(save_dir, f"line_{idx:03d}.jpg")
                cv2.imwrite(filename, crop)

        return self.lines, line_crops, line_masks


if __name__ == "__main__":
    def to_nested_hough_params(flat_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Преобразует "плоский" формат (hough_theta_min, subset1_min, nl_*)
        в формат параметров детектора (hough_theta_range, subset1_height_bounds, new_line_*).
        """
        return {
            "hough_theta_range": (
                float(flat_params["hough_theta_min"]),
                float(flat_params["hough_theta_max"]),
            ),
            "hough_rho_step_factor": float(flat_params["hough_rho_step_factor"]),
            "hough_max_votes_threshold": float(flat_params["hough_max_votes_threshold"]),
            "hough_secondary_threshold": float(flat_params["hough_secondary_threshold"]),
            "hough_angle_tolerance": float(flat_params["hough_angle_tolerance"]),
            "hough_neighborhood_radius": float(flat_params["hough_neighborhood_radius"]),
            "merge_distance_factor": float(flat_params["merge_distance_factor"]),
            "subset1_height_bounds": (
                float(flat_params["subset1_min"]),
                float(flat_params["subset1_max"]),
            ),
            "subset1_width_factor": float(flat_params["subset1_width_factor"]),
            "subset2_height_factor": float(flat_params["subset2_height_factor"]),
            "subset3_height_factor": float(flat_params["subset3_height_factor"]),
            "subset3_width_factor": float(flat_params["subset3_width_factor"]),
            "hough_small_dataset_threshold": float(flat_params["hough_small_dataset_threshold"]),
            "hough_large_dataset_threshold": float(flat_params["hough_large_dataset_threshold"]),
            "hough_min_max_votes": float(flat_params["hough_min_max_votes"]),
            "hough_min_secondary_votes": float(flat_params["hough_min_secondary_votes"]),
            "hough_max_max_votes": float(flat_params["hough_max_max_votes"]),
            "hough_max_secondary_votes": float(flat_params["hough_max_secondary_votes"]),
            "skew_expansion_threshold": float(flat_params["angle_filter"]),
            "new_line_lower_factor": float(flat_params["nl_lower"]),
            "new_line_upper_factor": float(flat_params["nl_upper"]),
            "new_line_vertical_grouping_factor": float(flat_params["nl_vert"]),
            "angle_filter_threshold": float(flat_params["angle_filter"]),
            "min_components_for_skew": float(flat_params["min_skew"]),
            "number_component_blocks_that_voted_for_the_line": float(
                flat_params["number_component_blocks_that_voted_for_the_line"]
            ),
        }

    start = time.time()
    img_path = 'datasets/school_notebooks_RU/images_base/21_225.JPG'
    img = cv2.imread(img_path)

    #загружаем параметры (простая optuna)
    # study = joblib.load("models/optuna/optuna_hough_transform.pkl")
    # best_params = study.best_params
    # params = {'binarization_method': 'is_binary', **best_params}

    #вычисляем параметры с помощью бустинга по фичам юнета
    # from collect_boost_hough_dataset import (
    #     extract_unet_avg_feature,
    #     load_unet_model,
    # )
    # from train_boost_hough_models import (
    #     load_boost_bundle,
    #     predict_params_from_feature,
    # )

    # bundle = load_boost_bundle("models/boost_hough/boost_models.joblib")

    # models = bundle["models"]
    # fixed_params = bundle["fixed_params"]

    # unet_model, unet_device = load_unet_model()

    # image_path = "datasets/school_notebooks_RU/images_base/1_11.JPG"
    # feature_vector = extract_unet_avg_feature(image_path, unet_model, unet_device)

    # params = predict_params_from_feature(feature_vector, models, fixed_params)
    # params['binarization_method'] = 'is_binary'

    # del unet_model
    # print(params)

    # тупо подбор параметров для одного изображения через optuna
    # params = {
    #   "hough_theta_min": -1.0,
    #   "hough_theta_max": 5.0,
    #   "hough_rho_step_factor": 0.21088694141085365,
    #   "hough_max_votes_threshold": 5.0,
    #   "hough_secondary_threshold": 7.0,
    #   "hough_angle_tolerance": 1.0,
    #   "hough_neighborhood_radius": 5.0,
    #   "merge_distance_factor": 0.4063791074678376,
    #   "subset1_min": 0.5983551243076958,
    #   "subset1_max": 2.8825111470822153,
    #   "subset1_width_factor": 0.374266329726502,
    #   "subset2_height_factor": 3.4776055890103637,
    #   "subset3_height_factor": 0.7970941527440096,
    #   "subset3_width_factor": 0.3054151039617877,
    #   "hough_small_dataset_threshold": 57.0,
    #   "hough_large_dataset_threshold": 192.0,
    #   "hough_min_max_votes": 4.0,
    #   "hough_min_secondary_votes": 4.0,
    #   "hough_max_max_votes": 9.0,
    #   "hough_max_secondary_votes": 15.0,
    #   "nl_lower": 0.2253320274850556,
    #   "nl_upper": 1.2285457629200196,
    #   "nl_vert": 0.6158249434772077,
    #   "angle_filter": 1.0350776657948941,
    #   "min_skew": 5.0,
    #   "number_component_blocks_that_voted_for_the_line": 2.0
    # }
    # params = to_nested_hough_params(params)

    pages, binary_pages = extract_pages_with_yolo(
        image_path=img_path,
        model_path='models/yolo_detect_notebook/yolo_detect_notebook_1_(1-architecture).pt',
        output_dir='debug_images',
        conf_threshold=0.7,
        return_binary = True
    )

    if img is None:
        print("Не удалось загрузить изображение")
        exit()

    for page in binary_pages:
        detector = TextLineDetector(page, debug=True)
        lines, _, _ = detector.detect_text_lines()
        print(f"Обнаружено строк: {len(lines)}")

    finish = time.time()
    print(f"Время выполнения: {finish - start} секунд")
