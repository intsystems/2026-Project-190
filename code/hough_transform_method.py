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
            skeleton_junction_removal_zone: зона удаления узлов скелета (доли от высоты)
            skeleton_junction_neighborhood: размер окрестности для удаления узлов
            hough_small_dataset_threshold: порог для адаптации порогов при малом количестве компонент
            hough_large_dataset_threshold: порог для адаптации порогов при большом количестве компонент
            hough_min_max_votes: минимальное значение для hough_max_votes_threshold после адаптации
            hough_min_secondary_votes: минимальное значение для hough_secondary_threshold после адаптации
            hough_max_max_votes: максимальное значение для hough_max_votes_threshold после адаптации
            hough_max_secondary_votes: максимальное значение для hough_secondary_threshold после адаптации
            skew_expansion_threshold: порог наклона для расширения диапазона углов Хафа
            new_line_lower_factor: нижний множитель среднего расстояния для определения кандидата в новую строку
            new_line_upper_factor: верхний множитель среднего расстояния для определения кандидата
            new_line_vertical_grouping_factor: множитель AH для группировки компонент по вертикали
            angle_filter_threshold: порог отклонения угла для отбрасывания строк (в градусах)
            min_components_for_skew: минимальное количество компонент для оценки наклона
        """
        defaults = {
            'binarization_method': 'u_net',
            'hough_theta_range': (85, 95),
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
            'skeleton_junction_removal_zone': (0.5, 1.5),
            'skeleton_junction_neighborhood': 3,
            'hough_small_dataset_threshold': 50,
            'hough_large_dataset_threshold': 200,
            'hough_min_max_votes': 3,
            'hough_min_secondary_votes': 5,
            'hough_max_max_votes': 10,
            'hough_max_secondary_votes': 15,
            'skew_expansion_threshold': 3,
            'new_line_lower_factor': 0.7,
            'new_line_upper_factor': 1.1,
            'new_line_vertical_grouping_factor': 0.8,
            'angle_filter_threshold': 10,
            'min_components_for_skew': 5,
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
            plt.show()
            hist_path = os.path.join(self.debug_output_dir, f"{self.debug_counter:03d}_height_histogram.png")
            plt.savefig(hist_path)
            plt.close()
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
        if abs(skew_est) > self.params['skew_expansion_threshold']:
            new_range = (85 - abs(skew_est), 95 + abs(skew_est))
            new_range = (max(70, new_range[0]), min(110, new_range[1]))
            self.params['hough_theta_range'] = new_range
            if self.debug:
                print(f"[DEBUG] Расширен диапазон углов до {new_range} из-за наклона {skew_est:.1f}°")

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

        while iter_count < max_iter:
            iter_count += 1
            if len(available_points) < self.params['hough_max_votes_threshold']:
                if self.debug:
                    print(f"[DEBUG] Hough stopped: only {len(available_points)} points left")
                break
            points_with_idx = [(idx, all_points[idx][0], all_points[idx][1]) for idx in available_points]
            acc, rho_bins, theta_bins = self._hough_vote(points_with_idx, theta_range, rho_step)
            if not acc:
                break
            max_cell = max(acc.items(), key=lambda kv: len(kv[1]))
            (rho_idx, theta_idx), point_indices = max_cell
            num_votes = len(point_indices)
            if num_votes < self.params['hough_max_votes_threshold']:
                break
            theta = theta_bins[theta_idx]
            rho_center = rho_bins[rho_idx]
            if num_votes < self.params['hough_secondary_threshold'] and lines:
                dominant_theta = np.median([line['theta'] for line in lines])
                if abs(theta - dominant_theta) > self.params['hough_angle_tolerance']:
                    for pt_idx in point_indices:
                        available_points.discard(pt_idx)
                    continue
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
            for comp_idx, vote_count in comp_votes.items():
                total_blocks = len(comp_points[comp_idx])
                if vote_count >= total_blocks / 2:
                    if comp_idx not in comp_assigned:
                        assigned_comps.append(comp_idx)
                        comp_assigned.add(comp_idx)
            if not assigned_comps:
                for pt_idx in neighbor_points:
                    available_points.discard(pt_idx)
                continue
            line = {
                'rho': rho_center,
                'theta': theta,
                'components': assigned_comps,
                'point_indices': list(neighbor_points)
            }
            lines.append(line)
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
        if len(self.lines) < 2:
            return
        h, w = self.binary.shape
        mid_x = w / 2.0
        for line in self.lines:
            theta_rad = np.deg2rad(line['theta'])
            sin_theta = np.sin(theta_rad)
            cos_theta = np.cos(theta_rad)
            if abs(sin_theta) < 1e-6:
                line['y_mid'] = 0
            else:
                line['y_mid'] = (line['rho'] - mid_x * cos_theta) / sin_theta
        self.lines.sort(key=lambda l: l['y_mid'])
        distances = []
        for i in range(len(self.lines)-1):
            d = abs(self.lines[i+1]['y_mid'] - self.lines[i]['y_mid'])
            distances.append(d)
        avg_dist = np.mean(distances) if distances else self.AH
        merged = []
        i = 0
        while i < len(self.lines):
            curr_line = self.lines[i]
            comps = curr_line['components'][:]
            j = i + 1
            while j < len(self.lines) and abs(self.lines[j]['y_mid'] - curr_line['y_mid']) < avg_dist * self.params['merge_distance_factor']:
                comps.extend(self.lines[j]['components'])
                j += 1
            merged_line = {
                'rho': curr_line['rho'],
                'theta': curr_line['theta'],
                'components': comps,
                'y_mid': curr_line['y_mid']
            }
            merged.append(merged_line)
            i = j
        self.lines = merged
        if self.debug:
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
        Разделяет вертикально связанную компоненту (обычно из subset2) на две части.

        Вход:
            comp (dict): компонента с маской 'mask'

        Выход:
            (part1, part2) (np.ndarray, np.ndarray): две маски-части.
        """
        mask = comp['mask']
        h, w = mask.shape

        skeleton = self._skeletonize(mask)
        junctions = self._find_junctions(skeleton)

        zone_ymin_factor, zone_ymax_factor = self.params['skeleton_junction_removal_zone']
        zone_y_min = int(h * zone_ymin_factor)
        zone_y_max = int(h * zone_ymax_factor)
        neigh_size = self.params['skeleton_junction_neighborhood']
        skeleton_mod = self._remove_junctions_in_zone(skeleton, junctions,
                                                      (zone_y_min, zone_y_max),
                                                      neigh_size)
        labeled_skeleton = measure.label(skeleton_mod, connectivity=2)
        upmost_label = None
        min_y = h
        for label in range(1, labeled_skeleton.max()+1):
            rows, _ = np.where(labeled_skeleton == label)
            if len(rows) > 0:
                y_min = np.min(rows)
                if y_min < min_y:
                    min_y = y_min
                    upmost_label = label
        if upmost_label is None:
            split_y = h // 2
            return mask[:split_y, :], mask[split_y:, :]
        upmost_mask = (labeled_skeleton == upmost_label)
        rest_mask = (labeled_skeleton != 0) & (labeled_skeleton != upmost_label)
        dist_up = ndimage.distance_transform_edt(~upmost_mask)
        dist_rest = ndimage.distance_transform_edt(~rest_mask)
        part1 = mask & (dist_up < dist_rest)
        part2 = mask & (dist_up >= dist_rest)

        if self.debug:
            # Создаём цветное изображение для отладки
            debug_img = self.image.copy() if len(self.image.shape) == 3 else cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            x, y, _, _ = comp['bbox']
            # Рисуем bounding box всей компоненты
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Накладываем part1 (красный) и part2 (синий) с прозрачностью
            overlay = debug_img.copy()
            # part1 и part2 — это маски (2D-массивы bool или uint8 с 0/1)
            part1_mask = (part1 > 0)
            part2_mask = (part2 > 0)
            overlay[y:y+h, x:x+w][part1_mask] = (0, 0, 255)   # красный для part1
            overlay[y:y+h, x:x+w][part2_mask] = (0, 255, 0)   # зелёный для part2 (чтобы лучше видеть)
            # Смешиваем с исходным
            debug_img = cv2.addWeighted(debug_img, 0.6, overlay, 0.4, 0)
            #self._save_debug_image(debug_img, "split_component")

        return part1, part2

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

    def postprocess_split_subset2(self):
        """
        Разделяет компоненты subset2 на части и присваивает их ближайшим строкам.
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

        for idx, comp in enumerate(self.subset2):
            x_comp, y_comp, w_comp, h_comp = comp['bbox']
            part1, part2 = self._split_vertically_connected(comp)
            
            best_line1, best_line2 = None, None

            # Вычисляем bounding box для part1
            rows1, cols1 = np.where(part1 > 0)
            if len(rows1) > 0:
                x1 = x_comp + np.min(cols1)
                y1 = y_comp + np.min(rows1)
                w1 = np.max(cols1) - np.min(cols1) + 1
                h1 = np.max(rows1) - np.min(rows1) + 1
                best_line1, _ = self._closest_lines_to_the_mask(x1, y1, w1, h1)
                if best_line1 is not None:
                    best_line1['all_components'].append(('split_parts', x_comp, y_comp, part1))

            # Вычисляем bounding box для part2
            rows2, cols2 = np.where(part2 > 0)
            if len(rows2) > 0:
                x2 = x_comp + np.min(cols2)
                y2 = y_comp + np.min(rows2)
                w2 = np.max(cols2) - np.min(cols2) + 1
                h2 = np.max(rows2) - np.min(rows2) + 1
                best_line2, _ = self._closest_lines_to_the_mask(x2, y2, w2, h2)
                if best_line2 is not None:
                    best_line2['all_components'].append(('split_parts', x_comp, y_comp, part2))

            if self.debug:
                debug_img = self.image.copy() if len(self.image.shape) == 3 else cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
                # Рамка компоненты
                cv2.rectangle(debug_img, (x_comp, y_comp), (x_comp + w_comp, y_comp + h_comp), (255, 255, 0), 2)

                # Наложение частей с прозрачностью
                overlay = debug_img.copy()
                part1_mask = (part1 > 0)
                part2_mask = (part2 > 0)
                overlay[y_comp:y_comp + h_comp, x_comp:x_comp + w_comp][part1_mask] = (0, 0, 255)   # красный
                overlay[y_comp:y_comp + h_comp, x_comp:x_comp + w_comp][part2_mask] = (0, 255, 0)   # зелёный
                debug_img = cv2.addWeighted(debug_img, 0.6, overlay, 0.4, 0)

                # Рисуем линии
                if best_line1 is not None:
                    self._draw_line(debug_img, best_line1, (255, 0, 0), thickness=2)      # синяя – ближайшая
                if best_line2 is not None and best_line2 != best_line1:
                    self._draw_line(debug_img, best_line2, (0, 255, 255), thickness=2)    # жёлтая – вторая

                #self._save_debug_image(debug_img, f"split_comp_{idx}")

    def _assign_all_components_to_lines(self):
        """
        Собирает все компоненты (subset1, subset2, subset3, split_parts) и присваивает их
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

        # Палитра из 30 хорошо различимых цветов (RGB)
        base_colors = [
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

        n = len(self.lines)
        colors = []
        for i in range(n):
            if i < 30:
                colors.append(base_colors[i])
            else:
                base = base_colors[i % 30]
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
        self.partition_components()
        self.block_based_hough()
        self.postprocess_merge_lines()
        self.postprocess_create_new_lines()
        self.postprocess_assign_subset3()
        self.postprocess_split_subset2()
        self._assign_all_components_to_lines()

        # Дополнительная фильтрация строк с большим наклоном после слияния
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
    img_path = 'datasets/school_notebooks_RU/images_base/1_11.JPG'
    img = cv2.imread(img_path)

    # загружаем параметры
    study = joblib.load("models/optuna/optuna_hough_transform.pkl")
    best_params = study.best_params
    all_params = {'binarization_method': 'u_net', **best_params}

    pages = extract_pages_with_yolo(
        image_path=img_path,
        model_path='models/yolo_detect_notebook/yolo_detect_notebook_1_(1-architecture).pt',
        output_dir='debug_images',
        conf_threshold=0.7
    )

    if img is None:
        print("Не удалось загрузить изображение")
        exit()

    for page in pages:
        detector = TextLineDetector(page, params=best_params, debug=True)
        lines, _, _ = detector.detect_text_lines()
        print(f"Обнаружено строк: {len(lines)}")