import heapq
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from post_processing import crop_line_rectangle
from processing import (
    extract_pages_with_yolo,
    correct_perspective,
    warp_binary_by_local_angles,
    warp_binary_by_local_angles_bijection,
)
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, peak_widths, savgol_filter


# Общая папка для всех отладочных изображений метода.
DEBUG_IMAGES_DIR = "debug_images"

# Подпапка только для debug-графиков HPP-профиля и разных вариантов его сглаживания.
HPP_DEBUG_DIR = os.path.join(DEBUG_IMAGES_DIR, "hpp")

# Размер окна скользящего среднего в debug-сравнении сглаживаний HPP.
# Чем больше окно, тем сильнее сглаживаются шумные пики, но тем сильнее размываются границы строк.
HPP_MOVING_AVERAGE_WINDOW = 21

# Sigma гауссова фильтра в debug-сравнении сглаживаний HPP.
# Большая sigma делает профиль плавнее и устойчивее к шуму, но может склеивать близкие строки.
HPP_GAUSSIAN_SIGMA_DEBUG = 3.0

# Размер окна медианного фильтра в debug-сравнении сглаживаний HPP.
# Медиана хорошо убирает одиночные выбросы, но при слишком большом окне съедает тонкие строки.
HPP_MEDIAN_FILTER_SIZE = 21

# Размер окна Savitzky-Golay фильтра в debug-сравнении сглаживаний HPP.
# Фильтр сглаживает профиль, стараясь сохранить форму пиков лучше, чем простое среднее.
HPP_SAVGOL_WINDOW = 31

# Степень полинома Savitzky-Golay фильтра.
# Чем выше степень, тем гибче аппроксимация, но тем выше риск подстроиться под шум.
HPP_SAVGOL_POLYORDER = 3

# Коэффициент гладкости сплайна в debug-сравнении HPP.
# Больше значение -- более гладкая кривая; меньше -- сплайн ближе повторяет исходный шумный профиль.
HPP_SPLINE_SMOOTH_FACTOR = 0.03

# Размер морфологического ядра для debug-сглаживания HPP-профиля.
# Используется как 1D open/close по профилю: помогает закрывать мелкие провалы и убирать мелкие пики.
HPP_MORPH_KERNEL_SIZE = 21

# Sigma гауссова сглаживания перед основным Otsu-порогом в _find_line_regions.
# Это уже рабочий гиперпараметр сегментации: больше sigma дает стабильнее порог, но может склеить строки.
HPP_OTSU_SMOOTH_SIGMA = 1.5

# Минимальная высота найденного TEXT-региона в пикселях.
# Все сегменты ниже этого размера считаются шумом/ложными строками и удаляются.
HPP_MIN_TEXT_REGION_HEIGHT = 4

# Максимальный вертикальный gap между соседними TEXT-регионами, при котором их склеиваем.
# Нужен, чтобы маленькая дырка внутри одной строки не разбивала ее на две разные строки.
HPP_MAX_GAP_TO_MERGE = 3

# Размер 1D-морфологического closing по TEXT/GAP маске после Otsu.
# Закрывает маленькие дырки внутри строк; слишком большое значение может склеивать соседние строки.
HPP_CLOSING_STRUCTURE_SIZE = 5

# Padding вокруг найденных HPP-регионов при повторном поиске строк.
# Нужен, чтобы первый проход удалил не только центр строки, но и близкие пиксели букв/шум вокруг нее.
HPP_SECOND_PASS_REGION_PADDING = 15

# Порог непроходимой энергии для A*.
# Черный текст дает около 2550, HPP-регионы строк получают еще +5000, поэтому такие клетки считаем стеной.
HPP_A_STAR_BLOCKED_ENERGY = 2000.0


class LineSegmentation:
    """
    Сегментирует строки в рукописных документах методом HPP и энергетических швов.
    """

    def __init__(self,
                 threshold: float = 0.4,
                 gaussian_sigma: float = 1.0,
                 debug: bool = True,
                 page_yolo_model: Any = None,
                 use_warp_binary_by_local_angles: bool = True,
                 use_bijection_warp: bool = False):
        """
        Короткое описание:
            задает параметры сегментации строк и режим сохранения отладки.
        Вход:
            threshold: float -- порог строки по нормализованному HPP.
            gaussian_sigma: float -- сигма гауссова сглаживания HPP.
            debug: bool -- сохранять отладочные файлы в debug_images.
            page_yolo_model: Any -- заранее загруженная YOLO-модель поиска страниц.
            use_warp_binary_by_local_angles: bool -- True: local warp, False: global correct_perspective.
            use_bijection_warp: bool -- True: использовать биективный local warp.
        Выход:
            None
        """
        self.threshold = threshold
        self.gaussian_sigma = gaussian_sigma
        self.debug = debug
        self.page_yolo_model = page_yolo_model
        self.use_warp_binary_by_local_angles = use_warp_binary_by_local_angles
        self.use_bijection_warp = use_bijection_warp

    def _horizontal_projection_profile(self, binary: np.ndarray) -> np.ndarray:
        """
        Короткое описание:
            вычисляет горизонтальный проекционный профиль бинарного изображения.
        Вход:
            binary: np.ndarray -- бинарное изображение, где текст равен 0, фон равен 255.
        Выход:
            np.ndarray -- количество текстовых пикселей в каждой строке.
        """
        return np.sum(binary == 0, axis=1).astype(np.float32)

    def _normalize_hpp(self, hpp: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Короткое описание:
            нормализует HPP minmax методом или после гауссова сглаживания.
        Вход:
            hpp: np.ndarray -- исходный горизонтальный проекционный профиль.
            method: str -- метод нормализации: minmax или gaussian.
        Выход:
            np.ndarray -- нормализованный профиль со значениями от 0 до 1.
        """
        if method == 'minmax':
            min_val = np.min(hpp)
            max_val = np.max(hpp)
            if max_val - min_val == 0:
                return np.zeros_like(hpp)
            return (hpp - min_val) / (max_val - min_val)
        elif method == 'gaussian':
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(hpp, sigma=self.gaussian_sigma)
            min_val = np.min(smoothed)
            max_val = np.max(smoothed)
            if max_val - min_val == 0:
                return np.zeros_like(smoothed)
            return (smoothed - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Неизвестный метод нормализации: {method}")

    def _normalize_profile_for_debug(self, profile: np.ndarray) -> np.ndarray:
        """
        Короткое описание:
            нормализует любой профиль в диапазон [0, 1] только для debug-графиков.
        Вход:
            profile: np.ndarray -- одномерный профиль.
        Выход:
            np.ndarray -- нормализованный профиль.
        """
        profile = np.asarray(profile, dtype=np.float32)
        min_val = float(np.min(profile))
        max_val = float(np.max(profile))
        if max_val - min_val < 1e-9:
            return np.zeros_like(profile, dtype=np.float32)
        return ((profile - min_val) / (max_val - min_val)).astype(np.float32)

    def _make_odd_window(self, requested_window: int, profile_length: int) -> int:
        """
        Короткое описание:
            подбирает нечетное окно сглаживания, которое помещается в профиль.
        Вход:
            requested_window: int -- желаемый размер окна.
            profile_length: int -- длина профиля.
        Выход:
            int -- безопасный нечетный размер окна.
        """
        window = max(3, min(int(requested_window), int(profile_length)))
        if window % 2 == 0:
            window -= 1
        return max(3, window)

    def _moving_average_profile(self, profile: np.ndarray, window_size: int) -> np.ndarray:
        """
        Короткое описание:
            сглаживает HPP скользящим средним.
        Вход:
            profile: np.ndarray -- исходный HPP.
            window_size: int -- размер окна.
        Выход:
            np.ndarray -- сглаженный профиль.
        """
        window_size = self._make_odd_window(window_size, len(profile))
        pad = window_size // 2
        padded = np.pad(profile.astype(np.float32), pad_width=pad, mode='edge')
        kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
        return np.convolve(padded, kernel, mode='valid').astype(np.float32)

    def _morphological_smooth_profile(self, profile: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Короткое описание:
            сглаживает профиль 1D-морфологией open+close по нормализованной кривой.
        Вход:
            profile: np.ndarray -- исходный HPP.
            kernel_size: int -- размер горизонтального морфологического ядра.
        Выход:
            np.ndarray -- сглаженный профиль в масштабе исходного HPP.
        """
        kernel_size = self._make_odd_window(kernel_size, len(profile))
        profile_min = float(np.min(profile))
        profile_max = float(np.max(profile))
        if profile_max - profile_min < 1e-9:
            return profile.astype(np.float32)

        normalized_u8 = (self._normalize_profile_for_debug(profile) * 255.0).astype(np.uint8).reshape(1, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
        closed = cv2.morphologyEx(normalized_u8, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        smoothed = opened.reshape(-1).astype(np.float32) / 255.0
        return smoothed * (profile_max - profile_min) + profile_min

    def _compute_hpp_smoothing_debug_profiles(self, hpp: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Короткое описание:
            считает несколько вариантов сглаживания HPP только для отладки.
        Вход:
            hpp: np.ndarray -- исходный горизонтальный проекционный профиль.
        Выход:
            Dict[str, np.ndarray] -- имя метода и сглаженный профиль.
        """
        hpp = hpp.astype(np.float32)
        profile_length = len(hpp)
        savgol_window = self._make_odd_window(HPP_SAVGOL_WINDOW, profile_length)
        median_size = self._make_odd_window(HPP_MEDIAN_FILTER_SIZE, profile_length)

        profiles = {
            "original": hpp,
            "moving_average": self._moving_average_profile(hpp, HPP_MOVING_AVERAGE_WINDOW),
            "gaussian": gaussian_filter1d(hpp, sigma=HPP_GAUSSIAN_SIGMA_DEBUG).astype(np.float32),
            "median": median_filter(hpp, size=median_size, mode='nearest').astype(np.float32),
            "morphological": self._morphological_smooth_profile(hpp, HPP_MORPH_KERNEL_SIZE),
        }

        if savgol_window > HPP_SAVGOL_POLYORDER:
            profiles["savitzky_golay"] = savgol_filter(
                hpp,
                window_length=savgol_window,
                polyorder=HPP_SAVGOL_POLYORDER,
                mode='nearest',
            ).astype(np.float32)

        if profile_length >= 4 and float(np.var(hpp)) > 1e-9:
            x = np.arange(profile_length, dtype=np.float32)
            smoothing_factor = float(profile_length) * float(np.var(hpp)) * HPP_SPLINE_SMOOTH_FACTOR
            spline = UnivariateSpline(x, hpp, s=smoothing_factor)
            profiles["spline"] = spline(x).astype(np.float32)

        return profiles

    def _save_hpp_smoothing_debug(self, hpp: np.ndarray, page_idx: int) -> None:
        """
        Короткое описание:
            сохраняет debug сравнение методов сглаживания HPP.
        Вход:
            hpp: np.ndarray -- исходный горизонтальный проекционный профиль.
            page_idx: int -- номер страницы.
        Выход:
            None
        """
        os.makedirs(HPP_DEBUG_DIR, exist_ok=True)
        profiles = self._compute_hpp_smoothing_debug_profiles(hpp)

        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        colors = {
            "original": "black",
            "moving_average": "tab:blue",
            "gaussian": "tab:orange",
            "median": "tab:green",
            "savitzky_golay": "tab:red",
            "spline": "tab:purple",
            "morphological": "tab:brown",
        }
        for name, profile in profiles.items():
            normalized = self._normalize_profile_for_debug(profile)
            linewidth = 2.2 if name != "original" else 1.0
            alpha = 0.9 if name != "original" else 0.45
            ax.plot(
                normalized,
                label=name,
                color=colors.get(name),
                linewidth=linewidth,
                alpha=alpha,
            )
            single_fig, single_ax = plt.subplots(1, 1, figsize=(14, 5))
            single_ax.plot(normalized, color=colors.get(name), linewidth=2.2)
            single_ax.axhline(
                self.threshold,
                color='gray',
                linestyle='--',
                linewidth=1.5,
                label=f'threshold={self.threshold:.2f}',
            )
            single_ax.set_title(f'HPP smoothing debug: {name}')
            single_ax.set_xlabel('Номер строки y')
            single_ax.set_ylabel('Нормализованное значение профиля')
            single_ax.grid(True, linestyle='--', alpha=0.5)
            single_ax.legend(loc='upper right')
            single_fig.tight_layout()
            single_fig.savefig(os.path.join(HPP_DEBUG_DIR, f'page_{page_idx:03d}_hpp_smoothing_{name}.jpg'))
            plt.close(single_fig)

        ax.axhline(self.threshold, color='gray', linestyle='--', linewidth=1.5, label=f'threshold={self.threshold:.2f}')
        ax.set_title('HPP smoothing debug: сравнение методов сглаживания')
        ax.set_xlabel('Номер строки y')
        ax.set_ylabel('Нормализованное значение профиля')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(os.path.join(HPP_DEBUG_DIR, f'page_{page_idx:03d}_hpp_smoothing_comparison.jpg'))
        plt.close(fig)

        csv_path = os.path.join(HPP_DEBUG_DIR, f'page_{page_idx:03d}_hpp_smoothing_profiles.csv')
        with open(csv_path, 'w', encoding='utf-8') as file:
            names = list(profiles.keys())
            file.write('y,' + ','.join(names) + '\n')
            for y_idx in range(len(hpp)):
                values = [self._normalize_profile_for_debug(profiles[name])[y_idx] for name in names]
                file.write(str(y_idx) + ',' + ','.join(f'{float(value):.6f}' for value in values) + '\n')

        summary_path = os.path.join(HPP_DEBUG_DIR, f'page_{page_idx:03d}_hpp_smoothing_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as file:
            file.write('Debug сглаживания горизонтального проекционного профиля HPP\n')
            file.write(f'profile_length={len(hpp)}\n')
            file.write(f'threshold={self.threshold}\n')
            file.write(f'moving_average_window={HPP_MOVING_AVERAGE_WINDOW}\n')
            file.write(f'gaussian_sigma={HPP_GAUSSIAN_SIGMA_DEBUG}\n')
            file.write(f'median_filter_size={HPP_MEDIAN_FILTER_SIZE}\n')
            file.write(f'savitzky_golay_window={HPP_SAVGOL_WINDOW}\n')
            file.write(f'savitzky_golay_polyorder={HPP_SAVGOL_POLYORDER}\n')
            file.write(f'spline_smooth_factor={HPP_SPLINE_SMOOTH_FACTOR}\n')
            file.write(f'morph_kernel_size={HPP_MORPH_KERNEL_SIZE}\n')

    def _find_line_regions(self,
                           normalized_hpp: np.ndarray,
                           debug_filename: Optional[str] = None) -> List[Tuple[int, int]]:
        """
        Короткое описание:
            находит области строк по нормализованному HPP.
        Вход:
            normalized_hpp: np.ndarray -- нормализованный профиль со значениями от 0 до 1.
            debug_filename: Optional[str] -- имя файла для debug-графика HPP.
        Выход:
            List[Tuple[int, int]] -- список пар start_row и end_row.
        """
        if len(normalized_hpp) == 0:
            return []
        if float(np.max(normalized_hpp)) <= 1e-9:
            return []

        # Otsu-модель: считаем, что HPP состоит из смеси двух классов TEXT/GAP.
        smoothed_hpp = gaussian_filter1d(
            normalized_hpp.astype(np.float32),
            sigma=HPP_OTSU_SMOOTH_SIGMA,
        )
        profile_u8 = np.clip(smoothed_hpp * 255.0, 0, 255).astype(np.uint8)
        otsu_threshold, _ = cv2.threshold(
            profile_u8,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        text_rows = profile_u8 > otsu_threshold

        # Закрываем мелкие дырки внутри строки в 1D-маске.
        kernel_size = max(1, int(HPP_CLOSING_STRUCTURE_SIZE))
        if kernel_size > 1:
            mask_u8 = text_rows.astype(np.uint8).reshape(1, -1) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
            text_rows = mask_u8.reshape(-1) > 0

        if not np.any(text_rows):
            return []

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

        # Удаляем короткие ложные сегменты.
        regions = [
            (start, end)
            for start, end in regions
            if end - start + 1 >= HPP_MIN_TEXT_REGION_HEIGHT
        ]

        merged = []
        for reg in regions:
            if not merged:
                merged.append(reg)
            else:
                prev_start, prev_end = merged[-1]
                curr_start, curr_end = reg
                if curr_start - prev_end <= HPP_MAX_GAP_TO_MERGE:
                    merged[-1] = (prev_start, curr_end)
                else:
                    merged.append(reg)
        if self.debug and debug_filename is not None:
            self._save_hpp_regions_debug(
                normalized_hpp,
                smoothed_hpp,
                text_rows,
                merged,
                otsu_threshold,
                debug_filename,
            )
        return merged

    def _save_hpp_regions_debug(self,
                                normalized_hpp: np.ndarray,
                                smoothed_hpp: np.ndarray,
                                text_rows: np.ndarray,
                                regions: List[Tuple[int, int]],
                                otsu_threshold: float,
                                debug_filename: str) -> None:
        """
        Короткое описание:
            сохраняет debug-график HPP-регионов для первого или повторного прохода.
        Вход:
            normalized_hpp: np.ndarray -- исходный нормализованный HPP.
            smoothed_hpp: np.ndarray -- сглаженный HPP перед Otsu.
            text_rows: np.ndarray -- бинарная TEXT/GAP маска.
            regions: List[Tuple[int, int]] -- найденные регионы.
            otsu_threshold: float -- Otsu-порог в шкале 0..255.
            debug_filename: str -- имя debug-файла.
        Выход:
            None
        """
        os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        x = np.arange(len(normalized_hpp))
        ax1.plot(x, normalized_hpp, color='tab:blue', linewidth=1.0, label='normalized_hpp')
        ax1.plot(x, smoothed_hpp, color='tab:orange', linewidth=2.0, label='gaussian_smoothed_hpp')
        ax1.axhline(float(otsu_threshold) / 255.0, color='tab:red', linestyle='--', label='otsu_threshold')
        for start, end in regions:
            ax1.axvspan(start, end, color='tab:green', alpha=0.18)
        ax1.set_title('HPP regions debug')
        ax1.set_ylabel('profile value')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right')

        ax2.plot(x, text_rows.astype(np.float32), color='tab:green', linewidth=2.0)
        for start, end in regions:
            ax2.axvspan(start, end, color='tab:green', alpha=0.18)
        ax2.set_xlabel('Номер строки y')
        ax2.set_ylabel('TEXT mask')
        ax2.grid(True, linestyle='--', alpha=0.5)

        fig.tight_layout()
        fig.savefig(os.path.join(DEBUG_IMAGES_DIR, debug_filename))
        plt.close(fig)

    def _find_line_regions_second_pass(self,
                                       binary: np.ndarray,
                                       line_regions: List[Tuple[int, int]],
                                       page_idx: int) -> List[Tuple[int, int]]:
        """
        Короткое описание:
            повторно ищет HPP-регионы после удаления уже найденных строк с padding.
        Вход:
            binary: np.ndarray -- бинарное изображение страницы.
            line_regions: List[Tuple[int, int]] -- регионы первого HPP-прохода.
            page_idx: int -- индекс страницы для debug-файлов.
        Выход:
            List[Tuple[int, int]] -- объединенный список регионов первого и второго прохода.
        """
        # Шаг 1: удаляем уже найденные строки, расширяя область на padding.
        remaining_binary = binary.copy()
        height = remaining_binary.shape[0]
        padding = max(0, int(HPP_SECOND_PASS_REGION_PADDING))
        removed_regions_with_padding = []
        for start, end in line_regions:
            y0 = max(0, int(start) - padding)
            y1 = min(height - 1, int(end) + padding)
            removed_regions_with_padding.append((y0, y1))
            remaining_binary[y0:y1 + 1, :] = 255
        cv2.imwrite(os.path.join(DEBUG_IMAGES_DIR, f'remaining_binary_{page_idx}.jpg'), remaining_binary)
        # Шаг 2: строим повторный HPP по оставшемуся тексту.
        second_hpp = self._horizontal_projection_profile(remaining_binary)
        second_norm_hpp = self._normalize_hpp(second_hpp, method='minmax')
        second_regions = self._find_line_regions(
            second_norm_hpp,
            debug_filename=f'page_{page_idx:03d}_hpp_second_pass_bases.jpg',
        )

        # Шаг 3: объединяем регионы и сортируем сверху вниз.
        merged_regions = sorted(
            line_regions + second_regions,
            key=lambda region: region[0],
        )

        if self.debug:
            second_mask = np.zeros(binary.shape[:2], dtype=np.uint8)
            for start, end in second_regions:
                second_mask[start:end + 1, :] = 255
            cv2.imwrite(
                os.path.join(DEBUG_IMAGES_DIR, f'page_{page_idx:03d}_hpp_second_pass_remaining.jpg'),
                remaining_binary,
            )
            cv2.imwrite(
                os.path.join(DEBUG_IMAGES_DIR, f'page_{page_idx:03d}_hpp_second_pass_new_regions_mask.jpg'),
                second_mask,
            )
            with open(
                os.path.join(DEBUG_IMAGES_DIR, f'page_{page_idx:03d}_hpp_second_pass_debug.txt'),
                'w',
                encoding='utf-8',
            ) as file:
                file.write(f'padding_only_for_second_vote={padding}\n')
                file.write(f'first_pass_regions_without_padding={line_regions}\n')
                file.write(f'removed_regions_with_padding={removed_regions_with_padding}\n')
                file.write(f'second_pass_regions_without_padding={second_regions}\n')
                file.write(f'final_line_regions_without_padding={merged_regions}\n')
                file.write('debug_images:\n')
                file.write(f'- page_{page_idx:03d}_hpp_second_pass_remaining.jpg: binary after removing first-pass regions with padding\n')
                file.write(f'- page_{page_idx:03d}_hpp_second_pass_new_regions_mask.jpg: only regions found on second pass\n')
                file.write(f'- page_{page_idx:03d}_hpp_second_pass_bases.jpg: second-pass HPP profile and TEXT mask\n')

        return merged_regions

    def _compute_energy_matrix(self,
                               binary: np.ndarray,
                               line_regions: List[Tuple[int, int]]) -> np.ndarray:
        """
        Короткое описание:
            строит энергетическую матрицу для поиска швов между строками.
        Вход:
            binary: np.ndarray -- бинарное изображение, где текст равен 0, фон равен 255.
            line_regions: List[Tuple[int, int]] -- области строк по HPP.
        Выход:
            np.ndarray -- энергетическая матрица размера H x W.
        """
        H, W = binary.shape
        energy = (255 - binary).astype(np.float32) * 10.0

        gray = binary if binary.dtype == np.uint8 else binary.astype(np.uint8)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        energy += 0.5 * gradient

        for start, end in line_regions:
            start = max(0, start - 2)
            end = min(H - 1, end + 5)
            energy[start:end+1, :] += 5000.0

        energy = np.nan_to_num(energy, nan=0.0, posinf=1e9, neginf=0.0)

        return energy

    def _compute_horizontal_min_energy_path_matrix(self, energy: np.ndarray) -> np.ndarray:
        """
        Короткое описание:
            вычисляет матрицу минимальных энергий горизонтальных путей.
        Вход:
            energy: np.ndarray -- энергетическая матрица размера H x W.
        Выход:
            np.ndarray -- матрица минимальных накопленных энергий размера H x W.
        """
        H, W = energy.shape
        min_energy = np.zeros((H, W), dtype=np.float32)
        min_energy[:, 0] = energy[:, 0]

        for x in range(1, W):
            for y in range(H):
                candidates = []
                for dy in (-1, 0, 1):
                    ny = y + dy
                    if 0 <= ny < H:
                        candidates.append(min_energy[ny, x-1])
                min_energy[y, x] = energy[y, x] + min(candidates)

        return min_energy

    def _extract_seam(self, min_energy_matrix: np.ndarray, start_y: int) -> List[int]:
        """
        Короткое описание:
            восстанавливает горизонтальный шов по матрице минимальных энергий.
        Вход:
            min_energy_matrix: np.ndarray -- матрица минимальных накопленных энергий.
            start_y: int -- строка старта шва на левом крае изображения.
        Выход:
            List[int] -- координаты y шва для каждого столбца x.
        """
        H, W = min_energy_matrix.shape
        seam = [0] * W
        y = start_y
        seam[0] = y

        y_cur = start_y
        for x in range(1, W):
            options = []
            for dy in (-1, 0, 1):
                ny = y_cur + dy
                if 0 <= ny < H:
                    options.append((ny, min_energy_matrix[ny, x]))
            best_ny, _ = min(options, key=lambda t: t[1])
            y_cur = best_ny
            seam[x] = y_cur
        return seam

    def _compute_min_energy_path_with_parents(self,
                                              energy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Короткое описание:
            создает матрицы минимальных энергий и предков для совместимости.
        Вход:
            energy: np.ndarray -- энергетическая матрица размера H x W.
        Выход:
            Tuple[np.ndarray, np.ndarray] -- матрица энергий и матрица предков.
        """
        H, W = energy.shape

        min_energy = np.full((H, W), np.inf, dtype=np.float32)
        parent = np.full((H, W), -1, dtype=np.int32)

        return min_energy, parent

    def _find_seam_a_star(self, energy: np.ndarray, start_y: int) -> List[int]:
        """
        Короткое описание:
            находит горизонтальный шов алгоритмом A* от левого до правого края, обходя запрещенные клетки.
        Вход:
            energy: np.ndarray -- энергетическая матрица размера H x W.
            start_y: int -- строка старта шва на левом крае изображения.
        Выход:
            List[int] -- координаты y шва для каждого столбца x.
        """
        H, W = energy.shape
        if not (0 <= start_y < H):
            return []

        blocked = energy >= HPP_A_STAR_BLOCKED_ENERGY
        directions = [-1, 0, 1]

        came_from = {}
        g_score = {}
        f_score = {}

        start = (start_y, 0)
        goal_x = W - 1

        # Шаг 1: если старт попал в запрещенную клетку, ищем ближайшую разрешенную строку в первом столбце.
        if blocked[start_y, 0]:
            free_ys = np.where(~blocked[:, 0])[0]
            if len(free_ys) == 0:
                return []
            start_y = int(free_ys[np.argmin(np.abs(free_ys - start_y))])
            start = (start_y, 0)

        g_score[start] = 0
        f_score[start] = 0

        open_set = []
        heapq.heappush(open_set, (0, start_y, 0))

        visited = set()

        while open_set:
            _, y, x = heapq.heappop(open_set)
            current = (y, x)

            if current in visited:
                continue
            visited.add(current)

            if x == goal_x:
                seam = [0] * W
                while current is not None:
                    cy, cx = current
                    seam[cx] = cy
                    current = came_from.get(current)
                return seam

            for dy in directions:
                ny = y + dy
                nx = x + 1
                if 0 <= ny < H and nx < W:
                    # Шаг 2: запрещаем A* проходить через текст и HPP-регионы с высокой энергией.
                    if blocked[ny, nx]:
                        continue
                    neighbor = (ny, nx)
                    tentative_g = g_score.get(current, np.inf) + energy[ny, nx]

                    if tentative_g < g_score.get(neighbor, np.inf):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g

                        h = (W - 1 - nx) * 1.0
                        f = tentative_g + h

                        f_score[neighbor] = f
                        heapq.heappush(open_set, (f, ny, nx))

        os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)
        warning_path = os.path.join(DEBUG_IMAGES_DIR, 'a_star_warnings.txt')
        with open(warning_path, 'a', encoding='utf-8') as file:
            file.write(
                f"Warning: A* не нашел путь для start_y={start_y}, "
                f"blocked_energy={HPP_A_STAR_BLOCKED_ENERGY}\n"
            )
        return []

    def _get_line_pixels_between_seams(self,
                                       binary: np.ndarray,
                                       upper_seam: List[int],
                                       lower_seam: List[int]) -> set:
        """
        Короткое описание:
            собирает текстовые пиксели между двумя горизонтальными швами.
        Вход:
            binary: np.ndarray -- бинарное изображение, где текст равен 0, фон равен 255.
            upper_seam: List[int] -- верхний шов как координата y для каждого x.
            lower_seam: List[int] -- нижний шов как координата y для каждого x.
        Выход:
            set -- множество координат текстовых пикселей в формате (x, y).
        """
        H, W = binary.shape
        pixels = set()
        for x in range(W):
            y_top = min(upper_seam[x], lower_seam[x])
            y_bot = max(upper_seam[x], lower_seam[x])
            for y in range(y_top + 1, y_bot):
                if 0 <= y < H and binary[y, x] == 0:
                        pixels.add((x, y))
        return pixels

    def _build_class_matrix(self,
                            shape: Tuple[int, int],
                            line_pixels: List[set]) -> np.ndarray:
        """
        Короткое описание:
            строит матрицу классов строк: 0 фон, 1 первая строка, 2 вторая строка и т.д.
        Вход:
            shape: Tuple[int, int] -- размер матрицы H x W.
            line_pixels: List[set] -- пиксели строк в координатах обработанного изображения.
        Выход:
            np.ndarray -- class-matrix типа int32.
        """
        class_matrix = np.zeros(shape, dtype=np.int32)
        for class_idx, pixels in enumerate(line_pixels, start=1):
            for x, y in pixels:
                if 0 <= y < shape[0] and 0 <= x < shape[1]:
                    class_matrix[y, x] = class_idx
        return class_matrix

    def _restore_class_matrix_to_input(self,
                                       class_matrix: np.ndarray,
                                       transform_sequence: dict) -> np.ndarray:
        """
        Короткое описание:
            возвращает class-matrix из координат warped-изображения в координаты входной страницы.
        Вход:
            class_matrix: np.ndarray -- классы строк после warp.
            transform_sequence: dict -- карты output_to_input_x/y из processing.
        Выход:
            np.ndarray -- class-matrix в координатах исходной бинарной страницы.
        """
        input_height, input_width = transform_sequence["input_shape"]
        restored = np.zeros((int(input_height), int(input_width)), dtype=np.int32)
        map_x = transform_sequence["output_to_input_x"]
        map_y = transform_sequence["output_to_input_y"]

        ys, xs = np.where(class_matrix > 0)
        for y, x in zip(ys, xs):
            source_x = int(round(float(map_x[y, x])))
            source_y = int(round(float(map_y[y, x])))
            if 0 <= source_y < restored.shape[0] and 0 <= source_x < restored.shape[1]:
                restored[source_y, source_x] = int(class_matrix[y, x])
        return restored

    def _save_class_matrix_debug(self,
                                 class_matrix: np.ndarray,
                                 page_idx: int,
                                 suffix: str) -> None:
        """
        Короткое описание:
            сохраняет цветную визуализацию матрицы классов строк.
        Вход:
            class_matrix: np.ndarray -- матрица классов.
            page_idx: int -- номер страницы.
            suffix: str -- суффикс имени файла.
        Выход:
            None
        """
        if not self.debug:
            return
        os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)
        vis = np.ones((*class_matrix.shape, 3), dtype=np.uint8) * 255
        random.seed(42)
        for class_idx in range(1, int(np.max(class_matrix)) + 1):
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            vis[class_matrix == class_idx] = color
        cv2.imwrite(os.path.join(DEBUG_IMAGES_DIR, f'page_{page_idx:03d}_{suffix}.jpg'), vis)

    def _save_full_class_matrix_debug(self,
                                      full_class_matrix: np.ndarray,
                                      image: np.ndarray = None) -> None:
        """
        Короткое описание:
            сохраняет подробный debug итоговой class-matrix в координатах исходного изображения.
        Вход:
            full_class_matrix: np.ndarray -- итоговая матрица классов.
            image: np.ndarray -- исходное изображение для overlay.
        Выход:
            None
        """
        if not self.debug or full_class_matrix is None:
            return
        os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)
        self._save_class_matrix_debug(full_class_matrix, 0, 'class_matrix_full_restored')
        np.save(os.path.join(DEBUG_IMAGES_DIR, 'full_class_matrix.npy'), full_class_matrix)

        classes, counts = np.unique(full_class_matrix, return_counts=True)
        summary_path = os.path.join(DEBUG_IMAGES_DIR, 'full_class_matrix_debug.txt')
        with open(summary_path, 'w', encoding='utf-8') as file:
            file.write(f'shape: {full_class_matrix.shape}\n')
            file.write(f'max_class: {int(np.max(full_class_matrix))}\n')
            file.write(f'text_pixels: {int(np.sum(full_class_matrix > 0))}\n')
            file.write('class_pixel_counts:\n')
            for class_idx, count in zip(classes, counts):
                file.write(f'  {int(class_idx)}: {int(count)}\n')

        if image is None:
            return
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        class_vis = np.ones_like(vis, dtype=np.uint8) * 255
        random.seed(42)
        for class_idx in range(1, int(np.max(full_class_matrix)) + 1):
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            class_vis[full_class_matrix == class_idx] = color
        text_mask = full_class_matrix > 0
        overlay = vis.copy()
        overlay[text_mask] = cv2.addWeighted(vis, 0.35, class_vis, 0.65, 0)[text_mask]
        cv2.imwrite(os.path.join(DEBUG_IMAGES_DIR, 'full_class_matrix_overlay.jpg'), overlay)

    def _paste_page_class_matrix(self,
                                 full_class_matrix: np.ndarray,
                                 page_class_matrix: np.ndarray,
                                 page_info: dict,
                                 class_offset: int) -> int:
        """
        Короткое описание:
            вклеивает class-matrix одной YOLO-страницы обратно в матрицу всего изображения.
        Вход:
            full_class_matrix: np.ndarray -- общая матрица исходного изображения.
            page_class_matrix: np.ndarray -- матрица классов в координатах bbox страницы.
            page_info: dict -- bbox страницы: x, y, w, h.
            class_offset: int -- сколько классов уже занято предыдущими страницами.
        Выход:
            int -- новый class_offset после вклейки страницы.
        """
        x0 = int(page_info["x"])
        y0 = int(page_info["y"])
        w = int(page_info["w"])
        h = int(page_info["h"])
        x1 = min(full_class_matrix.shape[1], x0 + w)
        y1 = min(full_class_matrix.shape[0], y0 + h)
        if x0 >= x1 or y0 >= y1:
            return class_offset

        page_fit = page_class_matrix[:y1 - y0, :x1 - x0].copy()
        nonzero = page_fit > 0
        if not np.any(nonzero):
            return class_offset

        page_fit[nonzero] += class_offset
        full_roi = full_class_matrix[y0:y1, x0:x1]
        full_roi[nonzero] = page_fit[nonzero]
        full_class_matrix[y0:y1, x0:x1] = full_roi
        return int(max(class_offset, int(np.max(page_fit))))

    def segment_lines(self,
                      image_path: str,
                      return_class_matrix: bool = False) -> List[set]:
        """
        Короткое описание:
            сегментирует строки на страницах рукописного документа.
        Вход:
            image_path: str -- путь к изображению документа.
        Выход:
            List[set] -- список множеств координат (x, y) текстовых пикселей строк.
        """
        image = cv2.imread(image_path) if (self.debug or return_class_matrix) else None
        if self.debug:
            os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(DEBUG_IMAGES_DIR, 'main_input.jpg'), image)

        # Шаг 1: находим страницы тетради и получаем бинарные изображения страниц.
        if return_class_matrix:
            pages, binary_pages, page_infos = extract_pages_with_yolo(
                image_path=image_path,
                model_path='models/yolo_detect_notebook/yolo_detect_notebook_1_(1-architecture).pt',
                output_dir=DEBUG_IMAGES_DIR,
                conf_threshold=0.8,
                return_binary=True,
                return_page_infos=True,
                yolo_model=self.page_yolo_model,
            )
        else:
            pages, binary_pages = extract_pages_with_yolo(
                image_path=image_path,
                model_path='models/yolo_detect_notebook/yolo_detect_notebook_1_(1-architecture).pt',
                output_dir=DEBUG_IMAGES_DIR,
                conf_threshold=0.8,
                return_binary=True,
                yolo_model=self.page_yolo_model,
            )
            page_infos = []
        # Инициализация списков для хранения пикселей строк и их обрезков
        lines_pixels = []
        lines_crops = []
        full_class_matrix = None
        class_offset = 0
        if return_class_matrix:
            if image is None:
                raise ValueError(f"Не удалось прочитать изображение: {image_path}")
            full_class_matrix = np.zeros(image.shape[:2], dtype=np.int32)
        for idx, page in enumerate(binary_pages):
            # Шаг 2: исправляем перспективу текущей страницы.
            # corrected_page, binary, _ = correct_perspective(
            #     page,
            #     debug=self.debug,
            #     debug_output_dir=os.path.join(DEBUG_IMAGES_DIR, f'page_{idx:03d}_perspective'),
            # )

            warp_function = (
                warp_binary_by_local_angles_bijection
                if self.use_bijection_warp
                else warp_binary_by_local_angles
            )

            if self.use_warp_binary_by_local_angles and return_class_matrix:
                binary, transform_sequence = warp_function(
                    page,
                    return_transform_sequence=True,
                )
            elif self.use_warp_binary_by_local_angles:
                binary = warp_function(page)
                transform_sequence = None
            else:
                _, binary, _, perspective_matrix = correct_perspective(
                    page,
                    debug=False,
                    return_matrix=True,
                )
                binary = np.where(binary < 128, 0, 255).astype(np.uint8)
                coord_x, coord_y = np.meshgrid(
                    np.arange(page.shape[1], dtype=np.float32),
                    np.arange(page.shape[0], dtype=np.float32),
                )
                coord_x = cv2.warpAffine(
                    coord_x,
                    perspective_matrix,
                    (binary.shape[1], binary.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=-1,
                )
                coord_y = cv2.warpAffine(
                    coord_y,
                    perspective_matrix,
                    (binary.shape[1], binary.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=-1,
                )
                transform_sequence = {
                    "input_shape": (int(page.shape[0]), int(page.shape[1])),
                    "output_to_input_x": coord_x.astype(np.float32),
                    "output_to_input_y": coord_y.astype(np.float32),
                }

            if self.debug:
                os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)
                debug_page_path = os.path.join(DEBUG_IMAGES_DIR, f'page_{idx:03d}_binary_page.jpg')
                debug_binary_path = os.path.join(
                    DEBUG_IMAGES_DIR,
                    f'page_{idx:03d}_binary_final.jpg',
                )
                cv2.imwrite(debug_page_path, page)
                cv2.imwrite(debug_binary_path, binary)

            # Шаг 3: строим HPP и нормализуем профиль.
            hpp = self._horizontal_projection_profile(binary)
            norm_hpp = self._normalize_hpp(hpp, method='minmax')

            if self.debug:
                self._save_hpp_smoothing_debug(hpp, idx)

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

                ax1.plot(hpp, color='blue')
                ax1.set_title('Горизонтальный проекционный профиль (HPP)')
                ax1.set_xlabel('Номер строки (y)')
                ax1.set_ylabel('Количество белых пикселей')
                ax1.grid(True, linestyle='--', alpha=0.7)

                ax2.plot(norm_hpp, color='green')
                ax2.set_title('Нормализованный HPP (min-max)')
                ax2.set_xlabel('Номер строки (y)')
                ax2.set_ylabel('Нормализованное значение')
                ax2.grid(True, linestyle='--', alpha=0.7)

                plt.tight_layout()
                fig.savefig(os.path.join(DEBUG_IMAGES_DIR, f'page_{idx:03d}_hpp_profile.jpg'))
                plt.close(fig)

            # Шаг 4: выделяем примерные вертикальные области строк по HPP.
            line_regions = self._find_line_regions(
                norm_hpp,
                debug_filename=f'page_{idx:03d}_hpp_bases.jpg',
            )
            if len(line_regions) == 0:
                if return_class_matrix:
                    self._save_full_class_matrix_debug(full_class_matrix, image)
                    return lines_pixels, lines_crops, full_class_matrix
                return []

            # Шаг 4.5: повторный HPP по остаточному изображению после вырезания найденных регионов.
            # line_regions = self._find_line_regions_second_pass(binary, line_regions, idx)

            # Шаг 5: строим энергетическую матрицу для поиска швов.
            energy = self._compute_energy_matrix(binary, line_regions)

            if self.debug:
                energy_debug = cv2.normalize(
                    energy, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                energy_path = os.path.join(DEBUG_IMAGES_DIR, f'page_{idx:03d}_energy_matrix.jpg')
                debug_text_path = os.path.join(DEBUG_IMAGES_DIR, f'page_{idx:03d}_debug.txt')
                cv2.imwrite(energy_path, energy_debug)
                with open(debug_text_path, 'w', encoding='utf-8') as file:
                    file.write(f'Максаимальная энергия {np.max(energy)}\n')

            _, parent = self._compute_min_energy_path_with_parents(energy)
            parent = parent.astype(np.int32)

            # Шаг 6: берем стартовые точки швов между соседними строками.
            start_points = []
            for i in range(len(line_regions) - 1):
                _, end_prev = line_regions[i]
                start_next, _ = line_regions[i+1]
                mid = (end_prev + start_next) // 2
                start_points.append(mid)

            if self.debug:
                with open(debug_text_path, 'a', encoding='utf-8') as file:
                    file.write(f'start_points {len(start_points)}\n')

            # Шаг 7: ищем швы между строками с помощью A*.
            seams = []
            for start_y in start_points:
                if 0 <= start_y < binary.shape[0]:
                    seam = self._find_seam_a_star(energy, start_y)
                    if seam:
                        seams.append(seam)

            if self.debug:
                if len(binary.shape) == 2:
                    vis_seams = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                else:
                    vis_seams = binary.copy()
                for seam in seams:
                    for x in range(1, len(seam)):
                        x_prev, y_prev = x - 1, seam[x - 1]
                        x_curr, y_curr = x, seam[x]
                        if (
                            0 <= y_prev < vis_seams.shape[0] and
                            0 <= x_prev < vis_seams.shape[1] and
                            0 <= y_curr < vis_seams.shape[0] and
                            0 <= x_curr < vis_seams.shape[1]
                        ):
                            cv2.line(
                                vis_seams,
                                (x_prev, y_prev),
                                (x_curr, y_curr),
                                (0, 0, 255),
                                thickness=2,
                            )

                energy_vis = np.zeros((energy.shape[0], energy.shape[1], 3), dtype=np.uint8)
                energy_vis[energy > 4000] = [255, 0, 0]
                vis_seams = cv2.addWeighted(vis_seams, 0.7, energy_vis, 0.3, 0)

                seams_path = os.path.join(DEBUG_IMAGES_DIR, f'page_{idx:03d}_seams_a_star.jpg')
                cv2.imwrite(seams_path, vis_seams)

            if len(seams) == 0:
                ys, xs = np.where(binary == 0)
                text_pixels = set(zip(xs.tolist(), ys.tolist()))
                if return_class_matrix:
                    warped_class_matrix = self._build_class_matrix(binary.shape, [text_pixels])
                    restored_class_matrix = self._restore_class_matrix_to_input(
                        warped_class_matrix,
                        transform_sequence,
                    )
                    if idx < len(page_infos):
                        class_offset = self._paste_page_class_matrix(
                            full_class_matrix,
                            restored_class_matrix,
                            page_infos[idx],
                            class_offset,
                        )
                    self._save_full_class_matrix_debug(full_class_matrix, image)
                    return [text_pixels], [], full_class_matrix
                return [text_pixels]

            # Шаг 8: сортируем найденные швы сверху вниз.
            seams = sorted(seams, key=lambda s: np.mean(s))

            # Шаг 9: добавляем верхнюю и нижнюю границы как виртуальные швы.
            seams_full = (
                [[0] * binary.shape[1]] +
                seams +
                [[binary.shape[0] - 1] * binary.shape[1]]
            )

            line_pixels = []
            line_crops = []

            # Шаг 10: собираем пиксели строк между соседними швами.
            for i in range(len(seams_full) - 1):
                upper = seams_full[i]
                lower = seams_full[i + 1]

                pixels = self._get_line_pixels_between_seams(binary, upper, lower)

                if len(pixels) == 0:
                    continue

                line_pixels.append(pixels)

                H, W = binary.shape
                white_image = np.ones((H, W, 3), dtype=np.uint8) * 255
                for x, y in pixels:
                    white_image[y, x] = (0, 0, 0)

                crop = crop_line_rectangle(white_image, pixels, debug=False, padding=0)
                if crop is None or crop.size == 0:
                    line_pixels.pop()
                    continue
                line_crops.append(crop)

            # Шаг 11: сохраняем прямоугольные изображения строк.
            save_dir = "output/lines"
            os.makedirs(save_dir, exist_ok=True)
            for line_idx, crop in enumerate(line_crops):
                if crop is None or crop.size == 0:
                    continue
                filename = os.path.join(save_dir, f"line_{line_idx:03d}.jpg")
                cv2.imwrite(filename, crop)
            lines_pixels.extend(line_pixels)
            lines_crops.extend(line_crops)
            if return_class_matrix:
                warped_class_matrix = self._build_class_matrix(binary.shape, line_pixels)
                restored_class_matrix = self._restore_class_matrix_to_input(
                    warped_class_matrix,
                    transform_sequence,
                )
                if idx < len(page_infos):
                    class_offset = self._paste_page_class_matrix(
                        full_class_matrix,
                        restored_class_matrix,
                        page_infos[idx],
                        class_offset,
                    )
                self._save_class_matrix_debug(warped_class_matrix, idx, 'class_matrix_warped')
                self._save_class_matrix_debug(restored_class_matrix, idx, 'class_matrix_restored')
    
            # if self.debug:
            #     with open(debug_text_path, 'a', encoding='utf-8') as file:
            #         file.write(f'Количество задетекшеных строк в странице {idx} {len(line_pixels)}\n')

            if self.debug:
                if len(binary.shape) == 2:
                    vis_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                else:
                    vis_image = binary.copy()

                random.seed(42)
                colors = []
                for _ in range(len(line_pixels)):
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    colors.append(color)

                for i, pixels in enumerate(line_pixels):
                    color = colors[i]
                    for (x, y) in pixels:
                        if 0 <= y < vis_image.shape[0] and 0 <= x < vis_image.shape[1]:
                            vis_image[y, x] = color

                cv2.imwrite(os.path.join(DEBUG_IMAGES_DIR, f'main_segmented_lines_{idx}.jpg'), vis_image)
                            

        if return_class_matrix:
            self._save_full_class_matrix_debug(full_class_matrix, image)
            return lines_pixels, lines_crops, full_class_matrix
        return lines_pixels, lines_crops


if __name__ == "__main__":
    image_path = '/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code/datasets/school_notebooks_RU/images_base/1_15.JPG'
    lineSegmentation = LineSegmentation(use_warp_binary_by_local_angles = True, use_bijection_warp=False)

    _, _ = lineSegmentation.segment_lines(image_path=image_path, return_class_matrix=True)
