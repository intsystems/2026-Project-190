import heapq
import os
import random
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from post_processing import crop_line_rectangle
from processing import extract_pages_with_yolo, correct_perspective
from scipy.signal import find_peaks, peak_widths


DEBUG_IMAGES_DIR = "debug_images"
ADDITIONAL_LINE_REGION_BLUR_KERNEL = (11, 11)


class LineSegmentation:
    """
    Сегментирует строки в рукописных документах методом HPP и энергетических швов.
    """

    def __init__(self, threshold: float = 0.4, gaussian_sigma: float = 1.0, debug: bool = True):
        """
        Короткое описание:
            задает параметры сегментации строк и режим сохранения отладки.
        Вход:
            threshold: float -- порог строки по нормализованному HPP.
            gaussian_sigma: float -- сигма гауссова сглаживания HPP.
            debug: bool -- сохранять отладочные файлы в debug_images.
        Выход:
            None
        """
        self.threshold = threshold
        self.gaussian_sigma = gaussian_sigma
        self.debug = debug

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
        text_rows = normalized_hpp > self.threshold
        # text_rows = np.zeros_like(normalized_hpp, dtype=bool)
        # peaks, properties = find_peaks(
        #     normalized_hpp,
        #     height=self.threshold,
        #     distance=5,
        #     prominence=0.05,
        # )
        # widths, width_heights, left_ips, right_ips = peak_widths(
        #     normalized_hpp,
        #     peaks,
        #     rel_height=0.5,
        # )
        # for peak, left_ip, right_ip in zip(peaks, left_ips, right_ips):
        #     start = max(0, int(np.floor(left_ip)))
        #     end = min(len(text_rows) - 1, int(np.ceil(right_ip)))
        #     text_rows[start:end+1] = True
        #     text_rows[peak] = True

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

        merged = []
        for reg in regions:
            if not merged:
                merged.append(reg)
            else:
                prev_start, prev_end = merged[-1]
                curr_start, curr_end = reg
                if curr_start - prev_end <= 3:
                    merged[-1] = (prev_start, curr_end)
                else:
                    merged.append(reg)
        return regions

    def _find_line_regions_additional(self,
                                      binary: np.ndarray,
                                      line_regions: List[Tuple[int, int]],
                                      page_idx: int) -> List[Tuple[int, int]]:
        """
        Короткое описание:
            повторно ищет строки после удаления уже найденных HPP-регионов.
        Вход:
            binary: np.ndarray -- бинарное изображение, где текст равен 0, фон равен 255.
            line_regions: List[Tuple[int, int]] -- уже найденные области строк.
            page_idx: int -- номер страницы для сохранения отладочных файлов.
        Выход:
            List[Tuple[int, int]] -- объединенный список найденных областей строк.
        """
        additional_binary = binary.copy()
        for start, end in line_regions:
            additional_binary[start:end+1, :] = 255

        blurred_binary = cv2.medianBlur(
            additional_binary,
            ADDITIONAL_LINE_REGION_BLUR_KERNEL[0],
        )
        additional_hpp = self._horizontal_projection_profile(blurred_binary)
        additional_norm_hpp = self._normalize_hpp(additional_hpp, method='minmax')
        additional_regions = self._find_line_regions(
            additional_norm_hpp,
            debug_filename=f'page_{page_idx:03d}_additional_hpp_bases.jpg',
        )
        line_regions = sorted(line_regions + additional_regions, key=lambda region: region[0])

        if self.debug:
            additional_mask = np.zeros(binary.shape[:2], dtype=np.uint8)
            for start, end in additional_regions:
                additional_mask[start:end+1, :] = 255

            cv2.imwrite(
                os.path.join(DEBUG_IMAGES_DIR, f'page_{page_idx:03d}_additional_binary.jpg'),
                additional_binary,
            )
            cv2.imwrite(
                os.path.join(DEBUG_IMAGES_DIR, f'page_{page_idx:03d}_additional_blur.jpg'),
                blurred_binary,
            )
            cv2.imwrite(
                os.path.join(DEBUG_IMAGES_DIR, f'page_{page_idx:03d}_additional_regions_mask.jpg'),
                additional_mask,
            )

        return line_regions

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
            находит горизонтальный шов алгоритмом A* от левого до правого края.
        Вход:
            energy: np.ndarray -- энергетическая матрица размера H x W.
            start_y: int -- строка старта шва на левом крае изображения.
        Выход:
            List[int] -- координаты y шва для каждого столбца x.
        """
        H, W = energy.shape
        if not (0 <= start_y < H):
            return []

        directions = [-1, 0, 1]

        came_from = {}
        g_score = {}
        f_score = {}

        start = (start_y, 0)
        goal_x = W - 1

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
            file.write(f"Warning: A* не нашел путь для start_y={start_y}\n")
        return list(range(start_y, start_y)) * W

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


    def segment_lines(self, image_path: str) -> List[set]:
        """
        Короткое описание:
            сегментирует строки на страницах рукописного документа.
        Вход:
            image_path: str -- путь к изображению документа.
        Выход:
            List[set] -- список множеств координат (x, y) текстовых пикселей строк.
        """
        if self.debug:
            image = cv2.imread(image_path)
            os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(DEBUG_IMAGES_DIR, 'main_input.jpg'), image)

        # Шаг 1: находим страницы тетради и получаем бинарные изображения страниц.
        pages, binary_pages = extract_pages_with_yolo(
            image_path=image_path,
            model_path='models/yolo_detect_notebook/yolo_detect_notebook_1_(1-architecture).pt',
            output_dir=DEBUG_IMAGES_DIR,
            conf_threshold=0.8,
            return_binary=True
        )
        # Инициализация списков для хранения пикселей строк и их обрезков
        lines_pixels = []
        lines_crops = []
        for idx, page in enumerate(binary_pages):
            # Шаг 2: исправляем перспективу текущей страницы.
            corrected_page, binary, _ = correct_perspective(
                page,
                debug=self.debug,
                debug_output_dir=os.path.join(DEBUG_IMAGES_DIR, f'page_{idx:03d}_perspective'),
            )

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
                return []

            # Шаг 4.5: повторно ищем строки после удаления уже найденных регионов.
            # line_regions.extend(self._find_line_regions_additional(binary, line_regions, idx))

            # if self.debug:
            #     mask = np.zeros(binary.shape[:2], dtype=np.uint8)
            #     for start, end in line_regions:
            #         mask[start:end+1, :] = 255
            #     mask_path = os.path.join(DEBUG_IMAGES_DIR, f'page_{idx:03d}_line_regions_mask.jpg')
            #     cv2.imwrite(mask_path, mask)

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
                text_pixels = set(zip(*np.where(binary == 0)))
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
                line_crops.append(crop)

            # Шаг 11: сохраняем прямоугольные изображения строк.
            save_dir = "output/lines"
            os.makedirs(save_dir, exist_ok=True)
            for line_idx, crop in enumerate(line_crops):
                filename = os.path.join(save_dir, f"line_{line_idx:03d}.jpg")
                cv2.imwrite(filename, crop)
            lines_pixels.extend(line_pixels)
            lines_crops.extend(line_crops)
    
            # if self.debug:
            #     with open(debug_text_path, 'a', encoding='utf-8') as file:
            #         file.write(f'Количество задетекшеных строк в странице {idx} {len(line_pixels)}\n')

            if self.debug:
                if len(corrected_page.shape) == 2:
                    vis_image = cv2.cvtColor(corrected_page, cv2.COLOR_GRAY2BGR)
                else:
                    vis_image = corrected_page.copy()

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
                            

        return lines_pixels, lines_crops


if __name__ == "__main__":
    image_path = '/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code/datasets/HWR200/hw_dataset/34/reuse12/ФотоТемное/3.JPG'
    lineSegmentation = LineSegmentation()

    _, _ = lineSegmentation.segment_lines(image_path=image_path)
