import numpy as np
import cv2
from typing import List, Tuple, Optional
from datasets import load_dataset
import cv2
from school_notebooks_RU import CocoMaskGenerator
import random
from typing import List, Tuple, Optional
import numpy as np
from u_net_binarization import binarize_image
import math
import heapq
from typing import Tuple, List
import os

class LineSegmentation:
    """
    Класс для сегментации строк в рукописных документах.
    Реализует комбинацию метода горизонтальной проекции (HPP) и seam carving с динамическим программированием,
    как описано в статье "Seam carving, horizontal projection profile and contour tracing for line and word
    segmentation of language independent handwritten documents" (Das & Panda, 2023).
    """

    def __init__(self, threshold: float = 0.3, gaussian_sigma: float = 1.0):
        """
        Инициализация параметров сегментации.

        Args:
            threshold (float): Порог для определения текстовых строк по нормализованной HPP.
                              Значения выше порога считаются принадлежащими тексту.
            gaussian_sigma (float): Сигма для гауссова сглаживания HPP (если применяется).
        """
        self.threshold = threshold
        self.gaussian_sigma = gaussian_sigma

    def _binarize(self, image: np.ndarray, method: str = 'otsu') -> np.ndarray:
        """
        Бинаризация входного изображения.

        Args:
            image (np.ndarray): Входное изображение (RGB, или бинарное).
            method (str): Метод бинаризации. Поддерживаются:
                          'simple' - простая пороговая обработка (127),
                          'otsu' - метод Оцу,
                          'adaptive' - адаптивная пороговая обработка.
                          По умолчанию 'otsu'.

        Returns:
            np.ndarray: Бинарное изображение, где текст = 0, фон = 255.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if method == 'simple':
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        elif method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        elif method == 'u_net':
            binary = binarize_image(image)
        else:
            raise ValueError(f"Неизвестный метод бинаризации: {method}")

        return binary

    def _horizontal_projection_profile(self, binary: np.ndarray) -> np.ndarray:
        """
        Вычисление горизонтального проекционного профиля (HPP).

        Args:
            binary (np.ndarray): Бинарное изображение (текст = 0, фон = 255).

        Returns:
            np.ndarray: Одномерный массив, где i-й элемент — сумма белых пикселей в i-й строке.
        """
        # Суммируем по столбцам (ось=1), делим на 255, чтобы получить количество белых пикселей
        return np.sum(binary == 0, axis=1).astype(np.float32)

    def _normalize_hpp(self, hpp: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Нормализация HPP.

        Args:
            hpp (np.ndarray): Входной профиль.
            method (str): Метод нормализации. 'minmax' - масштабирование в [0,1].
                           'gaussian' - сглаживание гауссовым фильтром (не реализовано).

        Returns:
            np.ndarray: Нормализованный профиль.
        """
        if method == 'minmax':
            min_val = np.min(hpp)
            max_val = np.max(hpp)
            if max_val - min_val == 0:
                return np.zeros_like(hpp)
            return (hpp - min_val) / (max_val - min_val)
        elif method == 'gaussian':
            # Простое сглаживание гауссовым ядром
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(hpp, sigma=self.gaussian_sigma)
            min_val = np.min(smoothed)
            max_val = np.max(smoothed)
            if max_val - min_val == 0:
                return np.zeros_like(smoothed)
            return (smoothed - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Неизвестный метод нормализации: {method}")

    def _find_line_regions(self, normalized_hpp: np.ndarray) -> List[Tuple[int, int]]:
        """
        Поиск текстовых областей строк по нормализованному HPP.

        Args:
            normalized_hpp (np.ndarray): Нормализованный HPP (значения в [0,1]).

        Returns:
            List[Tuple[int, int]]: Список кортежей (start_row, end_row) для каждой текстовой строки.
        """
        # Определяем строки, где HPP превышает порог
        text_rows = normalized_hpp > self.threshold
        if not np.any(text_rows):
            return []

        # Объединяем соседние строки в регионы
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

        # Объединяем очень близкие регионы (если расстояние между ними меньше 3 строк)
        merged = []
        for reg in regions:
            if not merged:
                merged.append(reg)
            else:
                prev_start, prev_end = merged[-1]
                curr_start, curr_end = reg
                if curr_start - prev_end <= 3:  # близкие строки
                    merged[-1] = (prev_start, curr_end)
                else:
                    merged.append(reg)
        return regions

    def _compute_energy_matrix(self, binary: np.ndarray, line_regions: List[Tuple[int, int]]) -> np.ndarray:
        """
        Улучшенная энергия:
        - Очень высокая (но НЕ inf) в местах, где есть текст.
        - Дополнительный штраф в HPP-регионах (чтобы шов предпочитал проходить между строк).
        - Небольшая градиентная энергия, чтобы шов огибал сильные края букв.
        """
        H, W = binary.shape
        # Базовая энергия: чем больше текста — тем выше энергия
        energy = (255 - binary).astype(np.float32) * 10.0          # текст = 2550, фон = 0

        gray = binary if binary.dtype == np.uint8 else binary.astype(np.uint8)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        energy += 0.5 * gradient

        for start, end in line_regions:
            start = max(0, start - 2)
            end   = min(H - 1, end + 5)
            energy[start:end+1, :] += 5000.0 

        energy = np.nan_to_num(energy, nan=0.0, posinf=1e9, neginf=0.0)
        
        return energy

    def _compute_horizontal_min_energy_path_matrix(self, energy: np.ndarray) -> np.ndarray:
        """
        Вычисление матрицы минимальных энергий для горизонтальных путей (слева направо).
        Алгоритм 2 из статьи.

        Args:
            energy (np.ndarray): Энергетическая матрица (H x W).

        Returns:
            np.ndarray: Матрица минимальных энергий (H x W), где M[x, y] — минимальная энергия пути
                        от левого края до пикселя (x, y). Обратите внимание: индексы в numpy — (row, col),
                        но в алгоритме ось X — столбцы, Y — строки. Для удобства будем использовать
                        матрицу, где строки — y, столбцы — x.
        """
        H, W = energy.shape
        # Инициализируем матрицу минимальных энергий копией первой колонки
        min_energy = np.zeros((H, W), dtype=np.float32)
        min_energy[:, 0] = energy[:, 0]

        # Проходим по столбцам слева направо
        for x in range(1, W):
            for y in range(H):
                # Рассматриваем возможные переходы из предыдущего столбца (x-1)
                candidates = []
                # Соседние строки: y-1, y, y+1 (с проверкой границ)
                for dy in (-1, 0, 1):
                    ny = y + dy
                    if 0 <= ny < H:
                        candidates.append(min_energy[ny, x-1])
                # Минимальная энергия пути до (x, y) = energy[y, x] + min(candidates)
                min_energy[y, x] = energy[y, x] + min(candidates)

        return min_energy

    def _extract_seam(self, min_energy_matrix: np.ndarray, start_y: int) -> List[int]:
        """
        Восстановление горизонтального шва (пути) по матрице минимальных энергий.
        Шов идёт от левого края (x=0) до правого края (x=W-1), проходя через start_y на левом крае.

        Args:
            min_energy_matrix (np.ndarray): Матрица минимальных энергий, полученная из compute_horizontal_min_energy_path_matrix.
            start_y (int): Координата y (строка) на левом крае (x=0), с которой начинается шов.

        Returns:
            List[int]: Список координат y для каждого столбца x от 0 до W-1.
        """
        H, W = min_energy_matrix.shape
        seam = [0] * W
        # Начинаем с заданной строки в первом столбце
        y = start_y
        seam[0] = y

        # Идём справа налево? Нет, идём слева направо, восстанавливая путь.
        # Для каждого следующего столбца выбираем соседнюю строку с минимальной энергией пути.
        # Восстанавливаем, зная, что в min_energy_matrix[y, x] хранится энергия пути,
        # но для восстановления пути нам нужно на каждом шаге выбирать переход с минимальной энергией.
        # Более надёжно: идём справа налево, выбирая предшественника.
        # Но проще: идём слева направо, на каждом шаге выбирая кандидата с наименьшим значением min_energy в следующем столбце.
        # Однако это может дать не совсем тот путь, который был вычислен, но в целом корректно.
        # Поскольку мы знаем start_y, можем идти слева направо, выбирая минимальный по энергии переход.
        # Альтернативно: можно восстановить, двигаясь справа налево, выбирая из предыдущего столбца пиксель,
        # который дал минимум для текущего. Для этого нужно хранить матрицу предков.
        # Реализуем с сохранением предков для большей точности.
        # Создадим матрицу предков при вычислении min_energy.
        # Переделаем compute_horizontal_min_energy_path_matrix, чтобы возвращать и предков.
        # Для простоты здесь пересоздадим путь, используя жадный выбор слева направо, но с учётом локального минимума.
        # Это будет приблизительно, но в целом работает.
        # Более строго: вызовем функцию, которая возвращает предков.
        # Сделаем отдельный метод, который возвращает матрицу предков.
        # Пока реализуем простой вариант.

        # Простой жадный путь (может немного отличаться от оптимального)
        y_cur = start_y
        for x in range(1, W):
            # Определяем возможные следующие y
            options = []
            for dy in (-1, 0, 1):
                ny = y_cur + dy
                if 0 <= ny < H:
                    options.append((ny, min_energy_matrix[ny, x]))
            # Выбираем кандидата с минимальной энергией
            best_ny, _ = min(options, key=lambda t: t[1])
            y_cur = best_ny
            seam[x] = y_cur
        return seam

    def _compute_min_energy_path_with_parents(self, energy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Находит минимальный энергетический путь (шов) с помощью A* для каждого start_y.
        Возвращаем min_energy и parent для совместимости с остальным кодом.
        """
        H, W = energy.shape
        
        # Для совместимости с твоим кодом создаём заглушки
        min_energy = np.full((H, W), np.inf, dtype=np.float32)
        parent = np.full((H, W), -1, dtype=np.int32)   # parent[y, x] = предыдущая y
        
        # Мы будем вычислять пути позже в segment_lines
        return min_energy, parent


    def _find_seam_a_star(self, energy: np.ndarray, start_y: int) -> List[int]:
        """
        Находит один горизонтальный шов с помощью A* от левого края (x=0, y=start_y)
        до правого края (x=W-1).
        """
        H, W = energy.shape
        if not (0 <= start_y < H):
            return []

        # Направления: можно идти в предыдущую, текущую или следующую строку
        directions = [-1, 0, 1]

        # Приоритетная очередь: (f_score, g_score, y, x, path) — но для экономии памяти храним только parent
        came_from = {}          # (y, x) -> (prev_y, prev_x)
        g_score = {}            # стоимость от старта
        f_score = {}            # g + эвристика

        start = (start_y, 0)
        goal_x = W - 1

        g_score[start] = 0
        f_score[start] = 0  # эвристика на старте = 0

        open_set = []       # приоритетная очередь
        heapq.heappush(open_set, (0, start_y, 0))  # (f, y, x)

        visited = set()

        while open_set:
            _, y, x = heapq.heappop(open_set)
            current = (y, x)

            if current in visited:
                continue
            visited.add(current)

            if x == goal_x:
                # Восстанавливаем путь
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
                        
                        # Эвристика: оставшееся расстояние по x + небольшая вертикальная компонента
                        h = (W - 1 - nx) * 1.0   # минимальная возможная стоимость (если энергия >=0)
                        f = tentative_g + h
                        
                        f_score[neighbor] = f
                        heapq.heappush(open_set, (f, ny, nx))

        # Если пути не найдено (крайне редко)
        print(f"Warning: A* не нашёл путь для start_y={start_y}")
        return list(range(start_y, start_y)) * W  # fallback


    def _get_line_pixels_between_seams(self, binary: np.ndarray, 
                                   upper_seam: List[int], 
                                   lower_seam: List[int]) -> set:
        """Собирает ровно пиксели текста между двумя швами (по столбцам)."""
        H, W = binary.shape
        pixels = set()
        for x in range(W):
            y_top = min(upper_seam[x], lower_seam[x])
            y_bot = max(upper_seam[x], lower_seam[x])
            # Берём только текстовые пиксели строго между швами
            for y in range(y_top + 1, y_bot):
                if 0 <= y < H and binary[y, x] == 0:
                    pixels.add((x, y))
        return pixels


    def crop_line_rectangle(self, image: np.ndarray, line_pixels: set, padding: int = 20) -> np.ndarray:
        """
        Строит минимальный повёрнутый прямоугольник ТОЛЬКО по текстовым пикселям строки,
        поворачивает изображение так, чтобы строка стала горизонтальной,
        и возвращает чистое выпрямленное изображение строки.
        """
        if not line_pixels:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        # Точки только текстовых пикселей
        points = np.array([[x, y] for x, y in line_pixels], dtype=np.float32)

        if len(points) < 5:  # слишком мало точек
            # fallback — обычный bounding box
            xs, ys = zip(*line_pixels)
            min_y, max_y = max(0, min(ys) - padding), min(image.shape[0]-1, max(ys) + padding)
            min_x, max_x = max(0, min(xs) - padding), min(image.shape[1]-1, max(xs) + padding)
            return image[min_y:max_y+1, min_x:max_x+1].copy()

        # Минимальный повёрнутый прямоугольник по текстовым пикселям
        rect = cv2.minAreaRect(points)
        (center_x, center_y), (width, height), angle = rect

        # Ориентируем так, чтобы строка была горизонтальной
        if width < height:
            angle += 90
            width, height = height, width

        # Матрица поворота
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

        # Новые размеры холста
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((image.shape[1] * cos) + (image.shape[0] * sin))
        new_h = int((image.shape[1] * sin) + (image.shape[0] * cos))

        rotation_matrix[0, 2] += new_w / 2 - center_x
        rotation_matrix[1, 2] += new_h / 2 - center_y

        # Поворачиваем исходное изображение
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))  # белый фон

        # Вычисляем координаты вырезания
        crop_x = int(new_w / 2 - width / 2 - padding)
        crop_y = int(new_h / 2 - height / 2 - padding)
        crop_w = int(width + 2 * padding)
        crop_h = int(height + 2 * padding)

        # Защита границ
        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)
        crop_w = min(crop_w, new_w - crop_x)
        crop_h = min(crop_h, new_h - crop_y)

        straightened = rotated[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        return straightened
    def segment_lines(self, image: np.ndarray, binarization_method: str = 'otsu', debug: bool = True) -> List[set]:
        """
        Сегментация строк в рукописном документе. Возвращает список множеств координат пикселей,
        принадлежащих каждой строке.

        Args:
            image (np.ndarray): Входное цветное или серое изображение.
            binarization_method (str): Метод бинаризации ('otsu', 'simple', 'adaptive').

        Returns:
            List[set]: Список множеств, каждое множество содержит координаты (x, y) текстовых пикселей
                    для одной строки. Координаты: x — номер столбца, y — номер строки.
        """
        # Шаг 1: Бинаризация
        binary = self._binarize(image, method=binarization_method)

        if debug:
            cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Window', 800, 600)
            cv2.imshow('Window', binary)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Шаг 2: HPP и нормализация
        hpp = self._horizontal_projection_profile(binary)
        norm_hpp = self._normalize_hpp(hpp, method='minmax')

        import matplotlib.pyplot as plt


        if debug:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

            # График исходного HPP
            ax1.plot(hpp, color='blue')
            ax1.set_title('Горизонтальный проекционный профиль (HPP)')
            ax1.set_xlabel('Номер строки (y)')
            ax1.set_ylabel('Количество белых пикселей')
            ax1.grid(True, linestyle='--', alpha=0.7)

            # График нормализованного HPP
            ax2.plot(norm_hpp, color='green')
            ax2.set_title('Нормализованный HPP (min–max)')
            ax2.set_xlabel('Номер строки (y)')
            ax2.set_ylabel('Нормализованное значение')
            ax2.grid(True, linestyle='--', alpha=0.7)

            # Отображаем оба графика
            plt.tight_layout()
            plt.show()

        # Шаг 3: Поиск текстовых регионов (приблизительное расположение строк)
        line_regions = self._find_line_regions(norm_hpp)
        if len(line_regions) == 0:
            return []  # Нет строк
        
        if debug:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for start, end in line_regions:
                mask[start:end+1, :] = 255
            cv2.namedWindow('Line Regions Mask', cv2.WINDOW_NORMAL)
            cv2.imshow('Line Regions Mask', mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Шаг 4: Энергетическая матрица с усилением энергии в текстовых областях
        energy = self._compute_energy_matrix(binary, line_regions)

        if debug:
            cv2.namedWindow('energy matrix', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('energy matrix', 800, 600)
            cv2.imshow('energy matrix', energy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print('Максаимальная энергия', np.max(energy))

        # Шаг 5: Матрицы минимальных энергий и предков для точного восстановления швов
        _, parent = self._compute_min_energy_path_with_parents(energy)
        parent = parent.astype(np.int32)

        # Шаг 6: Определение начальных точек для швов (середин между строками)
        start_points = []
        for i in range(len(line_regions) - 1):
            _, end_prev = line_regions[i]
            start_next, _ = line_regions[i+1]
            mid = (end_prev + start_next) // 2
            start_points.append(mid)

        if debug:
            print('start_points', len(start_points))

       # Шаг 7: Извлечение всех швов с помощью A*
        seams = []
        for start_y in start_points:
            if 0 <= start_y < binary.shape[0]:
                seam = self._find_seam_a_star(energy, start_y)
                if seam:
                    seams.append(seam)

        if debug:
            vis_seams = image.copy()
            for seam in seams:
                for x in range(1, len(seam)):
                    x_prev, y_prev = x - 1, seam[x - 1]
                    x_curr, y_curr = x, seam[x]
                    if (0 <= y_prev < vis_seams.shape[0] and 0 <= x_prev < vis_seams.shape[1] and
                        0 <= y_curr < vis_seams.shape[0] and 0 <= x_curr < vis_seams.shape[1]):
                        cv2.line(vis_seams, (x_prev, y_prev), (x_curr, y_curr), (0, 0, 255), thickness=2)

            # Синие области — высокая энергия
            energy_vis = np.zeros((energy.shape[0], energy.shape[1], 3), dtype=np.uint8)
            energy_vis[energy > 4000] = [255, 0, 0]   # BGR синий
            vis_seams = cv2.addWeighted(vis_seams, 0.7, energy_vis, 0.3, 0)

            cv2.namedWindow('Seams A*', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Seams A*', 1200, 800)
            cv2.imshow('Seams A*', vis_seams)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        if len(seams) == 0:
            text_pixels = set(zip(*np.where(binary == 0)))
            return [text_pixels]

        # Сортируем швы сверху вниз по средней высоте
        seams = sorted(seams, key=lambda s: np.mean(s))

        # Добавляем «виртуальные» швы сверху и снизу изображения
        seams_full = [[0] * binary.shape[1]] + seams + [[binary.shape[0] - 1] * binary.shape[1]]

        line_pixels = []
        line_crops = []   # новые прямоугольные изображения строк (по желанию)

        for i in range(len(seams_full) - 1):
            upper = seams_full[i]
            lower = seams_full[i + 1]
            
            pixels = self._get_line_pixels_between_seams(binary, upper, lower)
            line_pixels.append(pixels)
            
            H, W = binary.shape
            white_image = np.ones((H, W, 3), dtype=np.uint8) * 255
            for x, y in pixels:
                white_image[y, x] = (0, 0, 0)        # чёрный текст

            # Вырезаем и выпрямляем
            crop = self.crop_line_rectangle(white_image, pixels, padding=25)
            line_crops.append(crop)

        save_dir = "input/lines"
        os.makedirs(save_dir, exist_ok=True)
        for idx, crop in enumerate(line_crops):
            filename = os.path.join(save_dir, f"line_{idx:03d}.jpg")
            cv2.imwrite(filename, crop)

        if debug:
            print('Количество задетекшеных строк', len(line_pixels))

        return line_pixels   # если хочешь — можешь возвращать (line_pixels, line_crops)
    


URLS = {
    "images": "/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/library/datasets/school_notebooks_RU/images_base",
    "train_data": "datasets/school_notebooks_RU/annotations_train.json",
    "test_data": "datasets/school_notebooks_RU/annotations_test.json",
    "val_data": "datasets/school_notebooks_RU/annotations_val.json"
}

if __name__ == "__main__":
    lineSegmentation = LineSegmentation()

    # Чтение
    image = cv2.imread(URLS['images'] + '/2013.jpg')  # BGR порядок каналов

    # Отображение (в отдельном окне)
    cv2.namedWindow('Window', cv2.WINDOW_NORMAL)

    # (Необязательно) Устанавливаем желаемый размер окна, например 800x600
    cv2.resizeWindow('Window', 800, 600)
    cv2.imshow('Window', image)
    cv2.waitKey(0)                # ждём нажатия клавиши
    cv2.destroyAllWindows()

    line_images = lineSegmentation.segment_lines(image.copy(), binarization_method='u_net')
    # После того как line_images получен:
    # line_images = seg.segment_lines(image)

    # Создаём копию изображения для визуализации (BGR)
    vis_image = image.copy()

    # Генерируем случайные цвета для каждой строки (устойчивые при каждом запуске)
    random.seed(42)  # для воспроизводимости
    colors = []
    for _ in range(len(line_images)):
        # Случайный BGR цвет (от 0 до 255)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)

    # Накладываем маску: для каждого пикселя каждой строки
    for idx, pixels in enumerate(line_images):
        color = colors[idx]
        #print('Цвет первого региона ', color)
        for (x, y) in pixels:
            # Проверяем границы (на случай, если координаты выходят за пределы)
            if 0 <= y < vis_image.shape[0] and 0 <= x < vis_image.shape[1]:
                vis_image[y, x] = color  # закрашиваем пиксель

    # Отображение результата
    cv2.namedWindow('Segmented lines', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Segmented lines', 800, 600)
    cv2.imshow('Segmented lines', vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()