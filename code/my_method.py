import heapq
import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from post_processing import crop_line_rectangle
from processing import correct_perspective, extract_pages_with_yolo


# Папка для всех debug-артефактов метода:
# карты углов, графики HPP, energy и швы.
# На качество сегментации не влияет, только на удобство анализа.
DEBUG_IMAGES_DIR = "debug_images"

# Путь к модели YOLO, которая выделяет страницы тетради.
# Если модель слабая, все дальнейшие шаги метода будут работать по плохим page-crop.
YOLO_NOTEBOOK_MODEL_PATH = "models/yolo_detect_notebook/yolo_detect_notebook_1_(1-architecture).pt"

# Число ячеек сетки по вертикали для локальной оценки угла.
# Больше значение: выше локальная точность угла, но выше шум и вычислительная цена.
# Меньше значение: стабильнее, но хуже ловит локальную кривизну.
GRID_ROWS = 6

# Число ячеек сетки по горизонтали для локальной оценки угла.
# Логика такая же, как у GRID_ROWS: это баланс между детализацией и устойчивостью.
GRID_COLS = 8

# Минимум черных пикселей в ячейке, чтобы считать ее информативной для оценки угла.
# Нужен, чтобы не оценивать угол по почти пустым зонам фона.
# Если увеличить, станет меньше шумных углов; если переборщить, много ячеек уйдет в глобальный угол.
MIN_CELL_TEXT_PIXELS = 32

# Максимальное отклонение локального угла от глобального (в градусах).
# Это ограничитель от выбросов correct_perspective на маленьких фрагментах.
# Увеличение дает гибкость на сильно деформированных листах, но повышает риск ложных углов.
MAX_LOCAL_ANGLE_DELTA = 12.0

# Порог "практически нулевого" угла (в градусах).
# Если |угол| <= ANGLE_EPSILON, вместо Брезенхема берем обычную горизонтальную линию.
# Нужен для стабильности и скорости, чтобы не гонять Брезенхема там, где наклон почти отсутствует.
ANGLE_EPSILON = 0.25

# Порог нормализованного HPP для выделения строковых регионов.
# Выше порог: только выраженные текстовые полосы, риск пропуска слабых строк.
# Ниже порог: выше полнота, но больше ложных регионов.
HPP_THRESHOLD = 0.38

# Максимальный вертикальный разрыв между соседними HPP-регионами для их склейки.
# Нужен, чтобы не дробить одну строку на несколько частей из-за локальных провалов профиля.
LINE_REGION_MERGE_GAP = 5

# Делитель средней толщины строки для удаления подозрительно тонких регионов.
# Если высота региона меньше average_height / LINE_REGION_MIN_AVG_DIVISOR,
# считаем его ложным коротким шумом и не используем дальше.
LINE_REGION_MIN_AVG_DIVISOR = 3.0

# Вес штрафа текстовых пикселей в energy.
# Чем больше значение, тем сильнее A* избегает прохода через черный текст.
ENERGY_TEXT_WEIGHT = 8.0

# Вес штрафа градиента в energy.
# Добавляет чувствительность к резким переходам яркости, чтобы швы не резали символы по краю.
# Обычно ниже, чем ENERGY_TEXT_WEIGHT, чтобы главным оставался текстовый штраф.
ENERGY_GRADIENT_WEIGHT = 0.35

# Вес штрафа от angle-aware HPP трасс в energy.
# Это главный механизм привязки energy к наклонным линиям проекционного профиля.
# Больше значение сильнее удерживает шов между строками, но может переусилить локальные ошибки угла.
ENERGY_HPP_WEIGHT = 120.0

# Дополнительный коэффициент размазывания HPP-штрафа на соседние строки y-1 и y+1.
# Нужен для гладкости energy, чтобы A* не дергался между почти равными соседними траекториями.
ENERGY_HPP_NEIGHBOR_WEIGHT = 0.35

# Усиление energy внутри зон, уже классифицированных как текстовые регионы.
# Задача: не дать шву идти по телу строки, принуждая его искать межстрочное пространство.
# Также применяется к локальным HPP-строкам внутри ячейки, если они выше порога self.threshold.
# Это сохраняет короткие строки, которые видны локально, но теряются в глобальном HPP.
ENERGY_LINE_REGION_BOOST = 3500.0

# Отступ вверх от текстового региона, где тоже добавляется ENERGY_LINE_REGION_BOOST.
# Помогает защитить верхние выносные элементы символов.
ENERGY_LINE_REGION_MARGIN_TOP = 5

# Отступ вниз от текстового региона, где тоже добавляется ENERGY_LINE_REGION_BOOST.
# Обычно берется чуть больше верхнего, так как нижние выносные элементы и хвосты букв длиннее.
ENERGY_LINE_REGION_MARGIN_BOTTOM = 5

# Штраф за вертикальное смещение шага A* (dy = +/-1).
# Увеличение делает швы более гладкими и горизонтальными.
# Уменьшение позволяет шву агрессивнее обходить локальные препятствия.
A_STAR_DY_PENALTY = 4.0


class LineSegmentation:
    """
    Подробное описание:
        Сегментирует рукописные строки по схеме:
        1) коррекция перспективы страницы,
        2) оценка локального угла в ячейках сетки,
        3) angle-aware горизонтальный профиль проекции,
        4) построение энергетическои матрицы,
        5) поиск межстрочных швов алгоритмом A*.
    """

    def __init__(
        self,
        threshold: float = HPP_THRESHOLD,
        grid_rows: int = GRID_ROWS,
        grid_cols: int = GRID_COLS,
        debug: bool = True,
        use_tqdm: bool = True,
    ):
        """
        Короткое описание:
            Инициализирует параметры метода сегментации.
        Вход:
            threshold: float -- порог выделения строк по нормализованному HPP.
            grid_rows: int -- число ячеек сетки по оси Y.
            grid_cols: int -- число ячеек сетки по оси X.
            debug: bool -- флаг сохранения отладочных материалов.
            use_tqdm: bool -- флаг отображения прогресс-баров.
        Выход:
            None
        """
        self.threshold = threshold
        self.grid_rows = max(1, int(grid_rows))
        self.grid_cols = max(1, int(grid_cols))
        self.debug = debug
        self.use_tqdm = use_tqdm

    def _ensure_binary(self, binary: np.ndarray) -> np.ndarray:
        """
        Короткое описание:
            Нормализует изображение к бинарному формату uint8 с пикселями 0 и 255.
        Вход:
            binary: np.ndarray -- входное изображение страницы.
        Выход:
            np.ndarray -- нормализованное бинарное изображение.
        """
        # Шаг 1: переводим в grayscale при необходимости.
        if len(binary.shape) == 3:
            gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
        else:
            gray = binary.copy()

        # Шаг 2: приводим тип и жестко бинаризуем.
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        gray = np.where(gray < 128, 0, 255).astype(np.uint8)
        return gray

    def _build_grid(self, height: int, width: int) -> List[Tuple[int, int, int, int, int, int]]:
        """
        Короткое описание:
            Делит изображение на регулярную сетку ячеек.
        Вход:
            height: int -- высота изображения.
            width: int -- ширина изображения.
        Выход:
            List[Tuple[int, int, int, int, int, int]]:
                список ячеек в формате (y0, y1, x0, x1, gy, gx).
        """
        # Шаг 1: вычисляем границы по осям через равные отрезки.
        y_bounds = np.linspace(0, height, self.grid_rows + 1, dtype=np.int32)
        x_bounds = np.linspace(0, width, self.grid_cols + 1, dtype=np.int32)

        # Шаг 2: формируем список валидных ячеек.
        cells: List[Tuple[int, int, int, int, int, int]] = []
        for gy in range(self.grid_rows):
            for gx in range(self.grid_cols):
                y0, y1 = int(y_bounds[gy]), int(y_bounds[gy + 1])
                x0, x1 = int(x_bounds[gx]), int(x_bounds[gx + 1])
                if (y1 - y0) > 1 and (x1 - x0) > 1:
                    cells.append((y0, y1, x0, x1, gy, gx))
        return cells

    def _estimate_local_angle_map(
        self,
        binary: np.ndarray,
        global_angle: float,
        page_idx: int,
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, int, int]]]:
        """
        Короткое описание:
            Оценивает локальныи угол наклона в каждои ячейке сетки.
        Вход:
            binary: np.ndarray -- бинарное изображение страницы.
            global_angle: float -- глобальныи угол после correct_perspective.
            page_idx: int -- индекс страницы для debug-вывода.
        Выход:
            Tuple[np.ndarray, List[Tuple[int, int, int, int, int, int]]]:
                карта углов grid_rows x grid_cols и список ячеек сетки.
        """
        # Шаг 1: строим сетку и выделяем буфер карты углов.
        height, width = binary.shape
        cells = self._build_grid(height, width)
        angle_map = np.full((self.grid_rows, self.grid_cols), float(global_angle), dtype=np.float32)

        # Шаг 2: считаем локальныи угол для каждои ячейки.
        iterator = tqdm(
            cells,
            desc=f"Angles page {page_idx}",
            disable=not self.use_tqdm,
            leave=False,
        )
        for y0, y1, x0, x1, gy, gx in iterator:
            cell = binary[y0:y1, x0:x1]
            text_pixels = int(np.sum(cell == 0))
            if text_pixels < MIN_CELL_TEXT_PIXELS:
                angle_map[gy, gx] = float(global_angle)
                continue

            try:
                _, _, local_angle = correct_perspective(cell, debug=False)
                local_angle = float(local_angle)
            except Exception:
                local_angle = float(global_angle)

            min_allowed = float(global_angle) - MAX_LOCAL_ANGLE_DELTA
            max_allowed = float(global_angle) + MAX_LOCAL_ANGLE_DELTA
            local_angle = float(np.clip(local_angle, min_allowed, max_allowed))
            angle_map[gy, gx] = local_angle

        # Шаг 3: слегка сглаживаем карту углов для пространственнои согласованности.
        if angle_map.shape[0] > 1 and angle_map.shape[1] > 1:
            angle_map = cv2.GaussianBlur(angle_map, (3, 3), 0)

        # Шаг 4: сохраняем debug-материалы.
        if self.debug:
            os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)

            angle_min = float(np.min(angle_map))
            angle_max = float(np.max(angle_map))
            scale = max(angle_max - angle_min, 1e-6)
            angle_norm = ((angle_map - angle_min) / scale * 255.0).astype(np.uint8)
            angle_vis = cv2.applyColorMap(angle_norm, cv2.COLORMAP_JET)
            angle_vis = cv2.resize(
                angle_vis,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imwrite(
                os.path.join(DEBUG_IMAGES_DIR, f"page_{page_idx:03d}_my_method_angle_map.jpg"),
                angle_vis,
            )

            text_path = os.path.join(DEBUG_IMAGES_DIR, f"page_{page_idx:03d}_my_method_angles.txt")
            with open(text_path, "w", encoding="utf-8") as file:
                file.write(f"global_angle={float(global_angle):.6f}\n")
                for gy in range(angle_map.shape[0]):
                    row_values = " ".join(f"{float(v):.4f}" for v in angle_map[gy])
                    file.write(f"row_{gy:02d}: {row_values}\n")

        return angle_map, cells

    def _bresenham_line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> List[Tuple[int, int]]:
        """
        Короткое описание:
            Строит дискретную линию между двумя точками по алгоритму Брезенхема.
        Вход:
            x0: int -- x начальнои точки.
            y0: int -- y начальнои точки.
            x1: int -- x конечнои точки.
            y1: int -- y конечнои точки.
        Выход:
            List[Tuple[int, int]] -- список пикселеи линии в формате (x, y).
        """
        # Шаг 1: инициализируем переменные пошагового трассирования.
        points: List[Tuple[int, int]] = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        # Шаг 2: идем от старта к финишу и копим точки.
        x_cur, y_cur = x0, y0
        while True:
            points.append((x_cur, y_cur))
            if x_cur == x1 and y_cur == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x_cur += sx
            if e2 < dx:
                err += dx
                y_cur += sy
        return points

    def _compute_cell_slanted_hpp(
        self,
        binary: np.ndarray,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        angle_deg: float,
    ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """
        Короткое описание:
            Считает горизонтальныи профиль проекции в ячейке с учетом угла наклона.
        Вход:
            binary: np.ndarray -- бинарное изображение страницы.
            y0: int -- верхняя граница ячейки.
            y1: int -- нижняя граница ячейки.
            x0: int -- левая граница ячейки.
            x1: int -- правая граница ячейки.
            angle_deg: float -- угол строки в градусах для этои ячейки.
        Выход:
            Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
                профиль HPP в ячейке и список трасс линии для каждои локальнои строки.
        """
        # Шаг 1: готовим буферы для профиля и трассировок.
        cell_h = y1 - y0
        cell_w = x1 - x0
        profile = np.zeros(cell_h, dtype=np.float32)
        traces: List[List[Tuple[int, int]]] = [[] for _ in range(cell_h)]

        # Шаг 2: заранее считаем тангенс угла.
        angle_abs = abs(float(angle_deg))
        tan_value = float(np.tan(np.deg2rad(float(angle_deg)))) if angle_abs > ANGLE_EPSILON else 0.0

        # Шаг 3: для каждои строки ячейки считаем text-пиксели вдоль линии.
        for local_y in range(cell_h):
            y_start = y0 + local_y

            if angle_abs <= ANGLE_EPSILON:
                points = [(x, y_start) for x in range(x0, x1)]
            else:
                y_end = int(round(y_start + tan_value * (cell_w - 1)))
                points = self._bresenham_line(x0, y_start, x1 - 1, y_end)

            valid_points: List[Tuple[int, int]] = []
            text_count = 0.0
            for x, y in points:
                if x0 <= x < x1 and y0 <= y < y1:
                    valid_points.append((x, y))
                    if binary[y, x] == 0:
                        text_count += 1.0

            if len(valid_points) == 0:
                y_clip = int(np.clip(y_start, y0, y1 - 1))
                valid_points = [(x0, y_clip)]
                text_count = float(binary[y_clip, x0] == 0)

            profile[local_y] = text_count
            traces[local_y] = valid_points

        return profile, traces

    def _normalize_hpp(self, hpp: np.ndarray) -> np.ndarray:
        """
        Короткое описание:
            Нормализует профиль проекции в диапазон [0, 1].
        Вход:
            hpp: np.ndarray -- исходныи профиль.
        Выход:
            np.ndarray -- нормализованныи профиль.
        """
        # Шаг 1: обрабатываем вырожденныи случаи.
        hpp_min = float(np.min(hpp))
        hpp_max = float(np.max(hpp))
        delta = hpp_max - hpp_min
        if delta <= 1e-9:
            return np.zeros_like(hpp, dtype=np.float32)

        # Шаг 2: выполняем min-max нормализацию.
        normalized = (hpp.astype(np.float32) - hpp_min) / delta
        return normalized.astype(np.float32)

    def _build_angle_aware_hpp(
        self,
        binary: np.ndarray,
        angle_map: np.ndarray,
        cells: List[Tuple[int, int, int, int, int, int]],
        page_idx: int,
    ) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        """
        Короткое описание:
            Формирует глобальныи angle-aware HPP и структуру для energy.
        Вход:
            binary: np.ndarray -- бинарное изображение страницы.
            angle_map: np.ndarray -- карта локальных углов.
            cells: List[Tuple[int, int, int, int, int, int]] -- список ячеек.
            page_idx: int -- индекс страницы для debug.
        Выход:
            Tuple[np.ndarray, List[Dict[str, object]]]:
                глобальныи профиль по Y и информация по ячейкам.
        """
        # Шаг 1: подготавливаем буфер общего HPP и список данных ячеек.
        height, _ = binary.shape
        global_hpp = np.zeros(height, dtype=np.float32)
        cell_infos: List[Dict[str, object]] = []

        # Шаг 2: считаем локальныи HPP в каждои ячейке и агрегируем.
        iterator = tqdm(
            cells,
            desc=f"HPP page {page_idx}",
            disable=not self.use_tqdm,
            leave=False,
        )
        for y0, y1, x0, x1, gy, gx in iterator:
            local_angle = float(angle_map[gy, gx])
            profile, traces = self._compute_cell_slanted_hpp(binary, y0, y1, x0, x1, local_angle)
            global_hpp[y0:y1] += profile
            cell_infos.append(
                {
                    "y0": y0,
                    "y1": y1,
                    "x0": x0,
                    "x1": x1,
                    "gy": gy,
                    "gx": gx,
                    "angle": local_angle,
                    "profile": profile,
                    "traces": traces,
                }
            )

        # Шаг 3: сохраняем debug-график HPP.
        if self.debug:
            self._save_hpp_plot(global_hpp, page_idx)

        return global_hpp, cell_infos

    def _save_hpp_plot(self, hpp: np.ndarray, page_idx: int) -> None:
        """
        Короткое описание:
            Сохраняет график проекционного профиля средствами OpenCV.
        Вход:
            hpp: np.ndarray -- глобальныи HPP.
            page_idx: int -- индекс страницы.
        Выход:
            None
        """
        # Шаг 1: создаем белыи холст и базовые параметры.
        graph_h, graph_w = 420, 900
        pad_l, pad_r, pad_t, pad_b = 60, 20, 20, 50
        graph = np.full((graph_h, graph_w, 3), 255, dtype=np.uint8)

        # Шаг 2: переводим профиль в полилинию.
        x_min, x_max = 0, max(len(hpp) - 1, 1)
        y_min, y_max = float(np.min(hpp)), float(np.max(hpp))
        y_range = max(y_max - y_min, 1e-9)
        points: List[Tuple[int, int]] = []
        for x_idx, value in enumerate(hpp):
            x = int(pad_l + (x_idx - x_min) / max(1, x_max - x_min) * (graph_w - pad_l - pad_r))
            y = int(graph_h - pad_b - (float(value) - y_min) / y_range * (graph_h - pad_t - pad_b))
            points.append((x, y))

        # Шаг 3: рисуем кривую и рамку.
        for i in range(1, len(points)):
            cv2.line(graph, points[i - 1], points[i], (255, 0, 0), 2)
        cv2.rectangle(graph, (pad_l, pad_t), (graph_w - pad_r, graph_h - pad_b), (0, 0, 0), 1)
        cv2.imwrite(os.path.join(DEBUG_IMAGES_DIR, f"page_{page_idx:03d}_my_method_hpp.jpg"), graph)

    def _find_line_regions(self, normalized_hpp: np.ndarray) -> List[Tuple[int, int]]:
        """
        Короткое описание:
            Находит вертикальные интервалы строк по нормализованному HPP.
        Вход:
            normalized_hpp: np.ndarray -- профиль в диапазоне [0, 1].
        Выход:
            List[Tuple[int, int]] -- интервалы строк (start_y, end_y).
        """
        # Шаг 1: строим бинарную маску строк.
        text_rows = normalized_hpp > float(self.threshold)
        if not np.any(text_rows):
            return []

        # Шаг 2: преобразуем маску в интервалы.
        regions: List[Tuple[int, int]] = []
        in_region = False
        start = 0
        for idx, is_text in enumerate(text_rows):
            if is_text and not in_region:
                in_region = True
                start = idx
            elif (not is_text) and in_region:
                regions.append((start, idx - 1))
                in_region = False
        if in_region:
            regions.append((start, len(text_rows) - 1))

        # Шаг 3: склеиваем близкие интервалы.
        merged: List[Tuple[int, int]] = []
        for cur_start, cur_end in regions:
            if len(merged) == 0:
                merged.append((cur_start, cur_end))
                continue
            prev_start, prev_end = merged[-1]
            if (cur_start - prev_end) <= LINE_REGION_MERGE_GAP:
                merged[-1] = (prev_start, cur_end)
            else:
                merged.append((cur_start, cur_end))
        return merged

    def _filtred_lines_regions(
        self,
        line_regions: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        Короткое описание:
            Удаляет слишком тонкие ложные регионы строк по средней толщине.
        Вход:
            line_regions: List[Tuple[int, int]] -- исходные интервалы строк.
        Выход:
            List[Tuple[int, int]] -- интервалы строк после удаления тонких регионов.
        """
        # Шаг 1: если регионов мало, средняя толщина не дает надежного критерия.
        if len(line_regions) <= 1:
            return line_regions

        # Шаг 2: считаем толщину каждого региона и среднюю толщину.
        region_heights = np.array(
            [end - start + 1 for start, end in line_regions],
            dtype=np.float32,
        )
        average_height = float(np.mean(region_heights))
        min_allowed_height = average_height / LINE_REGION_MIN_AVG_DIVISOR

        # Шаг 3: выкидываем регионы тоньше среднего порога.
        filtred_regions: List[Tuple[int, int]] = []
        for region, height in zip(line_regions, region_heights):
            if float(height) >= min_allowed_height:
                filtred_regions.append(region)

        return filtred_regions

    def _extend_short_energy_components(self, energy: np.ndarray) -> np.ndarray:
        """
        Короткое описание:
            Протягивает неполные сильные компоненты energy через всю ширину страницы.
        Вход:
            energy: np.ndarray -- энергетическая матрица страницы.
        Выход:
            np.ndarray -- энергетическая матрица с протянутыми строковыми компонентами.
        """
        # Шаг 1: берем только сильную energy, чтобы не протягивать слабые градиенты букв.
        strong_mask = energy >= (ENERGY_LINE_REGION_BOOST * 0.5)
        if not np.any(strong_mask):
            return energy

        # Шаг 2: ищем связные компоненты сильных энергетических областей.
        num_labels, labels = cv2.connectedComponents(strong_mask.astype(np.uint8))
        extended_energy = energy.copy()
        height, width = energy.shape

        # Шаг 3: для неполной компоненты считаем средний y и проводим линию через страницу.
        for label_idx in range(1, num_labels):
            ys, xs = np.where(labels == label_idx)
            if len(xs) == 0:
                continue

            x_min = int(np.min(xs))
            x_max = int(np.max(xs))
            if x_min == 0 and x_max == width - 1:
                continue

            y_mean = int(np.clip(round(float(np.mean(ys))), 0, height - 1))
            extended_energy[y_mean, :] = np.maximum(
                extended_energy[y_mean, :],
                ENERGY_LINE_REGION_BOOST,
            )

        return extended_energy

    def _compute_energy_matrix(
        self,
        binary: np.ndarray,
        line_regions: List[Tuple[int, int]],
        cell_infos: List[Dict[str, object]],
        page_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Короткое описание:
            Считает energy отдельно в каждои ячейке сетки и склеивает их в общую матрицу.
        Вход:
            binary: np.ndarray -- бинарное изображение страницы.
            line_regions: List[Tuple[int, int]] -- интервалы текстовых строк.
            cell_infos: List[Dict[str, object]] -- локальные данные профилеи и трасс.
            page_idx: Optional[int] -- индекс страницы для debug-сохранения.
        Выход:
            np.ndarray -- энергетическая матрица H x W.
        """
        # Шаг 1: создаем пустую итоговую матрицу, которую будем заполнять ячейками.
        height, width = binary.shape
        energy = np.zeros((height, width), dtype=np.float32)

        # Шаг 2: для каждой ячейки строим свою локальную energy.
        for info in cell_infos:
            y0 = int(info["y0"])
            y1 = int(info["y1"])
            x0 = int(info["x0"])
            x1 = int(info["x1"])

            cell_binary = binary[y0:y1, x0:x1]
            cell_height, cell_width = cell_binary.shape
            cell_energy = np.zeros((cell_height, cell_width), dtype=np.float32)

            # Шаг 2.1: текстовые пиксели внутри клетки получают базовый штраф.
            cell_energy += (cell_binary == 0).astype(np.float32) * ENERGY_TEXT_WEIGHT

            # Шаг 2.2: градиент тоже считается локально внутри клетки.
            grad_x = cv2.Sobel(cell_binary, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(cell_binary, cv2.CV_32F, 0, 1, ksize=3)
            gradient = np.sqrt(grad_x * grad_x + grad_y * grad_y)
            cell_energy += ENERGY_GRADIENT_WEIGHT * gradient

            # Шаг 2.3: HPP-энергию рисуем прямо по трассам Брезенхема внутри клетки.
            profile = np.asarray(info["profile"], dtype=np.float32)
            traces = info["traces"]
            profile_norm = self._normalize_hpp(profile)
            for local_y, line_points in enumerate(traces):
                profile_value = float(profile_norm[local_y])
                value = profile_value * ENERGY_HPP_WEIGHT
                y_base = y0 + int(local_y)
                region_boost = 0.0
                for start, end in line_regions:
                    boost_y0 = int(start) - ENERGY_LINE_REGION_MARGIN_TOP
                    boost_y1 = int(end) + ENERGY_LINE_REGION_MARGIN_BOTTOM
                    if boost_y0 <= y_base <= boost_y1:
                        region_boost = ENERGY_LINE_REGION_BOOST
                        break

                # Локальная строка в ячейке может быть короткой и не попасть в глобальный HPP.
                # Если она достаточно сильная локально, усиливаем ее тем же способом, что и регион.
                if profile_value >= float(self.threshold):
                    region_boost = max(region_boost, ENERGY_LINE_REGION_BOOST)

                total_value = value + region_boost
                if total_value <= 0.0:
                    continue
                for x, y in line_points:
                    local_x = int(x) - x0
                    local_trace_y = int(y) - y0
                    if 0 <= local_x < cell_width and 0 <= local_trace_y < cell_height:
                        cell_energy[local_trace_y, local_x] += total_value
                        if local_trace_y - 1 >= 0:
                            cell_energy[local_trace_y - 1, local_x] += total_value * ENERGY_HPP_NEIGHBOR_WEIGHT
                        if local_trace_y + 1 < cell_height:
                            cell_energy[local_trace_y + 1, local_x] += total_value * ENERGY_HPP_NEIGHBOR_WEIGHT

            # Шаг 2.4: вклеиваем готовую локальную матрицу в общий холст.
            energy[y0:y1, x0:x1] = cell_energy

        # Шаг 3: убираем NaN и бесконечности.
        energy = np.nan_to_num(energy, nan=0.0, posinf=1e9, neginf=0.0).astype(np.float32)

        # Шаг 4: если сильная строковая компонента короткая, протягиваем ее через страницу.
        energy = self._extend_short_energy_components(energy)

        # Шаг 5: сохраняем подробный debug склеенной матрицы по ячейкам.
        if self.debug and page_idx is not None:
            self._save_cell_energy_debug(binary, energy, cell_infos, int(page_idx))

        return energy

    def _find_seam_a_star(self, energy: np.ndarray, start_y: int) -> List[int]:
        """
        Короткое описание:
            Ищет горизонтальныи шов от левого края до правого методом A*.
        Вход:
            energy: np.ndarray -- энергетическая матрица H x W.
            start_y: int -- начальная координата Y на левом крае.
        Выход:
            List[int] -- список координат Y шва для каждого X.
        """
        # Шаг 1: проверяем старт.
        height, width = energy.shape
        if not (0 <= int(start_y) < height):
            return []

        # Шаг 2: инициализируем структуры A*.
        start = (int(start_y), 0)
        goal_x = width - 1
        open_heap: List[Tuple[float, int, int]] = []
        heapq.heappush(open_heap, (0.0, int(start_y), 0))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        visited: set = set()

        # Шаг 3: основной цикл поиска.
        while len(open_heap) > 0:
            _, y_cur, x_cur = heapq.heappop(open_heap)
            current = (y_cur, x_cur)

            if current in visited:
                continue
            visited.add(current)

            if x_cur == goal_x:
                seam = [0] * width
                node: Optional[Tuple[int, int]] = current
                while node is not None:
                    y_node, x_node = node
                    seam[x_node] = y_node
                    node = came_from.get(node)
                return seam

            next_x = x_cur + 1
            if next_x >= width:
                continue

            for dy in (-1, 0, 1):
                next_y = y_cur + dy
                if not (0 <= next_y < height):
                    continue
                neighbor = (next_y, next_x)
                move_penalty = abs(dy) * A_STAR_DY_PENALTY
                candidate_g = g_score[current] + float(energy[next_y, next_x]) + move_penalty
                if candidate_g >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = candidate_g
                heuristic = float(goal_x - next_x)
                f_score = candidate_g + heuristic
                heapq.heappush(open_heap, (f_score, next_y, next_x))

        # Шаг 4: если путь не найден, возвращаем пустои шов.
        return []

    def _get_line_pixels_between_seams(
        self,
        binary: np.ndarray,
        upper_seam: List[int],
        lower_seam: List[int],
    ) -> set:
        """
        Короткое описание:
            Собирает текстовые пиксели между двумя соседними швами.
        Вход:
            binary: np.ndarray -- бинарное изображение.
            upper_seam: List[int] -- верхнии шов.
            lower_seam: List[int] -- нижнии шов.
        Выход:
            set -- множество координат (x, y) текстовых пикселеи.
        """
        # Шаг 1: инициализируем множество пикселеи строки.
        height, width = binary.shape
        pixels = set()

        # Шаг 2: идем по столбцам и выбираем черные пиксели между швами.
        for x in range(width):
            y_top = min(int(upper_seam[x]), int(lower_seam[x]))
            y_bottom = max(int(upper_seam[x]), int(lower_seam[x]))
            for y in range(y_top + 1, y_bottom):
                if 0 <= y < height and binary[y, x] == 0:
                    pixels.add((x, y))
        return pixels

    def _save_cell_energy_debug(
        self,
        binary: np.ndarray,
        energy: np.ndarray,
        cell_infos: List[Dict[str, object]],
        page_idx: int,
    ) -> None:
        """
        Короткое описание:
            Сохраняет debug-визуализацию energy, склееннои из отдельных ячеек.
        Вход:
            binary: np.ndarray -- бинарное изображение страницы.
            energy: np.ndarray -- итоговая энергетическая матрица.
            cell_infos: List[Dict[str, object]] -- данные ячеек сетки.
            page_idx: int -- индекс страницы.
        Выход:
            None
        """
        # Шаг 1: нормализуем energy и переводим ее в цветовую карту.
        os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)
        energy_norm = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(energy_norm, cv2.COLORMAP_TURBO)

        # Шаг 2: делаем overlay поверх бинарного изображения.
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(binary_bgr, 0.45, heatmap, 0.55, 0)
        traces_vis = binary_bgr.copy()

        # Шаг 3: рисуем границы ячеек, подписываем угол и показываем трассы Брезенхема.
        heatmap_grid = heatmap.copy()
        overlay_grid = overlay.copy()
        debug_text_path = os.path.join(DEBUG_IMAGES_DIR, f"page_{page_idx:03d}_my_method_cell_energy.txt")
        with open(debug_text_path, "w", encoding="utf-8") as file:
            file.write("gy gx y0 y1 x0 x1 angle bresenham_rows energy_min energy_max energy_mean\n")

            for info in cell_infos:
                y0 = int(info["y0"])
                y1 = int(info["y1"])
                x0 = int(info["x0"])
                x1 = int(info["x1"])
                gy = int(info["gy"])
                gx = int(info["gx"])
                angle = float(info["angle"])
                cell_energy = energy[y0:y1, x0:x1]
                profile = np.asarray(info["profile"], dtype=np.float32)
                profile_norm = self._normalize_hpp(profile)
                traces = info["traces"]
                bresenham_rows = 0

                cv2.rectangle(heatmap_grid, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 2)
                cv2.rectangle(overlay_grid, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
                cv2.rectangle(traces_vis, (x0, y0), (x1 - 1, y1 - 1), (0, 160, 255), 2)

                label = f"{gy},{gx} {angle:.1f}"
                label_x = min(x0 + 4, max(0, binary.shape[1] - 1))
                label_y = min(y0 + 18, max(0, binary.shape[0] - 1))
                cv2.putText(
                    heatmap_grid,
                    label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay_grid,
                    label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    traces_vis,
                    label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

                row_step = max(1, (y1 - y0) // 18)
                for local_y, line_points in enumerate(traces):
                    should_draw = (
                        local_y % row_step == 0 or
                        float(profile_norm[local_y]) >= float(self.threshold)
                    )
                    if not should_draw or len(line_points) < 2:
                        continue

                    uses_bresenham = abs(angle) > ANGLE_EPSILON
                    color = (0, 0, 255) if uses_bresenham else (255, 0, 0)
                    thickness = 2 if float(profile_norm[local_y]) >= float(self.threshold) else 1
                    if uses_bresenham:
                        bresenham_rows += 1

                    for point_idx in range(1, len(line_points)):
                        x_prev, y_prev = line_points[point_idx - 1]
                        x_cur, y_cur = line_points[point_idx]
                        cv2.line(
                            traces_vis,
                            (int(x_prev), int(y_prev)),
                            (int(x_cur), int(y_cur)),
                            color,
                            thickness,
                        )

                file.write(
                    f"{gy} {gx} {y0} {y1} {x0} {x1} {angle:.6f} {bresenham_rows} "
                    f"{float(np.min(cell_energy)):.6f} "
                    f"{float(np.max(cell_energy)):.6f} "
                    f"{float(np.mean(cell_energy)):.6f}\n"
                )

        # Шаг 4: сохраняем чистую матрицу, heatmap с сеткой и overlay.
        cv2.imwrite(
            os.path.join(DEBUG_IMAGES_DIR, f"page_{page_idx:03d}_my_method_cell_energy_gray.jpg"),
            energy_norm,
        )
        cv2.imwrite(
            os.path.join(DEBUG_IMAGES_DIR, f"page_{page_idx:03d}_my_method_cell_energy_grid.jpg"),
            heatmap_grid,
        )
        cv2.imwrite(
            os.path.join(DEBUG_IMAGES_DIR, f"page_{page_idx:03d}_my_method_cell_energy_overlay.jpg"),
            overlay_grid,
        )
        cv2.imwrite(
            os.path.join(DEBUG_IMAGES_DIR, f"page_{page_idx:03d}_my_method_bresenham_traces.jpg"),
            traces_vis,
        )

    def _save_energy_debug(self, energy: np.ndarray, page_idx: int) -> None:
        """
        Короткое описание:
            Сохраняет нормализованную визуализацию энергетическои матрицы.
        Вход:
            energy: np.ndarray -- энергетическая матрица.
            page_idx: int -- индекс страницы.
        Выход:
            None
        """
        # Шаг 1: нормализуем энергию для сохранения в uint8.
        energy_vis = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Шаг 2: сохраняем debug-файл.
        cv2.imwrite(
            os.path.join(DEBUG_IMAGES_DIR, f"page_{page_idx:03d}_my_method_energy.jpg"),
            energy_vis,
        )

    def _save_seams_debug(
        self,
        binary: np.ndarray,
        seams: List[List[int]],
        page_idx: int,
    ) -> None:
        """
        Короткое описание:
            Сохраняет визуализацию найденных A* швов.
        Вход:
            binary: np.ndarray -- бинарное изображение.
            seams: List[List[int]] -- список швов.
            page_idx: int -- индекс страницы.
        Выход:
            None
        """
        # Шаг 1: готовим BGR-подложку.
        vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # Шаг 2: рисуем каждый шов.
        for seam in seams:
            for x in range(1, len(seam)):
                cv2.line(
                    vis,
                    (x - 1, int(seam[x - 1])),
                    (x, int(seam[x])),
                    (0, 0, 255),
                    2,
                )

        # Шаг 3: сохраняем визуализацию.
        cv2.imwrite(
            os.path.join(DEBUG_IMAGES_DIR, f"page_{page_idx:03d}_my_method_seams.jpg"),
            vis,
        )

    def _save_start_points_debug(
        self,
        binary: np.ndarray,
        line_regions: List[Tuple[int, int]],
        start_points: List[int],
        page_idx: int,
    ) -> None:
        """
        Короткое описание:
            Сохраняет debug стартовых точек A* швов между строками.
        Вход:
            binary: np.ndarray -- бинарное изображение страницы.
            line_regions: List[Tuple[int, int]] -- интервалы найденных строк.
            start_points: List[int] -- стартовые y-точки швов.
            page_idx: int -- индекс страницы.
        Выход:
            None
        """
        # Шаг 1: готовим цветную подложку.
        if len(binary.shape) == 2:
            vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            vis = binary.copy()
        height, width = binary.shape[:2]

        # Шаг 2: показываем границы регионов строк тонкими цветными линиями.
        for start, end in line_regions:
            cv2.line(vis, (0, int(start)), (width - 1, int(start)), (255, 180, 0), 1)
            cv2.line(vis, (0, int(end)), (width - 1, int(end)), (255, 180, 0), 1)

        # Шаг 3: стартовые точки рисуем жирно и заметно.
        for point_idx, start_y in enumerate(start_points):
            y = int(np.clip(start_y, 0, height - 1))
            cv2.line(vis, (0, y), (width - 1, y), (0, 0, 255), 4)
            cv2.circle(vis, (12, y), 9, (0, 255, 255), -1)
            cv2.circle(vis, (12, y), 9, (0, 0, 255), 2)
            cv2.putText(
                vis,
                f"{point_idx}: y={y}",
                (26, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Шаг 4: сохраняем изображение и текстовый список точек.
        image_path = os.path.join(DEBUG_IMAGES_DIR, f"page_{page_idx:03d}_my_method_start_points.jpg")
        text_path = os.path.join(DEBUG_IMAGES_DIR, f"page_{page_idx:03d}_my_method_start_points.txt")
        cv2.imwrite(image_path, vis)
        with open(text_path, "w", encoding="utf-8") as file:
            file.write(f"start_points_count={len(start_points)}\n")
            for point_idx, start_y in enumerate(start_points):
                file.write(f"{point_idx}: y={int(start_y)}\n")

    def _save_line_regions_debug(
        self,
        binary: np.ndarray,
        line_regions: List[Tuple[int, int]],
        page_idx: int,
        name: str = "line_regions",
    ) -> None:
        """
        Короткое описание:
            Сохраняет debug найденных HPP-регионов строк.
        Вход:
            binary: np.ndarray -- бинарное изображение страницы.
            line_regions: List[Tuple[int, int]] -- интервалы найденных строк.
            page_idx: int -- индекс страницы.
        Выход:
            None
        """
        # Шаг 1: готовим цветную подложку.
        if len(binary.shape) == 2:
            vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            vis = binary.copy()
        height, width = binary.shape[:2]

        # Шаг 2: полупрозрачно закрашиваем каждый регион.
        overlay = vis.copy()
        for region_idx, (start, end) in enumerate(line_regions):
            y0 = int(np.clip(start, 0, height - 1))
            y1 = int(np.clip(end, 0, height - 1))
            color = (0, 180, 255) if region_idx % 2 == 0 else (0, 255, 120)
            cv2.rectangle(overlay, (0, y0), (width - 1, y1), color, -1)
            cv2.line(overlay, (0, y0), (width - 1, y0), (0, 0, 255), 2)
            cv2.line(overlay, (0, y1), (width - 1, y1), (0, 0, 255), 2)
            cv2.putText(
                overlay,
                f"{region_idx}: {y0}-{y1}",
                (10, max(18, y0 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        vis = cv2.addWeighted(vis, 0.55, overlay, 0.45, 0)

        # Шаг 3: сохраняем изображение и текстовую сводку.
        image_path = os.path.join(DEBUG_IMAGES_DIR, f"page_{page_idx:03d}_{name}.jpg")
        text_path = os.path.join(DEBUG_IMAGES_DIR, f"page_{page_idx:03d}_{name}.txt")
        cv2.imwrite(image_path, vis)
        with open(text_path, "w", encoding="utf-8") as file:
            file.write(f"line_regions_count={len(line_regions)}\n")
            for region_idx, (start, end) in enumerate(line_regions):
                file.write(f"{region_idx}: start={int(start)}, end={int(end)}, height={int(end - start + 1)}\n")

    def segment_lines(self, image_path: str) -> Tuple[List[set], List[np.ndarray]]:
        """
        Короткое описание:
            Выполняет сегментацию строк на всех страницах изображения документа.
        Вход:
            image_path: str -- путь к входному изображению.
        Выход:
            Tuple[List[set], List[np.ndarray]]:
                список множеств пикселеи строк и список прямоугольных crop строк.
        """
        # Шаг 1: готовим debug-директорию.
        if self.debug:
            os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)
            source = cv2.imread(image_path)
            if source is not None:
                cv2.imwrite(os.path.join(DEBUG_IMAGES_DIR, "my_method_main_input.jpg"), source)

        # Шаг 2: выделяем страницы и бинарные страницы.
        _, binary_pages = extract_pages_with_yolo(
            image_path=image_path,
            model_path=YOLO_NOTEBOOK_MODEL_PATH,
            output_dir=DEBUG_IMAGES_DIR,
            conf_threshold=0.8,
            return_binary=True,
        )

        # Шаг 3: инициализируем итоговые структуры.
        lines_pixels: List[set] = []
        lines_crops: List[np.ndarray] = []

        # Шаг 4: обрабатываем каждую страницу.
        page_iterator = tqdm(
            list(enumerate(binary_pages)),
            desc="Pages my_method",
            disable=not self.use_tqdm,
        )
        for page_idx, page_binary in page_iterator:
            binary_input = self._ensure_binary(page_binary)

            corrected_page, binary_corrected, global_angle = correct_perspective(
                binary_input,
                debug=self.debug,
                debug_output_dir=os.path.join(DEBUG_IMAGES_DIR, f"page_{page_idx:03d}_my_method_perspective"),
            )
            binary_corrected = self._ensure_binary(binary_corrected)

            angle_map, cells = self._estimate_local_angle_map(
                binary_corrected,
                float(global_angle),
                page_idx,
            )
            global_hpp, cell_infos = self._build_angle_aware_hpp(
                binary_corrected,
                angle_map,
                cells,
                page_idx,
            )
            normalized_hpp = self._normalize_hpp(global_hpp)
            line_regions = self._find_line_regions(normalized_hpp)
            if self.debug:
                self._save_line_regions_debug(binary_corrected, line_regions, page_idx, name="line_regions_before_filter")
            line_regions = self._filtred_lines_regions(line_regions)

            if len(line_regions) == 0:
                ys, xs = np.where(binary_corrected == 0)
                text_pixels = {(int(x), int(y)) for y, x in zip(ys, xs)}
                if len(text_pixels) == 0:
                    continue
                lines_pixels.append(text_pixels)
                white_image = np.ones((binary_corrected.shape[0], binary_corrected.shape[1], 3), dtype=np.uint8) * 255
                for x, y in text_pixels:
                    white_image[y, x] = (0, 0, 0)
                line_crop = crop_line_rectangle(white_image, text_pixels, debug=False, padding=0)
                lines_crops.append(line_crop)
                continue

            energy = self._compute_energy_matrix(
                binary_corrected,
                line_regions,
                cell_infos,
                page_idx=page_idx,
            )
            cv2.imwrite('debug_images/img_energy.png', energy)
            if self.debug:
                self._save_energy_debug(energy, page_idx)

            # Шаг 5: формируем стартовые точки швов между соседними строками.
            start_points: List[int] = []
            for reg_idx in range(len(line_regions) - 1):
                _, prev_end = line_regions[reg_idx]
                next_start, _ = line_regions[reg_idx + 1]
                start_points.append((prev_end + next_start) // 2)

            if self.debug:
                self._save_line_regions_debug(binary_corrected, line_regions, page_idx)
                self._save_start_points_debug(binary_corrected, line_regions, start_points, page_idx)

            # Шаг 6: ищем A* швы.
            seams: List[List[int]] = []
            for start_y in tqdm(
                start_points,
                desc=f"Seams page {page_idx}",
                disable=not self.use_tqdm,
                leave=False,
            ):
                seam = self._find_seam_a_star(energy, int(start_y))
                if len(seam) == binary_corrected.shape[1]:
                    seams.append(seam)

            if self.debug:
                self._save_seams_debug(binary_corrected, seams, page_idx)

            if len(seams) == 0:
                text_pixels = {(x, y) for y, x in zip(*np.where(binary_corrected == 0))}
                if len(text_pixels) > 0:
                    lines_pixels.append(text_pixels)
                continue

            # Шаг 7: сортируем швы и дополняем верхнеи и нижнеи границами.
            seams = sorted(seams, key=lambda seam: float(np.mean(seam)))
            seams_full = (
                [[0] * binary_corrected.shape[1]]
                + seams
                + [[binary_corrected.shape[0] - 1] * binary_corrected.shape[1]]
            )

            # Шаг 8: собираем пиксели строк и их crop.
            page_line_pixels: List[set] = []
            page_line_crops: List[np.ndarray] = []
            for seam_idx in range(len(seams_full) - 1):
                upper = seams_full[seam_idx]
                lower = seams_full[seam_idx + 1]
                pixels = self._get_line_pixels_between_seams(binary_corrected, upper, lower)
                if len(pixels) == 0:
                    continue
                page_line_pixels.append(pixels)

                white = np.ones((binary_corrected.shape[0], binary_corrected.shape[1], 3), dtype=np.uint8) * 255
                for x, y in pixels:
                    white[y, x] = (0, 0, 0)
                crop = crop_line_rectangle(white, pixels, debug=False, padding=0)
                page_line_crops.append(crop)

            # Шаг 9: сохраняем постраничныи debug и агрегируем итоги.
            if self.debug and corrected_page is not None and len(page_line_pixels) > 0:
                if len(corrected_page.shape) == 2:
                    vis = cv2.cvtColor(corrected_page, cv2.COLOR_GRAY2BGR)
                else:
                    vis = corrected_page.copy()
                random.seed(42)
                colors = [
                    (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
                    for _ in range(len(page_line_pixels))
                ]
                for line_idx, pixels in enumerate(page_line_pixels):
                    color = colors[line_idx]
                    for x, y in pixels:
                        if 0 <= y < vis.shape[0] and 0 <= x < vis.shape[1]:
                            vis[y, x] = color
                cv2.imwrite(
                    os.path.join(DEBUG_IMAGES_DIR, f"page_{page_idx:03d}_my_method_segmented_lines.jpg"),
                    vis,
                )

            lines_pixels.extend(page_line_pixels)
            lines_crops.extend(page_line_crops)

        # Шаг 10: сохраняем crop строк.
        save_dir = "output/lines"
        os.makedirs(save_dir, exist_ok=True)
        for line_idx, crop in enumerate(lines_crops):
            cv2.imwrite(os.path.join(save_dir, f"my_method_line_{line_idx:03d}.jpg"), crop)

        return lines_pixels, lines_crops


if __name__ == "__main__":
    IMAGE_PATH = "/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code/datasets/school_notebooks_RU/images_base/50_504.JPG"
    segmenter = LineSegmentation(debug=True, use_tqdm=True)
    segmenter.segment_lines(image_path=IMAGE_PATH)
