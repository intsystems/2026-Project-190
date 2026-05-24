"""
Короткое описание:
    Прогоняет my_small_size_das_panda_hpp_seam_exact.py по test split HWR200
    и считает те же метрики: polygon IoU matching, precision, recall, hmean.
"""

import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from tqdm import tqdm

from typing import List



EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
import das_panda_hpp_seam_exact as exact
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

# 1) метод постообработаки

def _box_sides(box: np.ndarray) -> List[float]:
    """
    Длины 4 сторон rotated rectangle.

    box:
        np.ndarray shape=(4, 2), результат cv2.boxPoints(...)

    Возвращает:
        [side01, side12, side23, side30]
    """
    box = np.asarray(box, dtype=np.float32).reshape(4, 2)
    return [
        float(np.linalg.norm(box[(i + 1) % 4] - box[i]))
        for i in range(4)
    ]


def rect_height_min_side(box: np.ndarray) -> float:
    """
    Высота прямоугольника в твоём смысле:
        высота = наименьшая сторона rotated rectangle.

    Это не bbox по X/Y, а именно сторона повёрнутого прямоугольника.
    """
    sides = _box_sides(box)
    return float(min(sides))


def mean_rect_height_min_side(boxes: List[np.ndarray]) -> float:
    """
    Средняя высота по всем предсказанным прямоугольникам.

    Используется как нормальная высота строки.
    Потом всё, что выше mean * 1.2, считаем подозрительно высоким.
    """
    if len(boxes) == 0:
        return 0.0

    heights = [rect_height_min_side(box) for box in boxes]
    return float(np.mean(heights))


def _rect_local_axes(box: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Строит локальные оси rotated rectangle.

    Нам нужны две оси:
        u -- вдоль длинной стороны строки;
        v -- вдоль короткой стороны, то есть по высоте строки.

    Любая точка p внутри rect может быть записана так:
        p = center + s * u + t * v

    где:
        s -- координата вдоль строки;
        t -- координата по высоте строки.

    Именно по t мы будем обрезать слишком высокий rect.
    """
    box = np.asarray(box, dtype=np.float32).reshape(4, 2)
    center = np.mean(box, axis=0).astype(np.float32)

    edges = []
    for i in range(4):
        edge = box[(i + 1) % 4] - box[i]
        length = float(np.linalg.norm(edge))
        edges.append((length, edge))

    # Длинная сторона.
    long_len, long_edge = max(edges, key=lambda item: item[0])

    if long_len < 1e-6:
        u = np.array([1.0, 0.0], dtype=np.float32)
    else:
        u = (long_edge / long_len).astype(np.float32)

    # Короткая ось перпендикулярна длинной.
    v = np.array([-u[1], u[0]], dtype=np.float32)

    rel = box - center[None, :]
    s_values = rel @ u
    t_values = rel @ v

    return {
        "center": center,
        "u": u,
        "v": v,
        "s_min": float(np.min(s_values)),
        "s_max": float(np.max(s_values)),
        "t_min": float(np.min(t_values)),
        "t_max": float(np.max(t_values)),
    }


def _box_from_local_axes(
    center: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    s_min: float,
    s_max: float,
    t_min: float,
    t_max: float,
) -> np.ndarray:
    """
    Собирает rotated rectangle обратно из локальных координат.

    Возвращает:
        np.ndarray shape=(4, 2)
    """
    return np.array(
        [
            center + s_min * u + t_min * v,
            center + s_max * u + t_min * v,
            center + s_max * u + t_max * v,
            center + s_min * u + t_max * v,
        ],
        dtype=np.float32,
    )


def _crop_tall_box_by_own_class_mass(
    box: np.ndarray,
    class_pixels_xy: np.ndarray,
    target_height: float,
) -> np.ndarray:
    """
    Обрезает один слишком высокий rect по центру масс пикселей его же класса.

    class_pixels_xy:
        np.ndarray shape=(N, 2), точки [x, y] для конкретного class_id:
            xs, ys = np.where(class_matrix == class_id)

    Логика:
        - есть высокий rect;
        - считаем центр масс пикселей этой строки;
        - смотрим, к какой стороне по высоте этот центр ближе;
        - противоположную, дальнюю сторону режем;
        - новая высота примерно target_height.

    Тут нет threshold, нет grayscale, нет поиска чёрных пикселей.
    Всё берётся из class_matrix.
    """
    box = np.asarray(box, dtype=np.float32).reshape(4, 2)

    if class_pixels_xy.shape[0] == 0:
        return box

    axes = _rect_local_axes(box)
    center = axes["center"]
    u = axes["u"]
    v = axes["v"]

    s_min = axes["s_min"]
    s_max = axes["s_max"]
    t_min = axes["t_min"]
    t_max = axes["t_max"]

    # Центр масс пикселей именно этой строки/class_id.
    mass_center = np.mean(class_pixels_xy.astype(np.float32), axis=0)

    # Координата центра масс по высоте rect.
    t_mass = float((mass_center - center) @ v)

    dist_to_top = abs(t_mass - t_min)
    dist_to_bottom = abs(t_max - t_mass)

    if dist_to_top <= dist_to_bottom:
        # Масса ближе к верхней стороне.
        # Значит нижняя сторона дальше, её и режем.
        new_t_min = t_min
        new_t_max = t_min + target_height

        # Защита: центр масс не должен оказаться снаружи после обрезки.
        if t_mass > new_t_max:
            shift = t_mass - new_t_max
            new_t_min += shift
            new_t_max += shift
    else:
        # Масса ближе к нижней стороне.
        # Значит верхняя сторона дальше, её и режем.
        new_t_max = t_max
        new_t_min = t_max - target_height

        # Защита: центр масс не должен оказаться снаружи после обрезки.
        if t_mass < new_t_min:
            shift = new_t_min - t_mass
            new_t_min -= shift
            new_t_max -= shift

    # Не даём выйти за исходный rect.
    new_t_min = max(new_t_min, t_min)
    new_t_max = min(new_t_max, t_max)

    # Если что-то схлопнулось, лучше вернуть старый box.
    if new_t_max - new_t_min < 2.0:
        return box

    return _box_from_local_axes(
        center=center,
        u=u,
        v=v,
        s_min=s_min,
        s_max=s_max,
        t_min=new_t_min,
        t_max=new_t_max,
    )


def _side_midpoints(box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Берёт midpoint левой и правой стороны rect.

    Для каждой из 4 сторон считаем midpoint.
    Левая сторона = midpoint с минимальным x.
    Правая сторона = midpoint с максимальным x.

    Возвращает:
        left_mid, right_mid
    """
    box = np.asarray(box, dtype=np.float32).reshape(4, 2)

    mids = []
    for i in range(4):
        mids.append((box[i] + box[(i + 1) % 4]) * 0.5)

    mids = np.asarray(mids, dtype=np.float32)

    left_mid = mids[int(np.argmin(mids[:, 0]))]
    right_mid = mids[int(np.argmax(mids[:, 0]))]

    return left_mid, right_mid


def _kmeans_1d_2clusters(values: np.ndarray, max_iter: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Очень простой 1D k-means на 2 кластера.

    Нужен, чтобы не тащить sklearn.

    values:
        X-координаты midpoint'ов левых/правых сторон.

    Возвращает:
        labels  -- 0/1 для каждой точки
        centers -- два центра кластеров
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    if values.size < 2:
        return np.zeros(values.size, dtype=np.int32), np.array([0.0, 0.0], dtype=np.float32)

    centers = np.array([float(np.min(values)), float(np.max(values))], dtype=np.float32)

    for _ in range(max_iter):
        d0 = np.abs(values - centers[0])
        d1 = np.abs(values - centers[1])
        labels = (d1 < d0).astype(np.int32)

        new_centers = centers.copy()

        if np.any(labels == 0):
            new_centers[0] = float(np.mean(values[labels == 0]))
        if np.any(labels == 1):
            new_centers[1] = float(np.mean(values[labels == 1]))

        if np.allclose(new_centers, centers, atol=1e-3):
            break

        centers = new_centers

    return labels, centers


def _crop_box_by_vertical_cut(box: np.ndarray, cut_x: float, remove_left: bool) -> np.ndarray:
    """
    Обрезает rotated rectangle вертикальной границей cut_x.

    remove_left=True:
        удаляем левую часть, оставляем x >= cut_x.

    remove_left=False:
        удаляем правую часть, оставляем x <= cut_x.

    Сделано через локальную ось u, чтобы не просто clamp'ить углы по X.
    """
    box = np.asarray(box, dtype=np.float32).reshape(4, 2)
    axes = _rect_local_axes(box)

    center = axes["center"]
    u = axes["u"]
    v = axes["v"]

    s_min = axes["s_min"]
    s_max = axes["s_max"]
    t_min = axes["t_min"]
    t_max = axes["t_max"]

    # Если строка почти вертикальная, обрезка по X нестабильна.
    # Для строк текста это почти не должно случаться.
    if abs(float(u[0])) < 1e-6:
        return box

    # x = center_x + s * u_x
    # значит s для вертикальной границы:
    cut_s = float((cut_x - float(center[0])) / float(u[0]))

    if not (s_min < cut_s < s_max):
        return box

    left_s = s_min if (center + s_min * u)[0] <= (center + s_max * u)[0] else s_max
    right_s = s_max if left_s == s_min else s_min

    new_s_min = s_min
    new_s_max = s_max

    if remove_left:
        # Нужно убрать сторону, которая геометрически слева.
        if left_s == s_min:
            new_s_min = cut_s
        else:
            new_s_max = cut_s
    else:
        # Нужно убрать сторону, которая геометрически справа.
        if right_s == s_max:
            new_s_max = cut_s
        else:
            new_s_min = cut_s

    if abs(new_s_max - new_s_min) < 3.0:
        return box

    return _box_from_local_axes(
        center=center,
        u=u,
        v=v,
        s_min=new_s_min,
        s_max=new_s_max,
        t_min=t_min,
        t_max=t_max,
    )


def _crop_side_page_noise_by_midpoint_clustering(
    boxes: List[np.ndarray],
    min_small_cluster_fraction: float = 0.05,
    max_small_cluster_fraction: float = 0.35,
    separation_factor: float = 1.0,
) -> List[np.ndarray]:
    """
    Убирает боковой кусок другой страницы.

    Идея:
        - у каждого rect берём две точки:
            midpoint левой стороны,
            midpoint правой стороны;
        - кластеризуем их X-координаты на 2 кластера;
        - если один кластер маленький и хорошо отделён,
          считаем его краем чужой страницы;
        - режем rect по границе между кластерами.

    Критерий отделимости:
        inter_distance > intra_distance * separation_factor

    То есть межкластерное расстояние должно быть больше внутрикластерного.
    """
    if len(boxes) < 2:
        return boxes

    points = []

    for box in boxes:
        left_mid, right_mid = _side_midpoints(box)
        points.append(left_mid)
        points.append(right_mid)

    points = np.asarray(points, dtype=np.float32)
    x = points[:, 0]

    labels, centers = _kmeans_1d_2clusters(x)

    count0 = int(np.sum(labels == 0))
    count1 = int(np.sum(labels == 1))

    if count0 == 0 or count1 == 0:
        return boxes

    total = count0 + count1
    small_label = 0 if count0 < count1 else 1
    main_label = 1 - small_label
    small_fraction = min(count0, count1) / max(total, 1)

    # Если маленький кластер слишком маленький, это может быть случайная точка.
    # Если слишком большой, это уже не "залезший край", а нормальная геометрия.
    if small_fraction < min_small_cluster_fraction:
        return boxes
    if small_fraction > max_small_cluster_fraction:
        return boxes

    intra_values = []
    for label in [0, 1]:
        cluster_x = x[labels == label]
        center_x = centers[label]
        if cluster_x.size > 0:
            intra_values.extend(np.abs(cluster_x - center_x).tolist())

    intra_distance = float(np.mean(intra_values)) if intra_values else 0.0
    inter_distance = float(abs(centers[0] - centers[1]))

    if inter_distance <= intra_distance * separation_factor:
        return boxes

    cut_x = float((centers[0] + centers[1]) * 0.5)

    # Если маленький кластер левее основного — удаляем левую часть.
    # Если правее — удаляем правую часть.
    small_is_left = centers[small_label] < centers[main_label]

    return [
        _crop_box_by_vertical_cut(
            box=box,
            cut_x=cut_x,
            remove_left=small_is_left,
        )
        for box in boxes
    ]


def class_matrix_to_postprocessed_polygons(
    class_matrix: np.ndarray,
    x_offset: int,
    y_offset: int,
    tall_factor: float = 1.2,
) -> List[np.ndarray]:
    """
    Что делает:
        1. По каждому class_id строит cv2.minAreaRect.
        2. Считает среднюю высоту rect как среднюю меньшую сторону.
        3. Если rect выше mean_height * tall_factor,
           режет дальнюю сторону по центру масс пикселей этого class_id.
        4. По midpoint'ам левых/правых сторон режет боковой мусор другой страницы.
        5. Только в самом конце добавляет x_offset/y_offset,
           чтобы перейти из координат page crop в координаты исходного изображения.

    Почему offset в конце:
        class_matrix живёт в координатах page crop.
        GT живёт в координатах исходного изображения.
        Значит для IoU нужен bbox offset, но после всей постобработки.
    """
    if class_matrix.size == 0:
        return []

    boxes: List[np.ndarray] = []
    class_points: List[np.ndarray] = []

    max_class = int(np.max(class_matrix))

    MIN_COMPONENT_AREA = 6

    for class_id in range(1, max_class + 1):
        class_mask = (class_matrix == class_id).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            class_mask,
            connectivity=8
        )

        clean_mask = np.zeros_like(class_mask, dtype=np.uint8)

        for label_id in range(1, num_labels):  # 0 — это background
            area = stats[label_id, cv2.CC_STAT_AREA]

            if area >= MIN_COMPONENT_AREA:
                clean_mask[labels == label_id] = 1

        ys, xs = np.where(clean_mask > 0)

        if len(xs) < 3:
            continue

        points_xy = np.column_stack(
            [xs.astype(np.float32), ys.astype(np.float32)]
        )

        rect = cv2.minAreaRect(points_xy)
        box = cv2.boxPoints(rect).astype(np.float32)

        boxes.append(box)
        class_points.append(points_xy)
    if len(boxes) == 0:
        return []

    mean_height = mean_rect_height_min_side(boxes)

    cropped_boxes: List[np.ndarray] = []

    for box, points_xy in zip(boxes, class_points):
        height = rect_height_min_side(box)

        if mean_height > 0 and height > mean_height * tall_factor:
            box = _crop_tall_box_by_own_class_mass(
                box=box,
                class_pixels_xy=points_xy,
                target_height=mean_height,
            )

        cropped_boxes.append(box.astype(np.float32))

    cropped_boxes = _crop_side_page_noise_by_midpoint_clustering(
        boxes=cropped_boxes,
        min_small_cluster_fraction=0.05,
        max_small_cluster_fraction=0.35,
        separation_factor=1.0,
    )

    offset = np.array([float(x_offset), float(y_offset)], dtype=np.float32)

    return [
        box.astype(np.float32) + offset
        for box in cropped_boxes
    ]

# метод 2
def class_matrix_to_center_mass_cropped_polygons(
    class_matrix: np.ndarray,
    x_offset: int,
    y_offset: int,
    min_component_area: int = 6,
) -> List[np.ndarray]:
    """
    Новый метод обработки боксов.

    Идея:
        1. Для каждого class_id удаляем мелкие связные компоненты.
        2. По оставшимся пикселям считаем центр масс.
        3. По оставшимся пикселям строим cv2.minAreaRect.
        4. Обрезаем box по большей удалённой стороне так,
           чтобы центр масс стал центром box по локальной высоте.
        5. Добавляем offset в координаты исходного изображения.
    """
    if class_matrix.size == 0:
        return []

    def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        mask = mask.astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        clean_mask = np.zeros_like(mask, dtype=np.uint8)

        for label_id in range(1, num_labels):  # 0 — background
            area = int(stats[label_id, cv2.CC_STAT_AREA])

            if area >= min_area:
                clean_mask[labels == label_id] = 1

        return clean_mask

    def get_local_axes(box: np.ndarray):
        """
        Для rotated box строим локальные оси:
            u — вдоль длинной стороны;
            v — вдоль короткой стороны, то есть по высоте.
        """
        box = np.asarray(box, dtype=np.float32).reshape(4, 2)

        center = np.mean(box, axis=0).astype(np.float32)

        edges = []
        for i in range(4):
            edge = box[(i + 1) % 4] - box[i]
            length = float(np.linalg.norm(edge))
            edges.append((length, edge))

        long_len, long_edge = max(edges, key=lambda item: item[0])

        if long_len < 1e-6:
            u = np.array([1.0, 0.0], dtype=np.float32)
        else:
            u = (long_edge / long_len).astype(np.float32)

        v = np.array([-u[1], u[0]], dtype=np.float32)

        rel = box - center[None, :]
        s_values = rel @ u
        t_values = rel @ v

        return {
            "center": center,
            "u": u,
            "v": v,
            "s_min": float(np.min(s_values)),
            "s_max": float(np.max(s_values)),
            "t_min": float(np.min(t_values)),
            "t_max": float(np.max(t_values)),
        }

    def make_box_from_local(
        center: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        s_min: float,
        s_max: float,
        t_min: float,
        t_max: float,
    ) -> np.ndarray:
        return np.array(
            [
                center + s_min * u + t_min * v,
                center + s_max * u + t_min * v,
                center + s_max * u + t_max * v,
                center + s_min * u + t_max * v,
            ],
            dtype=np.float32,
        )

    def crop_box_to_make_mass_center_y_center(
        box: np.ndarray,
        points_xy: np.ndarray,
    ) -> np.ndarray:
        """
        Обрезает box так, чтобы центр масс стал центром по локальной высоте.

        Если центр масс ближе к верхней стороне — режем нижнюю.
        Если центр масс ближе к нижней стороне — режем верхнюю.
        """
        box = np.asarray(box, dtype=np.float32).reshape(4, 2)

        if points_xy.shape[0] == 0:
            return box

        axes = get_local_axes(box)

        center = axes["center"]
        u = axes["u"]
        v = axes["v"]

        s_min = axes["s_min"]
        s_max = axes["s_max"]
        t_min = axes["t_min"]
        t_max = axes["t_max"]

        mass_center = np.mean(points_xy.astype(np.float32), axis=0)

        # Координата центра масс по локальной высоте.
        t_mass = float((mass_center - center) @ v)

        if t_mass <= t_min or t_mass >= t_max:
            return box

        dist_to_top = t_mass - t_min
        dist_to_bottom = t_max - t_mass

        # Берём меньшую дистанцию.
        # Большая сторона будет обрезана.
        half_height = min(dist_to_top, dist_to_bottom)

        new_t_min = t_mass - half_height
        new_t_max = t_mass + half_height

        if new_t_max - new_t_min < 2.0:
            return box

        return make_box_from_local(
            center=center,
            u=u,
            v=v,
            s_min=s_min,
            s_max=s_max,
            t_min=new_t_min,
            t_max=new_t_max,
        )

    polygons: List[np.ndarray] = []
    max_class = int(np.max(class_matrix))

    for class_id in range(1, max_class + 1):
        class_mask = (class_matrix == class_id).astype(np.uint8)

        clean_mask = remove_small_components(
            mask=class_mask,
            min_area=min_component_area,
        )

        ys, xs = np.where(clean_mask > 0)

        if len(xs) < 3:
            continue

        points_xy = np.column_stack(
            [
                xs.astype(np.float32),
                ys.astype(np.float32),
            ]
        )

        rect = cv2.minAreaRect(points_xy)
        box = cv2.boxPoints(rect).astype(np.float32)

        box = crop_box_to_make_mass_center_y_center(
            box=box,
            points_xy=points_xy,
        )

        offset = np.array(
            [float(x_offset), float(y_offset)],
            dtype=np.float32,
        )

        polygons.append(box.astype(np.float32) + offset)

    return polygons


# метод 3

def remove_small_components(mask: np.ndarray, min_area: int = 6) -> np.ndarray:
    mask = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    clean = np.zeros_like(mask, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            clean[labels == label_id] = 1

    return clean

# метод 4

def class_matrix_to_top_polygons(
    class_matrix: np.ndarray,
    x_offset: int,
    y_offset: int,
    min_component_area: int = 6,
    keep_pixel_fraction: float = 0.80,
) -> List[np.ndarray]:
    """
    Новый тип постпроцессинга.

    Идея:
        1. Для каждого class_id удаляем мелкие связные компоненты.
        2. По оставшимся пикселям строим cv2.minAreaRect.
        3. Переводим пиксели в локальные координаты box.
        4. Двигаем только верхнюю границу box так,
           чтобы внутри осталось keep_pixel_fraction пикселей строки.
        5. Нижняя, левая и правая границы не меняются.
        6. Добавляем x_offset/y_offset.

    keep_pixel_fraction=0.80:
        значит отрезаем верхние 20% пикселей по локальной высоте.
    """
    if class_matrix.size == 0:
        return []

    def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        mask = mask.astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        clean_mask = np.zeros_like(mask, dtype=np.uint8)

        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])

            if area >= min_area:
                clean_mask[labels == label_id] = 1

        return clean_mask

    def get_local_axes(box: np.ndarray):
        """
        u — вдоль длинной стороны строки.
        v — по высоте строки.
        """
        box = np.asarray(box, dtype=np.float32).reshape(4, 2)

        center = np.mean(box, axis=0).astype(np.float32)

        edges = []
        for i in range(4):
            edge = box[(i + 1) % 4] - box[i]
            length = float(np.linalg.norm(edge))
            edges.append((length, edge))

        long_len, long_edge = max(edges, key=lambda item: item[0])

        if long_len < 1e-6:
            u = np.array([1.0, 0.0], dtype=np.float32)
        else:
            u = (long_edge / long_len).astype(np.float32)

        v = np.array([-u[1], u[0]], dtype=np.float32)

        rel_box = box - center[None, :]
        s_values = rel_box @ u
        t_values = rel_box @ v

        return {
            "center": center,
            "u": u,
            "v": v,
            "s_min": float(np.min(s_values)),
            "s_max": float(np.max(s_values)),
            "t_min": float(np.min(t_values)),
            "t_max": float(np.max(t_values)),
        }

    def make_box_from_local(
        center: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        s_min: float,
        s_max: float,
        t_min: float,
        t_max: float,
    ) -> np.ndarray:
        return np.array(
            [
                center + s_min * u + t_min * v,
                center + s_max * u + t_min * v,
                center + s_max * u + t_max * v,
                center + s_min * u + t_max * v,
            ],
            dtype=np.float32,
        )

    def move_top_border_keep_pixels(
        box: np.ndarray,
        points_xy: np.ndarray,
        keep_fraction: float,
    ) -> np.ndarray:
        """
        Двигает только верхнюю границу box.

        Если keep_fraction=0.80:
            новая верхняя граница = 20-й percentile по t.
            Значит ниже неё остаётся 80% пикселей.
        """
        box = np.asarray(box, dtype=np.float32).reshape(4, 2)

        if points_xy.shape[0] == 0:
            return box

        keep_fraction = float(np.clip(keep_fraction, 0.0, 1.0))

        axes = get_local_axes(box)

        center = axes["center"]
        u = axes["u"]
        v = axes["v"]

        s_min = axes["s_min"]
        s_max = axes["s_max"]
        t_min = axes["t_min"]
        t_max = axes["t_max"]

        rel_points = points_xy.astype(np.float32) - center[None, :]
        t_points = rel_points @ v

        cut_percentile = (1.0 - keep_fraction) * 100.0
        new_t_min = float(np.percentile(t_points, cut_percentile))

        # Защита: верхняя граница не должна выйти за исходный box.
        new_t_min = max(t_min, min(new_t_min, t_max))

        # Защита от схлопывания.
        if t_max - new_t_min < 2.0:
            return box

        return make_box_from_local(
            center=center,
            u=u,
            v=v,
            s_min=s_min,
            s_max=s_max,
            t_min=new_t_min,
            t_max=t_max,
        )

    polygons: List[np.ndarray] = []
    max_class = int(np.max(class_matrix))
    offset = np.array([float(x_offset), float(y_offset)], dtype=np.float32)

    for class_id in range(1, max_class + 1):
        class_mask = (class_matrix == class_id).astype(np.uint8)

        clean_mask = remove_small_components(
            mask=class_mask,
            min_area=min_component_area,
        )

        ys, xs = np.where(clean_mask > 0)

        if len(xs) < 3:
            continue

        points_xy = np.column_stack(
            [
                xs.astype(np.float32),
                ys.astype(np.float32),
            ]
        )

        rect = cv2.minAreaRect(points_xy)
        box = cv2.boxPoints(rect).astype(np.float32)

        box = move_top_border_keep_pixels(
            box=box,
            points_xy=points_xy,
            keep_fraction=keep_pixel_fraction,
        )

        polygons.append(box.astype(np.float32) + offset)

    return polygons

def pca_box_from_points(
    points_xy: np.ndarray,
    q_long_low: float = 1.0,
    q_long_high: float = 99.0,
    q_short_low: float = 0.0,
    q_short_high: float = 100.0,
    min_height: float = 3.0,
) -> np.ndarray:
    """
    Строит rotated box по пикселям строки через PCA.

    points_xy:
        np.ndarray shape=(N, 2), точки [x, y]

    Идея:
        - PCA даёт направление строки;
        - проецируем все точки на ось длины и высоты;
        - берём не min/max, а percentile, чтобы шум не раздувал box.
    """
    points = points_xy.astype(np.float32)

    center = np.mean(points, axis=0)

    centered = points - center[None, :]

    cov = np.cov(centered.T)

    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]

    u = eigvecs[:, order[0]].astype(np.float32)  # вдоль строки
    v = eigvecs[:, order[1]].astype(np.float32)  # по высоте

    # Чтобы направление было стабильнее слева-направо
    if u[0] < 0:
        u = -u
        v = -v

    s = centered @ u
    t = centered @ v

    s_min = float(np.percentile(s, q_long_low))
    s_max = float(np.percentile(s, q_long_high))

    t_min = float(np.percentile(t, q_short_low))
    t_max = float(np.percentile(t, q_short_high))

    if t_max - t_min < min_height:
        mid = 0.5 * (t_min + t_max)
        t_min = mid - min_height / 2.0
        t_max = mid + min_height / 2.0

    box = np.array(
        [
            center + s_min * u + t_min * v,
            center + s_max * u + t_min * v,
            center + s_max * u + t_max * v,
            center + s_min * u + t_max * v,
        ],
        dtype=np.float32,
    )

    return box

def class_matrix_to_pca_detection_boxes(
    class_matrix: np.ndarray,
    x_offset: int,
    y_offset: int,
    min_component_area: int = 6,
    min_points: int = 10,
) -> List[np.ndarray]:
    """
    Конвертирует хорошую сегментацию class_matrix в detection boxes.

    Для каждого class_id:
        1. Берём маску класса.
        2. Удаляем мелкие компоненты.
        3. По оставшимся пикселям строим PCA rotated box.
        4. Добавляем offset страницы.
    """
    if class_matrix.size == 0:
        return []

    polygons: List[np.ndarray] = []

    max_class = int(np.max(class_matrix))
    offset = np.array([float(x_offset), float(y_offset)], dtype=np.float32)

    for class_id in range(1, max_class + 1):
        mask = (class_matrix == class_id).astype(np.uint8)

        clean_mask = remove_small_components(
            mask,
            min_area=min_component_area,
        )

        ys, xs = np.where(clean_mask > 0)

        if len(xs) < min_points:
            continue

        points_xy = np.column_stack(
            [
                xs.astype(np.float32),
                ys.astype(np.float32),
            ]
        )

        box = pca_box_from_points(
            points_xy,
            q_long_low=1.0,
            q_long_high=99.0,
            q_short_low=5.0,
            q_short_high=95.0,
            min_height=3.0,
        )

        polygons.append(box + offset)

    return polygons

# метод 5
def class_matrix_to_pca_top_polygons(
    class_matrix: np.ndarray,
    x_offset: int,
    y_offset: int,
    min_component_area: int = 6,
    keep_pixel_fraction: float = 0.80,
    min_points: int = 3,
    min_height: float = 2.0,
) -> List[np.ndarray]:
    """
    Новый тип постпроцессинга.

    Идея:
        1. Для каждого class_id чистим мелкие связные компоненты.
        2. По оставшимся пикселям получаем направление строки через PCA.
        3. На основе PCA строим rotated rectangle, который содержит
           все очищенные пиксели класса.
        4. Двигаем только верхнюю границу box так,
           чтобы внутри осталось keep_pixel_fraction пикселей строки.
        5. Нижнюю, левую и правую границы не трогаем.
        6. Добавляем offset в координаты исходного изображения.
    """
    if class_matrix.size == 0:
        return []

    def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        mask = mask.astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        clean_mask = np.zeros_like(mask, dtype=np.uint8)

        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])

            if area >= min_area:
                clean_mask[labels == label_id] = 1

        return clean_mask

    def pca_full_box_from_points(points_xy: np.ndarray) -> Dict[str, Any]:
        """
        Через PCA получаем оси:
            u — вдоль строки;
            v — по высоте строки.

        Потом строим box, который содержит все points_xy:
            s_min/s_max = min/max проекции на u
            t_min/t_max = min/max проекции на v
        """
        points = points_xy.astype(np.float32)

        center = np.mean(points, axis=0).astype(np.float32)
        centered = points - center[None, :]

        # Если точек слишком мало или они вырождены,
        # np.cov/eigh может дать нестабильный результат.
        if points.shape[0] < 3:
            u = np.array([1.0, 0.0], dtype=np.float32)
            v = np.array([0.0, 1.0], dtype=np.float32)
        else:
            cov = np.cov(centered.T)

            if not np.all(np.isfinite(cov)):
                u = np.array([1.0, 0.0], dtype=np.float32)
                v = np.array([0.0, 1.0], dtype=np.float32)
            else:
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = np.argsort(eigvals)[::-1]

                u = eigvecs[:, order[0]].astype(np.float32)
                v = eigvecs[:, order[1]].astype(np.float32)

                # Стабилизируем направление: u смотрит слева направо.
                if u[0] < 0:
                    u = -u
                    v = -v

        s_values = centered @ u
        t_values = centered @ v

        s_min = float(np.min(s_values))
        s_max = float(np.max(s_values))
        t_min = float(np.min(t_values))
        t_max = float(np.max(t_values))

        if t_max - t_min < min_height:
            t_mid = 0.5 * (t_min + t_max)
            t_min = t_mid - min_height / 2.0
            t_max = t_mid + min_height / 2.0

        return {
            "center": center,
            "u": u,
            "v": v,
            "s_min": s_min,
            "s_max": s_max,
            "t_min": t_min,
            "t_max": t_max,
            "s_values": s_values,
            "t_values": t_values,
        }

    def make_box_from_local(
        center: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        s_min: float,
        s_max: float,
        t_min: float,
        t_max: float,
    ) -> np.ndarray:
        return np.array(
            [
                center + s_min * u + t_min * v,
                center + s_max * u + t_min * v,
                center + s_max * u + t_max * v,
                center + s_min * u + t_max * v,
            ],
            dtype=np.float32,
        )

    def move_top_border_keep_fraction(
        geom: Dict[str, Any],
        keep_fraction: float,
    ) -> np.ndarray:
        """
        Двигает только верхнюю границу.

        Если keep_fraction = 0.80:
            cut_percentile = 20
            новая верхняя граница = 20-й percentile по t_values

        Тогда 80% пикселей имеют t >= new_t_min,
        то есть остаются внутри box.
        """
        keep_fraction = float(np.clip(keep_fraction, 0.0, 1.0))

        center = geom["center"]
        u = geom["u"]
        v = geom["v"]

        s_min = geom["s_min"]
        s_max = geom["s_max"]
        old_t_min = geom["t_min"]
        t_max = geom["t_max"]
        t_values = geom["t_values"]

        cut_percentile = (1.0 - keep_fraction) * 100.0
        new_t_min = float(np.percentile(t_values, cut_percentile))

        # Двигаем только вверхнюю границу внутрь старого box.
        new_t_min = max(old_t_min, min(new_t_min, t_max))

        # Защита от схлопывания.
        if t_max - new_t_min < min_height:
            new_t_min = old_t_min

        return make_box_from_local(
            center=center,
            u=u,
            v=v,
            s_min=s_min,
            s_max=s_max,
            t_min=new_t_min,
            t_max=t_max,
        )

    polygons: List[np.ndarray] = []
    offset = np.array([float(x_offset), float(y_offset)], dtype=np.float32)

    max_class = int(np.max(class_matrix))

    for class_id in range(1, max_class + 1):
        class_mask = (class_matrix == class_id).astype(np.uint8)

        clean_mask = remove_small_components(
            mask=class_mask,
            min_area=min_component_area,
        )

        ys, xs = np.where(clean_mask > 0)

        if len(xs) < min_points:
            continue

        points_xy = np.column_stack(
            [
                xs.astype(np.float32),
                ys.astype(np.float32),
            ]
        )

        geom = pca_full_box_from_points(points_xy)

        box = move_top_border_keep_fraction(
            geom=geom,
            keep_fraction=keep_pixel_fraction,
        )

        polygons.append(box.astype(np.float32) + offset)

    return polygons

# метод 6

def class_matrix_to_morph_pca_polygons_debug(
    class_matrix: np.ndarray,
    x_offset: int,
    y_offset: int,
    min_component_area: int = 6,
    min_points: int = 3,
    kernel_size: int = 3,
    morph_iterations: int = 2,
    min_height: float = 2.0,
    debug_dir: Path | None = None,
    debug_name: str = "morph_pca",
) -> List[np.ndarray]:
    """
    Постпроцессинг class_matrix с debug.

    Что делает:
        1. Для каждого class_id берёт маску.
        2. Удаляет мелкие компоненты.
        3. Делает dilation -> erosion.
        4. Ещё раз удаляет мелкие компоненты.
        5. Собирает новую morph_class_matrix после операций.
        6. По morph_mask считает PCA.
        7. Строит rotated rectangle по PCA.
        8. Сохраняет debug:
            - class_matrix_before.png
            - class_matrix_after_morph.png
            - rects_on_morph_matrix.png
        9. Возвращает polygons с offset.
    """
    if class_matrix.size == 0:
        return []

    def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        mask = mask.astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        clean_mask = np.zeros_like(mask, dtype=np.uint8)

        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])

            if area >= min_area:
                clean_mask[labels == label_id] = 1

        return clean_mask

    def iterative_erosion_dilation(
            mask: np.ndarray,
            kernel_size: int,
            iterations: int,
        ) -> np.ndarray:
            """
            Итеративно делает erosion -> dilation.

            mask:
                1 = текущий class_id / строка
                0 = фон

            erosion уменьшает строку:
                - убирает мелкий шум;
                - откусывает тонкие выступы;
                - может разорвать слабые мостики.

            dilation потом расширяет строку обратно:
                - возвращает размер реальных компонент;
                - частично восстанавливает толщину.

            Последовательность erosion -> dilation = opening.
            """
            mask = mask.astype(np.uint8)

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            )

            result = mask.copy()

            result = cv2.erode(result, kernel, iterations=iterations)
            result = cv2.dilate(result, kernel, iterations=iterations)

            return result

    def pca_minrect_from_points(points_xy: np.ndarray) -> np.ndarray:
        points = points_xy.astype(np.float32)

        center = np.mean(points, axis=0).astype(np.float32)
        centered = points - center[None, :]

        if points.shape[0] < 3:
            u = np.array([1.0, 0.0], dtype=np.float32)
            v = np.array([0.0, 1.0], dtype=np.float32)
        else:
            cov = np.cov(centered.T)

            if not np.all(np.isfinite(cov)):
                u = np.array([1.0, 0.0], dtype=np.float32)
                v = np.array([0.0, 1.0], dtype=np.float32)
            else:
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = np.argsort(eigvals)[::-1]

                u = eigvecs[:, order[0]].astype(np.float32)
                v = eigvecs[:, order[1]].astype(np.float32)

                if u[0] < 0:
                    u = -u
                    v = -v

        s_values = centered @ u
        t_values = centered @ v

        s_min = float(np.min(s_values))
        s_max = float(np.max(s_values))
        t_min = float(np.min(t_values))
        t_max = float(np.max(t_values))

        if t_max - t_min < min_height:
            t_mid = 0.5 * (t_min + t_max)
            t_min = t_mid - min_height / 2.0
            t_max = t_mid + min_height / 2.0

        box = np.array(
            [
                center + s_min * u + t_min * v,
                center + s_max * u + t_min * v,
                center + s_max * u + t_max * v,
                center + s_min * u + t_max * v,
            ],
            dtype=np.float32,
        )

        return box

    def colorize_class_matrix(matrix: np.ndarray) -> np.ndarray:
        """
        Делает цветную картинку из class_matrix.

        0 = чёрный фон.
        class_id > 0 = псевдоцвет.
        """
        matrix = matrix.astype(np.int32)
        h, w = matrix.shape

        color = np.zeros((h, w, 3), dtype=np.uint8)

        ids = np.unique(matrix)
        ids = ids[ids > 0]

        for class_id in ids:
            # Детерминированный псевдоцвет по class_id.
            r = (37 * int(class_id) + 17) % 255
            g = (97 * int(class_id) + 53) % 255
            b = (157 * int(class_id) + 91) % 255

            color[matrix == class_id] = (b, g, r)

        return color

    polygons: List[np.ndarray] = []
    offset = np.array([float(x_offset), float(y_offset)], dtype=np.float32)

    h, w = class_matrix.shape[:2]
    morph_class_matrix = np.zeros((h, w), dtype=np.int32)

    max_class = int(np.max(class_matrix))

    # box-ы без offset — для рисования на morph_class_matrix
    boxes_local: List[np.ndarray] = []

    for class_id in range(1, max_class + 1):
        class_mask = (class_matrix == class_id).astype(np.uint8)

        clean_mask = remove_small_components(
            mask=class_mask,
            min_area=min_component_area,
        )

        morph_mask = iterative_erosion_dilation(
            mask=clean_mask,
            kernel_size=kernel_size,
            iterations=morph_iterations,
        )

        morph_mask = remove_small_components(
            mask=morph_mask,
            min_area=min_component_area,
        )

        # ВАЖНО:
        # Собираем новую class_matrix после операций.
        # Если после dilation/erosion классы где-то пересеклись,
        # более поздний class_id перезапишет предыдущий.
        morph_class_matrix[morph_mask > 0] = class_id

        ys, xs = np.where(morph_mask > 0)

        if len(xs) < min_points:
            continue

        points_xy = np.column_stack(
            [
                xs.astype(np.float32),
                ys.astype(np.float32),
            ]
        )

        box = pca_minrect_from_points(points_xy)

        boxes_local.append(box.astype(np.float32))
        polygons.append(box.astype(np.float32) + offset)

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        before_img = colorize_class_matrix(class_matrix)
        after_img = colorize_class_matrix(morph_class_matrix)

        rect_img = after_img.copy()

        for box in boxes_local:
            pts = np.round(box).astype(np.int32).reshape(-1, 1, 2)

            cv2.polylines(
                rect_img,
                [pts],
                isClosed=True,
                color=(0, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            center = np.mean(box, axis=0)
            cv2.circle(
                rect_img,
                tuple(np.round(center).astype(np.int32)),
                radius=3,
                color=(0, 0, 255),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        cv2.imwrite(
            str(debug_dir / f"{debug_name}_class_matrix_before.png"),
            before_img,
        )

        cv2.imwrite(
            str(debug_dir / f"{debug_name}_class_matrix_after_morph.png"),
            after_img,
        )

        cv2.imwrite(
            str(debug_dir / f"{debug_name}_rects_on_morph_matrix.png"),
            rect_img,
        )

        # Удобная общая картинка рядом: before | after | rects
        concat_img = np.concatenate([before_img, after_img, rect_img], axis=1)

        cv2.imwrite(
            str(debug_dir / f"{debug_name}_debug_panel.png"),
            concat_img,
        )

    return polygons