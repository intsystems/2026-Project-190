import cv2
import numpy as np
from multiprocessing import Pool
import os
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from u_net_binarization import binarize_image

# Пытаюсь рандомно выбирать точки на бинарном изображении чтоб получить мин рект притом робастый

def _ransac_iteration(args):
    """Вспомогательная функция для одной итерации RANSAC (запускается в отдельном процессе)."""
    points, sample_size, inlier_threshold = args
    # Случайная выборка
    idx = np.random.choice(len(points), sample_size, replace=False)
    sample = points[idx]
    rect = cv2.minAreaRect(sample)
    box = cv2.boxPoints(rect).astype(np.float32)
    inliers = 0
    for pt in points:
        if point_to_polygon_distance(pt, box) < inlier_threshold:
            inliers += 1
    return (inliers, rect)

def robust_min_area_rect(points, outlier_ratio=0.05, ransac_iter=20, inlier_threshold=3.0, num_processes=20):
    """
    Находит повернутый прямоугольник, устойчивый к выбросам, используя RANSAC с распараллеливанием.
    """
    if len(points) < 4:
        return cv2.minAreaRect(points)

    # Определяем размер выборки
    sample_size = max(5, min(20, int(0.05 * len(points))))
    sample_size = min(sample_size, len(points))

    # Создаём список аргументов для каждой итерации
    args_list = [(points, sample_size, inlier_threshold) for _ in range(ransac_iter)]

    # Распараллеливаем с помощью пула процессов
    with Pool(processes=num_processes) as pool:
        results = pool.map(_ransac_iteration, args_list)

    # Находим итерацию с максимальным количеством inliers
    best_inliers, best_rect = max(results, key=lambda x: x[0])

    # Проверяем долю inliers
    if best_inliers / len(points) >= (1 - outlier_ratio):
        # Пересчитываем прямоугольник по всем inliers
        box = cv2.boxPoints(best_rect).astype(np.float32)
        inlier_pts = []
        for pt in points:
            if point_to_polygon_distance(pt, box) < inlier_threshold:
                inlier_pts.append(pt)
        if len(inlier_pts) >= 4:
            refined_rect = cv2.minAreaRect(np.array(inlier_pts))
            return refined_rect
        return best_rect
    else:
        return cv2.minAreaRect(points)

def point_to_polygon_distance(pt, polygon):
    """
    Вычисляет минимальное расстояние от точки pt (x, y) до контура polygon (4 точки).
    polygon: массив (4,2) углов прямоугольника.
    """
    pt = np.array(pt, dtype=np.float32)
    min_dist = float('inf')
    for i in range(4):
        p1 = polygon[i]
        p2 = polygon[(i+1)%4]
        # Расстояние от точки до отрезка
        dist = point_to_segment_distance(pt, p1, p2)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def point_to_segment_distance(pt, p1, p2):
    """
    Расстояние от точки pt до отрезка (p1-p2).
    """
    # Вектор отрезка
    seg_vec = p2 - p1
    # Вектор от p1 до pt
    pt_vec = pt - p1
    seg_len_sq = np.dot(seg_vec, seg_vec)
    if seg_len_sq == 0:
        return np.linalg.norm(pt_vec)
    # Проекция
    t = np.dot(pt_vec, seg_vec) / seg_len_sq
    if t < 0:
        nearest = p1
    elif t > 1:
        nearest = p2
    else:
        nearest = p1 + t * seg_vec
    return np.linalg.norm(pt - nearest)

# Пытаемся с помощью анализа hpp белых пикселей найти разрез (работаем с корекцией поворотов 0 90 180 270)
# Вот так использовать (p.s работает фигово)

# gray = cv2.cvtColor(straightened, cv2.COLOR_BGR2GRAY)
# binary = binarize_image(gray)
# global_angle = detect_best_orientation(binary)
# cv2.imwrite('debug_images/binary_straightened.jpg', binary)
# if global_angle != 0:
#     if global_angle == 90:
#         straightened = cv2.rotate(straightened, cv2.ROTATE_90_CLOCKWISE)
#     elif global_angle == -90:
#         straightened = cv2.rotate(straightened, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     elif global_angle == 180:
#         straightened = cv2.rotate(straightened, cv2.ROTATE_180)


def count_lines(binary, angle):
    """Возвращает количество строк текста для заданного поворота (0, 90, 180, 270)."""
    if angle == 0:
        img = binary
    elif angle == 90:
        img = cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        img = cv2.rotate(binary, cv2.ROTATE_180)
    elif angle == 270:
        img = cv2.rotate(binary, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return 0

    # Горизонтальная проекция (сумма чёрных пикселей по строкам)
    row_sum = (img == 255).sum(axis=1).astype(float) # считаем по беллым и это гениално. По вертикалебудет мало а по горезонтале много так как промежутки между строок
    if row_sum.max() == 0:
        return 0
    row_sum /= row_sum.max()
    row_sum = median_filter(row_sum, size=5)

    # Поиск пиков (строк) с минимальной высотой и расстоянием
    peaks = find_peaks(row_sum, prominence=0.1, distance=5)[0]
    return len(peaks)

def detect_best_orientation(binary):
    """
    Определяет, на сколько градусов нужно повернуть изображение,
    чтобы текст стал горизонтальным и не перевёрнутым.
    Возвращает угол: 0, 90, 180, -90 (270 против часовой).
    """
    counts = {0: count_lines(binary, 0),
              90: count_lines(binary, 90)} # ,180: count_lines(binary, 180), 270: count_lines(binary, 270)
    print("Количество строк:", counts)  # для отладки

    # Выбираем ориентацию с максимальным числом строк
    best = max(counts, key=counts.get)

    # Если лучшие 0 и 180 дают одинаковое количество (горизонтальные ориентации)
    if best == 180 and counts[180] == counts[0] and counts[0] >= max(counts[90], counts[270]):
        # Дополнительная проверка: смотрим, где находится первая строка
        def first_peak_y(b):
            row_sum = (b == 255).sum(axis=1).astype(float)
            if row_sum.max() == 0:
                return 0
            row_sum /= row_sum.max()
            row_sum = median_filter(row_sum, size=5)
            peaks = find_peaks(row_sum, prominence=0.1, distance=5)[0]
            return peaks[0] if len(peaks) > 0 else 0

        first_0 = first_peak_y(binary)
        first_180 = first_peak_y(cv2.rotate(binary, cv2.ROTATE_180))
        if first_0 > first_180:
            best = 0
        else:
            best = 0

    if best == 0:
        return 0
    elif best == 90:
        return -90
    elif best == 180:
        return 180
    elif best == 270:
        return 90



def crop_line_rectangle(image: np.ndarray, line_pixels: set, debug: bool, padding: int = 100, robust: bool = False
                        , correct_global_angle: bool = False) -> np.ndarray:
        """
        Строит минимальный повёрнутый прямоугольник ТОЛЬКО по текстовым пикселям строки,
        поворачивает изображение так, чтобы строка стала горизонтальной,
        и возвращает чистое выпрямленное изображение строки.

        СНАЧАЛО y потом x. То есть пара (y, x) а не (x, y)!!!!!!!!
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
        rect = None
        if not robust:
            rect = cv2.minAreaRect(points)
            (center_x, center_y), (width, height), angle = rect
        else:
            rect = robust_min_area_rect(points)
            (center_x, center_y), (width, height), angle = rect


        if debug:
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            image_deg = image.copy()

            cv2.drawContours(image_deg, [box], 0, (0, 0, 255), thickness=3)

            # Показываем результат
            cv2.imwrite('debug_images/result_rect.jpg', image_deg)

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

        h, w = straightened.shape[:2]
        if h > w:
            straightened = cv2.rotate(straightened, cv2.ROTATE_90_CLOCKWISE)

        if correct_global_angle:
            gray = cv2.cvtColor(straightened, cv2.COLOR_BGR2GRAY)
            binary = binarize_image(gray)
            global_angle = detect_best_orientation(binary)
            if global_angle != 0:
                if global_angle == 90:
                    straightened = cv2.rotate(straightened, cv2.ROTATE_90_CLOCKWISE)
                elif global_angle == -90:
                    straightened = cv2.rotate(straightened, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif global_angle == 180:
                    straightened = cv2.rotate(straightened, cv2.ROTATE_180)

        return straightened