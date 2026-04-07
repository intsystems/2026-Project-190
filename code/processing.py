import cv2
import numpy as np
import os
from u_net_binarization import binarize_image
from post_processing import crop_line_rectangle
import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import time
from contextlib import contextmanager

# Используя yolo модель выделяем маску стрниц, бинаризуем изображени поворачивам соотвествующие станницы  разрезам

def extract_pages_with_yolo(image_path, model_path, output_dir="debug_images", conf_threshold=0.7, debug=False):
    os.makedirs(output_dir, exist_ok=True)

    # Загрузка изображения
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить: {image_path}")
    else:
        img = image_path.copy()

    # Бинаризация
    binary = binarize_image(img)
    if debug:
        cv2.imwrite(os.path.join(output_dir, "01_binary.jpg"), binary)

    # YOLO
    model = YOLO(model_path)
    results = model(img, conf=conf_threshold, verbose=False)
    if not (results and len(results) > 0 and results[0].masks is not None):
        print("Маски не найдены")
        return []

    annotated = results[0].plot()
    if debug:
        cv2.imwrite(os.path.join(output_dir, "02_yolo_result.jpg"), annotated)

    masks = results[0].masks.data.cpu().numpy()
    orig_h, orig_w = binary.shape
    pages = []

    for i, mask in enumerate(masks):
        mask_bin = (mask > 0.5).astype(np.uint8) * 255
        mask_big = cv2.resize(mask_bin, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        if debug:
            cv2.imwrite(os.path.join(output_dir, f"03_mask_big_{i}.jpg"), mask_big)

        contours, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        # Вырезаем из *цветного* изображения
        page_roi = img[y:y+h, x:x+w]
        mask_roi = mask_big[y:y+h, x:x+w] 

        # Создаём белый фон размером с вырезанную область
        white_bg = np.full_like(page_roi, 255)   # (h, w, 3) все пиксели = 255

        # Накладываем цветные пиксели только там, где маска == 255
        mask_3ch = np.stack([mask_roi] * 3, axis=-1)  # (h, w, 3) для поэлементного выбора
        result_page = np.where(mask_3ch == 255, page_roi, white_bg)

        if debug:
            cv2.imwrite(os.path.join(output_dir, f"07_page_{i}.jpg"), result_page)
        pages.append(result_page)

    return pages


# Значит тут пытаемся с помощью hpp найти разрез - фигня

#  if len(img.shape) == 3:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = img.copy()
#     binary = binarize_image(gray)
#     result = detect_spread_or_single(binary)
#     if result['is_spread']:
#         print(f"Разворот, оптимальный угол: {result['best_angle']}°")
#         cv2.imwrite("debug_images/Left page.jpg", result['left_img'])
#         cv2.imwrite("debug_images/Right page.jpg", result['right_img'])
#     else:
#         print("Одна страница")

def detect_spread_or_single(binary_img, angle_step=10, angle_range=90):
    """
    Определяет, является ли бинарное изображение разворотом двух страниц или одной страницей.

    Параметры:
    ----------
    binary_img : np.ndarray
        Бинарное изображение (0 - чёрный, 255 - белый)
    angle_step : int
        Шаг поворота в градусах (по умолчанию 3)
    angle_range : int
        Диапазон углов от -angle_range до angle_range (по умолчанию 15)

    Возвращает:
    -----------
    dict: {
        'is_spread': bool,      # True если разворот, False если одна страница
        'best_angle': float,    # угол, при котором найден наилучший разрез
        'left_img': np.ndarray или None,   # левая страница (если разворот)
        'right_img': np.ndarray или None   # правая страница
    }
    """
    h, w = binary_img.shape
    best_angle = 0
    best_score = 0
    best_center_peak = None

    # Перебираем углы
    for angle in range(-angle_range, angle_range + 1, angle_step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rotated = cv2.warpAffine(binary_img, M, (w, h), borderValue=255)

        cv2.imwrite(f"debug_images/rotated{angle}.jpg", rotated)

        vertical_profile = np.sum(rotated == 255, axis=0)

        kernel = np.ones(5) / 5
        smoothed = np.convolve(vertical_profile, kernel, mode='same')

        threshold = np.max(smoothed) * 0.3
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1] and smoothed[i] > threshold:
                peaks.append(i)
        if len(peaks) >= 3:
            center_candidate = min(peaks, key=lambda x: abs(x - w / 2))
            left_candidate = min(peaks, key=lambda x: abs(x - w * 0.25))
            right_candidate = min(peaks, key=lambda x: abs(x - w * 0.75))

            if (abs(left_candidate - w * 0.25) < w * 0.1 and
                abs(center_candidate - w * 0.5) < w * 0.1 and
                abs(right_candidate - w * 0.75) < w * 0.1):
                score = (1 - abs(left_candidate - w*0.25)/(w*0.1)) + \
                        (1 - abs(center_candidate - w*0.5)/(w*0.1)) + \
                        (1 - abs(right_candidate - w*0.75)/(w*0.1))
                if score > best_score:
                    best_score = score
                    best_angle = angle
                    best_center_peak = center_candidate

    if best_score > 2.0:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), best_angle, 1)
        rotated = cv2.warpAffine(binary_img, M, (w, h), borderValue=255)

        left_img = rotated[:, :best_center_peak]
        right_img = rotated[:, best_center_peak:]
        return {
            'is_spread': True,
            'best_angle': best_angle,
            'left_img': left_img,
            'right_img': right_img
        }
    else:
        return {
            'is_spread': False,
            'best_angle': None,
            'left_img': None,
            'right_img': None
        }

# Значит тут пытаемся с помощью размытия гауса найти старицы - фигня так как школота может писать не на всей странице

def correct_perspective(image, binary=None, debug=False, correct_global_angle=False):
    """
    Исправляет перспективные искажения (наклон) изображения, находя самый большой
    четырёхугольный контур и выпрямляя его во фронтальный вид.
    """
    if debug:
        os.makedirs('debug_images', exist_ok=True)

    # 1. Предобработка для поиска контуров
    # Проверяем, цветное ли изображение
    if binary == None:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        binary = binarize_image(gray)
    median = cv2.medianBlur(binary, 17)

    if debug:
        cv2.imwrite('debug_images/binary.jpg', binary)
        cv2.imwrite('debug_images/Median_Blur.jpg', median)


    black_coords = set((y, x) for x, y in np.argwhere(median == 0))
    start_time = time.time()
    straightened = crop_line_rectangle(image, black_coords, debug=debug, robust=False, correct_global_angle=correct_global_angle)
    end_time = time.time()
    print(end_time - start_time)
    return straightened

# Значит это кореция шума - бесполезная вешь u-net как и otse работает ещё более фиговие (размывается разница между
# текстом и синими линиями) - в общем такое

def normalize_illumination(image, clip_limit=2.0, tile_grid_size=(8,8), gamma=0.3):
    """
    Выравнивает неравномерную освещённость и ослабляет засвеченные участки.

    Параметры:
        image: входное изображение (BGR или grayscale)
        clip_limit: порог ограничения контраста для CLAHE (больше – сильнее контраст)
        tile_grid_size: размер сетки (число тайлов по высоте и ширине)
        gamma: показатель гамма-коррекции (<1 осветляет тени, >1 затемняет)

    Возвращает:
        normalized: изображение с нормализованной освещённостью (uint8)
    """
    # Если изображение цветное, переводим в LAB и работаем только с каналом яркости
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_norm = _apply_clahe_and_gamma(l, clip_limit, tile_grid_size, gamma)
        lab_norm = cv2.merge((l_norm, a, b))
        normalized = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
    else:
        normalized = _apply_clahe_and_gamma(image, clip_limit, tile_grid_size, gamma)

    return normalized

def _apply_clahe_and_gamma(channel, clip_limit, tile_grid_size, gamma):
    # CLAHE – выравнивание локальной гистограммы
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    equalized = clahe.apply(channel)

    # Гамма-коррекция для подавления засветов
    look_up = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    corrected = cv2.LUT(equalized, look_up)
    return corrected

if __name__ == "__main__":
    img_path = 'datasets/school_notebooks_RU/images_base/2_28.JPG'
    img = cv2.imread(img_path)

    #Нормализация освещения
    img_corrected = normalize_illumination(img, clip_limit=4, gamma=0.2)

    #Исправление наклона
    start_time = time.time()
    img_final = correct_perspective(img)
    end_time = time.time()
    print(end_time - start_time)
    cv2.imwrite('debug_images/img_final.jpg', img_final)
    cv2.imwrite('debug_images/img_corrected.jpg', img_corrected)

    # pages = extract_pages_with_yolo(
    #     image_path='datasets/school_notebooks_RU/images_base/2_31.JPG',
    #     model_path='models/yolo_detect_notebook/yolo_detect_notebook_1_(1-architecture).pt',
    #     output_dir='debug_images',
    #     conf_threshold=0.7
    # )
    # for idx, page in enumerate(pages):
    #     img_final = correct_perspective(page, debug=True, correct_global_angle=True)
    #     cv2.imwrite(f'debug_images/img_final{idx}.jpg', img_final)
    #     print(f"Страница {idx+1}: размер {page.shape}")