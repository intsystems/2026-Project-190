import os
import sys
import cv2
import numpy as np
import optuna
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
from typing import List, Dict, Tuple, Any
from contextlib import contextmanager
import tempfile

from processing import extract_pages_with_yolo
from hough_transform_method import TextLineDetector
from post_processing import crop_line_rectangle   # предполагается, что эта функция доступна

# ----------------------------------------------------------------------
# Подавление вывода
# ----------------------------------------------------------------------
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# ----------------------------------------------------------------------
# Загрузка ground truth (вырезанные бинарные изображения строк)
# ----------------------------------------------------------------------
def load_ground_truth_masks(gt_dir: str) -> List[np.ndarray]:
    """
    Загружает все бинарные маски строк из папки gt_dir.
    Каждая маска – 2D массив (uint8) с текстом = 0, фон = 255.
    """
    masks = []
    for fname in sorted(os.listdir(gt_dir)):
        if not fname.endswith('.png'):
            continue
        mask = cv2.imread(os.path.join(gt_dir, fname), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks.append(mask)
    return masks

def iou_binary_images(img1: np.ndarray, img2: np.ndarray, target_size: Tuple[int, int] = (64, 64)) -> float:
    """
    Вычисляет IoU между двумя бинарными изображениями (0/255).
    Приводит их к единому размеру target_size (интерполяция ближайшего соседа).
    """
    if img1.size == 0 or img2.size == 0:
        return 0.0
    # Изменяем размер
    h, w = target_size
    img1_resized = cv2.resize(img1, (w, h), interpolation=cv2.INTER_NEAREST)
    img2_resized = cv2.resize(img2, (w, h), interpolation=cv2.INTER_NEAREST)
    # Бинаризация (на всякий случай)
    _, img1_bin = cv2.threshold(img1_resized, 127, 1, cv2.THRESH_BINARY)
    _, img2_bin = cv2.threshold(img2_resized, 127, 1, cv2.THRESH_BINARY)
    inter = np.logical_and(img1_bin, img2_bin).sum()
    union = np.logical_or(img1_bin, img2_bin).sum()
    return inter / union if union > 0 else 0.0

def evaluate_image(fname: str, img: np.ndarray, gt_masks: List[np.ndarray],
                   model_path: str, params: Dict[str, Any]) -> float:
    """
    Обрабатывает изображение, получает страницы, для каждой страницы детектирует строки,
    вырезает их (как crop_line_rectangle) и сравнивает с ground truth.
    Возвращает средний IoU по всем страницам.
    """
    # Получаем страницы и их bbox (предполагается, что extract_pages_with_yolo возвращает bbox)
    pages, bboxes = extract_pages_with_yolo(
        image_path=img,
        model_path=model_path,
        output_dir='debug_optuna',
        conf_threshold=0.7,
        return_bboxes=True
    )
    if not pages:
        return 0.0

    total_score = 0.0
    n_pages = 0
    for page, bbox in zip(pages, bboxes):
        with suppress_stdout():
            detector = TextLineDetector(page, params=params, debug=False)
            detector.detect_text_lines()
            # Получаем маски строк (пиксели) из детектора (он должен сохранять их в line_masks)
            if hasattr(detector, 'line_masks'):
                pred_points_list = [np.array(pts, dtype=np.int32) for pts in detector.line_masks if pts]
            else:
                # Если нет line_masks, получаем через _create_colored_segmentation
                _, line_masks = detector._create_colored_segmentation()
                pred_points_list = [np.array(pts, dtype=np.int32) for pts in line_masks if pts]

        if not pred_points_list:
            continue

        # Для каждой предсказанной строки вырезаем её бинарное изображение
        pred_images = []
        for pts in pred_points_list:
            # Создаём белое изображение размером со страницу, рисуем чёрные точки
            white_bg = np.ones((page.shape[0], page.shape[1]), dtype=np.uint8) * 255
            for (x, y) in pts:
                if 0 <= y < white_bg.shape[0] and 0 <= x < white_bg.shape[1]:
                    white_bg[y, x] = 0
            # Вырезаем и выпрямляем строку
            crop = crop_line_rectangle(white_bg, pts, debug=False, padding=5)
            if crop is not None and crop.size > 0:
                pred_images.append(crop)

        if not pred_images or not gt_masks:
            continue

        # Сравниваем предсказанные изображения с ground truth (матрица IoU)
        n_pred = len(pred_images)
        n_gt = len(gt_masks)
        cost = np.zeros((n_pred, n_gt))
        for i, p_img in enumerate(pred_images):
            for j, g_img in enumerate(gt_masks):
                cost[i, j] = -iou_binary_images(p_img, g_img)
        row, col = linear_sum_assignment(cost)
        total_iou = -cost[row, col].sum()
        page_score = total_iou / max(n_pred, n_gt)
        total_score += page_score
        n_pages += 1

    return total_score / n_pages if n_pages > 0 else 0.0

# ----------------------------------------------------------------------
# Оптимизируемая функция (без изменений, кроме вызова evaluate_image)
# ----------------------------------------------------------------------
def objective(trial, images_dict, ground_truth_masks, model_path, fixed_params):
    params = fixed_params.copy()
    # ... (все suggest_* такие же, как в предыдущем коде) ...
    # Для краткости опускаю, но они должны быть здесь

    def process_one(fname, img, gt_masks):
        return evaluate_image(fname, img, gt_masks, model_path, params)

    results = Parallel(n_jobs=-1)(
        delayed(process_one)(fname, img, ground_truth_masks[fname])
        for fname, img in images_dict.items() if fname in ground_truth_masks
    )
    return -np.mean(results)

# ----------------------------------------------------------------------
# Главная функция
# ----------------------------------------------------------------------
def main():
    images_dir = "datasets/school_notebooks_RU/images_base"
    target_base = "datasets/school_notebooks_RU/images_target"
    model_path = "models/yolo_detect_notebook/yolo_detect_notebook_1_(1-architecture).pt"

    images_dict = {}
    for fname in os.listdir(images_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(images_dir, fname))
            if img is not None:
                images_dict[fname] = img

    ground_truth_masks = {}
    for fname in images_dict:
        base = os.path.splitext(fname)[0]
        gt_dir = os.path.join(target_base, base)
        if os.path.isdir(gt_dir):
            masks = load_ground_truth_masks(gt_dir)
            if masks:
                ground_truth_masks[fname] = masks

    print(f"Найдено изображений: {len(images_dict)}")
    print(f"Из них с ground truth: {len(ground_truth_masks)}")

    fixed_params = {'binarization_method': 'u_net'}
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(trial, images_dict, ground_truth_masks, model_path, fixed_params),
        n_trials=50, n_jobs=1, show_progress_bar=True
    )
    print("Лучшие параметры:", study.best_params)
    print("Лучший IoU:", -study.best_value)

if __name__ == "__main__":
    main()