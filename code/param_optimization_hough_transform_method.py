import os
import sys
import cv2
import numpy as np
import optuna
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
from typing import List, Tuple, Dict, Any
import multiprocessing
import gc
from processing import extract_pages_with_yolo
from hough_transform_method import TextLineDetector
from post_processing import crop_line_rectangle
import joblib

def ensure_grayscale(img):
    if len(img.shape) == 2:
        return img
    return img[:, :, 0]

# Загрузка размеченных истенных строк
def load_ground_truth_masks(gt_dir: str) -> List[np.ndarray]:
    masks = []
    for fname in sorted(os.listdir(gt_dir)):
        if not fname.endswith('.png'):
            continue
        mask = cv2.imread(os.path.join(gt_dir, fname), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks.append(mask)

    masks = [ensure_grayscale(c) for c in masks if c is not None and c.size > 0]

    return masks

# Поворот бинарного изображения
def rotate_binary_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
    return rotated

# IoU двух бинарных изображений приводим к единому размеру
def iou_binary_images(img1: np.ndarray, img2: np.ndarray, target_size: Tuple[int, int] = (32, 256)) -> float:
    if img1.size == 0 or img2.size == 0:
        return 0.0
    h, w = target_size
    img1_resized = cv2.resize(img1, (w, h), interpolation=cv2.INTER_NEAREST)
    img2_resized = cv2.resize(img2, (w, h), interpolation=cv2.INTER_NEAREST)
    # cv2.imwrite('debug_images/img1_resized.jpg', img1_resized)
    # cv2.imwrite('debug_images/img2_resized.jpg', img2_resized)
    inter = np.logical_and(img1_resized, img2_resized).sum()
    union = np.logical_or(img1_resized, img2_resized).sum()
    return inter / union if union > 0 else 0.0

# Максимальный IoU с поворотом
def best_iou_with_rotations(img1: np.ndarray, img2: np.ndarray,
                            target_size: Tuple[int, int] = (32, 256),
                            step: int = 6) -> float:
    best = 0.0
    for angle in range(0, 180, step):
        rotated = rotate_binary_image(img1, angle)
        iou = iou_binary_images(rotated, img2, target_size)
        if iou > best:
            best = iou
    return best

# Оценка одного изображения (разворота)
def evaluate_image(img_path: str, gt_masks: List[np.ndarray],
                   model_path: str, params: Dict[str, Any],
                   debug: bool = False, debug_dir: str = "debug_images") -> float:
    img = cv2.imread(img_path)
    if img is None:
        return 0.0

    if debug:
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Получаем страницы
    _, binary_pages = extract_pages_with_yolo(
        image_path=img,
        model_path=model_path,
        output_dir=debug_dir if debug else '/tmp',
        conf_threshold=0.7,
        return_binary = True
    )
    if not binary_pages:
        return 0.0

    all_pred_images = [] # сюда собираем вырезанные строки
    all_pred_points = [] # для отладки (точки)
    for page_idx, page in enumerate(binary_pages):
        detector = TextLineDetector(page, params=params, debug=False)

        _, crops, masks = detector.detect_text_lines()

        crops = [ensure_grayscale(c) for c in crops if c is not None and c.size > 0]

        all_pred_images.extend(crops)
        if debug:
            all_pred_points.extend(masks)

        if debug:
            # Рисуем боксы на странице
            debug_img = page.copy()
            for pts in masks:
                pts_arr = np.array(pts, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts_arr)
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(debug_dir, f"{base_name}_page{page_idx}_pred_boxes.jpg"), debug_img)

    if not all_pred_images or not gt_masks:
        return 0.0
    # Матрица IoU с поворотами
    def compute_pair(i, j):
        return -best_iou_with_rotations(all_pred_images[i], gt_masks[j])

    n_pred = len(all_pred_images)
    n_gt = len(gt_masks)

    # Генерируем все пары индексов
    pairs = [(i, j) for i in range(n_pred) for j in range(n_gt)]
    
    results = Parallel(n_jobs=8)(
        delayed(compute_pair)(i, j) for (i, j) in pairs
    )

    # Собираем матрицу cost
    cost = np.zeros((n_pred, n_gt))
    for (i, j), val in zip(pairs, results):
        cost[i, j] = val

    row, col = linear_sum_assignment(cost)
    total_iou = -cost[row, col].sum()
    score = total_iou / max(n_pred, n_gt)

    if debug:
        # Сохраняем лучшие соответствия
        for i in range(n_pred):
            matched_gt_idx = col[np.where(row == i)[0][0]] if i in row else None
            if matched_gt_idx is not None:
                best_iou = -cost[i, matched_gt_idx]
                h1, w1 = all_pred_images[i].shape
                h2, w2 = gt_masks[matched_gt_idx].shape
                collage = np.ones((max(h1, h2), w1 + w2 + 10), dtype=np.uint8) * 255
                collage[:h1, :w1] = all_pred_images[i]
                collage[:h2, w1+10:w1+10+w2] = gt_masks[matched_gt_idx]
                cv2.imwrite(os.path.join(debug_dir, f"{base_name}_pred{i}_gt{matched_gt_idx}_iou{best_iou:.3f}.jpg"), collage)

    del img, binary_pages
    gc.collect()
    return score


# Целевая функция для Optuna
def objective(trial: optuna.Trial,
             image_files: List[Tuple[str, str, List[np.ndarray]]],
             model_path: str,
             fixed_params: Dict[str, Any]) -> float:

    params = fixed_params.copy()
    # Пространство поиска гиперпараметров
    params['hough_theta_range'] = (
        trial.suggest_int('hough_theta_min', 70, 90),
        trial.suggest_int('hough_theta_max', 90, 110)
    )
    params['hough_rho_step_factor'] = trial.suggest_float('hough_rho_step_factor', 0.1, 0.3)
    params['hough_max_votes_threshold'] = trial.suggest_int('hough_max_votes_threshold', 3, 20)
    params['hough_secondary_threshold'] = trial.suggest_int('hough_secondary_threshold', 5, 25)
    params['hough_angle_tolerance'] = trial.suggest_int('hough_angle_tolerance', 1, 5)
    params['hough_neighborhood_radius'] = trial.suggest_int('hough_neighborhood_radius', 3, 10)
    params['merge_distance_factor'] = trial.suggest_float('merge_distance_factor', 0.5, 1.5)
    params['subset1_height_bounds'] = (
        trial.suggest_float('subset1_min', 0.3, 0.7),
        trial.suggest_float('subset1_max', 2.5, 4.0)
    )
    params['subset1_width_factor'] = trial.suggest_float('subset1_width_factor', 0.3, 0.8)
    params['subset2_height_factor'] = trial.suggest_float('subset2_height_factor', 2.5, 4.5)
    params['subset3_height_factor'] = trial.suggest_float('subset3_height_factor', 0.2, 0.8)
    params['subset3_width_factor'] = trial.suggest_float('subset3_width_factor', 0.2, 0.8)
    params['skeleton_junction_removal_zone'] = (
        trial.suggest_float('skeleton_zone_min', 0.3, 0.7),
        trial.suggest_float('skeleton_zone_max', 1.0, 2.0)
    )
    params['skeleton_junction_neighborhood'] = trial.suggest_int('skeleton_neigh', 1, 5)
    params['hough_small_dataset_threshold'] = trial.suggest_int('small_thresh', 30, 100)
    params['hough_large_dataset_threshold'] = trial.suggest_int('large_thresh', 150, 300)
    params['hough_min_max_votes'] = trial.suggest_int('min_max_votes', 2, 6)
    params['hough_min_secondary_votes'] = trial.suggest_int('min_sec_votes', 3, 8)
    params['hough_max_max_votes'] = trial.suggest_int('max_max_votes', 8, 15)
    params['hough_max_secondary_votes'] = trial.suggest_int('max_sec_votes', 10, 20)
    params['skew_expansion_threshold'] = trial.suggest_float('skew_exp', 2.0, 6.0)
    params['new_line_lower_factor'] = trial.suggest_float('nl_lower', 0.5, 0.9)
    params['new_line_upper_factor'] = trial.suggest_float('nl_upper', 1.0, 1.5)
    params['new_line_vertical_grouping_factor'] = trial.suggest_float('nl_vert', 0.5, 1.2)
    params['angle_filter_threshold'] = trial.suggest_int('angle_filter', 5, 20)
    params['min_components_for_skew'] = trial.suggest_int('min_skew', 3, 10)

    def process_one(fname, full_path, gt_dir):
        gt_masks = load_ground_truth_masks(gt_dir)
        return evaluate_image(full_path, gt_masks, model_path, params, debug=False)
    n_jobs = min(2, multiprocessing.cpu_count())
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_one)(fname, path, gt_dir) for (fname, path, gt_dir) in image_files
    )
    avg_score = np.mean(results) if results else 0.0
    gc.collect()
    return -avg_score

# Главная функция
def main():
    images_dir = "datasets/school_notebooks_RU/images_base"
    target_base = "datasets/school_notebooks_RU/images_target"
    model_path = "models/yolo_detect_notebook/yolo_detect_notebook_1_(1-architecture).pt"

    image_files = []
    image_files_len = 50 
    for idx, fname in enumerate(os.listdir(images_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        full_path = os.path.join(images_dir, fname)
        base = os.path.splitext(fname)[0]
        gt_dir = os.path.join(target_base, base)
        if os.path.isdir(gt_dir):
            masks = load_ground_truth_masks(gt_dir)
            if masks:
                image_files.append((fname, full_path, gt_dir))

        if idx > image_files_len:
            break

    print(f"Найдено изображений с ground truth: {len(image_files)}")
    if len(image_files) == 0:
        print("Нет изображений для оптимизации, выходим.")
        return
    
    image_files = image_files[:10]

    fixed_params = {'binarization_method': 'is_binary'}
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(trial, image_files, model_path, fixed_params),
        n_trials=30,
        n_jobs=1,
        show_progress_bar=True
    )
    print("\nЛучшие параметры:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Лучший средний IoU: {-study.best_value:.4f}")

    joblib.dump(study, "optuna_hough_transform.pkl")

if __name__ == "__main__":
    # Отладка на одном изображении
    # test_img = "datasets/school_notebooks_RU/images_base/ru_hw2022_22_IMG_6120.JPG"
    # test_base = os.path.splitext(os.path.basename(test_img))[0]
    # test_gt_dir = os.path.join("datasets/school_notebooks_RU/images_target", test_base)
    # if os.path.isdir(test_gt_dir):
    #     test_masks = load_ground_truth_masks(test_gt_dir)
    #     test_params = {'binarization_method': 'u_net'}
    #     score = evaluate_image(test_img, test_masks,
    #                            "models/yolo_detect_notebook/yolo_detect_notebook_1_(1-architecture).pt",
    #                            test_params, debug=True)
    #     print(f"Test IoU = {score:.4f}")
    # else:
    #     print("Нет ground truth для тестового изображения")
    main()