import argparse
import gc
import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import optuna
import torch
from ultralytics import YOLO
from tqdm import tqdm

from hough_method.param_optimization_hough_transform_method import evaluate_image, load_ground_truth_masks
from u_net_binarization import UNetTiny


DATASET_RUNS_DIR = "datasets/boost_hough_datasets"
FEATURES_DIR_NAME = "features"
DATASET_ROWS_FILENAME = "dataset_rows.json"
METADATA_FILENAME = "metadata.json"
UNET_MODEL_PATH = "models/u_net/unet_binarization_3_(6-architecture).pth"
YOLO_MODEL_PATH = "models/yolo_detect_notebook/yolo_detect_notebook_1_(1-architecture).pt"
IMAGES_DIR = "datasets/school_notebooks_RU/images_base"
TARGET_BASE_DIR = "datasets/school_notebooks_RU/images_target"
TMP_DIR = "tmp_files/boost_hough"
UNET_TARGET_SIZE = (3000, 3000)
PER_IMAGE_OPTUNA_TRIALS = 25
MAX_IMAGES = 100
FIXED_PARAMS = {"binarization_method": "u_net"}

DEFAULT_DETECTOR_PARAMS = {
    "hough_theta_range": (-10, 10),
    "hough_rho_step_factor": 0.15,
    "hough_max_votes_threshold": 5,
    "hough_secondary_threshold": 10,
    "hough_angle_tolerance": 2,
    "hough_neighborhood_radius": 5,
    "merge_distance_factor": 1.0,
    "subset1_height_bounds": (0.5, 3.0),
    "subset1_width_factor": 0.5,
    "subset2_height_factor": 3.0,
    "subset3_height_factor": 0.5,
    "subset3_width_factor": 0.5,
    "hough_small_dataset_threshold": 50,
    "hough_large_dataset_threshold": 200,
    "hough_min_max_votes": 3,
    "hough_min_secondary_votes": 5,
    "hough_max_max_votes": 10,
    "hough_max_secondary_votes": 15,
    "new_line_lower_factor": 0.7,
    "new_line_upper_factor": 1.1,
    "new_line_vertical_grouping_factor": 0.8,
    "angle_filter_threshold": 1,
    "min_components_for_skew": 5,
    "number_component_blocks_that_voted_for_the_line": 3,
}


def sanitize_stem(name: str) -> str:
    """
    Короткое описание:
        Преобразует имя файла в безопасный stem для временных файлов.

    Вход:
        name (str): исходное имя файла.

    Выход:
        safe_name (str): безопасное имя без проблемных символов.
    """
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def ensure_tmp_dir(tmp_dir: str = TMP_DIR) -> str:
    """
    Короткое описание:
        Создаёт временную директорию для служебных файлов.

    Вход:
        tmp_dir (str): путь до временной директории.

    Выход:
        tmp_dir (str): путь до существующей директории.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def feature_cache_path(fname: str, tmp_dir: str = TMP_DIR) -> str:
    """
    Короткое описание:
        Возвращает путь до кэша признаков изображения.

    Вход:
        fname (str): имя изображения.
        tmp_dir (str): директория для `.npy` файлов.

    Выход:
        path (str): путь до файла признаков.
    """
    stem = sanitize_stem(os.path.splitext(fname)[0])
    return os.path.join(tmp_dir, f"{stem}_feature.npy")


def flatten_params(params: Dict[str, Any]) -> Dict[str, float]:
    """
    Короткое описание:
        Превращает словарь параметров детектора в плоский словарь чисел.

    Вход:
        params (Dict[str, Any]): словарь параметров детектора строк.

    Выход:
        flat_params (Dict[str, float]): плоский словарь числовых параметров.
    """
    return {
        "hough_theta_min": float(params["hough_theta_range"][0]),
        "hough_theta_max": float(params["hough_theta_range"][1]),
        "hough_rho_step_factor": float(params["hough_rho_step_factor"]),
        "hough_max_votes_threshold": float(params["hough_max_votes_threshold"]),
        "hough_secondary_threshold": float(params["hough_secondary_threshold"]),
        "hough_angle_tolerance": float(params["hough_angle_tolerance"]),
        "hough_neighborhood_radius": float(params["hough_neighborhood_radius"]),
        "merge_distance_factor": float(params["merge_distance_factor"]),
        "subset1_min": float(params["subset1_height_bounds"][0]),
        "subset1_max": float(params["subset1_height_bounds"][1]),
        "subset1_width_factor": float(params["subset1_width_factor"]),
        "subset2_height_factor": float(params["subset2_height_factor"]),
        "subset3_height_factor": float(params["subset3_height_factor"]),
        "subset3_width_factor": float(params["subset3_width_factor"]),
        "hough_small_dataset_threshold": float(params["hough_small_dataset_threshold"]),
        "hough_large_dataset_threshold": float(params["hough_large_dataset_threshold"]),
        "hough_min_max_votes": float(params["hough_min_max_votes"]),
        "hough_min_secondary_votes": float(params["hough_min_secondary_votes"]),
        "hough_max_max_votes": float(params["hough_max_max_votes"]),
        "hough_max_secondary_votes": float(params["hough_max_secondary_votes"]),
        "nl_lower": float(params["new_line_lower_factor"]),
        "nl_upper": float(params["new_line_upper_factor"]),
        "nl_vert": float(params["new_line_vertical_grouping_factor"]),
        "angle_filter": float(params["angle_filter_threshold"]),
        "min_skew": float(params["min_components_for_skew"]),
        "number_component_blocks_that_voted_for_the_line": float(
            params["number_component_blocks_that_voted_for_the_line"]
        ),
    }


def build_params_from_trial(
    trial: optuna.Trial,
    fixed_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Короткое описание:
        Собирает словарь гиперпараметров детектора из trial Optuna.

    Вход:
        trial (optuna.Trial): текущий trial.
        fixed_params (Dict[str, Any]): фиксированные параметры.

    Выход:
        params (Dict[str, Any]): итоговый словарь параметров детектора.
    """
    params = {**DEFAULT_DETECTOR_PARAMS, **fixed_params}
    params["hough_theta_range"] = (
        trial.suggest_int("hough_theta_min", -10, 0),
        trial.suggest_int("hough_theta_max", 0, 10),
    )
    params["hough_rho_step_factor"] = trial.suggest_float("hough_rho_step_factor", 0.1, 0.3)
    params["hough_max_votes_threshold"] = trial.suggest_int("hough_max_votes_threshold", 1, 30)
    params["hough_secondary_threshold"] = trial.suggest_int("hough_secondary_threshold", 1, 30)
    params["hough_angle_tolerance"] = trial.suggest_int("hough_angle_tolerance", 1, 5)
    params["hough_neighborhood_radius"] = trial.suggest_int("hough_neighborhood_radius", 3, 10)
    params["merge_distance_factor"] = trial.suggest_float("merge_distance_factor", 0.1, 2.0)
    params["subset1_height_bounds"] = (
        trial.suggest_float("subset1_min", 0.3, 0.7),
        trial.suggest_float("subset1_max", 2.5, 4.0),
    )
    params["subset1_width_factor"] = trial.suggest_float("subset1_width_factor", 0.3, 0.8)
    params["subset2_height_factor"] = trial.suggest_float("subset2_height_factor", 2.5, 4.5)
    params["subset3_height_factor"] = trial.suggest_float("subset3_height_factor", 0.2, 0.8)
    params["subset3_width_factor"] = trial.suggest_float("subset3_width_factor", 0.2, 0.8)
    params["hough_small_dataset_threshold"] = trial.suggest_int("hough_small_dataset_threshold", 20, 100)
    params["hough_large_dataset_threshold"] = trial.suggest_int("hough_large_dataset_threshold", 120, 350)
    params["hough_min_max_votes"] = trial.suggest_int("hough_min_max_votes", 2, 6)
    params["hough_min_secondary_votes"] = trial.suggest_int("hough_min_secondary_votes", 3, 10)
    params["hough_max_max_votes"] = trial.suggest_int("hough_max_max_votes", 8, 15)
    params["hough_max_secondary_votes"] = trial.suggest_int("hough_max_secondary_votes", 12, 25)
    params["new_line_lower_factor"] = trial.suggest_float("nl_lower", 0.1, 0.9)
    params["new_line_upper_factor"] = trial.suggest_float("nl_upper", 1.0, 1.5)
    params["new_line_vertical_grouping_factor"] = trial.suggest_float("nl_vert", 0.5, 1.2)
    params["angle_filter_threshold"] = trial.suggest_float("angle_filter", 0.1, 7)
    params["min_components_for_skew"] = trial.suggest_int("min_skew", 3, 10)
    params["number_component_blocks_that_voted_for_the_line"] = trial.suggest_int(
        "number_component_blocks_that_voted_for_the_line", 1, 8
    )
    return params


def load_unet_model(
    model_path: str = UNET_MODEL_PATH,
    device: Optional[str] = None,
) -> Tuple[UNetTiny, torch.device]:
    """
    Короткое описание:
        Загружает U-Net для извлечения признаков страницы.

    Вход:
        model_path (str): путь до весов U-Net.
        device (Optional[str]): целевое устройство.

    Выход:
        model (UNetTiny): загруженная модель.
        torch_device (torch.device): устройство инференса.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    model = UNetTiny(in_channels=1, out_channels=1).to(torch_device)
    model.eval()
    state_dict = torch.load(model_path, map_location=torch_device, weights_only=True)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model, torch_device


def preprocess_for_unet(
    image: np.ndarray,
    target_size: Tuple[int, int] = UNET_TARGET_SIZE,
) -> torch.Tensor:
    """
    Короткое описание:
        Подготавливает изображение к подаче в U-Net как в основном биноризаторе.

    Вход:
        image (np.ndarray): входное изображение.
        target_size (Tuple[int, int]): итоговый размер.

    Выход:
        input_tensor (torch.Tensor): тензор размера `(1, 1, H, W)`.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        gray = image.squeeze()
    else:
        gray = image

    h, w = gray.shape
    target_h, target_w = target_size
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_h, target_w), 255, dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    img_norm = padded.astype(np.float32) / 255.0
    img_norm = (img_norm - 0.5) / 0.5
    return torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)


def extract_unet_avg_feature(
    image_path: str,
    model: UNetTiny,
    device: torch.device,
    target_size: Tuple[int, int] = UNET_TARGET_SIZE,
) -> np.ndarray:
    """
    Короткое описание:
        Извлекает усреднённый bottleneck-вектор U-Net из изображения.

    Вход:
        image_path (str): путь до изображения.
        model (UNetTiny): U-Net модель.
        device (torch.device): устройство инференса.
        target_size (Tuple[int, int]): размер препроцессинга.

    Выход:
        avg_vector (np.ndarray): embedding-вектор изображения.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

    input_tensor = preprocess_for_unet(image, target_size=target_size).to(device)
    with torch.no_grad():
        x1 = model.inc(input_tensor)
        x2 = model.down1(x1)
        x3 = model.down2(x2)
        x4 = model.down3(x3)
        avg_vector = x4.mean(dim=(2, 3)).squeeze(0).cpu().numpy().astype(np.float32)
    return avg_vector


def collect_image_files(
    images_dir: str,
    target_base: str,
    max_images: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    """
    Короткое описание:
        Собирает изображения, для которых есть папка с GT-разметкой.

    Вход:
        images_dir (str): папка с изображениями.
        target_base (str): базовая папка GT.
        max_images (Optional[int]): ограничение по числу изображений.

    Выход:
        image_files (List[Tuple[str, str, str]]): список `(file, image_path, gt_dir)`.
    """
    image_files = []
    fnames = [
        # "2_40.JPG",
        # "13_150.JPG",
        # "4_55.JPG",
        # "8_109.JPG",
        # "9_115.JPG",
        # "21_225.JPG",
        # "22_233.JPG",
        # "30_284.jpeg",
        # "31_305.JPG",
        # "31_317.JPG",
        # "36_354.JPG",
        # "37_365.JPG",
        # "50_503.JPG",
        # "50_512.JPG",
        # "52_520.jpg",
        # "58_567.JPG",
        # "76_738.jpg",
        # "82_780.JPG",
        # "88_861.JPG",
        # "99_994.JPG",
        # "2231.jpg",
        # "2367.jpg",
        # "2383.jpg",
        # "2633.jpg",
        # "2630.jpg",
        # "2619.jpg",
        # "2659.jpg",
        # "2720.jpg",
        # "2788.jpg",
        # "2804.jpg",
        # "2820.jpg",
        # "2823.jpg",
        # "2884.jpg",
        # "ru_hw2022_12_IMG_6598.JPG",
        # "1_23.JPG",
        # "2_33.JPG",
        # "3_44.JPG",
        # "5_74.JPG",
        # "5_81.JPG",
        # "5_88.JPG",
        # "6_89.JPG",
        # "6_93.JPG",
        # "9_120.JPG",
        # "11_130.JPG",
        # "15_166.JPG",
        "17_181.JPG",
        "18_193.JPG",
        "18_205.JPG",
        "19_206.JPG",
        "21_221.JPG",
        "21_224.JPG",
        "22_226.JPG",
        "22_228.JPG",
        "23_238.jpg",
        "25_250.jpeg",
        "26_257.jpg",
        "30_303.jpeg",
        "32_319.JPG",
        "34_333.jpg",
        "35_344.JPG",
        "35_347.JPG",
        "35_348.JPG",
        "36_356.JPG",
        "43_423.JPG",
        "ru_hw2022_16_IMG_6447.JPG",
        "2917.jpg",
        "2916.jpg",
        "2895.jpg",
        "2880.jpg",
        "2878.jpg",
        "2868.jpg",
        "2858.jpg",
        "2853.jpg",
        "2847.jpg",
        "2818.jpg",
        "2013.jpg",
        "2039.jpg"
    ]
    for fname in fnames:
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        full_path = os.path.join(images_dir, fname)
        gt_dir = os.path.join(target_base, os.path.splitext(fname)[0])
        if os.path.isdir(gt_dir) and any(mask_name.endswith(".png") for mask_name in os.listdir(gt_dir)):
            image_files.append((fname, full_path, gt_dir))
        if max_images is not None and len(image_files) >= max_images:
            break
    return image_files


def optimize_single_image(
    image_path: str,
    gt_dir: str,
    run_dir: str,
    model_path: str,
    fixed_params: Dict[str, Any],
    n_trials: int,
    seed: int,
) -> Tuple[Dict[str, Any], float]:
    """
    Короткое описание:
        Подбирает oracle-параметры для одного изображения через Optuna.

    Вход:
        image_path (str): путь до изображения.
        gt_dir (str): путь до GT-папки.
        model_path (str): путь до YOLO-модели.
        fixed_params (Dict[str, Any]): фиксированные параметры.
        n_trials (int): число trial.
        seed (int): seed для sampler.

    Выход:
        best_params (Dict[str, Any]): лучший словарь гиперпараметров.
        best_score (float): лучшее значение исходной метрики.
    """
    gt_masks = load_ground_truth_masks(gt_dir)
    yolo_model = None
    unet_model = None
    unet_device = None

    try:
        # Переиспользуем модели в рамках оптимизации одного изображения,
        # чтобы не пересоздавать их на каждом trial.
        yolo_model = YOLO(model_path)
        unet_model, unet_device = load_unet_model(UNET_MODEL_PATH)

        def objective_single(trial: optuna.Trial) -> float:
            params = build_params_from_trial(trial, fixed_params)
            score = evaluate_image(
                image_path,
                gt_masks,
                model_path,
                params,
                debug=False,
                yolo_model=yolo_model,
                unet_model=unet_model,
                unet_device=unet_device,
            )
            return -score

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective_single, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

        # fig = optuna.visualization.plot_slice(study)
        # fig.show()
        filename = os.path.basename(image_path).split('.')[0]
        os.makedirs(f"{run_dir}/param_importances", exist_ok=True)
        try:
            importances = optuna.importance.get_param_importances(study)
        except Exception as exc:
            importances = {"_error": str(exc)}
        with open(f"{run_dir}/param_importances/{filename}.json", "w") as f:
            json.dump(importances, f, indent=4, ensure_ascii=False)

        return build_params_from_trial(study.best_trial, fixed_params), -study.best_value
    finally:
        if yolo_model is not None:
            del yolo_model
        if unet_model is not None:
            del unet_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cache_features_to_disk(
    image_files: List[Tuple[str, str, str]],
    model: UNetTiny,
    device: torch.device,
    tmp_dir: str = TMP_DIR,
) -> int:
    """
    Короткое описание:
        Извлекает и сохраняет признаки изображений на диск.

    Вход:
        image_files (List[Tuple[str, str, str]]): список изображений.
        model (UNetTiny): U-Net модель.
        device (torch.device): устройство инференса.
        tmp_dir (str): директория для кэша.

    Выход:
        feature_dim (int): размерность признака.
    """
    ensure_tmp_dir(tmp_dir)
    feature_dim: Optional[int] = None
    for fname, full_path, _ in tqdm(image_files, desc="Extract U-Net features", unit="image"):
        cache_path = feature_cache_path(fname, tmp_dir)
        if os.path.exists(cache_path):
            if feature_dim is None:
                feature_dim = int(np.load(cache_path, mmap_mode="r").shape[0])
            continue
        feature_vector = extract_unet_avg_feature(full_path, model, device)
        np.save(cache_path, feature_vector)
        feature_dim = int(feature_vector.shape[0])
        del feature_vector
        gc.collect()
    if feature_dim is None:
        raise RuntimeError("Не удалось определить размерность признаков.")
    return feature_dim


def create_run_name(prefix: str) -> str:
    """
    Короткое описание:
        Создаёт уникальное имя папки прогона по текущему времени.

    Вход:
        prefix (str): текстовый префикс имени прогона.

    Выход:
        run_name (str): имя папки прогона.
    """
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def ensure_run_dir(run_name: str) -> str:
    """
    Короткое описание:
        Создаёт папку прогона датасета и подпапку для признаков.

    Вход:
        run_name (str): имя папки прогона.

    Выход:
        run_dir (str): путь до созданной папки прогона.
    """
    run_dir = os.path.join(DATASET_RUNS_DIR, run_name)
    os.makedirs(os.path.join(run_dir, FEATURES_DIR_NAME), exist_ok=True)
    return run_dir


def load_json_or_default(path: str, default_value: Any) -> Any:
    """
    Короткое описание:
        Загружает JSON-файл или возвращает значение по умолчанию.

    Вход:
        path (str): путь до JSON-файла.
        default_value (Any): значение, которое нужно вернуть при отсутствии файла.

    Выход:
        value (Any): загруженные данные или default_value.
    """
    if not os.path.exists(path):
        return default_value
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str, data: Any) -> None:
    """
    Короткое описание:
        Сохраняет данные в JSON-файл.

    Вход:
        path (str): путь до файла.
        data (Any): сохраняемые данные.

    Выход:
        None: функция сохраняет данные на диск.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def dataset_rows_path(run_dir: str) -> str:
    """
    Короткое описание:
        Возвращает путь до файла с записями датасета.

    Вход:
        run_dir (str): папка прогона.

    Выход:
        path (str): путь до dataset_rows.json.
    """
    return os.path.join(run_dir, DATASET_ROWS_FILENAME)


def metadata_path(run_dir: str) -> str:
    """
    Короткое описание:
        Возвращает путь до файла metadata.json.

    Вход:
        run_dir (str): папка прогона.

    Выход:
        path (str): путь до metadata.json.
    """
    return os.path.join(run_dir, METADATA_FILENAME)


def build_feature_target_dir(run_dir: str) -> str:
    """
    Короткое описание:
        Возвращает путь до подпапки с embedding-признаками прогона.

    Вход:
        run_dir (str): папка прогона.

    Выход:
        features_dir (str): путь до подпапки features.
    """
    return os.path.join(run_dir, FEATURES_DIR_NAME)


def build_dataset_row(
    fname: str,
    full_path: str,
    gt_dir: str,
    feature_path: str,
    best_score: float,
    best_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Короткое описание:
        Формирует одну запись датасета для последующего обучения бустинга.

    Вход:
        fname (str): имя изображения.
        full_path (str): полный путь до изображения.
        gt_dir (str): путь до директории ground truth.
        feature_path (str): путь до файла с embedding-признаками.
        best_score (float): лучшее значение метрики для изображения.
        best_params (Dict[str, Any]): лучший словарь гиперпараметров.

    Выход:
        row (Dict[str, Any]): словарь записи датасета.
    """
    return {
        "file": fname,
        "image_path": full_path,
        "gt_dir": gt_dir,
        "feature_path": feature_path,
        "oracle_score": float(best_score),
        "oracle_params": flatten_params(best_params),
    }


def load_existing_rows(run_dir: str) -> List[Dict[str, Any]]:
    """
    Короткое описание:
        Загружает существующие записи датасета из папки прогона.

    Вход:
        run_dir (str): папка прогона.

    Выход:
        rows (List[Dict[str, Any]]): список записей датасета.
    """
    return load_json_or_default(dataset_rows_path(run_dir), [])


def update_run_metadata(
    run_dir: str,
    mode: str,
    image_files: List[Tuple[str, str, str]],
    rows: List[Dict[str, Any]],
) -> None:
    """
    Короткое описание:
        Обновляет metadata.json для папки прогона.

    Вход:
        run_dir (str): папка прогона.
        mode (str): режим запуска.
        image_files (List[Tuple[str, str, str]]): список доступных изображений.
        rows (List[Dict[str, Any]]): текущие записи датасета.

    Выход:
        None: функция сохраняет metadata.json.
    """
    data = {
        "mode": mode,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "images_available": len(image_files),
        "images_optimized": len(rows),
        "images_pending": max(0, len(image_files) - len(rows)),
        "features_dir": build_feature_target_dir(run_dir),
        "dataset_rows_path": dataset_rows_path(run_dir),
    }
    save_json(metadata_path(run_dir), data)


def bootstrap_legacy_dataset(run_name: Optional[str] = None) -> str:
    """
    Короткое описание:
        Собирает датасет из уже существующих legacy-артефактов в results и tmp_files.

    Вход:
        run_name (Optional[str]): имя создаваемой папки прогона.

    Выход:
        run_dir (str): путь до созданного прогона с агрегированным датасетом.
    """
    if run_name is None:
        run_name = create_run_name("legacy_aggregate")

    run_dir = ensure_run_dir(run_name)
    features_dir = build_feature_target_dir(run_dir)

    merged_rows: Dict[str, Dict[str, Any]] = {}

    legacy_oracle_path = os.path.join("results", "boost_hough", "oracle_rows.json")
    legacy_oracle_rows = load_json_or_default(legacy_oracle_path, [])
    for row in legacy_oracle_rows:
        fname = row["file"]
        image_path = os.path.join(IMAGES_DIR, fname)
        gt_dir = os.path.join(TARGET_BASE_DIR, os.path.splitext(fname)[0])
        src_feature_path = feature_cache_path(fname, "tmp_files/boost_hough")
        if not os.path.exists(src_feature_path):
            continue

        dst_feature_path = os.path.join(features_dir, os.path.basename(src_feature_path))
        if not os.path.exists(dst_feature_path):
            shutil.copy2(src_feature_path, dst_feature_path)

        merged_rows[fname] = {
            "file": fname,
            "image_path": image_path,
            "gt_dir": gt_dir,
            "feature_path": dst_feature_path,
            "oracle_score": float(row["oracle_score"]),
            "oracle_params": row["oracle_params"],
            "source": "legacy_results_tmp",
        }

    for existing_dir_name in sorted(os.listdir(DATASET_RUNS_DIR)) if os.path.isdir(DATASET_RUNS_DIR) else []:
        existing_dir = os.path.join(DATASET_RUNS_DIR, existing_dir_name)
        existing_rows = load_json_or_default(dataset_rows_path(existing_dir), [])
        for row in existing_rows:
            fname = row["file"]
            best_prev = merged_rows.get(fname)
            if best_prev is None or row["oracle_score"] > best_prev["oracle_score"]:
                src_feature_path = row["feature_path"]
                if not os.path.exists(src_feature_path):
                    continue
                dst_feature_path = os.path.join(features_dir, os.path.basename(src_feature_path))
                if not os.path.exists(dst_feature_path):
                    shutil.copy2(src_feature_path, dst_feature_path)
                new_row = dict(row)
                new_row["feature_path"] = dst_feature_path
                new_row["source"] = f"dataset_run:{existing_dir_name}"
                merged_rows[fname] = new_row

    rows = sorted(merged_rows.values(), key=lambda item: item["file"])
    save_json(dataset_rows_path(run_dir), rows)
    update_run_metadata(run_dir, "bootstrap_legacy", [(r["file"], r["image_path"], r["gt_dir"]) for r in rows], rows)
    return run_dir


def collect_dataset_run(
    run_dir: str,
    max_images: int,
    n_trials: int,
    resume: bool,
) -> str:
    """
    Короткое описание:
        Собирает или дозаполняет датасет оптимизированных изображений для обучения бустинга.

    Вход:
        run_dir (str): папка прогона датасета.
        max_images (int): максимальное число изображений для обработки.
        n_trials (int): число trial Optuna на одно изображение.
        resume (bool): нужно ли продолжать существующий прогон.

    Выход:
        run_dir (str): путь до папки прогона.
    """
    ensure_run_dir(os.path.basename(run_dir))
    features_dir = build_feature_target_dir(run_dir)
    image_files = collect_image_files(IMAGES_DIR, TARGET_BASE_DIR, max_images=max_images)

    rows = load_existing_rows(run_dir) if resume else []
    processed = {row["file"] for row in rows}

    unet_model, unet_device = load_unet_model(UNET_MODEL_PATH)
    cache_features_to_disk(image_files, unet_model, unet_device, features_dir)
    del unet_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pending_files = [item for item in image_files if item[0] not in processed]
    for idx, (fname, full_path, gt_dir) in enumerate(
        tqdm(pending_files, desc="Collect dataset rows", unit="image")
    ):
        print(f"[dataset] {idx + 1}/{len(pending_files)} {fname}")
        error_text = None
        try:
            best_params, best_score = optimize_single_image(
                image_path=full_path,
                run_dir=run_dir,
                gt_dir=gt_dir,
                model_path=YOLO_MODEL_PATH,
                fixed_params=FIXED_PARAMS,
                n_trials=n_trials,
                seed=42 + len(rows) + idx,
            )
        except Exception as exc:
            error_text = str(exc)
            print(f"[dataset][ERROR] {fname}: {error_text}")
            best_params = {**DEFAULT_DETECTOR_PARAMS, **FIXED_PARAMS}
            best_score = 0.0
        row = build_dataset_row(
            fname=fname,
            full_path=full_path,
            gt_dir=gt_dir,
            feature_path=feature_cache_path(fname, features_dir),
            best_score=best_score,
            best_params=best_params,
        )
        if error_text is not None:
            row["error"] = error_text
        rows.append(row)
        rows = sorted(rows, key=lambda item: item["file"])
        save_json(dataset_rows_path(run_dir), rows)
        update_run_metadata(run_dir, "collect", image_files, rows)
        gc.collect()

    update_run_metadata(run_dir, "collect", image_files, rows)
    return run_dir


def parse_args() -> argparse.Namespace:
    """
    Короткое описание:
        Разбирает аргументы командной строки для сборки датасета.

    Вход:
        None: используются аргументы командной строки.

    Выход:
        args (argparse.Namespace): распарсенные аргументы запуска.
    """
    parser = argparse.ArgumentParser(description="Сбор датасета для обучения boost-моделей параметров Hough.")
    parser.add_argument("--mode", choices=["collect", "bootstrap_legacy"], default="collect")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume-run", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=15)
    parser.add_argument("--n-trials", type=int, default=PER_IMAGE_OPTUNA_TRIALS)
    return parser.parse_args()


def main() -> None:
    """
    Короткое описание:
        Точка входа для отдельного сценария сборки датасета бустинга.

    Вход:
        None: использует аргументы командной строки.

    Выход:
        None: сохраняет датасет очередного прогона в datasets.
    """
    os.makedirs(DATASET_RUNS_DIR, exist_ok=True)
    args = parse_args()

    if args.mode == "bootstrap_legacy":
        run_dir = bootstrap_legacy_dataset(run_name=args.run_name)
        print(f"Собран legacy-датасет: {run_dir}")
        return

    if args.resume_run is not None:
        run_dir = os.path.join(DATASET_RUNS_DIR, args.resume_run)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Не найдена папка прогона для дозаполнения: {run_dir}")
    else:
        run_name = args.run_name or create_run_name("dataset")
        run_dir = ensure_run_dir(run_name)

    run_dir = collect_dataset_run(
        run_dir=run_dir,
        max_images=MAX_IMAGES,
        n_trials=args.n_trials,
        resume=True,
    )
    print(f"Датасет сохранён: {run_dir}")


if __name__ == "__main__":
    main()
