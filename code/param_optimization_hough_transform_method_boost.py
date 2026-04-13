import json
import os
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import optuna
import torch
from tqdm import tqdm

from param_optimization_hough_transform_method import (
    evaluate_image,
    load_ground_truth_masks,
)
from u_net_binarization import UNetTiny


UNET_MODEL_PATH = "models/u_net/unet_binarization_3_(6-architecture).pth"
YOLO_MODEL_PATH = "models/yolo_detect_notebook/yolo_detect_notebook_1_(1-architecture).pt"
IMAGES_DIR = "datasets/school_notebooks_RU/images_base"
TARGET_BASE_DIR = "datasets/school_notebooks_RU/images_target"
OUTPUT_DIR = "boost_hough_results"
TMP_DIR = "tmp_files/boost_hough"
UNET_TARGET_SIZE = (3000, 3000)

MAX_IMAGES = 25
TRAIN_COUNT = 21
PER_IMAGE_OPTUNA_TRIALS = 12
FIXED_PARAMS = {"binarization_method": "is_binary"}


PARAM_SPECS = {
    "hough_theta_min": {"type": "int", "low": 70, "high": 90},
    "hough_theta_max": {"type": "int", "low": 90, "high": 110},
    "hough_rho_step_factor": {"type": "float", "low": 0.1, "high": 0.3},
    "hough_max_votes_threshold": {"type": "int", "low": 3, "high": 20},
    "hough_secondary_threshold": {"type": "int", "low": 5, "high": 25},
    "hough_angle_tolerance": {"type": "int", "low": 1, "high": 5},
    "hough_neighborhood_radius": {"type": "int", "low": 3, "high": 10},
    "merge_distance_factor": {"type": "float", "low": 0.5, "high": 1.5},
    "subset1_min": {"type": "float", "low": 0.3, "high": 0.7},
    "subset1_max": {"type": "float", "low": 2.5, "high": 4.0},
    "subset1_width_factor": {"type": "float", "low": 0.3, "high": 0.8},
    "subset2_height_factor": {"type": "float", "low": 2.5, "high": 4.5},
    "subset3_height_factor": {"type": "float", "low": 0.2, "high": 0.8},
    "subset3_width_factor": {"type": "float", "low": 0.2, "high": 0.8},
    "skeleton_zone_min": {"type": "float", "low": 0.3, "high": 0.7},
    "skeleton_zone_max": {"type": "float", "low": 1.0, "high": 2.0},
    "skeleton_neigh": {"type": "int", "low": 1, "high": 5},
    "small_thresh": {"type": "int", "low": 30, "high": 100},
    "large_thresh": {"type": "int", "low": 150, "high": 300},
    "min_max_votes": {"type": "int", "low": 2, "high": 6},
    "min_sec_votes": {"type": "int", "low": 3, "high": 8},
    "max_max_votes": {"type": "int", "low": 8, "high": 15},
    "max_sec_votes": {"type": "int", "low": 10, "high": 20},
    "skew_exp": {"type": "float", "low": 2.0, "high": 6.0},
    "nl_lower": {"type": "float", "low": 0.5, "high": 0.9},
    "nl_upper": {"type": "float", "low": 1.0, "high": 1.5},
    "nl_vert": {"type": "float", "low": 0.5, "high": 1.2},
    "angle_filter": {"type": "int", "low": 5, "high": 20},
    "min_skew": {"type": "int", "low": 3, "high": 10},
}


def sanitize_stem(name: str) -> str:
    """
    Короткое описание:
        Превращает имя файла в безопасный идентификатор для временных файлов.

    Вход:
        name (str): исходное имя файла или stem.

    Выход:
        safe_name (str): безопасное имя для использования в путях.
    """
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def ensure_tmp_dir(tmp_dir: str = TMP_DIR) -> str:
    """
    Короткое описание:
        Создаёт директорию для временных артефактов бустинга.

    Вход:
        tmp_dir (str): путь до временной директории.

    Выход:
        tmp_dir (str): путь до существующей временной директории.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def feature_cache_path(fname: str, tmp_dir: str = TMP_DIR) -> str:
    """
    Короткое описание:
        Возвращает путь до файла с кэшем признаков изображения.

    Вход:
        fname (str): имя исходного изображения.
        tmp_dir (str): временная директория.

    Выход:
        path (str): путь до `.npy` файла с embedding-вектором.
    """
    stem = sanitize_stem(os.path.splitext(fname)[0])
    return os.path.join(tmp_dir, f"{stem}_feature.npy")


def records_cache_path(name: str, tmp_dir: str = TMP_DIR) -> str:
    """
    Короткое описание:
        Возвращает путь до временного JSON-файла с накопленными записями.

    Вход:
        name (str): логическое имя набора записей.
        tmp_dir (str): временная директория.

    Выход:
        path (str): путь до JSON-файла.
    """
    return os.path.join(tmp_dir, f"{sanitize_stem(name)}.json")


def build_params_from_trial(
    trial: optuna.Trial,
    fixed_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Короткое описание:
        Собирает словарь гиперпараметров детектора строк из trial Optuna.

    Вход:
        trial (optuna.Trial): объект trial, из которого сэмплируются гиперпараметры.
        fixed_params (Dict[str, Any]): фиксированные параметры, которые не оптимизируются.

    Выход:
        params (Dict[str, Any]): словарь параметров в формате, подходящем для TextLineDetector.
    """
    params = fixed_params.copy()
    params["hough_theta_range"] = (
        trial.suggest_int("hough_theta_min", 70, 90),
        trial.suggest_int("hough_theta_max", 90, 110),
    )
    params["hough_rho_step_factor"] = trial.suggest_float("hough_rho_step_factor", 0.1, 0.3)
    params["hough_max_votes_threshold"] = trial.suggest_int("hough_max_votes_threshold", 3, 20)
    params["hough_secondary_threshold"] = trial.suggest_int("hough_secondary_threshold", 5, 25)
    params["hough_angle_tolerance"] = trial.suggest_int("hough_angle_tolerance", 1, 5)
    params["hough_neighborhood_radius"] = trial.suggest_int("hough_neighborhood_radius", 3, 10)
    params["merge_distance_factor"] = trial.suggest_float("merge_distance_factor", 0.5, 1.5)
    params["subset1_height_bounds"] = (
        trial.suggest_float("subset1_min", 0.3, 0.7),
        trial.suggest_float("subset1_max", 2.5, 4.0),
    )
    params["subset1_width_factor"] = trial.suggest_float("subset1_width_factor", 0.3, 0.8)
    params["subset2_height_factor"] = trial.suggest_float("subset2_height_factor", 2.5, 4.5)
    params["subset3_height_factor"] = trial.suggest_float("subset3_height_factor", 0.2, 0.8)
    params["subset3_width_factor"] = trial.suggest_float("subset3_width_factor", 0.2, 0.8)
    params["skeleton_junction_removal_zone"] = (
        trial.suggest_float("skeleton_zone_min", 0.3, 0.7),
        trial.suggest_float("skeleton_zone_max", 1.0, 2.0),
    )
    params["skeleton_junction_neighborhood"] = trial.suggest_int("skeleton_neigh", 1, 5)
    params["hough_small_dataset_threshold"] = trial.suggest_int("small_thresh", 30, 100)
    params["hough_large_dataset_threshold"] = trial.suggest_int("large_thresh", 150, 300)
    params["hough_min_max_votes"] = trial.suggest_int("min_max_votes", 2, 6)
    params["hough_min_secondary_votes"] = trial.suggest_int("min_sec_votes", 3, 8)
    params["hough_max_max_votes"] = trial.suggest_int("max_max_votes", 8, 15)
    params["hough_max_secondary_votes"] = trial.suggest_int("max_sec_votes", 10, 20)
    params["skew_expansion_threshold"] = trial.suggest_float("skew_exp", 2.0, 6.0)
    params["new_line_lower_factor"] = trial.suggest_float("nl_lower", 0.5, 0.9)
    params["new_line_upper_factor"] = trial.suggest_float("nl_upper", 1.0, 1.5)
    params["new_line_vertical_grouping_factor"] = trial.suggest_float("nl_vert", 0.5, 1.2)
    params["angle_filter_threshold"] = trial.suggest_int("angle_filter", 5, 20)
    params["min_components_for_skew"] = trial.suggest_int("min_skew", 3, 10)
    return params


def flatten_params(params: Dict[str, Any]) -> Dict[str, float]:
    """
    Короткое описание:
        Переводит сложный словарь гиперпараметров в плоский набор числовых признаков.

    Вход:
        params (Dict[str, Any]): словарь параметров детектора строк.

    Выход:
        flat_params (Dict[str, float]): плоский словарь scalar-значений для обучения бустинга.
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
        "skeleton_zone_min": float(params["skeleton_junction_removal_zone"][0]),
        "skeleton_zone_max": float(params["skeleton_junction_removal_zone"][1]),
        "skeleton_neigh": float(params["skeleton_junction_neighborhood"]),
        "small_thresh": float(params["hough_small_dataset_threshold"]),
        "large_thresh": float(params["hough_large_dataset_threshold"]),
        "min_max_votes": float(params["hough_min_max_votes"]),
        "min_sec_votes": float(params["hough_min_secondary_votes"]),
        "max_max_votes": float(params["hough_max_max_votes"]),
        "max_sec_votes": float(params["hough_max_secondary_votes"]),
        "skew_exp": float(params["skew_expansion_threshold"]),
        "nl_lower": float(params["new_line_lower_factor"]),
        "nl_upper": float(params["new_line_upper_factor"]),
        "nl_vert": float(params["new_line_vertical_grouping_factor"]),
        "angle_filter": float(params["angle_filter_threshold"]),
        "min_skew": float(params["min_components_for_skew"]),
    }


def unflatten_params(
    flat_params: Dict[str, float],
    fixed_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Короткое описание:
        Восстанавливает словарь гиперпараметров детектора из плоских предсказаний бустинга.

    Вход:
        flat_params (Dict[str, float]): плоский словарь предсказанных scalar-параметров.
        fixed_params (Dict[str, Any]): фиксированные параметры, которые не предсказываются.

    Выход:
        params (Dict[str, Any]): словарь параметров в формате TextLineDetector.
    """
    clamped: Dict[str, float] = {}
    for name, spec in PARAM_SPECS.items():
        value = flat_params[name]
        value = max(spec["low"], min(spec["high"], value))
        if spec["type"] == "int":
            value = int(round(value))
        else:
            value = float(value)
        clamped[name] = value

    if clamped["hough_theta_min"] >= clamped["hough_theta_max"]:
        clamped["hough_theta_max"] = max(
            clamped["hough_theta_min"] + 1,
            PARAM_SPECS["hough_theta_max"]["low"],
        )
        clamped["hough_theta_max"] = min(
            clamped["hough_theta_max"],
            PARAM_SPECS["hough_theta_max"]["high"],
        )

    if clamped["subset1_min"] >= clamped["subset1_max"]:
        clamped["subset1_max"] = min(max(clamped["subset1_min"] + 0.1, 2.5), 4.0)

    if clamped["skeleton_zone_min"] >= clamped["skeleton_zone_max"]:
        clamped["skeleton_zone_max"] = min(max(clamped["skeleton_zone_min"] + 0.1, 1.0), 2.0)

    if clamped["small_thresh"] >= clamped["large_thresh"]:
        clamped["large_thresh"] = min(max(clamped["small_thresh"] + 1, 150), 300)

    params = fixed_params.copy()
    params["hough_theta_range"] = (clamped["hough_theta_min"], clamped["hough_theta_max"])
    params["hough_rho_step_factor"] = clamped["hough_rho_step_factor"]
    params["hough_max_votes_threshold"] = clamped["hough_max_votes_threshold"]
    params["hough_secondary_threshold"] = clamped["hough_secondary_threshold"]
    params["hough_angle_tolerance"] = clamped["hough_angle_tolerance"]
    params["hough_neighborhood_radius"] = clamped["hough_neighborhood_radius"]
    params["merge_distance_factor"] = clamped["merge_distance_factor"]
    params["subset1_height_bounds"] = (clamped["subset1_min"], clamped["subset1_max"])
    params["subset1_width_factor"] = clamped["subset1_width_factor"]
    params["subset2_height_factor"] = clamped["subset2_height_factor"]
    params["subset3_height_factor"] = clamped["subset3_height_factor"]
    params["subset3_width_factor"] = clamped["subset3_width_factor"]
    params["skeleton_junction_removal_zone"] = (
        clamped["skeleton_zone_min"],
        clamped["skeleton_zone_max"],
    )
    params["skeleton_junction_neighborhood"] = clamped["skeleton_neigh"]
    params["hough_small_dataset_threshold"] = clamped["small_thresh"]
    params["hough_large_dataset_threshold"] = clamped["large_thresh"]
    params["hough_min_max_votes"] = clamped["min_max_votes"]
    params["hough_min_secondary_votes"] = clamped["min_sec_votes"]
    params["hough_max_max_votes"] = clamped["max_max_votes"]
    params["hough_max_secondary_votes"] = clamped["max_sec_votes"]
    params["skew_expansion_threshold"] = clamped["skew_exp"]
    params["new_line_lower_factor"] = clamped["nl_lower"]
    params["new_line_upper_factor"] = clamped["nl_upper"]
    params["new_line_vertical_grouping_factor"] = clamped["nl_vert"]
    params["angle_filter_threshold"] = clamped["angle_filter"]
    params["min_components_for_skew"] = clamped["min_skew"]
    return params


def load_unet_model(
    model_path: str = UNET_MODEL_PATH,
    device: Optional[str] = None,
) -> Tuple[UNetTiny, torch.device]:
    """
    Короткое описание:
        Загружает U-Net модель, из bottleneck которой потом извлекаются признаки изображения.

    Вход:
        model_path (str): путь до файла с весами U-Net.
        device (Optional[str]): устройство для загрузки модели, например 'cpu' или 'cuda'.

    Выход:
        model (UNetTiny): загруженная U-Net модель.
        torch_device (torch.device): устройство, на которое загружена модель.
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
        Повторяет preprocessing из u_net_binarization.py для согласованного извлечения признаков.

    Вход:
        image (np.ndarray): входное изображение страницы.
        target_size (Tuple[int, int]): размер, к которому приводится изображение перед подачей в U-Net.

    Выход:
        input_tensor (torch.Tensor): подготовленный тензор размера (1, 1, H, W).
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
        Извлекает средний bottleneck-вектор U-Net и использует его как embedding изображения.

    Вход:
        image_path (str): путь до входного изображения.
        model (UNetTiny): загруженная U-Net модель.
        device (torch.device): устройство, на котором выполняется инференс.
        target_size (Tuple[int, int]): размер preprocessing перед подачей в модель.

    Выход:
        avg_vector (np.ndarray): вектор признаков изображения, полученный усреднением bottleneck-тензора.
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


@dataclass
class TreeNode:
    """
    Короткое описание:
        Хранит один узел простого регрессионного дерева.

    Вход:
        feature_idx (Optional[int]): индекс признака для разбиения.
        threshold (Optional[float]): порог разбиения.
        left (Optional["TreeNode"]): левый дочерний узел.
        right (Optional["TreeNode"]): правый дочерний узел.
        value (Optional[float]): значение в листе дерева.

    Выход:
        TreeNode: объект узла дерева.
    """

    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Короткое описание:
            Сериализует узел дерева в обычный словарь.

        Вход:
            None: использует поля текущего узла.

        Выход:
            node_dict (Dict[str, Any]): словарь с данными узла и дочерних узлов.
        """
        return {
            "feature_idx": self.feature_idx,
            "threshold": self.threshold,
            "value": self.value,
            "left": None if self.left is None else self.left.to_dict(),
            "right": None if self.right is None else self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, node_dict: Dict[str, Any]) -> "TreeNode":
        """
        Короткое описание:
            Восстанавливает узел дерева из словаря.

        Вход:
            node_dict (Dict[str, Any]): сериализованный словарь узла.

        Выход:
            node (TreeNode): восстановленный узел дерева.
        """
        if node_dict is None:
            return None
        return cls(
            feature_idx=node_dict["feature_idx"],
            threshold=node_dict["threshold"],
            value=node_dict["value"],
            left=cls.from_dict(node_dict["left"]),
            right=cls.from_dict(node_dict["right"]),
        )


class SimpleRegressionTree:
    """
    Короткое описание:
        Реализует небольшое регрессионное дерево для использования внутри самописного бустинга.

    Вход:
        max_depth (int): максимальная глубина дерева.
        min_samples_leaf (int): минимальное количество объектов в листе.
        max_thresholds (int): максимальное число кандидатных порогов при поиске split.

    Выход:
        SimpleRegressionTree: обучаемый объект регрессионного дерева.
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 2,
        max_thresholds: int = 16,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.root: Optional[TreeNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleRegressionTree":
        """
        Короткое описание:
            Обучает регрессионное дерево на признаках и целевых значениях.

        Вход:
            X (np.ndarray): матрица признаков размера (N, D).
            y (np.ndarray): вектор целевых значений размера (N,).

        Выход:
            self (SimpleRegressionTree): обученный объект дерева.
        """
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        if (
            depth >= self.max_depth
            or len(y) < 2 * self.min_samples_leaf
            or np.allclose(y, y[0])
        ):
            return TreeNode(value=float(np.mean(y)))

        best_feature = None
        best_threshold = None
        best_score = np.inf
        best_left_mask = None

        parent_var = np.var(y) * len(y)
        if parent_var <= 1e-12:
            return TreeNode(value=float(np.mean(y)))

        for feature_idx in range(X.shape[1]):
            values = X[:, feature_idx]
            unique_vals = np.unique(values)
            if len(unique_vals) <= 1:
                continue

            if len(unique_vals) > self.max_thresholds:
                quantiles = np.linspace(0.05, 0.95, self.max_thresholds)
                thresholds = np.unique(np.quantile(unique_vals, quantiles))
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                left_count = int(np.sum(left_mask))
                right_count = int(np.sum(right_mask))
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue

                left_y = y[left_mask]
                right_y = y[right_mask]
                score = np.var(left_y) * left_count + np.var(right_y) * right_count
                if score < best_score:
                    best_score = score
                    best_feature = feature_idx
                    best_threshold = float(threshold)
                    best_left_mask = left_mask

        if best_feature is None or best_left_mask is None or best_score >= parent_var:
            return TreeNode(value=float(np.mean(y)))

        left_node = self._build(X[best_left_mask], y[best_left_mask], depth + 1)
        right_node = self._build(X[~best_left_mask], y[~best_left_mask], depth + 1)
        return TreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node,
        )

    def _predict_one(self, x: np.ndarray, node: TreeNode) -> float:
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Короткое описание:
            Предсказывает значения для набора объектов с помощью обученного дерева.

        Вход:
            X (np.ndarray): матрица признаков размера (N, D).

        Выход:
            pred (np.ndarray): массив предсказанных значений размера (N,).
        """
        return np.array([self._predict_one(x, self.root) for x in X], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """
        Короткое описание:
            Сериализует регрессионное дерево в словарь без pickle-классов.

        Вход:
            None: использует состояние текущего дерева.

        Выход:
            tree_dict (Dict[str, Any]): сериализованное представление дерева.
        """
        return {
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_thresholds": self.max_thresholds,
            "root": None if self.root is None else self.root.to_dict(),
        }

    @classmethod
    def from_dict(cls, tree_dict: Dict[str, Any]) -> "SimpleRegressionTree":
        """
        Короткое описание:
            Восстанавливает регрессионное дерево из словаря.

        Вход:
            tree_dict (Dict[str, Any]): сериализованное представление дерева.

        Выход:
            tree (SimpleRegressionTree): восстановленное дерево.
        """
        tree = cls(
            max_depth=tree_dict["max_depth"],
            min_samples_leaf=tree_dict["min_samples_leaf"],
            max_thresholds=tree_dict["max_thresholds"],
        )
        tree.root = TreeNode.from_dict(tree_dict["root"])
        return tree


class SimpleGradientBoostingRegressor:
    """
    Короткое описание:
        Реализует простой градиентный бустинг на базе маленьких регрессионных деревьев.

    Вход:
        n_estimators (int): количество деревьев в ансамбле.
        learning_rate (float): шаг, с которым добавляется вклад каждого дерева.
        max_depth (int): максимальная глубина каждого дерева.
        min_samples_leaf (int): минимальный размер листа дерева.
        max_thresholds (int): число кандидатных порогов для поиска разбиений.

    Выход:
        SimpleGradientBoostingRegressor: объект бустинга для регрессии.
    """

    def __init__(
        self,
        n_estimators: int = 64,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_samples_leaf: int = 2,
        max_thresholds: int = 16,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.base_value: float = 0.0
        self.trees: List[SimpleRegressionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleGradientBoostingRegressor":
        """
        Короткое описание:
            Обучает бустинг на признаках и целевых значениях.

        Вход:
            X (np.ndarray): матрица признаков размера (N, D).
            y (np.ndarray): вектор целевых значений размера (N,).

        Выход:
            self (SimpleGradientBoostingRegressor): обученный объект бустинга.
        """
        self.base_value = float(np.mean(y))
        pred = np.full(len(y), self.base_value, dtype=np.float32)
        self.trees = []

        for _ in range(self.n_estimators):
            residual = y - pred
            tree = SimpleRegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_thresholds=self.max_thresholds,
            )
            tree.fit(X, residual)
            pred = pred + self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Короткое описание:
            Предсказывает значения для набора объектов с помощью ансамбля деревьев.

        Вход:
            X (np.ndarray): матрица признаков размера (N, D).

        Выход:
            pred (np.ndarray): массив предсказанных значений размера (N,).
        """
        pred = np.full(X.shape[0], self.base_value, dtype=np.float32)
        for tree in self.trees:
            pred = pred + self.learning_rate * tree.predict(X)
        return pred

    def to_dict(self) -> Dict[str, Any]:
        """
        Короткое описание:
            Сериализует бустинг в словарь из обычных Python-структур.

        Вход:
            None: использует состояние текущего бустинга.

        Выход:
            model_dict (Dict[str, Any]): сериализованное представление бустинга.
        """
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_thresholds": self.max_thresholds,
            "base_value": self.base_value,
            "trees": [tree.to_dict() for tree in self.trees],
        }

    @classmethod
    def from_dict(cls, model_dict: Dict[str, Any]) -> "SimpleGradientBoostingRegressor":
        """
        Короткое описание:
            Восстанавливает бустинг из словаря.

        Вход:
            model_dict (Dict[str, Any]): сериализованное представление бустинга.

        Выход:
            model (SimpleGradientBoostingRegressor): восстановленный бустинг.
        """
        model = cls(
            n_estimators=model_dict["n_estimators"],
            learning_rate=model_dict["learning_rate"],
            max_depth=model_dict["max_depth"],
            min_samples_leaf=model_dict["min_samples_leaf"],
            max_thresholds=model_dict["max_thresholds"],
        )
        model.base_value = model_dict["base_value"]
        model.trees = [SimpleRegressionTree.from_dict(tree_dict) for tree_dict in model_dict["trees"]]
        return model


def collect_image_files(
    images_dir: str,
    target_base: str,
    max_images: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    """
    Короткое описание:
        Собирает изображения, для которых существует директория с ground truth масками.

    Вход:
        images_dir (str): директория с исходными изображениями.
        target_base (str): базовая директория с поддиректориями ground truth.
        max_images (Optional[int]): максимальное число изображений для сбора.

    Выход:
        image_files (List[Tuple[str, str, str]]): список кортежей
            (имя файла, полный путь до изображения, путь до папки с GT).
    """
    image_files = []
    fnames = [  # это хорошие изображения для обучения
        "1_11.JPG",
        "ru_hw2022_16_IMG_6418.JPG",
        "2911.jpg",
        "2883.jpg",
        "2881.jpg",
        "2858.jpg",
        "2820.jpg",
        "2821.jpg",
        "1_13.JPG",
        "1_18.JPG",
        "1_22.JPG",
        "7_96.JPG",
        "2878.jpg",
        "81_764.JPG",
        "8_110.JPG",
        "2_31.JPG",
        "9_115.JPG",
        "11_139.JPG",
        "ru_hw2022_23_IMG_6467.JPG",
        "ru_hw2022_30_IMG_5644.JPG",
        "3_47.JPG",
        "43_423.JPG",
        "25_256.jpeg",
        "4_60.JPG",
        "2_40.JPG",
        "64_626.jpg",
        "28_268.JPG",
        "1_10.JPG",
        "1_12.JPG",
        "5_74.JPG"
    ]
    for fname in fnames: # sorted(os.listdir(images_dir))
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        full_path = os.path.join(images_dir, fname)
        base = os.path.splitext(fname)[0]
        gt_dir = os.path.join(target_base, base)

        if os.path.isdir(gt_dir):
            has_gt = any(mask_name.endswith(".png") for mask_name in os.listdir(gt_dir))
            if has_gt:
                image_files.append((fname, full_path, gt_dir))

        if max_images is not None and len(image_files) >= max_images:
            break
    return image_files


def optimize_single_image(
    image_path: str,
    gt_dir: str,
    model_path: str,
    fixed_params: Dict[str, Any],
    n_trials: int,
    seed: int,
) -> Tuple[Dict[str, Any], float]:
    """
    Ищет лучшие гиперпараметры для одного изображения.

    Короткое описание:
        Подбирает лучшие гиперпараметры для одного изображения через Optuna.

    Вход:
        image_path (str): путь до изображения.
        gt_dir (str): путь до директории с GT-масками.
        model_path (str): путь до YOLO-модели для выделения страниц.
        fixed_params (Dict[str, Any]): фиксированные параметры детектора.
        n_trials (int): число trial для Optuna.
        seed (int): seed для sampler Optuna.

    Выход:
        best_params (Dict[str, Any]): лучший найденный словарь гиперпараметров.
        best_score (float): лучшее значение исходной метрики evaluate_image.
    """
    gt_masks = load_ground_truth_masks(gt_dir)

    def objective_single(trial: optuna.Trial) -> float:
        params = build_params_from_trial(trial, fixed_params)
        score = evaluate_image(image_path, gt_masks, model_path, params, debug=False)
        return -score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective_single, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    best_params = build_params_from_trial(study.best_trial, fixed_params)
    best_score = -study.best_value
    return best_params, best_score


def fit_boost_models(
    X_train: np.ndarray,
    target_rows: List[Dict[str, float]],
) -> Dict[str, SimpleGradientBoostingRegressor]:
    """
    Короткое описание:
        Обучает отдельную бустинг-модель для каждого гиперпараметра.

    Вход:
        X_train (np.ndarray): матрица признаков train-изображений размера (N, D).
        target_rows (List[Dict[str, float]]): список словарей с целевыми гиперпараметрами.

    Выход:
        models (Dict[str, SimpleGradientBoostingRegressor]): словарь обученных моделей по именам параметров.
    """
    target_names = list(target_rows[0].keys())
    targets = {
        name: np.array([row[name] for row in target_rows], dtype=np.float32)
        for name in target_names
    }

    models = {}
    for name in target_names:
        model = SimpleGradientBoostingRegressor(
            n_estimators=96,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=max(2, min(4, len(X_train) // 4)),
            max_thresholds=16,
        )
        model.fit(X_train, targets[name])
        models[name] = model
    return models


def predict_params_from_feature(
    feature_vector: np.ndarray,
    models: Dict[str, SimpleGradientBoostingRegressor],
    fixed_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Короткое описание:
        Предсказывает полный словарь гиперпараметров по embedding-вектору изображения.

    Вход:
        feature_vector (np.ndarray): вектор признаков изображения.
        models (Dict[str, SimpleGradientBoostingRegressor]): словарь обученных моделей бустинга.
        fixed_params (Dict[str, Any]): фиксированные параметры детектора.

    Выход:
        params (Dict[str, Any]): предсказанный словарь гиперпараметров для TextLineDetector.
    """
    feature_vector = feature_vector.reshape(1, -1)
    flat_pred = {
        name: float(model.predict(feature_vector)[0])
        for name, model in models.items()
    }
    return unflatten_params(flat_pred, fixed_params)


def evaluate_predicted_params(
    image_files: List[Tuple[str, str, str]],
    models: Dict[str, SimpleGradientBoostingRegressor],
    model_path: str,
    fixed_params: Dict[str, Any],
    desc: str,
    tmp_dir: str = TMP_DIR,
) -> List[Dict[str, Any]]:
    """
    Короткое описание:
        Проверяет предсказанные бустингом параметры на исходной функции оценки.

    Вход:
        image_files (List[Tuple[str, str, str]]): список изображений и путей к их GT.
        models (Dict[str, SimpleGradientBoostingRegressor]): обученные модели бустинга.
        model_path (str): путь до YOLO-модели для выделения страниц.
        fixed_params (Dict[str, Any]): фиксированные параметры детектора.
        desc (str): подпись progress bar.
        tmp_dir (str): директория, где лежат сохранённые на диск признаки.

    Выход:
        rows (List[Dict[str, Any]]): список записей с именем файла, score и предсказанными параметрами.
    """
    rows = []
    for fname, full_path, gt_dir in tqdm(
        image_files,
        desc=desc,
        unit="image",
    ):
        gt_masks = load_ground_truth_masks(gt_dir)
        feature_vector = np.load(feature_cache_path(fname, tmp_dir))
        predicted_params = predict_params_from_feature(feature_vector, models, fixed_params)
        score = evaluate_image(full_path, gt_masks, model_path, predicted_params, debug=False)
        rows.append(
            {
                "file": fname,
                "predicted_score": float(score),
                "predicted_params": flatten_params(predicted_params),
            }
        )
        del gt_masks, feature_vector
        gc.collect()
    return rows


def cache_features_to_disk(
    image_files: List[Tuple[str, str, str]],
    model: UNetTiny,
    device: torch.device,
    tmp_dir: str = TMP_DIR,
) -> int:
    """
    Короткое описание:
        Извлекает embedding-признаки для изображений и сохраняет их на диск в tmp_files.

    Вход:
        image_files (List[Tuple[str, str, str]]): список изображений и путей к ним.
        model (UNetTiny): загруженная U-Net модель.
        device (torch.device): устройство инференса.
        tmp_dir (str): директория для временного кэша.

    Выход:
        feature_dim (int): размерность сохранённых embedding-векторов.
    """
    ensure_tmp_dir(tmp_dir)
    feature_dim: Optional[int] = None

    for fname, full_path, _ in tqdm(
        image_files,
        desc="Extract U-Net features",
        unit="image",
    ):
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


def load_feature_matrix(
    image_files: List[Tuple[str, str, str]],
    tmp_dir: str = TMP_DIR,
) -> np.ndarray:
    """
    Короткое описание:
        Загружает матрицу признаков для указанного списка изображений из временного кэша.

    Вход:
        image_files (List[Tuple[str, str, str]]): список изображений.
        tmp_dir (str): директория с сохранёнными `.npy` признаками.

    Выход:
        X (np.ndarray): матрица признаков размера (N, D).
    """
    vectors = [np.load(feature_cache_path(fname, tmp_dir)) for fname, _, _ in image_files]
    X = np.stack(vectors, axis=0)
    del vectors
    gc.collect()
    return X


def save_records(
    rows: List[Dict[str, Any]],
    path: str,
) -> None:
    """
    Короткое описание:
        Сохраняет список записей в JSON-файл во временной директории.

    Вход:
        rows (List[Dict[str, Any]]): список записей.
        path (str): путь до JSON-файла.

    Выход:
        None: функция сохраняет записи на диск.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)


def build_report(
    train_files: List[Tuple[str, str, str]],
    valid_files: List[Tuple[str, str, str]],
    X_train: np.ndarray,
    oracle_rows: List[Dict[str, Any]],
    train_eval: List[Dict[str, Any]],
    valid_eval: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Короткое описание:
        Собирает итоговый JSON-отчёт по oracle-параметрам и качеству предсказаний бустинга.

    Вход:
        train_files (List[Tuple[str, str, str]]): train-изображения.
        valid_files (List[Tuple[str, str, str]]): validation-изображения.
        X_train (np.ndarray): матрица train-признаков.
        oracle_rows (List[Dict[str, Any]]): лучшие параметры, найденные Optuna для train.
        train_eval (List[Dict[str, Any]]): оценка бустинга на train.
        valid_eval (List[Dict[str, Any]]): оценка бустинга на validation.

    Выход:
        report (Dict[str, Any]): итоговый словарь отчёта для сохранения в JSON.
    """
    return {
        "config": {
            "max_images": MAX_IMAGES,
            "train_count": len(train_files),
            "valid_count": len(valid_files),
            "per_image_optuna_trials": PER_IMAGE_OPTUNA_TRIALS,
            "feature_dim": int(X_train.shape[1]),
            "feature_source": "U-Net bottleneck x4 mean over spatial dimensions",
            "evaluation_source": (
                "Imported evaluate_image from "
                "param_optimization_hough_transform_method.py without changing metric"
            ),
        },
        "oracle_train": oracle_rows,
        "predicted_train": train_eval,
        "predicted_valid": valid_eval,
        "mean_scores": {
            "oracle_train_mean": float(np.mean([row["oracle_score"] for row in oracle_rows])) if oracle_rows else 0.0,
            "predicted_train_mean": float(np.mean([row["predicted_score"] for row in train_eval])) if train_eval else 0.0,
            "predicted_valid_mean": float(np.mean([row["predicted_score"] for row in valid_eval])) if valid_eval else 0.0,
        },
    }


def serialize_model_bundle(model_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Короткое описание:
        Преобразует пакет моделей в сериализуемый словарь без пользовательских классов.

    Вход:
        model_bundle (Dict[str, Any]): словарь с обученными моделями и метаинформацией.

    Выход:
        serialized_bundle (Dict[str, Any]): словарь, пригодный для безопаского joblib/json сохранения.
    """
    serialized_bundle = dict(model_bundle)
    serialized_bundle["models"] = {
        name: model.to_dict()
        for name, model in model_bundle["models"].items()
    }
    serialized_bundle["serialization_format"] = "plain_dict_v1"
    return serialized_bundle


def deserialize_model_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Короткое описание:
        Восстанавливает пакет моделей из сериализованного словаря.

    Вход:
        bundle (Dict[str, Any]): словарь, загруженный с диска.

    Выход:
        restored_bundle (Dict[str, Any]): словарь с восстановленными объектами бустинга.
    """
    restored_bundle = dict(bundle)
    serialization_format = restored_bundle.get("serialization_format")
    if serialization_format == "plain_dict_v1":
        restored_bundle["models"] = {
            name: SimpleGradientBoostingRegressor.from_dict(model_dict)
            for name, model_dict in restored_bundle["models"].items()
        }
    return restored_bundle


def load_boost_bundle(model_path: str) -> Dict[str, Any]:
    """
    Короткое описание:
        Загружает пакет бустинговых моделей с диска в безопасном формате.

    Вход:
        model_path (str): путь до файла с сохранённым пакетом моделей.

    Выход:
        bundle (Dict[str, Any]): словарь с восстановленными моделями и метаинформацией.
    """
    bundle = joblib.load(model_path)
    return deserialize_model_bundle(bundle)


def save_artifacts(
    report: Dict[str, Any],
    model_bundle: Dict[str, Any],
    output_dir: str,
) -> None:
    """
    Короткое описание:
        Сохраняет JSON-отчёт и сериализованный пакет моделей бустинга.

    Вход:
        report (Dict[str, Any]): итоговый отчёт.
        model_bundle (Dict[str, Any]): словарь с моделями и метаинформацией.
        output_dir (str): директория, куда сохраняются артефакты.

    Выход:
        None: функция сохраняет файлы на диск.
    """
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "boost_report.json")
    model_path = os.path.join(output_dir, "boost_models.joblib")

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    serialized_bundle = serialize_model_bundle(model_bundle)
    joblib.dump(serialized_bundle, model_path)

    print(f"Сохранён отчёт: {report_path}")
    print(f"Сохранены модели: {model_path}")


def main() -> None:
    """
    Короткое описание:
        Запускает полный пайплайн: извлечение U-Net признаков, поиск oracle-параметров и обучение бустинга.

    Вход:
        None: использует константы файла для путей, размеров выборки и числа trial.

    Выход:
        None: сохраняет отчёт и модели бустинга на диск.
    """
    image_files = collect_image_files(IMAGES_DIR, TARGET_BASE_DIR, max_images=MAX_IMAGES)
    print(f"Найдено изображений с GT: {len(image_files)}")
    if len(image_files) < 3:
        print("Слишком мало изображений для обучения бустинга.")
        return

    ensure_tmp_dir(TMP_DIR)

    train_files = image_files[:TRAIN_COUNT]
    valid_files = image_files[TRAIN_COUNT:] if len(image_files) > TRAIN_COUNT else image_files[-2:]

    unet_model, unet_device = load_unet_model(UNET_MODEL_PATH)
    feature_dim = cache_features_to_disk(image_files, unet_model, unet_device, TMP_DIR)
    del unet_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_target_rows: List[Dict[str, float]] = []
    oracle_rows: List[Dict[str, Any]] = []
    for idx, (fname, full_path, gt_dir) in enumerate(
        tqdm(train_files, desc="Optuna per image", unit="image")
    ):
        print(f"[optuna] {idx + 1}/{len(train_files)} {fname}")
        best_params, best_score = optimize_single_image(
            image_path=full_path,
            gt_dir=gt_dir,
            model_path=YOLO_MODEL_PATH,
            fixed_params=FIXED_PARAMS,
            n_trials=PER_IMAGE_OPTUNA_TRIALS,
            seed=42 + idx,
        )
        flat_params = flatten_params(best_params)
        train_target_rows.append(flat_params)
        oracle_rows.append(
            {
                "file": fname,
                "oracle_score": float(best_score),
                "oracle_params": flat_params,
            }
        )
        save_records(oracle_rows, records_cache_path("oracle_rows", TMP_DIR))
        save_records(train_target_rows, records_cache_path("train_target_rows", TMP_DIR))
        gc.collect()

    X_train = load_feature_matrix(train_files, TMP_DIR)
    models = fit_boost_models(X_train, train_target_rows)

    train_eval = evaluate_predicted_params(
        train_files,
        models,
        YOLO_MODEL_PATH,
        FIXED_PARAMS,
        desc="Evaluate predicted params on train",
        tmp_dir=TMP_DIR,
    )
    valid_eval = evaluate_predicted_params(
        valid_files,
        models,
        YOLO_MODEL_PATH,
        FIXED_PARAMS,
        desc="Evaluate predicted params on valid",
        tmp_dir=TMP_DIR,
    )
    save_records(train_eval, records_cache_path("train_eval", TMP_DIR))
    save_records(valid_eval, records_cache_path("valid_eval", TMP_DIR))

    report = build_report(
        train_files=train_files,
        valid_files=valid_files,
        X_train=X_train,
        oracle_rows=oracle_rows,
        train_eval=train_eval,
        valid_eval=valid_eval,
    )

    model_bundle = {
        "models": models,
        "fixed_params": FIXED_PARAMS,
        "param_specs": PARAM_SPECS,
        "train_files": train_files,
        "valid_files": valid_files,
        "feature_dim": feature_dim,
        "unet_feature_description": "Average pooled bottleneck tensor x4 from UNetTiny",
    }

    save_artifacts(report, model_bundle, OUTPUT_DIR)
    print(json.dumps(report["mean_scores"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
