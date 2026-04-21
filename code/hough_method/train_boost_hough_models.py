import argparse
import json
import os
from typing import Any, Dict, List, Tuple

from catboost import CatBoostRegressor
import joblib
import numpy as np
from tqdm import tqdm

from param_optimization_hough_transform_method import evaluate_image, load_ground_truth_masks
from collect_boost_hough_dataset import (
    FIXED_PARAMS,
    UNET_MODEL_PATH,
    extract_unet_avg_feature,
    flatten_params,
    load_unet_model,
    YOLO_MODEL_PATH,
)


DATASET_RUNS_DIR = "datasets/boost_hough_datasets/2"
MODEL_DIR = "models/boost_hough"
DEBUG_REPORT_DIR = "debug_report"
DATASET_ROWS_FILENAME = "dataset_rows.json"
MODEL_FILENAME = "boost_models.joblib"
REPORT_FILENAME = "boost_report.json"
CATBOOST_ITERATIONS = 100
CATBOOST_DEPTH = 6
CATBOOST_LEARNING_RATE = 0.03
CATBOOST_LOSS_FUNCTION = "RMSE"
CATBOOST_VERBOSE = False
TRAIN_COUNT_DEFAULT = 21

DEFAULT_DETECTOR_PARAMS = {
    "hough_theta_range": (85, 95),
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
    "skew_expansion_threshold": 3,
    "new_line_lower_factor": 0.7,
    "new_line_upper_factor": 1.1,
    "new_line_vertical_grouping_factor": 0.8,
    "angle_filter_threshold": 10,
    "min_components_for_skew": 5,
}

PARAM_SPECS = {
    "hough_theta_min": {"type": "int", "low": 70, "high": 90},
    "hough_theta_max": {"type": "int", "low": 90, "high": 110},
    "hough_rho_step_factor": {"type": "float", "low": 0.1, "high": 0.3},
    "hough_max_votes_threshold": {"type": "int", "low": 3, "high": 20},
    "hough_secondary_threshold": {"type": "int", "low": 5, "high": 25},
    "hough_angle_tolerance": {"type": "int", "low": 1, "high": 5},
    "hough_neighborhood_radius": {"type": "int", "low": 3, "high": 10},
    "subset1_min": {"type": "float", "low": 0.3, "high": 0.7},
    "subset1_max": {"type": "float", "low": 2.5, "high": 4.0},
    "subset1_width_factor": {"type": "float", "low": 0.3, "high": 0.8},
    "subset2_height_factor": {"type": "float", "low": 2.5, "high": 4.5},
    "subset3_height_factor": {"type": "float", "low": 0.2, "high": 0.8},
    "subset3_width_factor": {"type": "float", "low": 0.2, "high": 0.8},
    "skew_exp": {"type": "float", "low": 2.0, "high": 6.0},
    "nl_lower": {"type": "float", "low": 0.5, "high": 0.9},
    "nl_upper": {"type": "float", "low": 1.0, "high": 1.5},
    "nl_vert": {"type": "float", "low": 0.5, "high": 1.2},
    "angle_filter": {"type": "int", "low": 5, "high": 20},
    "min_skew": {"type": "int", "low": 3, "high": 10},
}


def load_json(path: str) -> Any:
    """
    Короткое описание:
        Загружает JSON-файл с диска.

    Вход:
        path (str): путь до JSON-файла.

    Выход:
        data (Any): загруженные данные.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def sanitize_stem(name: str) -> str:
    """
    Короткое описание:
        Приводит имя файла к безопасному stem для имени feature-файла.

    Вход:
        name (str): исходное имя файла.

    Выход:
        safe_name (str): безопасный stem без проблемных символов.
    """
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def expected_feature_path(dataset_run_dir: str, file_name: str) -> str:
    """
    Короткое описание:
        Возвращает ожидаемый путь feature-файла внутри текущего dataset-run.

    Вход:
        dataset_run_dir (str): путь до папки датасет-прогона.
        file_name (str): имя изображения.

    Выход:
        feature_path (str): путь до `.npy` файла признаков.
    """
    stem = sanitize_stem(os.path.splitext(file_name)[0])
    return os.path.join(dataset_run_dir, "features", f"{stem}_feature.npy")


def save_json(path: str, data: Any) -> None:
    """
    Короткое описание:
        Сохраняет данные в JSON-файл.

    Вход:
        path (str): путь до файла.
        data (Any): сохраняемые данные.

    Выход:
        None: функция сохраняет файл на диск.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)



def ensure_output_dirs() -> None:
    """
    Короткое описание:
        Создаёт директории для финальной модели и debug-отчёта.

    Вход:
        None: использует константные пути.

    Выход:
        None: создаёт директории на диске.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DEBUG_REPORT_DIR, exist_ok=True)


def unflatten_params(
    flat_params: Dict[str, float],
    fixed_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Короткое описание:
        Восстанавливает полный словарь параметров детектора из плоских предсказаний.

    Вход:
        flat_params (Dict[str, float]): плоский словарь числовых параметров.
        fixed_params (Dict[str, Any]): фиксированные параметры.

    Выход:
        params (Dict[str, Any]): словарь параметров детектора строк.
    """
    clamped: Dict[str, float] = {}
    for name, spec in PARAM_SPECS.items():
        value = max(spec["low"], min(spec["high"], flat_params[name]))
        clamped[name] = int(round(value)) if spec["type"] == "int" else float(value)

    if clamped["hough_theta_min"] >= clamped["hough_theta_max"]:
        clamped["hough_theta_max"] = min(
            PARAM_SPECS["hough_theta_max"]["high"],
            max(clamped["hough_theta_min"] + 1, PARAM_SPECS["hough_theta_max"]["low"]),
        )

    if clamped["subset1_min"] >= clamped["subset1_max"]:
        clamped["subset1_max"] = min(max(clamped["subset1_min"] + 0.1, 2.5), 4.0)

    params = {**DEFAULT_DETECTOR_PARAMS, **fixed_params}
    params["hough_theta_range"] = (clamped["hough_theta_min"], clamped["hough_theta_max"])
    params["hough_rho_step_factor"] = clamped["hough_rho_step_factor"]
    params["hough_max_votes_threshold"] = clamped["hough_max_votes_threshold"]
    params["hough_secondary_threshold"] = clamped["hough_secondary_threshold"]
    params["hough_angle_tolerance"] = clamped["hough_angle_tolerance"]
    params["hough_neighborhood_radius"] = clamped["hough_neighborhood_radius"]
    params["subset1_height_bounds"] = (clamped["subset1_min"], clamped["subset1_max"])
    params["subset1_width_factor"] = clamped["subset1_width_factor"]
    params["subset2_height_factor"] = clamped["subset2_height_factor"]
    params["subset3_height_factor"] = clamped["subset3_height_factor"]
    params["subset3_width_factor"] = clamped["subset3_width_factor"]
    params["skew_expansion_threshold"] = clamped["skew_exp"]
    params["new_line_lower_factor"] = clamped["nl_lower"]
    params["new_line_upper_factor"] = clamped["nl_upper"]
    params["new_line_vertical_grouping_factor"] = clamped["nl_vert"]
    params["angle_filter_threshold"] = clamped["angle_filter"]
    params["min_components_for_skew"] = clamped["min_skew"]
    return params


def fit_boost_models(
    X_train: np.ndarray,
    target_rows: List[Dict[str, float]],
) -> Dict[str, CatBoostRegressor]:
    """
    Короткое описание:
        Обучает отдельный CatBoost для каждого предсказываемого параметра.

    Вход:
        X_train (np.ndarray): матрица признаков train.
        target_rows (List[Dict[str, float]]): целевые параметры.

    Выход:
        models (Dict[str, CatBoostRegressor]): словарь обученных моделей.
    """
    target_names = list(target_rows[0].keys())
    targets = {name: np.array([row[name] for row in target_rows], dtype=np.float32) for name in target_names}

    models: Dict[str, CatBoostRegressor] = {}
    for name in target_names:
        model = CatBoostRegressor(
            iterations=CATBOOST_ITERATIONS,
            depth=CATBOOST_DEPTH,
            learning_rate=CATBOOST_LEARNING_RATE,
            loss_function=CATBOOST_LOSS_FUNCTION,
            verbose=CATBOOST_VERBOSE,
        )
        model.fit(X_train, targets[name])
        models[name] = model
    return models


def predict_params_from_feature(
    feature_vector: np.ndarray,
    models: Dict[str, CatBoostRegressor],
    fixed_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Короткое описание:
        Предсказывает полный набор параметров детектора по признаку изображения.

    Вход:
        feature_vector (np.ndarray): embedding-вектор изображения.
        models (Dict[str, CatBoostRegressor]): обученные модели.
        fixed_params (Dict[str, Any]): фиксированные параметры.

    Выход:
        params (Dict[str, Any]): итоговый словарь параметров детектора.
    """
    flat_pred = {name: float(model.predict(feature_vector.reshape(1, -1))[0]) for name, model in models.items()}
    return unflatten_params(flat_pred, fixed_params)


def load_boost_bundle(model_path: str = os.path.join(MODEL_DIR, MODEL_FILENAME)) -> Dict[str, Any]:
    """
    Короткое описание:
        Загружает сохранённый пакет бустинговых моделей.

    Вход:
        model_path (str): путь до `.joblib` файла.

    Выход:
        bundle (Dict[str, Any]): словарь с моделями и метаданными.
    """
    return joblib.load(model_path)


def load_dataset_rows(dataset_run_dir: str) -> List[Dict[str, Any]]:
    """
    Короткое описание:
        Загружает строки датасета из папки выбранного прогона.

    Вход:
        dataset_run_dir (str): путь до папки датасета.

    Выход:
        rows (List[Dict[str, Any]]): список записей датасета.
    """
    dataset_path = os.path.join(dataset_run_dir, DATASET_ROWS_FILENAME)
    rows = load_json(dataset_path)
    if not rows:
        raise ValueError(f"Датасет пустой: {dataset_path}")
    return rows


def normalize_feature_paths(
    rows: List[Dict[str, Any]],
    dataset_run_dir: str,
) -> List[Dict[str, Any]]:
    """
    Короткое описание:
        Нормализует feature_path под текущий dataset-run и восстанавливает отсутствующие `.npy`.

    Вход:
        rows (List[Dict[str, Any]]): строки датасета.
        dataset_run_dir (str): путь до папки датасет-прогона.

    Выход:
        rows (List[Dict[str, Any]]): строки с валидными путями до признаков.
    """
    changed = False
    missing_rows: List[Dict[str, Any]] = []

    for row in rows:
        current_path = row["feature_path"]
        run_local_path = expected_feature_path(dataset_run_dir, row["file"])

        if os.path.exists(run_local_path):
            if current_path != run_local_path:
                row["feature_path"] = run_local_path
                changed = True
            continue

        if os.path.exists(current_path):
            os.makedirs(os.path.dirname(run_local_path), exist_ok=True)
            np.save(run_local_path, np.load(current_path))
            row["feature_path"] = run_local_path
            changed = True
            continue

        row["feature_path"] = run_local_path
        missing_rows.append(row)
        changed = True

    if missing_rows:
        unet_model, unet_device = load_unet_model(UNET_MODEL_PATH)
        for row in tqdm(missing_rows, desc="Rebuild missing features", unit="image"):
            feature_vector = extract_unet_avg_feature(row["image_path"], unet_model, unet_device)
            os.makedirs(os.path.dirname(row["feature_path"]), exist_ok=True)
            np.save(row["feature_path"], feature_vector)
        del unet_model

    if changed:
        save_json(os.path.join(dataset_run_dir, DATASET_ROWS_FILENAME), rows)
    return rows


def split_dataset_rows(
    rows: List[Dict[str, Any]],
    train_count: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Короткое описание:
        Делит строки датасета на train и valid части.

    Вход:
        rows (List[Dict[str, Any]]): строки датасета.
        train_count (int): желаемое число train-объектов.

    Выход:
        train_rows (List[Dict[str, Any]]): train-часть.
        valid_rows (List[Dict[str, Any]]): valid-часть.
    """
    rows = sorted(rows, key=lambda item: item["file"])
    train_count = min(train_count, len(rows))
    train_rows = rows[:train_count]
    valid_rows = rows[train_count:]

    if not valid_rows and len(rows) >= 3:
        valid_size = min(2, len(rows) - 1)
        train_rows = rows[:-valid_size]
        valid_rows = rows[-valid_size:]

    return train_rows, valid_rows


def load_feature_matrix_from_rows(rows: List[Dict[str, Any]]) -> np.ndarray:
    """
    Короткое описание:
        Загружает матрицу признаков по строкам датасета.

    Вход:
        rows (List[Dict[str, Any]]): строки датасета с путями до feature-файлов.

    Выход:
        X (np.ndarray): матрица признаков размера (N, D).
    """
    vectors = [np.load(row["feature_path"]) for row in rows]
    return np.stack(vectors, axis=0)


def build_target_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    Короткое описание:
        Формирует целевые строки гиперпараметров для обучения бустинга.

    Вход:
        rows (List[Dict[str, Any]]): строки датасета.

    Выход:
        target_rows (List[Dict[str, float]]): список плоских словарей гиперпараметров.
    """
    return [dict(row["oracle_params"]) for row in rows]


def evaluate_rows(
    rows: List[Dict[str, Any]],
    models: Dict[str, Any],
    desc: str,
) -> List[Dict[str, Any]]:
    """
    Короткое описание:
        Оценивает предсказанные бустингом параметры на выбранных строках датасета.

    Вход:
        rows (List[Dict[str, Any]]): строки датасета.
        models (Dict[str, Any]): обученные бустинговые модели.
        desc (str): подпись progress bar.

    Выход:
        eval_rows (List[Dict[str, Any]]): список записей с метрикой и предсказанными параметрами.
    """
    eval_rows = []
    for row in tqdm(rows, desc=desc, unit="image"):
        gt_masks = load_ground_truth_masks(row["gt_dir"])
        feature_vector = np.load(row["feature_path"])
        params = predict_params_from_feature(feature_vector, models, FIXED_PARAMS)
        score = evaluate_image(row["image_path"], gt_masks, YOLO_MODEL_PATH, params, debug=False)
        eval_rows.append(
            {
                "file": row["file"],
                "predicted_score": float(score),
                "predicted_params": flatten_params(params),
                "oracle_score": float(row["oracle_score"]),
            }
        )
    return eval_rows


def build_report(
    dataset_run_dir: str,
    train_rows: List[Dict[str, Any]],
    valid_rows: List[Dict[str, Any]],
    train_eval: List[Dict[str, Any]],
    valid_eval: List[Dict[str, Any]],
    feature_dim: int,
) -> Dict[str, Any]:
    """
    Короткое описание:
        Собирает отчёт обучения бустинговых моделей.

    Вход:
        dataset_run_dir (str): путь до датасета, на котором было обучение.
        train_rows (List[Dict[str, Any]]): train-часть датасета.
        valid_rows (List[Dict[str, Any]]): valid-часть датасета.
        train_eval (List[Dict[str, Any]]): оценка на train.
        valid_eval (List[Dict[str, Any]]): оценка на valid.
        feature_dim (int): размерность признакового вектора.

    Выход:
        report (Dict[str, Any]): итоговый отчёт обучения.
    """
    return {
        "dataset_run_dir": dataset_run_dir,
        "catboost_config": {
            "iterations": CATBOOST_ITERATIONS,
            "depth": CATBOOST_DEPTH,
            "learning_rate": CATBOOST_LEARNING_RATE,
            "loss_function": CATBOOST_LOSS_FUNCTION,
            "verbose": CATBOOST_VERBOSE,
        },
        "train_count": len(train_rows),
        "valid_count": len(valid_rows),
        "feature_dim": feature_dim,
        "train_files": [row["file"] for row in train_rows],
        "valid_files": [row["file"] for row in valid_rows],
        "predicted_train": train_eval,
        "predicted_valid": valid_eval,
        "mean_scores": {
            "oracle_train_mean": float(np.mean([row["oracle_score"] for row in train_rows])) if train_rows else 0.0,
            "predicted_train_mean": float(np.mean([row["predicted_score"] for row in train_eval])) if train_eval else 0.0,
            "oracle_valid_mean": float(np.mean([row["oracle_score"] for row in valid_rows])) if valid_rows else 0.0,
            "predicted_valid_mean": float(np.mean([row["predicted_score"] for row in valid_eval])) if valid_eval else 0.0,
        },
    }


def parse_args() -> argparse.Namespace:
    """
    Короткое описание:
        Разбирает аргументы командной строки для обучения бустинга.

    Вход:
        None: использует аргументы командной строки.

    Выход:
        args (argparse.Namespace): распарсенные аргументы.
    """
    parser = argparse.ArgumentParser(description="Обучение бустинговых моделей по готовому датасету.")
    parser.add_argument("--dataset-run", type=str, default=None)
    parser.add_argument("--train-count", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    """
    Короткое описание:
        Точка входа для отдельного сценария обучения бустинговых моделей.

    Вход:
        None: использует аргументы командной строки.

    Выход:
        None: сохраняет модель и отчёт на диск.
    """
    ensure_output_dirs()

    dataset_run_dir = DATASET_RUNS_DIR
    if not os.path.isdir(dataset_run_dir):
        raise FileNotFoundError(f"Не найдена папка датасета: {dataset_run_dir}")

    rows = load_dataset_rows(dataset_run_dir)
    rows = normalize_feature_paths(rows, dataset_run_dir)
    train_rows, valid_rows = split_dataset_rows(rows, TRAIN_COUNT_DEFAULT)
    X_train = load_feature_matrix_from_rows(train_rows)
    target_rows = build_target_rows(train_rows)
    models = fit_boost_models(X_train, target_rows)

    # train_eval = evaluate_rows(train_rows, models, "Evaluate train")
    # valid_eval = evaluate_rows(valid_rows, models, "Evaluate valid") if valid_rows else []

    feature_dim = int(X_train.shape[1])
    # report = build_report(dataset_run_dir, train_rows, valid_rows, train_eval, valid_eval, feature_dim)
    # report_path = os.path.join(DEBUG_REPORT_DIR, REPORT_FILENAME)
    # save_json(report_path, report)

    model_bundle = {
        "models": models,
        "fixed_params": FIXED_PARAMS,
        "param_specs": PARAM_SPECS,
        "feature_dim": feature_dim,
        "dataset_run_dir": dataset_run_dir,
        "train_files": [row["file"] for row in train_rows],
        "valid_files": [row["file"] for row in valid_rows],
    }
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    joblib.dump(model_bundle, model_path)

    print(f"Модели сохранены: {model_path}")
    # print(f"Отчёт сохранён: {report_path}")
    # print(json.dumps(report["mean_scores"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
