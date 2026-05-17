import json
import math
from pathlib import Path
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "datasets/boost_hough_datasets/3/param_importances"
DATASET_ROWS_PATH = BASE_DIR / "datasets/boost_hough_datasets/3/dataset_rows.json"
OUTPUT_IMPORTANCES_PATH = BASE_DIR / "debug_images/param_importances_mean_variance_cv.json"
OUTPUT_HYPERPARAMS_PATH = BASE_DIR / "debug_images/hyperparams_mean_variance_cv.json"
EPS = 1e-12
IMPORTANT_THRESHOLD = 0.15


def compute_stats(
    values: dict[str, list[float]],
    add_importance_label: bool = False,
) -> dict[str, dict[str, float | int | None | str]]:
    """
    Короткое описание:
        Считает mean, variance, std и cv для словаря списков чисел.
    Вход:
        values (dict[str, list[float]]): словарь вида параметр -> список значений.
        add_importance_label (bool): если True, добавляет метку важности через IMPORTANT_THRESHOLD.
    Выход:
        dict[str, dict[str, float | int | None | str]]: статистика по каждому параметру.
    """
    stats = {}
    for key, arr in values.items():
        if not arr:
            continue
        mean = sum(arr) / len(arr)
        variance = sum((x - mean) ** 2 for x in arr) / len(arr)
        std = math.sqrt(variance)
        cv = (std / abs(mean)) if abs(mean) > EPS else None
        row = {
            "mean": mean,
            "variance": variance,
            "std": std,
            "cv": cv,
            "max_importance": max(arr),
            "count": len(arr),
        }
        if add_importance_label:
            row["importance_label"] = "не важен" if row["max_importance"] <= IMPORTANT_THRESHOLD else "важен"
        stats[key] = row
    return stats


def main() -> None:
    """
    Короткое описание:
        Считает mean/variance/std/cv для param_importances и для oracle_params из dataset_rows.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    importances_values = defaultdict(list)
    importance_files = sorted(INPUT_DIR.glob("*.json"))
    for json_path in importance_files:
        with open(json_path, "r", encoding="utf-8") as fp:
            row = json.load(fp)
        for key, value in row.items():
            if isinstance(value, (int, float)):
                importances_values[key].append(float(value))

    hyperparams_values = defaultdict(list)
    with open(DATASET_ROWS_PATH, "r", encoding="utf-8") as fp:
        dataset_rows = json.load(fp)
    for row in dataset_rows:
        oracle_params = row.get("oracle_params", {})
        if not isinstance(oracle_params, dict):
            continue
        for key, value in oracle_params.items():
            if isinstance(value, (int, float)):
                hyperparams_values[key].append(float(value))

    OUTPUT_IMPORTANCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_IMPORTANCES_PATH, "w", encoding="utf-8") as fp:
        json.dump(compute_stats(importances_values, add_importance_label=True), fp, ensure_ascii=False, indent=2)
    with open(OUTPUT_HYPERPARAMS_PATH, "w", encoding="utf-8") as fp:
        json.dump(compute_stats(hyperparams_values), fp, ensure_ascii=False, indent=2)

    print(f"Read importance files: {len(importance_files)} from {INPUT_DIR}")
    print(f"Read dataset rows: {len(dataset_rows)} from {DATASET_ROWS_PATH}")
    print(f"Wrote importance stats: {len(importances_values)} params -> {OUTPUT_IMPORTANCES_PATH}")
    print(f"Wrote hyperparam stats: {len(hyperparams_values)} params -> {OUTPUT_HYPERPARAMS_PATH}")


if __name__ == "__main__":
    main()
