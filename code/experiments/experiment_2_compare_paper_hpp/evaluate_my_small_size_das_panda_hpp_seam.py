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


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import compare_das_panda_hpp_exact_vs_my as compare
import das_panda_hpp_seam_exact as exact


MODULE_PATH = EXPERIMENT_DIR / "my_small_size_das_panda_hpp_seam_exact.py"
OUTPUT_DIR = PROJECT_ROOT / "debug_images" / "experiment_2_compare_paper_hpp" / "evaluate_my_small_size_das_panda_hpp_seam"
OUTPUT_JSON_PATH = OUTPUT_DIR / "summary.json"
TARGET_VIS_DIR = OUTPUT_DIR / "targets"

# None: весь test.txt. Число: первые N.
EVALUATE_N_IMAGES = None
IOU_THRESHOLD = 0.5
CLEAR_OUTPUT_DIR_ON_START = True


def load_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось загрузить модуль: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_compare_helpers() -> None:
    compare.COMPARE_N_IMAGES = EVALUATE_N_IMAGES
    compare.IOU_THRESHOLD = IOU_THRESHOLD
    compare.OUTPUT_DIR = OUTPUT_DIR
    compare.OUTPUT_JSON_PATH = OUTPUT_JSON_PATH
    compare.TARGET_VIS_DIR = TARGET_VIS_DIR


def evaluate_module(
    module: Any,
    rows: List[Dict[str, Any]],
    yolo_model: Any,
    unet_model: Any,
    unet_device: Any,
) -> Dict[str, Any]:
    method_name = "my_small_size_das_panda_hpp_seam_exact"
    per_image = []
    skipped_missing = 0
    skipped_runtime = 0

    for row in tqdm(rows, desc=f"Evaluate {method_name}"):
        if cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR) is None:
            skipped_missing += 1
            continue
        try:
            start_time = time.perf_counter()
            class_matrix, lines, page_info = module.run_on_image(
                str(row["image_path"]),
                debug=False,
                yolo_model=yolo_model,
                unet_model=unet_model,
                unet_device=unet_device,
                return_page_info=True,
            )
            runtime_sec = time.perf_counter() - start_time
        except Exception as error:
            skipped_runtime += 1
            per_image.append(
                {
                    "relative_path": row["relative_path"],
                    "error": repr(error),
                    "runtime_sec": None,
                    "metrics": {
                        "tp": 0.0,
                        "fp": 0.0,
                        "fn": float(len(row["gt_polygons"])),
                        "precision": 0.0,
                        "recall": 0.0,
                        "hmean": 0.0,
                    },
                    "line_count": 0,
                }
            )
            continue

        bbox = page_info["bbox"]
        predicted = compare.class_matrix_to_polygons(class_matrix, int(bbox["x"]), int(bbox["y"]))
        metrics = compare.match_polygons(predicted, row["gt_polygons"])
        compare.save_method_visuals(row, method_name, predicted, metrics)
        per_image.append(
            {
                "relative_path": row["relative_path"],
                "metrics": metrics,
                "runtime_sec": float(runtime_sec),
                "method_timing_seconds": page_info.get("timing_seconds"),
                "line_count": int(len(lines)),
                "page_bbox": bbox,
                "page_confidence": page_info.get("confidence"),
            }
        )

    return {
        "method": method_name,
        "aggregate": compare.aggregate(per_image),
        "runtime": compare.aggregate_runtime(per_image),
        "runtime_scope": "module.run_on_image: YOLO page crop + U-Net binarization + my_small_size method logic",
        "per_image": per_image,
        "skipped_missing": int(skipped_missing),
        "skipped_runtime": int(skipped_runtime),
    }


def main() -> None:
    configure_compare_helpers()
    if CLEAR_OUTPUT_DIR_ON_START and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_VIS_DIR.mkdir(parents=True, exist_ok=True)

    module = load_module("my_small_size_das_panda_hpp_eval", MODULE_PATH)
    rows = compare.build_rows()

    for row in tqdm(rows, desc="Save target GT overlays"):
        compare.save_target_visualization(row)

    from ultralytics import YOLO
    from u_net_binarization import load_unet_model

    yolo_model = YOLO(str(exact.YOLO_PAGE_SEGMENTATION_MODEL_PATH))
    unet_model, unet_device = load_unet_model(
        model_path=str(exact.UNET_BINARIZATION_MODEL_PATH),
        device=exact.UNET_DEVICE,
    )

    result = evaluate_module(module, rows, yolo_model, unet_model, unet_device)
    summary = {
        "evaluate_n_images": EVALUATE_N_IMAGES,
        "rows_count": len(rows),
        "skipped_input_rows": compare.SKIPPED_INPUT_ROWS,
        "iou_threshold": IOU_THRESHOLD,
        "module_path": str(MODULE_PATH),
        "result": result,
    }
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    aggregate = result["aggregate"]
    runtime = result["runtime"]
    print(
        "[OK] My small-size hmean: "
        f"{aggregate['hmean']:.6f}, precision: {aggregate['precision']:.6f}, recall: {aggregate['recall']:.6f}"
    )
    print(f"[OK] TP/FP/FN: {aggregate['tp']:.0f}/{aggregate['fp']:.0f}/{aggregate['fn']:.0f}")
    print(f"[OK] Runtime mean/median: {runtime['mean_runtime_sec']:.3f}s / {runtime['median_runtime_sec']:.3f}s")
    print(f"[OK] Summary: {OUTPUT_JSON_PATH}")
    print(f"[OK] Targets/debug: {TARGET_VIS_DIR}")


if __name__ == "__main__":
    main()
