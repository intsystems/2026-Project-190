"""
Короткое описание:
    My-версия Das/Panda HPP + seam carving для сегментации строк.

Отличия от das_panda_hpp_seam_exact.py только те, что заданы в задаче:
    1. Перед HPP применяется correct_perspective и warp_binary_by_local_angles
       из processing.py с фиксированной сеткой 6 x 6.
    2. HPP-регионы ищутся после морфологического сглаживания профиля двумя
       ядрами: AH / 2 и AH / 3. Выбирается вариант, давший больше строк.
    3. Остальная логика наследуется от exact-метода.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import das_panda_hpp_seam_exact as exact
from processing import correct_perspective, warp_binary_by_local_angles


DEBUG_IMAGES_DIR = str(PROJECT_ROOT / "debug_images" / "experiment_2_compare_paper_hpp" / "my_das_panda_hpp_seam_exact")
INPUT_IMAGE_PATH = exact.INPUT_IMAGE_PATH
DEBUG = True

USE_YOLO_PAGE_SEGMENTATION = exact.USE_YOLO_PAGE_SEGMENTATION
YOLO_PAGE_SEGMENTATION_MODEL_PATH = exact.YOLO_PAGE_SEGMENTATION_MODEL_PATH
YOLO_PAGE_CONF = exact.YOLO_PAGE_CONF
YOLO_PAGE_IMGSZ = exact.YOLO_PAGE_IMGSZ
YOLO_PAGE_DEVICE = exact.YOLO_PAGE_DEVICE

UNET_BINARIZATION_MODEL_PATH = exact.UNET_BINARIZATION_MODEL_PATH
UNET_TARGET_SIZE = exact.UNET_TARGET_SIZE
UNET_THRESHOLD = exact.UNET_THRESHOLD
UNET_DEVICE = exact.UNET_DEVICE

# Оставляем таким же, как в hpp_method.py.
HPP_MIN_PLATEAU_RISE_FRACTION = 0.05

# По задаче сетка local warp фиксирована и не подбирается.
LOCAL_WARP_GRID_ROWS = 6
LOCAL_WARP_GRID_COLS = 6


class MyDasPandaHPPSeamSegmenter(exact.DasPandaHPPSeamSegmenter):
    """
    Наследует exact-реализацию seam carving. Меняются только бинаризация/warp и
    способ получения text regions из HPP.
    """

    def __init__(
        self,
        image: np.ndarray,
        debug: bool = False,
        debug_output_dir: str = DEBUG_IMAGES_DIR,
        unet_model: Any = None,
        unet_device: Any = None,
    ):
        super().__init__(
            image=image,
            debug=debug,
            debug_output_dir=debug_output_dir,
            unet_model=unet_model,
            unet_device=unet_device,
        )
        self.average_character_height = 10.0
        self.hpp_smoothed_variants: List[Dict[str, Any]] = []
        self.selected_hpp_kernel_size = 0
        self.selected_hpp_smoothed: Optional[np.ndarray] = None
        self.output_to_page_x: Optional[np.ndarray] = None
        self.output_to_page_y: Optional[np.ndarray] = None
        self.page_input_shape: Tuple[int, int] = self.image.shape[:2]

    def detect(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        self.binary_article = self.binarize_with_unet_as_article_binary()
        self.average_character_height = self.estimate_average_character_height_from_article_binary(self.binary_article)

        self.hpp = self.compute_hpp(self.binary_article)
        self.hpp_normalized = self.normalize_hpp_minmax(self.hpp)
        self.text_regions = self.extract_text_regions(self.hpp_normalized)
        self.clustered_regions = self.cluster_line_regions(self.text_regions)
        self.start_points = self.compute_seam_starting_points(self.clustered_regions)

        self.modified_energy = self.compute_modified_energy(self.binary_article, self.clustered_regions)
        self.min_energy_matrix = self.compute_horizontal_min_energy_path_matrix(self.modified_energy)
        self.seams = [self.trace_horizontal_seam(self.min_energy_matrix, start_row) for start_row in self.start_points]

        self.line_bands = self.compute_line_bands(self.seams, self.binary_article.shape[0])
        class_matrix, lines = self.build_class_matrix_from_seams(self.binary_article, self.seams)
        restored_class_matrix = self.restore_class_matrix_to_page_coordinates(class_matrix)

        if self.debug:
            self.save_all_debug(class_matrix, lines)
            self._save_debug_image(exact.colorize_class_matrix(restored_class_matrix), "10_my_restored_class_matrix_page_coords")
            self.save_detection_rectangles_debug(restored_class_matrix)
        return restored_class_matrix, lines

    def binarize_with_unet_as_article_binary(self) -> np.ndarray:
        """
        U-Net -> correct_perspective -> warp_binary_by_local_angles(6x6) ->
        article binary convention: background=0, text=255.
        """
        from u_net_binarization import binarize_image_with_loaded_model, load_unet_model

        if self.unet_model is None or self.unet_device is None:
            model, device = load_unet_model(str(UNET_BINARIZATION_MODEL_PATH), device=UNET_DEVICE)
            owns_model = True
        else:
            model, device = self.unet_model, self.unet_device
            owns_model = False

        try:
            unet_mask = binarize_image_with_loaded_model(
                image=self.image,
                model=model,
                device=device,
                target_size=UNET_TARGET_SIZE,
                threshold=UNET_THRESHOLD,
                debug=False,
            )
        finally:
            if owns_model:
                del model
                gc.collect()

        unet_mask = np.where(unet_mask < 128, 0, 255).astype(np.uint8)
        _, corrected_binary, global_angle, first_matrix = correct_perspective(
            unet_mask,
            debug=self.debug,
            debug_output_dir=self.debug_output_dir,
            return_matrix=True,
        )
        warped_binary, transform_sequence = warp_binary_by_local_angles(
            corrected_binary,
            grid_rows=LOCAL_WARP_GRID_ROWS,
            grid_cols=LOCAL_WARP_GRID_COLS,
            hyperparameter_selection=False,
            debug=self.debug,
            debug_output_dir=self.debug_output_dir,
            return_transform_sequence=True,
        )
        warped_binary = np.where(warped_binary < 128, 0, 255).astype(np.uint8)
        article_binary = np.where(warped_binary == 0, 255, 0).astype(np.uint8)
        self.prepare_output_to_page_mapping(
            transform_sequence=transform_sequence,
            first_matrix=first_matrix,
            page_shape=unet_mask.shape,
        )

        self.gray = warped_binary.copy()
        if self.debug:
            self._save_debug_image(unet_mask, "01_unet_raw_text_black")
            self._save_debug_image(corrected_binary, "02_correct_perspective_binary_text_black")
            self._save_debug_image(warped_binary, "03_local_warp_6x6_binary_text_black")
            self._save_debug_image(article_binary, "04_article_binary_after_warp_text_white")
            with open(os.path.join(self.debug_output_dir, "my_preprocessing.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "correct_perspective_global_angle": float(global_angle),
                        "local_warp_grid_rows": int(LOCAL_WARP_GRID_ROWS),
                        "local_warp_grid_cols": int(LOCAL_WARP_GRID_COLS),
                    },
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
        return article_binary

    def prepare_output_to_page_mapping(
        self,
        transform_sequence: Dict[str, Any],
        first_matrix: np.ndarray,
        page_shape: Tuple[int, int],
    ) -> None:
        """
        Строит карту warped-output -> исходный page crop.
        """
        output_to_corrected_x = transform_sequence["output_to_input_x"].astype(np.float32)
        output_to_corrected_y = transform_sequence["output_to_input_y"].astype(np.float32)
        inverse_first_matrix = cv2.invertAffineTransform(first_matrix)

        page_x = (
            inverse_first_matrix[0, 0] * output_to_corrected_x
            + inverse_first_matrix[0, 1] * output_to_corrected_y
            + inverse_first_matrix[0, 2]
        )
        page_y = (
            inverse_first_matrix[1, 0] * output_to_corrected_x
            + inverse_first_matrix[1, 1] * output_to_corrected_y
            + inverse_first_matrix[1, 2]
        )
        self.output_to_page_x = page_x.astype(np.float32)
        self.output_to_page_y = page_y.astype(np.float32)
        self.page_input_shape = (int(page_shape[0]), int(page_shape[1]))

    def restore_class_matrix_to_page_coordinates(self, class_matrix: np.ndarray) -> np.ndarray:
        """
        Возвращает class_matrix из координат warped-бинарки в координаты исходного
        page crop, чтобы сравнение с GT шло в одной системе координат.
        """
        if self.output_to_page_x is None or self.output_to_page_y is None:
            return class_matrix

        restored = np.zeros(self.page_input_shape, dtype=np.int32)
        ys, xs = np.where(class_matrix > 0)
        if len(xs) == 0:
            return restored

        mapped_x = np.rint(self.output_to_page_x[ys, xs]).astype(np.int32)
        mapped_y = np.rint(self.output_to_page_y[ys, xs]).astype(np.int32)
        valid = (
            (mapped_x >= 0)
            & (mapped_x < restored.shape[1])
            & (mapped_y >= 0)
            & (mapped_y < restored.shape[0])
        )
        restored[mapped_y[valid], mapped_x[valid]] = class_matrix[ys[valid], xs[valid]]
        return restored

    def save_detection_rectangles_debug(self, class_matrix: np.ndarray) -> None:
        """
        Сохраняет minAreaRect-прямоугольники по восстановленной class_matrix:
        это ровно те прямоугольники, которые compare потом оценивает через IoU.
        """
        if not self.debug:
            return
        if len(self.image.shape) == 2:
            canvas = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = self.image.copy()

        max_class = int(np.max(class_matrix)) if class_matrix.size else 0
        rng = np.random.default_rng(12345)
        colors = rng.integers(40, 230, size=(max_class + 1, 3), dtype=np.uint8)
        rectangles = []
        for class_index in range(1, max_class + 1):
            ys, xs = np.where(class_matrix == class_index)
            if len(xs) < 3:
                continue
            points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect).astype(np.int32)
            color = tuple(int(value) for value in colors[class_index])
            cv2.polylines(canvas, [box.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(class_index),
                (int(box[0, 0]), max(18, int(box[0, 1]) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            rectangles.append(
                {
                    "class_id": int(class_index),
                    "points": [[int(x), int(y)] for x, y in box.tolist()],
                    "pixel_count": int(len(xs)),
                }
            )

        self._save_debug_image(canvas, "11_my_detection_minrectangles_page_coords")
        with open(os.path.join(self.debug_output_dir, "my_detection_rectangles.json"), "w", encoding="utf-8") as file:
            json.dump({"rectangles": rectangles}, file, indent=2, ensure_ascii=False)

    def estimate_average_character_height_from_article_binary(self, binary_article: np.ndarray) -> float:
        """
        Как в my_louloudis: высоты connected components, удаляем первый bin,
        считаем weighted mean.
        """
        text_mask = (binary_article > 0).astype(np.uint8)
        labels_count, _, stats, _ = cv2.connectedComponentsWithStats(text_mask, connectivity=8)
        heights = [
            int(stats[label, cv2.CC_STAT_HEIGHT])
            for label in range(1, labels_count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= 1
        ]
        if len(heights) == 0:
            return 10.0

        values_all, counts_all = np.unique(np.asarray(heights, dtype=np.int32), return_counts=True)
        values_used = values_all[1:] if len(values_all) > 1 else values_all
        counts_used = counts_all[1:] if len(counts_all) > 1 else counts_all
        average_height = float(
            np.sum(values_used.astype(np.float64) * counts_used.astype(np.float64))
            / max(1.0, float(np.sum(counts_used)))
        )

        if self.debug:
            with open(os.path.join(self.debug_output_dir, "my_average_character_height.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "average_character_height": float(average_height),
                        "height_histogram_all": {
                            "values": [int(value) for value in values_all.tolist()],
                            "counts": [int(value) for value in counts_all.tolist()],
                        },
                        "height_histogram_used_for_mean": {
                            "values": [int(value) for value in values_used.tolist()],
                            "counts": [int(value) for value in counts_used.tolist()],
                        },
                    },
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
        return average_height

    def normalize_hpp_minmax(self, hpp: np.ndarray) -> np.ndarray:
        hpp = np.asarray(hpp, dtype=np.float32)
        min_value = float(np.min(hpp)) if hpp.size else 0.0
        max_value = float(np.max(hpp)) if hpp.size else 0.0
        if max_value - min_value < 1e-9:
            return np.zeros_like(hpp, dtype=np.float32)
        return ((hpp - min_value) / (max_value - min_value)).astype(np.float32)

    def _make_odd_window(self, requested_window: int, profile_length: int) -> int:
        window = max(3, min(int(requested_window), int(profile_length)))
        if window % 2 == 0:
            window -= 1
        return max(3, window)

    def _morphological_smooth_profile(self, profile: np.ndarray, kernel_size: int) -> np.ndarray:
        kernel_size = self._make_odd_window(kernel_size, len(profile))
        profile_min = float(np.min(profile))
        profile_max = float(np.max(profile))
        if profile_max - profile_min < 1e-9:
            return profile.astype(np.float32)

        normalized_u8 = (self.normalize_hpp_minmax(profile) * 255.0).astype(np.uint8).reshape(1, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
        closed = cv2.morphologyEx(normalized_u8, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        smoothed = opened.reshape(-1).astype(np.float32) / 255.0
        return smoothed * (profile_max - profile_min) + profile_min

    def _find_plateau_rows_by_kernel(self, normalized_hpp: np.ndarray, kernel_size: int) -> Tuple[np.ndarray, np.ndarray]:
        smoothed_hpp = self._morphological_smooth_profile(normalized_hpp.astype(np.float32), kernel_size)
        smoothed_u8 = np.clip(self.normalize_hpp_minmax(smoothed_hpp) * 255.0, 0, 255).astype(np.uint8)
        min_rise = int(np.ceil(255.0 * HPP_MIN_PLATEAU_RISE_FRACTION))

        plateau_rows = np.zeros_like(smoothed_u8, dtype=bool)
        row_idx = 0
        while row_idx < len(smoothed_u8):
            plateau_start = row_idx
            plateau_value = int(smoothed_u8[row_idx])
            while row_idx + 1 < len(smoothed_u8) and int(smoothed_u8[row_idx + 1]) == plateau_value:
                row_idx += 1
            plateau_end = row_idx

            has_left_rise = plateau_start > 0 and int(smoothed_u8[plateau_start - 1]) < plateau_value
            has_right_fall = plateau_end + 1 < len(smoothed_u8) and int(smoothed_u8[plateau_end + 1]) < plateau_value
            if plateau_value > 0 and has_left_rise and has_right_fall:
                left_base = plateau_start - 1
                while left_base > 0 and int(smoothed_u8[left_base - 1]) <= int(smoothed_u8[left_base]):
                    left_base -= 1

                right_base = plateau_end + 1
                while right_base + 1 < len(smoothed_u8) and int(smoothed_u8[right_base + 1]) <= int(smoothed_u8[right_base]):
                    right_base += 1

                left_rise = plateau_value - int(smoothed_u8[left_base])
                right_rise = plateau_value - int(smoothed_u8[right_base])
                if min(left_rise, right_rise) >= min_rise:
                    plateau_rows[plateau_start:plateau_end + 1] = True

            row_idx += 1
        return smoothed_hpp, plateau_rows

    def _regions_from_mask(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        regions: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for row_index, value in enumerate(mask):
            if bool(value) and start is None:
                start = row_index
            elif (not bool(value)) and start is not None:
                regions.append((int(start), int(row_index - 1)))
                start = None
        if start is not None:
            regions.append((int(start), int(len(mask) - 1)))
        return regions

    def extract_text_regions(self, normalized_hpp: np.ndarray) -> List[Tuple[int, int]]:
        ah = max(3.0, float(self.average_character_height))
        kernel_candidates = [
            max(3, int(round(ah / 2.0))),
            max(3, int(round(ah / 3.0))),
        ]

        variants: List[Dict[str, Any]] = []
        for kernel_size in kernel_candidates:
            smoothed, plateau_rows = self._find_plateau_rows_by_kernel(normalized_hpp, kernel_size)
            regions = self._regions_from_mask(plateau_rows)
            variants.append(
                {
                    "kernel_size": int(self._make_odd_window(kernel_size, len(normalized_hpp))),
                    "smoothed": smoothed,
                    "regions": regions,
                    "line_count": int(len(regions)),
                }
            )

        selected = max(variants, key=lambda item: item["line_count"])
        self.hpp_smoothed_variants = variants
        self.selected_hpp_kernel_size = int(selected["kernel_size"])
        self.selected_hpp_smoothed = selected["smoothed"]

        if self.debug:
            with open(os.path.join(self.debug_output_dir, "my_hpp_smoothing_selection.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "average_character_height": float(self.average_character_height),
                        "hpp_min_plateau_rise_fraction": float(HPP_MIN_PLATEAU_RISE_FRACTION),
                        "variants": [
                            {
                                "kernel_size": int(item["kernel_size"]),
                                "line_count": int(item["line_count"]),
                                "regions": [[int(a), int(b)] for a, b in item["regions"]],
                            }
                            for item in variants
                        ],
                        "selected_kernel_size": int(self.selected_hpp_kernel_size),
                    },
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
        return [(int(a), int(b)) for a, b in selected["regions"]]

    def save_hpp_debug(self) -> None:
        super().save_hpp_debug()
        if not self.debug or self.selected_hpp_smoothed is None:
            return
        values = self.selected_hpp_smoothed.astype(np.float32)
        height = len(values)
        width = 900
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        normalized = self.normalize_hpp_minmax(values)
        for row, value in enumerate(normalized):
            x = int(round(float(value) * (width - 1)))
            cv2.line(canvas, (0, row), (x, row), (0, 0, 0), 1)
        self._save_debug_image(canvas, "09_my_selected_smoothed_hpp")

    def save_summary(self, class_matrix: np.ndarray, lines: List[Dict[str, Any]]) -> None:
        super().save_summary(class_matrix, lines)
        path = os.path.join(self.debug_output_dir, "my_summary.json")
        summary = {
            "method": "my_das_panda_hpp_seam_exact",
            "changes": [
                "correct_perspective before HPP",
                "warp_binary_by_local_angles with fixed 6x6 grid",
                "morphological HPP smoothing with AH/2 and AH/3 kernels, choose more lines",
            ],
            "average_character_height": float(self.average_character_height),
            "selected_hpp_kernel_size": int(self.selected_hpp_kernel_size),
            "line_count": int(len(lines)),
            "class_count": int(np.max(class_matrix)) if class_matrix.size else 0,
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)


def run_on_image(
    image_path: str,
    debug: bool = DEBUG,
    debug_output_dir: str = DEBUG_IMAGES_DIR,
    yolo_model: Any = None,
    unet_model: Any = None,
    unet_device: Any = None,
    return_page_info: bool = False,
    **_: Any,
) -> Any:
    run_start_time = time.perf_counter()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    if USE_YOLO_PAGE_SEGMENTATION:
        page_result = exact.extract_largest_page_with_yolo(
            image,
            debug=debug,
            debug_output_dir=debug_output_dir,
            yolo_model=yolo_model,
        )
        if page_result is not None:
            image = page_result["page_image"]
        else:
            page_result = {
                "page_image": image,
                "page_mask": np.ones(image.shape[:2], dtype=np.uint8) * 255,
                "bbox": {"x": 0, "y": 0, "w": int(image.shape[1]), "h": int(image.shape[0])},
                "confidence": None,
            }
    else:
        page_result = {
            "page_image": image,
            "page_mask": np.ones(image.shape[:2], dtype=np.uint8) * 255,
            "bbox": {"x": 0, "y": 0, "w": int(image.shape[1]), "h": int(image.shape[0])},
            "confidence": None,
        }

    detector = MyDasPandaHPPSeamSegmenter(
        image=image,
        debug=debug,
        debug_output_dir=debug_output_dir,
        unet_model=unet_model,
        unet_device=unet_device,
    )
    detector_start_time = time.perf_counter()
    class_matrix, lines = detector.detect()
    detector_runtime_sec = time.perf_counter() - detector_start_time
    page_result["timing_seconds"] = {
        "total_run_on_image": float(time.perf_counter() - run_start_time),
        "detector": float(detector_runtime_sec),
    }
    if return_page_info:
        return class_matrix, lines, page_result
    return class_matrix, lines


def main() -> None:
    if not INPUT_IMAGE_PATH:
        print("Укажи INPUT_IMAGE_PATH или импортируй run_on_image(...) из этого файла.")
        return
    class_matrix, lines, page_info = run_on_image(
        INPUT_IMAGE_PATH,
        debug=DEBUG,
        debug_output_dir=DEBUG_IMAGES_DIR,
        return_page_info=True,
    )
    print(f"[OK] lines={len(lines)}, classes={int(np.max(class_matrix)) if class_matrix.size else 0}")
    print(f"[OK] page_info={page_info.get('bbox')}")
    print(f"[OK] timing={page_info.get('timing_seconds')}")
    print(f"[OK] debug={DEBUG_IMAGES_DIR}")


if __name__ == "__main__":
    main()
