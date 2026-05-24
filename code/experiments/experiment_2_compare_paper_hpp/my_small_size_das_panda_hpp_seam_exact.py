"""
Small-size вариант my_das_panda_hpp_seam_exact.py.

Меняем только то, что задано:
    1. Local warp оценивается на уменьшенной corrected-бинарке.
    2. HPP и A* работают на уменьшенной warped-бинарке.
    3. Результирующая small class-matrix масштабируется обратно на полный размер.
    4. HPP_SMALL_HEIGHT фиксирован, HPP_SMALL_WIDTH считается по aspect ratio.
"""

import heapq
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
import my_das_panda_hpp_seam_exact as base_my
from processing import correct_perspective


DEBUG_IMAGES_DIR = str(
    PROJECT_ROOT
    / "debug_images"
    / "experiment_2_compare_paper_hpp"
    / "my_small_size_das_panda_hpp_seam_exact"
)
INPUT_IMAGE_PATH = exact.INPUT_IMAGE_PATH
DEBUG = True

USE_YOLO_PAGE_SEGMENTATION = exact.USE_YOLO_PAGE_SEGMENTATION
YOLO_PAGE_CONF = exact.YOLO_PAGE_CONF
YOLO_PAGE_IMGSZ = exact.YOLO_PAGE_IMGSZ
YOLO_PAGE_DEVICE = exact.YOLO_PAGE_DEVICE


UNET_BINARIZATION_MODEL_PATH = base_my.UNET_BINARIZATION_MODEL_PATH
UNET_TARGET_SIZE = base_my.UNET_TARGET_SIZE
UNET_THRESHOLD = base_my.UNET_THRESHOLD
UNET_DEVICE = base_my.UNET_DEVICE


# None: ширина считается по пропорции от HPP_SMALL_HEIGHT и aspect ratio текущего изображения.
HPP_SMALL_WIDTH = None
HPP_SMALL_HEIGHT = 640


class MySmallSizeDasPandaHPPSeamSegmenter(base_my.MyDasPandaHPPSeamSegmenter):
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
        self.small_binary_article: Optional[np.ndarray] = None
        self.small_hpp: Optional[np.ndarray] = None
        self.small_hpp_normalized: Optional[np.ndarray] = None
        self.small_text_regions: List[Tuple[int, int]] = []
        self.small_average_character_height = 10.0
        self.small_to_full_y_scale = 1.0
        self.small_to_full_x_scale = 1.0
        self.current_small_width = 0
        self.current_small_height = 0
        self.timing_seconds: Dict[str, float] = {}

    def get_proportional_small_size(self, full_width: int, full_height: int) -> Tuple[int, int]:
        small_height = max(1, int(HPP_SMALL_HEIGHT))
        if HPP_SMALL_WIDTH is None:
            small_width = int(round(float(full_width) * float(small_height) / max(1.0, float(full_height))))
        else:
            small_width = int(HPP_SMALL_WIDTH)
        return max(1, small_width), small_height

    def binarize_with_unet_as_article_binary(self) -> np.ndarray:
        """
        U-Net -> correct_perspective на полном изображении -> local warp,
        оцененный на small-копии, но примененный к полной corrected-бинарке.
        """
        from u_net_binarization import binarize_image_with_loaded_model, load_unet_model

        if self.unet_model is None or self.unet_device is None:
            model, device = load_unet_model(str(base_my.UNET_BINARIZATION_MODEL_PATH), device=base_my.UNET_DEVICE)
            owns_model = True
        else:
            model, device = self.unet_model, self.unet_device
            owns_model = False

        try:
            stage_t0 = time.perf_counter()
            unet_mask = binarize_image_with_loaded_model(
                image=self.image,
                model=model,
                device=device,
                target_size=base_my.UNET_TARGET_SIZE,
                threshold=base_my.UNET_THRESHOLD,
                debug=False,
            )
            self.timing_seconds["unet_binarization"] = time.perf_counter() - stage_t0
        finally:
            if owns_model:
                del model

        unet_mask = np.where(unet_mask < 128, 0, 255).astype(np.uint8)
        stage_t0 = time.perf_counter()
        _, corrected_binary, global_angle, first_matrix = correct_perspective(
            unet_mask,
            debug=self.debug,
            debug_output_dir=self.debug_output_dir,
            return_matrix=True,
        )
        corrected_binary = np.where(corrected_binary < 128, 0, 255).astype(np.uint8)
        self.timing_seconds["correct_perspective"] = time.perf_counter() - stage_t0

        stage_t0 = time.perf_counter()
        warped_binary, transform_sequence, small_warp_debug = self.apply_small_local_warp_to_full_binary(
            corrected_binary=corrected_binary,
            first_matrix=first_matrix,
            page_shape=unet_mask.shape,
        )
        self.timing_seconds["small_local_warp_estimate_and_apply"] = time.perf_counter() - stage_t0
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
            self._save_debug_image(small_warp_debug["small_corrected_binary"], "03_small_for_local_warp_text_black")
            self._save_debug_image(small_warp_debug["small_displacement_vis"], "04_small_warp_displacement_y")
            self._save_debug_image(small_warp_debug["full_displacement_vis"], "05_full_resized_warp_displacement_y")
            self._save_debug_image(warped_binary, "06_small_estimated_local_warp_full_binary_text_black")
            self._save_debug_image(article_binary, "07_article_binary_after_small_warp_text_white")
            with open(os.path.join(self.debug_output_dir, "my_small_preprocessing.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "correct_perspective_global_angle": float(global_angle),
                        "small_warp_width": int(self.current_small_width),
                        "small_warp_height": int(self.current_small_height),
                        "small_width_hyperparameter": HPP_SMALL_WIDTH,
                        "small_height_hyperparameter": int(HPP_SMALL_HEIGHT),
                        "local_warp_grid_rows": int(base_my.LOCAL_WARP_GRID_ROWS),
                        "local_warp_grid_cols": int(base_my.LOCAL_WARP_GRID_COLS),
                        "angle_map": small_warp_debug["angle_map"].astype(float).tolist(),
                    },
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
        return article_binary

    def apply_small_local_warp_to_full_binary(
        self,
        corrected_binary: np.ndarray,
        first_matrix: np.ndarray,
        page_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, np.ndarray]]:
        full_height, full_width = corrected_binary.shape[:2]
        small_width, small_height = self.get_proportional_small_size(full_width, full_height)
        self.current_small_width = int(small_width)
        self.current_small_height = int(small_height)
        small_binary = cv2.resize(
            corrected_binary,
            (int(small_width), int(small_height)),
            interpolation=cv2.INTER_NEAREST,
        )
        small_displacement_y, angle_map = self.estimate_small_local_displacement_y(small_binary)
        full_displacement_y = cv2.resize(
            small_displacement_y,
            (full_width, full_height),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)
        full_displacement_y *= float(full_height) / float(small_height)
        full_displacement_y -= np.mean(full_displacement_y, axis=1, keepdims=True)

        grid_x, grid_y = np.meshgrid(
            np.arange(full_width, dtype=np.float32),
            np.arange(full_height, dtype=np.float32),
        )
        map_x = grid_x
        map_y = grid_y + full_displacement_y
        warped_binary = cv2.remap(
            corrected_binary,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        warped_binary = np.where(warped_binary < 128, 0, 255).astype(np.uint8)

        coord_x, coord_y = np.meshgrid(
            np.arange(full_width, dtype=np.float32),
            np.arange(full_height, dtype=np.float32),
        )
        coord_x = cv2.remap(
            coord_x,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1,
        )
        coord_y = cv2.remap(
            coord_y,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1,
        )
        transform_sequence = {
            "name": "small_size_local_warp_applied_to_full",
            "input_shape": (int(page_shape[0]), int(page_shape[1])),
            "output_shape": (int(warped_binary.shape[0]), int(warped_binary.shape[1])),
            "output_to_input_x": coord_x.astype(np.float32),
            "output_to_input_y": coord_y.astype(np.float32),
        }
        debug_info = {
            "small_corrected_binary": small_binary,
            "small_displacement_vis": self.displacement_to_vis(small_displacement_y),
            "full_displacement_vis": self.displacement_to_vis(full_displacement_y),
            "angle_map": angle_map,
        }
        return warped_binary, transform_sequence, debug_info

    def estimate_small_local_displacement_y(self, small_binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = small_binary.shape[:2]
        grid_rows = max(1, int(base_my.LOCAL_WARP_GRID_ROWS))
        grid_cols = max(1, int(base_my.LOCAL_WARP_GRID_COLS))
        y_bounds = np.linspace(0, height, grid_rows + 1, dtype=np.int32)
        x_bounds = np.linspace(0, width, grid_cols + 1, dtype=np.int32)
        angle_map = np.zeros((grid_rows, grid_cols), dtype=np.float32)

        for gy in range(grid_rows):
            for gx in range(grid_cols):
                y0, y1 = int(y_bounds[gy]), int(y_bounds[gy + 1])
                x0, x1 = int(x_bounds[gx]), int(x_bounds[gx + 1])
                cell = small_binary[y0:y1, x0:x1]
                if int(np.sum(cell == 0)) < 32:
                    angle_map[gy, gx] = 0.0
                    continue
                try:
                    _, _, local_angle = correct_perspective(cell, debug=False)
                except Exception:
                    local_angle = 0.0
                angle_map[gy, gx] = float(np.clip(float(local_angle), -8.0, 8.0))

        dense_angles = cv2.resize(angle_map, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        slopes = np.tan(np.deg2rad(dense_angles)).astype(np.float32)
        displacement_y = np.zeros((height, width), dtype=np.float32)
        for x in range(1, width):
            displacement_y[:, x] = displacement_y[:, x - 1] + slopes[:, x - 1]
        displacement_y -= np.mean(displacement_y, axis=1, keepdims=True)
        return displacement_y.astype(np.float32), angle_map

    def displacement_to_vis(self, displacement_y: np.ndarray) -> np.ndarray:
        disp_min = float(np.min(displacement_y))
        disp_max = float(np.max(displacement_y))
        disp_range = max(disp_max - disp_min, 1e-6)
        disp_u8 = ((displacement_y - disp_min) / disp_range * 255.0).astype(np.uint8)
        return cv2.applyColorMap(disp_u8, cv2.COLORMAP_TURBO)

    def detect(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        total_t0 = time.perf_counter()

        stage_t0 = time.perf_counter()
        self.binary_article = self.binarize_with_unet_as_article_binary()
        self.timing_seconds["preprocessing_unet_perspective_local_warp"] = time.perf_counter() - stage_t0

        stage_t0 = time.perf_counter()
        self.average_character_height = self.estimate_average_character_height_from_article_binary(self.binary_article)
        full_height, full_width = self.binary_article.shape[:2]
        small_width, small_height = self.get_proportional_small_size(full_width, full_height)
        self.current_small_width = int(small_width)
        self.current_small_height = int(small_height)
        self.small_binary_article = cv2.resize(
            self.binary_article,
            (int(small_width), int(small_height)),
            interpolation=cv2.INTER_NEAREST,
        )
        self.small_to_full_y_scale = float(full_height) / float(small_height)
        self.small_to_full_x_scale = float(full_width) / float(small_width)
        self.small_average_character_height = max(
            3.0,
            float(self.average_character_height) / max(1e-6, self.small_to_full_y_scale),
        )
        self.timing_seconds["resize_to_small_hpp"] = time.perf_counter() - stage_t0

        stage_t0 = time.perf_counter()
        self.hpp = self.compute_hpp(self.small_binary_article)
        self.hpp_normalized = self.normalize_hpp_minmax(self.hpp)
        self.small_hpp = self.compute_hpp(self.small_binary_article)
        self.small_hpp_normalized = self.normalize_hpp_minmax(self.small_hpp)
        self.small_text_regions = self.extract_text_regions_for_small_hpp(self.small_hpp_normalized)
        self.text_regions = self.scale_small_regions_to_full(self.small_text_regions, full_height)
        self.clustered_regions = self.cluster_line_regions(self.small_text_regions)
        self.start_points = self.compute_seam_starting_points(self.clustered_regions)
        self.timing_seconds["small_hpp_regions_and_scale_to_full"] = time.perf_counter() - stage_t0

        stage_t0 = time.perf_counter()
        self.modified_energy = self.compute_modified_energy(self.small_binary_article, self.clustered_regions)
        self.seams = [self.trace_horizontal_seam_a_star(self.modified_energy, start_row) for start_row in self.start_points]
        self.timing_seconds["energy_matrix_and_a_star"] = time.perf_counter() - stage_t0

        stage_t0 = time.perf_counter()
        self.line_bands = self.compute_line_bands(self.seams, small_height)
        small_class_matrix, small_lines = self.build_class_matrix_from_seams(self.small_binary_article, self.seams)
        class_matrix = self.scale_class_matrix_to_full(small_class_matrix, full_width, full_height)
        lines = self.build_lines_from_class_matrix(class_matrix)
        restored_class_matrix = self.restore_class_matrix_to_page_coordinates(class_matrix)
        self.timing_seconds["class_matrix_and_restore_coordinates"] = time.perf_counter() - stage_t0
        self.timing_seconds["total_detect"] = time.perf_counter() - total_t0

        if self.debug:
            self.save_small_size_debug(
                small_class_matrix=small_class_matrix,
                full_class_matrix=class_matrix,
                restored_class_matrix=restored_class_matrix,
                lines=lines,
                small_lines=small_lines,
            )
            self._save_debug_image(exact.colorize_class_matrix(restored_class_matrix), "13_small_restored_class_matrix_page_coords")
            self.save_detection_rectangles_debug(restored_class_matrix)
        return restored_class_matrix, lines

    def scale_class_matrix_to_full(self, small_class_matrix: np.ndarray, full_width: int, full_height: int) -> np.ndarray:
        return cv2.resize(
            small_class_matrix.astype(np.int32),
            (int(full_width), int(full_height)),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)

    def build_lines_from_class_matrix(self, class_matrix: np.ndarray) -> List[Dict[str, Any]]:
        lines: List[Dict[str, Any]] = []
        max_class = int(np.max(class_matrix)) if class_matrix.size else 0
        for line_id in range(1, max_class + 1):
            ys, xs = np.where(class_matrix == line_id)
            if len(xs) == 0:
                continue
            lines.append(
                {
                    "id": int(line_id),
                    "top": int(np.min(ys)),
                    "bottom": int(np.max(ys)),
                    "left": int(np.min(xs)),
                    "right": int(np.max(xs)),
                    "source": "small_hpp_a_star_scaled_to_full",
                    "text_pixels": int(len(xs)),
                }
            )
        return lines

    def extract_text_regions_for_small_hpp(self, normalized_hpp: np.ndarray) -> List[Tuple[int, int]]:
        previous_ah = float(self.average_character_height)
        try:
            self.average_character_height = float(self.small_average_character_height)
            return self.extract_text_regions(normalized_hpp)
        finally:
            self.average_character_height = previous_ah

    def scale_small_regions_to_full(
        self,
        small_regions: List[Tuple[int, int]],
        full_height: int,
    ) -> List[Tuple[int, int]]:
        scaled: List[Tuple[int, int]] = []
        for start, end in small_regions:
            full_start = int(np.floor(float(start) * self.small_to_full_y_scale))
            full_end = int(np.ceil(float(end + 1) * self.small_to_full_y_scale)) - 1
            full_start = int(np.clip(full_start, 0, full_height - 1))
            full_end = int(np.clip(full_end, 0, full_height - 1))
            if full_start <= full_end:
                scaled.append((full_start, full_end))
        return self.merge_overlapping_regions(scaled)

    def merge_overlapping_regions(self, regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not regions:
            return []
        regions = sorted((int(a), int(b)) for a, b in regions)
        merged: List[Tuple[int, int]] = []
        current_start, current_end = regions[0]
        for start, end in regions[1:]:
            if start <= current_end + 1:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))
        return merged

    # def trace_horizontal_seam_a_star(self, energy: np.ndarray, start_row: int) -> np.ndarray:
    #     rows, cols = energy.shape
    #     start_y = int(np.clip(start_row, 0, rows - 1))
    #     start = (start_y, 0)
    #     goal_col = cols - 1

    #     open_set: List[Tuple[float, int, int]] = []
    #     heapq.heappush(open_set, (float(energy[start_y, 0]), start_y, 0))
    #     came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    #     g_score: Dict[Tuple[int, int], float] = {start: float(energy[start_y, 0])}
    #     visited = set()

    #     while open_set:
    #         _, y, x = heapq.heappop(open_set)
    #         current = (y, x)
    #         if current in visited:
    #             continue
    #         visited.add(current)

    #         if x == goal_col:
    #             points: List[Tuple[int, int]] = []
    #             while True:
    #                 cy, cx = current
    #                 points.append((cx, cy))
    #                 if current == start:
    #                     break
    #                 current = came_from[current]
    #             points.reverse()
    #             return np.asarray(points, dtype=np.int32)

    #         next_x = x + 1
    #         if next_x >= cols:
    #             continue
    #         for dy in (-1, 0, 1):
    #             next_y = y + dy
    #             if not (0 <= next_y < rows):
    #                 continue
    #             neighbor = (next_y, next_x)
    #             tentative_g = g_score[current] + float(energy[next_y, next_x])
    #             if tentative_g >= g_score.get(neighbor, np.inf):
    #                 continue
    #             came_from[neighbor] = current
    #             g_score[neighbor] = tentative_g
    #             heuristic = float(goal_col - next_x)
    #             heapq.heappush(open_set, (tentative_g + heuristic, next_y, next_x))

    #     return self.trace_horizontal_seam(self.compute_horizontal_min_energy_path_matrix(energy), start_row)

    def trace_horizontal_seam_a_star(self, energy: np.ndarray, start_row: int) -> np.ndarray:
        rows, cols = energy.shape

        start_y = int(np.clip(start_row, 0, rows - 1))
        start = (start_y, 0)
        goal_col = cols - 1

        vertical_penalty = 2.0
        drift_penalty_weight = 0.05

        open_set: List[Tuple[float, float, float, int, int]] = []
        heapq.heappush(open_set, (float(energy[start_y, 0]), 0.0, 0.0, start_y, 0))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {
            start: float(energy[start_y, 0])
        }

        visited = set()

        while open_set:
            _, _, _, y, x = heapq.heappop(open_set)

            current = (y, x)

            if current in visited:
                continue

            visited.add(current)

            if x == goal_col:
                points: List[Tuple[int, int]] = []

                while True:
                    cy, cx = current
                    points.append((cx, cy))

                    if current == start:
                        break

                    current = came_from[current]

                points.reverse()
                return np.asarray(points, dtype=np.int32)

            next_x = x + 1

            if next_x >= cols:
                continue

            # ВАЖНО:
            # Сначала прямо, потом вверх/вниз.
            for dy in (0, -1, 1):
                next_y = y + dy

                if not (0 <= next_y < rows):
                    continue

                neighbor = (next_y, next_x)

                move_penalty = vertical_penalty * abs(dy)
                drift_penalty = drift_penalty_weight * abs(next_y - start_y)

                tentative_g = (
                    g_score[current]
                    + float(energy[next_y, next_x])
                    + move_penalty
                    + drift_penalty
                )

                if tentative_g >= g_score.get(neighbor, np.inf):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                heuristic = float(goal_col - next_x)

                heapq.heappush(
                    open_set,
                    (
                        tentative_g + heuristic,
                        abs(next_y - start_y),
                        abs(dy),
                        next_y,
                        next_x,
                    )
                )

        return self.trace_horizontal_seam(
            self.compute_horizontal_min_energy_path_matrix(energy),
            start_row,
        )

    def save_small_size_debug(
        self,
        small_class_matrix: np.ndarray,
        full_class_matrix: np.ndarray,
        restored_class_matrix: np.ndarray,
        lines: List[Dict[str, Any]],
        small_lines: List[Dict[str, Any]],
    ) -> None:
        if self.small_binary_article is not None:
            self._save_debug_image(self.small_binary_article, "09_small_hpp_article_binary_text_white")
        if self.small_hpp_normalized is not None:
            self.save_small_hpp_debug()
        if self.modified_energy is not None:
            self._save_debug_image(exact.normalize_to_uint8(self.modified_energy), "11_small_modified_energy")
        if self.small_binary_article is not None:
            seam_vis = cv2.cvtColor(np.where(self.small_binary_article > 0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for start, end in self.clustered_regions:
                cv2.rectangle(seam_vis, (0, int(start)), (seam_vis.shape[1] - 1, int(end)), (0, 180, 0), 1)
            for seam in self.seams:
                cv2.polylines(seam_vis, [seam.reshape(-1, 1, 2)], False, (0, 0, 255), 2)
            self._save_debug_image(seam_vis, "12_small_regions_and_a_star_seams")
        self._save_debug_image(exact.colorize_class_matrix(small_class_matrix), "12_small_final_class_matrix")
        self._save_debug_image(exact.colorize_class_matrix(full_class_matrix), "12_full_scaled_class_matrix")

        with open(os.path.join(self.debug_output_dir, "my_small_size_summary.json"), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "method": "my_small_size_das_panda_hpp_seam_exact",
                    "base_file": "my_das_panda_hpp_seam_exact.py",
                    "small_width_hyperparameter": HPP_SMALL_WIDTH,
                    "small_height_hyperparameter": int(HPP_SMALL_HEIGHT),
                    "actual_small_width": int(self.current_small_width),
                    "actual_small_height": int(self.current_small_height),
                    "full_binary_shape": [int(v) for v in self.binary_article.shape[:2]] if self.binary_article is not None else None,
                    "small_to_full_y_scale": float(self.small_to_full_y_scale),
                    "small_to_full_x_scale": float(self.small_to_full_x_scale),
                    "average_character_height_full": float(self.average_character_height),
                    "average_character_height_small": float(self.small_average_character_height),
                    "small_hpp_regions": [[int(a), int(b)] for a, b in self.small_text_regions],
                    "scaled_full_hpp_regions": [[int(a), int(b)] for a, b in self.text_regions],
                    "clustered_regions_small": [[int(a), int(b)] for a, b in self.clustered_regions],
                    "start_points": [int(v) for v in self.start_points],
                    "timing_seconds": {key: float(value) for key, value in self.timing_seconds.items()},
                    "line_count": int(len(lines)),
                    "small_line_count": int(len(small_lines)),
                    "class_count_small": int(np.max(small_class_matrix)) if small_class_matrix.size else 0,
                    "class_count_full_scaled": int(np.max(full_class_matrix)) if full_class_matrix.size else 0,
                    "class_count_restored": int(np.max(restored_class_matrix)) if restored_class_matrix.size else 0,
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

    def save_small_hpp_debug(self) -> None:
        values = self.small_hpp_normalized.astype(np.float32)
        height = len(values)
        width = 900
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        normalized = self.normalize_hpp_minmax(values)
        for row, value in enumerate(normalized):
            x = int(round(float(value) * (width - 1)))
            cv2.line(canvas, (0, row), (x, row), (0, 0, 0), 1)
        for start, end in self.small_text_regions:
            cv2.rectangle(canvas, (0, int(start)), (width - 1, int(end)), (0, 180, 0), 1)
        self._save_debug_image(canvas, "10_small_normalized_hpp_regions")


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
    run_t0 = time.perf_counter()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    if USE_YOLO_PAGE_SEGMENTATION:
        yolo_t0 = time.perf_counter()
        page_result = exact.extract_largest_page_with_yolo(
            image,
            debug=debug,
            debug_output_dir=debug_output_dir,
            yolo_model=yolo_model,
        )
        yolo_runtime_sec = time.perf_counter() - yolo_t0
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
        yolo_runtime_sec = 0.0
        page_result = {
            "page_image": image,
            "page_mask": np.ones(image.shape[:2], dtype=np.uint8) * 255,
            "bbox": {"x": 0, "y": 0, "w": int(image.shape[1]), "h": int(image.shape[0])},
            "confidence": None,
        }

    detector = MySmallSizeDasPandaHPPSeamSegmenter(
        image=image,
        debug=debug,
        debug_output_dir=debug_output_dir,
        unet_model=unet_model,
        unet_device=unet_device,
    )
    class_matrix, lines = detector.detect()
    page_result["timing_seconds"] = {
        "total_run_on_image": float(time.perf_counter() - run_t0),
        "yolo_page_segmentation": float(yolo_runtime_sec),
        "detector": {key: float(value) for key, value in detector.timing_seconds.items()},
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
