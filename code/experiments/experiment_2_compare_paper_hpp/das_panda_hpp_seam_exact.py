"""
Короткое описание:
    Реализация метода Das & Panda 2023:
    "Seam carving, horizontal projection profile and contour tracing for line and
    word segmentation of language independent handwritten documents".

    В этом файле реализована только часть line segmentation: HPP + seam carving.

Важно:
    В статье явно описаны HPP, Gaussian normalization, modified energy matrix и
    DP seam carving. Но численные значения threshold для normalized HPP и правило
    "nearby text-lines are clustered" не указаны. Эти места вынесены в отдельные
    гиперпараметры и отмечены в debug summary.
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEBUG_IMAGES_DIR = str(PROJECT_ROOT / "debug_images" / "experiment_2_compare_paper_hpp" / "das_panda_hpp_seam_exact")
INPUT_IMAGE_PATH = "/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code/datasets/HWR200/hw_dataset/3/reuse9/ФотоСветлое/1.JPG"
DEBUG = True

# В задаче договорились: вместо Otsu из статьи используем YOLO page + U-Net binarization.
USE_YOLO_PAGE_SEGMENTATION = True
YOLO_PAGE_SEGMENTATION_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolo_segment_notebook"
    / "yolo_segment_notebook_3_(2-architecture).pt"
)
YOLO_PAGE_CONF = 0.05
YOLO_PAGE_IMGSZ = 640
YOLO_PAGE_DEVICE = None

UNET_BINARIZATION_MODEL_PATH = PROJECT_ROOT / "models" / "u_net" / "unet_binarization_3_(6-architecture).pth"
UNET_TARGET_SIZE = (3000, 3000)
UNET_THRESHOLD = 0.5
UNET_DEVICE = None

# Ниже параметры не указаны численно в статье.
# Статья: "If the normalized HPP value of a row is greater than a threshold..."
HPP_ZSCORE_THRESHOLD = 0.0

# Статья: "Nearby text-lines are clustered into text-line-regions."
REGION_MERGE_GAP_ROWS = 0

# Фильтр слишком тонких регионов тоже не указан. По умолчанию выключен.
MIN_TEXT_REGION_HEIGHT = 1

class DasPandaHPPSeamSegmenter:
    """
    Подробное описание:
        Линейная сегментация по статье Das & Panda 2023:
        1. Binary image.
        2. Horizontal projection profile.
        3. Gaussian normalization: (hist - histMean) / histStd.
        4. Rows with normalized HPP above threshold become text rows.
        5. Text rows are grouped into text-line-regions.
        6. Energy matrix Dm starts from binary image; pixels in text-line-regions
           get energy increased by 255.
        7. Horizontal minimum energy path matrix is computed right-to-left.
        8. Seam starting points are midpoints between consecutive line regions.
        9. Seams split the page into line images.
    """

    def __init__(
        self,
        image: np.ndarray,
        debug: bool = False,
        debug_output_dir: str = DEBUG_IMAGES_DIR,
        unet_model: Any = None,
        unet_device: Any = None,
    ):
        self.image = image.copy()
        self.debug = bool(debug)
        self.debug_output_dir = str(debug_output_dir)
        self.unet_model = unet_model
        self.unet_device = unet_device
        self.debug_counter = 0

        self.gray = self._to_gray(self.image)
        self.binary_article: Optional[np.ndarray] = None
        self.hpp: Optional[np.ndarray] = None
        self.hpp_normalized: Optional[np.ndarray] = None
        self.text_regions: List[Tuple[int, int]] = []
        self.clustered_regions: List[Tuple[int, int]] = []
        self.start_points: List[int] = []
        self.modified_energy: Optional[np.ndarray] = None
        self.min_energy_matrix: Optional[np.ndarray] = None
        self.seams: List[np.ndarray] = []
        self.line_bands: List[Tuple[int, int]] = []

        if self.debug:
            os.makedirs(self.debug_output_dir, exist_ok=True)

    def detect(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Короткое описание:
            Запускает line segmentation и возвращает class-matrix строк.
        Вход:
            None
        Выход:
            Tuple[np.ndarray, List[Dict[str, Any]]] -- class_matrix и line metadata.
        """
        # Шаг 1-2 статьи: Binary Image. Здесь U-Net заменяет Otsu по условию задачи.
        self.binary_article = self.binarize_with_unet_as_article_binary()

        # Шаг 3-7 статьи: HPP, normalization, text regions, CR, seam starting points.
        self.hpp = self.compute_hpp(self.binary_article)
        self.hpp_normalized = self.gaussian_normalize_histogram(self.hpp)
        self.text_regions = self.extract_text_regions(self.hpp_normalized)
        self.clustered_regions = self.cluster_line_regions(self.text_regions)
        self.start_points = self.compute_seam_starting_points(self.clustered_regions)

        # Шаг 8-10 статьи: modified energy, horizontal DP, paths.
        self.modified_energy = self.compute_modified_energy(self.binary_article, self.clustered_regions)
        self.min_energy_matrix = self.compute_horizontal_min_energy_path_matrix(self.modified_energy)
        self.seams = [self.trace_horizontal_seam(self.min_energy_matrix, start_row) for start_row in self.start_points]

        # Шаг 11 статьи: draw paths and extract line images.
        self.line_bands = self.compute_line_bands(self.seams, self.binary_article.shape[0])
        class_matrix, lines = self.build_class_matrix_from_seams(self.binary_article, self.seams)

        if self.debug:
            self.save_all_debug(class_matrix, lines)
        return class_matrix, lines

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            gray = image.copy()
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.clip(gray, 0, 255).astype(np.uint8)

    def _save_debug_image(self, image: np.ndarray, name: str) -> None:
        if not self.debug:
            return
        os.makedirs(self.debug_output_dir, exist_ok=True)
        path = os.path.join(self.debug_output_dir, f"{self.debug_counter:03d}_{name}.png")
        cv2.imwrite(path, image)
        self.debug_counter += 1

    def binarize_with_unet_as_article_binary(self) -> np.ndarray:
        """
        Короткое описание:
            Получает binary image в соглашении статьи: background=0, text=255.
        Вход:
            None
        Выход:
            np.ndarray -- uint8 binary, где текст 255.
        """
        from u_net_binarization import binarize_image_with_loaded_model, load_unet_model

        if self.unet_model is None or self.unet_device is None:
            model, device = load_unet_model(str(UNET_BINARIZATION_MODEL_PATH), device=UNET_DEVICE)
            owns_model = True
        else:
            model, device = self.unet_model, self.unet_device
            owns_model = False

        try:
            # В существующей U-Net функции foreground/text обычно 0, background 255.
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

        article_binary = np.where(unet_mask == 0, 255, 0).astype(np.uint8)
        if self.debug:
            self._save_debug_image(unet_mask, "01_unet_raw_text_black")
            self._save_debug_image(article_binary, "02_article_binary_text_white")
        return article_binary

    def compute_hpp(self, binary_article: np.ndarray) -> np.ndarray:
        """
        Статья Eq. (1): HPP(x) = sum_y f(x, y), 1 <= x <= m.
        """
        return np.sum(binary_article.astype(np.float32), axis=1)

    def gaussian_normalize_histogram(self, hist: np.ndarray) -> np.ndarray:
        """
        Algorithm 1 step 4: (hist - histMean) / histStd.
        """
        hist_mean = float(np.mean(hist))
        hist_std = float(np.std(hist))
        if hist_std <= 1e-9:
            return np.zeros_like(hist, dtype=np.float32)
        return ((hist - hist_mean) / hist_std).astype(np.float32)

    def extract_text_regions(self, normalized_hpp: np.ndarray) -> List[Tuple[int, int]]:
        """
        Algorithm 1 step 5-6: rows with normalized HPP > threshold form text-line regions.
        """
        text_rows = normalized_hpp > float(HPP_ZSCORE_THRESHOLD)
        regions: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for row_index, is_text in enumerate(text_rows):
            if is_text and start is None:
                start = row_index
            elif (not is_text) and start is not None:
                end = row_index - 1
                if end - start + 1 >= int(MIN_TEXT_REGION_HEIGHT):
                    regions.append((int(start), int(end)))
                start = None
        if start is not None:
            end = len(text_rows) - 1
            if end - start + 1 >= int(MIN_TEXT_REGION_HEIGHT):
                regions.append((int(start), int(end)))
        return regions

    def cluster_line_regions(self, regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Algorithm 1 step 7: line regions are clustered together into CR.
        Точный критерий "nearby" в статье не указан.
        """
        if not regions:
            return []
        merge_gap = int(REGION_MERGE_GAP_ROWS)
        clustered: List[Tuple[int, int]] = []
        current_start, current_end = regions[0]
        for start, end in regions[1:]:
            if start - current_end - 1 <= merge_gap:
                current_end = end
            else:
                clustered.append((int(current_start), int(current_end)))
                current_start, current_end = start, end
        clustered.append((int(current_start), int(current_end)))
        return clustered

    def compute_seam_starting_points(self, clustered_regions: List[Tuple[int, int]]) -> List[int]:
        """
        Algorithm 1 step 7: SP = (CR[i][0] + CR[i - 1][1]) / 2.
        """
        starts: List[int] = []
        for index in range(1, len(clustered_regions)):
            previous_end = clustered_regions[index - 1][1]
            current_start = clustered_regions[index][0]
            starts.append(int(round((current_start + previous_end) / 2.0)))
        return starts

    def compute_modified_energy(self, binary_article: np.ndarray, clustered_regions: List[Tuple[int, int]]) -> np.ndarray:
        """
        Algorithm 1 step 8:
            Copy binary image as energy matrix, then increase energy of pixels of IB
            falling in text-line regions by 255.
        """
        energy = binary_article.astype(np.float32).copy()
        for start, end in clustered_regions:
            energy[start:end + 1, :] += 255.0
        return energy

    def compute_horizontal_min_energy_path_matrix(self, energy: np.ndarray) -> np.ndarray:
        """
        Algorithm 2:
            Pm[j][i] = Em[j][i] + min(Pm[j][i+1], Pm[j-1][i+1], Pm[j+1][i+1]).
        """
        rows, cols = energy.shape
        pm = energy.astype(np.float32).copy()
        for col in range(cols - 2, -1, -1):
            for row in range(rows):
                candidates = [pm[row, col + 1]]
                if row > 0:
                    candidates.append(pm[row - 1, col + 1])
                if row < rows - 1:
                    candidates.append(pm[row + 1, col + 1])
                pm[row, col] = energy[row, col] + min(candidates)
        return pm

    def trace_horizontal_seam(self, pm: np.ndarray, start_row: int) -> np.ndarray:
        """
        Algorithm 1 step 10:
            Compute path from min-energy path matrix, starting at SP in column 0.
        """
        rows, cols = pm.shape
        seam = np.zeros((cols, 2), dtype=np.int32)
        row = int(np.clip(start_row, 0, rows - 1))
        for col in range(cols):
            seam[col] = (col, row)
            if col == cols - 1:
                break
            next_candidates = [(pm[row, col + 1], row)]
            if row > 0:
                next_candidates.append((pm[row - 1, col + 1], row - 1))
            if row < rows - 1:
                next_candidates.append((pm[row + 1, col + 1], row + 1))
            _, row = min(next_candidates, key=lambda item: item[0])
        return seam

    def compute_line_bands(self, seams: List[np.ndarray], image_height: int) -> List[Tuple[int, int]]:
        """
        Converts separating seams into top/bottom bands. A pixel row belongs to the
        band between two neighboring seam rows.
        """
        if not seams:
            return [(0, image_height - 1)]
        seam_rows = [int(round(float(np.median(seam[:, 1])))) for seam in seams]
        seam_rows = sorted(set(int(np.clip(row, 0, image_height - 1)) for row in seam_rows))
        boundaries = [-1] + seam_rows + [image_height]
        bands: List[Tuple[int, int]] = []
        for index in range(len(boundaries) - 1):
            top = boundaries[index] + 1
            bottom = boundaries[index + 1] - 1
            if top <= bottom:
                bands.append((int(top), int(bottom)))
        return bands

    def build_class_matrix_from_seams(
        self,
        binary_article: np.ndarray,
        seams: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        The paper draws seam paths to separate lines. For evaluation, each text
        pixel is assigned to the vertical interval between neighboring seam paths
        at the same x-column.
        """
        class_matrix = np.zeros(binary_article.shape, dtype=np.int32)
        lines: List[Dict[str, Any]] = []
        text_mask = binary_article > 0
        height, width = binary_article.shape

        if not seams:
            class_matrix[text_mask] = 1
            ys, xs = np.where(text_mask)
            if len(xs) > 0:
                lines.append(
                    {
                        "id": 1,
                        "top": int(np.min(ys)),
                        "bottom": int(np.max(ys)),
                        "left": int(np.min(xs)),
                        "right": int(np.max(xs)),
                        "source": "whole_page_no_seams",
                        "text_pixels": int(np.sum(text_mask)),
                    }
                )
            return class_matrix, lines

        seam_rows_by_col: List[List[int]] = [[] for _ in range(width)]
        for seam in seams:
            for x, y in seam:
                if 0 <= x < width:
                    seam_rows_by_col[int(x)].append(int(np.clip(y, 0, height - 1)))
        for col in range(width):
            seam_rows_by_col[col] = sorted(seam_rows_by_col[col])

        text_y, text_x = np.where(text_mask)
        for y, x in zip(text_y, text_x):
            separators_above = 0
            for seam_y in seam_rows_by_col[int(x)]:
                if int(y) > seam_y:
                    separators_above += 1
            class_matrix[int(y), int(x)] = separators_above + 1

        max_class = int(np.max(class_matrix)) if class_matrix.size else 0
        for line_id in range(1, max_class + 1):
            assigned = class_matrix == line_id
            if int(np.sum(assigned)) == 0:
                continue
            ys, xs = np.where(assigned)
            lines.append(
                {
                    "id": int(line_id),
                    "top": int(np.min(ys)),
                    "bottom": int(np.max(ys)),
                    "left": int(np.min(xs)),
                    "right": int(np.max(xs)),
                    "source": "hpp_seam_paths",
                    "text_pixels": int(np.sum(assigned)),
                }
            )
        return class_matrix, lines

    def save_all_debug(self, class_matrix: np.ndarray, lines: List[Dict[str, Any]]) -> None:
        self._save_debug_image(self.gray, "00_gray")
        if self.binary_article is not None:
            self._save_debug_image(self.binary_article, "03_binary_article")
        if self.modified_energy is not None:
            energy_vis = normalize_to_uint8(self.modified_energy)
            self._save_debug_image(energy_vis, "04_modified_energy")
        if self.min_energy_matrix is not None:
            self._save_debug_image(normalize_to_uint8(self.min_energy_matrix), "05_min_energy_matrix")

        seam_vis = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        for seam in self.seams:
            cv2.polylines(seam_vis, [seam.reshape(-1, 1, 2)], False, (0, 0, 255), 2)
        for start, end in self.clustered_regions:
            cv2.rectangle(seam_vis, (0, start), (seam_vis.shape[1] - 1, end), (0, 180, 0), 1)
        self._save_debug_image(seam_vis, "06_text_regions_and_seams")

        class_vis = colorize_class_matrix(class_matrix)
        self._save_debug_image(class_vis, "07_final_class_matrix")

        self.save_hpp_debug()
        self.save_summary(class_matrix, lines)

    def save_hpp_debug(self) -> None:
        if not self.debug or self.hpp_normalized is None:
            return
        height = len(self.hpp_normalized)
        width = 900
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        values = self.hpp_normalized
        min_value = float(np.min(values)) if values.size else 0.0
        max_value = float(np.max(values)) if values.size else 1.0
        scale = max(max_value - min_value, 1e-6)
        for row, value in enumerate(values):
            x = int(round((float(value) - min_value) / scale * (width - 1)))
            cv2.line(canvas, (0, row), (x, row), (0, 0, 0), 1)
        threshold_x = int(round((float(HPP_ZSCORE_THRESHOLD) - min_value) / scale * (width - 1)))
        threshold_x = int(np.clip(threshold_x, 0, width - 1))
        cv2.line(canvas, (threshold_x, 0), (threshold_x, height - 1), (0, 0, 255), 2)
        self._save_debug_image(canvas, "08_normalized_hpp")

    def save_summary(self, class_matrix: np.ndarray, lines: List[Dict[str, Any]]) -> None:
        summary = {
            "paper": "Das & Panda 2023, Results in Engineering 18, 101110",
            "implemented_scope": "line segmentation only: HPP + horizontal seam carving",
            "input_shape": [int(v) for v in self.image.shape],
            "binary_convention": "article convention: background=0, text=255; U-Net text-black mask is inverted",
            "hpp_regions": [[int(a), int(b)] for a, b in self.text_regions],
            "clustered_regions": [[int(a), int(b)] for a, b in self.clustered_regions],
            "start_points": [int(v) for v in self.start_points],
            "line_bands": [[int(a), int(b)] for a, b in self.line_bands],
            "line_count": int(len(lines)),
            "class_count": int(np.max(class_matrix)) if class_matrix.size else 0,
            "paper_parameters_without_numeric_values": {
                "HPP_ZSCORE_THRESHOLD": float(HPP_ZSCORE_THRESHOLD),
                "REGION_MERGE_GAP_ROWS": int(REGION_MERGE_GAP_ROWS),
                "MIN_TEXT_REGION_HEIGHT": int(MIN_TEXT_REGION_HEIGHT),
            },
        }
        path = os.path.join(self.debug_output_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)


def normalize_to_uint8(matrix: np.ndarray) -> np.ndarray:
    values = matrix.astype(np.float32)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value <= min_value:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.round((values - min_value) / (max_value - min_value) * 255.0).astype(np.uint8)


def colorize_class_matrix(class_matrix: np.ndarray) -> np.ndarray:
    height, width = class_matrix.shape[:2]
    result = np.full((height, width, 3), 255, dtype=np.uint8)
    max_class = int(np.max(class_matrix)) if class_matrix.size else 0
    rng = np.random.default_rng(12345)
    colors = rng.integers(40, 230, size=(max_class + 1, 3), dtype=np.uint8)
    for class_index in range(1, max_class + 1):
        result[class_matrix == class_index] = colors[class_index]
    return result


def run_on_image(
    image_path: str,
    debug: bool = DEBUG,
    debug_output_dir: str = DEBUG_IMAGES_DIR,
    yolo_model: Any = None,
    unet_model: Any = None,
    unet_device: Any = None,
    return_page_info: bool = False,
) -> Any:
    """
    Короткое описание:
        Запускает YOLO page crop + U-Net + Das/Panda HPP seam line segmentation.
    """
    run_start_time = time.perf_counter()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    if USE_YOLO_PAGE_SEGMENTATION:
        page_result = extract_largest_page_with_yolo(image, debug=debug, debug_output_dir=debug_output_dir, yolo_model=yolo_model)
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

    detector = DasPandaHPPSeamSegmenter(
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


def extract_largest_page_with_yolo(
    image: np.ndarray,
    debug: bool = DEBUG,
    debug_output_dir: str = DEBUG_IMAGES_DIR,
    yolo_model: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Короткое описание:
        Находит страницу через YOLO segmentation. При нескольких кандидатах выбирает
        максимальную confidence, как в experiment_1.
    """
    if not YOLO_PAGE_SEGMENTATION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найдена YOLO-модель страницы: {YOLO_PAGE_SEGMENTATION_MODEL_PATH}")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError("Для YOLO-сегментации нужен ultralytics в /home/sasha/Documents/venv") from error

    owns_model = yolo_model is None
    model = YOLO(str(YOLO_PAGE_SEGMENTATION_MODEL_PATH)) if owns_model else yolo_model
    result = model.predict(
        image,
        conf=YOLO_PAGE_CONF,
        imgsz=YOLO_PAGE_IMGSZ,
        device=YOLO_PAGE_DEVICE,
        verbose=False,
    )[0]

    if debug:
        os.makedirs(debug_output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_output_dir, "yolo_00_input.jpg"), image)
        cv2.imwrite(os.path.join(debug_output_dir, "yolo_01_result.jpg"), result.plot())

    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return None

    image_height, image_width = image.shape[:2]
    masks = result.masks.data.cpu().numpy()
    confidences = []
    if result.boxes is not None and result.boxes.conf is not None:
        confidences = [float(value) for value in result.boxes.conf.detach().cpu().numpy().tolist()]

    best_mask = None
    best_area = -1
    best_confidence = -1.0
    candidates: List[Dict[str, Any]] = []
    for mask_index, mask in enumerate(masks):
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        mask_full = cv2.resize(mask_binary, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        area = int(np.sum(mask_full > 0))
        confidence = confidences[mask_index] if mask_index < len(confidences) else 0.0
        candidates.append({"index": int(mask_index), "confidence": float(confidence), "mask_area": int(area)})
        if confidence > best_confidence or (abs(confidence - best_confidence) < 1e-9 and area > best_area):
            best_confidence = float(confidence)
            best_area = area
            best_mask = mask_full

    if best_mask is None or best_area <= 0:
        return None

    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    page_roi = image[y:y + height, x:x + width]
    mask_roi = best_mask[y:y + height, x:x + width]
    white_page = np.full_like(page_roi, 255)
    page_image = np.where(mask_roi[:, :, None] > 0, page_roi, white_page)

    if debug:
        cv2.imwrite(os.path.join(debug_output_dir, "yolo_02_page_mask.png"), best_mask)
        cv2.imwrite(os.path.join(debug_output_dir, "yolo_03_page_crop.jpg"), page_image)
        info = {
            "model_path": str(YOLO_PAGE_SEGMENTATION_MODEL_PATH),
            "conf": YOLO_PAGE_CONF,
            "imgsz": YOLO_PAGE_IMGSZ,
            "bbox": {"x": int(x), "y": int(y), "w": int(width), "h": int(height)},
            "mask_area": int(best_area),
            "selected_confidence": float(best_confidence),
            "selection_rule": "max_confidence_then_area",
            "candidates": candidates,
        }
        with open(os.path.join(debug_output_dir, "yolo_page_info.json"), "w", encoding="utf-8") as file:
            json.dump(info, file, indent=2, ensure_ascii=False)

    if owns_model:
        del model
        gc.collect()

    return {
        "page_image": page_image,
        "page_mask": mask_roi,
        "bbox": {"x": int(x), "y": int(y), "w": int(width), "h": int(height)},
        "confidence": float(best_confidence),
        "candidates": candidates,
    }


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
