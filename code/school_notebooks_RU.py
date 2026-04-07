import json
from typing import List, Tuple, Optional
import cv2
import numpy as np
import random
from processing import normalize_illumination
import os


class CocoMaskGenerator:
    """
    Класс для работы с аннотациями в формате COCO: создание масок, визуализация,
    извлечение аннотаций по изображению, а также вычисление метрик сравнения масок.
    """

    def __init__(self, json_path: str):
        """
        Загружает JSON-файл с аннотациями COCO.

        Параметры:
            json_path (str): Путь к JSON-файлу.

        Выход:
            None
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # Создаём словарь для быстрого доступа к аннотациям по image_id
        self.annotations_by_image = {}
        for ann in self.data.get('annotations', []):
            img_id = ann['image_id']
            self.annotations_by_image.setdefault(img_id, []).append(ann)

        # Словарь для быстрого доступа к информации об изображениях по image_id
        self.image_info = {}
        for img in self.data.get('images', []):
            self.image_info[img['id']] = img

    def get_image_id_by_filename(self, filename: str) -> Optional[int]:
        """
        Возвращает image_id по имени файла.

        Параметры:
            filename (str): Имя файла (например, '0_0.jpg').

        Выход:
            int или None: image_id, если файл найден, иначе None.
        """
        for img_id, info in self.image_info.items():
            if info.get('file_name') == filename:
                return img_id
        return None

    def get_annotations(self, image_id: int, category_id: Optional[int] = None) -> List[dict]:
        """
        Возвращает список аннотаций для указанного изображения.

        Параметры:
            image_id (int): Идентификатор изображения.
            category_id (int, optional): Если указан, возвращаются только аннотации с данной категорией.

        Выход:
            List[dict]: Список аннотаций.
        """
        anns = self.annotations_by_image.get(image_id, [])
        if category_id is not None:
            anns = [a for a in anns if a.get('category_id') == category_id]
        return anns

    def get_image_shape(self, image_id: int) -> Tuple[int, int]:
        """
        Возвращает высоту и ширину изображения из JSON.

        Параметры:
            image_id (int): Идентификатор изображения.

        Выход:
            Tuple[int, int]: (height, width).
        """
        info = self.image_info.get(image_id)
        if info is None:
            raise ValueError(f"Изображение с id {image_id} не найдено в JSON.")
        return info['height'], info['width']

    def create_mask(self, image_id: int, category_id: int = 0,
                    height: Optional[int] = None, width: Optional[int] = None) -> np.ndarray:
        """
        Создаёт бинарную маску (0/255) для указанного изображения и категории объектов.
        Если не переданы height и width, они берутся из JSON.

        Параметры:
            image_id (int): Идентификатор изображения.
            category_id (int): Категория объектов (по умолчанию 4 — text_line).
            height (int, optional): Высота маски. Если не указана, берётся из JSON.
            width (int, optional): Ширина маски. Если не указана, берётся из JSON.

        Выход:
            np.ndarray: Бинарная маска (uint8), значения 0 или 255.
        """
        if height is None or width is None:
            h, w = self.get_image_shape(image_id)
            height = height or h
            width = width or w

        mask = np.zeros((height, width), dtype=np.uint8)
        annotations = self.get_annotations(image_id, category_id)

        for ann in annotations:
            seg = ann.get('segmentation')
            if not seg:
                continue
            # В формате COCO segmentation — список полигонов. Берём первый.

            polygon = seg[0]
            pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.fillPoly(mask, [pts], color=color)

        return mask

    def overlay_mask(self, image: np.ndarray, mask: np.ndarray,
                     color: Tuple[int, int, int] = (0, 255, 0),
                     alpha: float = 0.4) -> np.ndarray:
        """
        Накладывает полупрозрачную маску на изображение.

        Параметры:
            image (np.ndarray): Исходное изображение (BGR или RGB).
            mask (np.ndarray): Бинарная маска (0/255).
            color (tuple): Цвет маски в формате (B, G, R).
            alpha (float): Прозрачность (0.0 – полностью прозрачная, 1.0 – полностью непрозрачная).

        Выход:
            np.ndarray: Изображение с наложенной маской.
        """
        overlay = image.copy()
        overlay[mask > 0] = color
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        return result

    def get_all_polygons(self, image_id: int, category_id: int = 0) -> list:
        """

        """
        annotations = self.get_annotations(image_id, category_id)


        polygons = []
        for ann in annotations:
            seg = ann.get('segmentation')
            if not seg:
                continue
            # В формате COCO segmentation — список полигонов. Берём первый.
            for polygon in seg:
                polygons.append(polygon)

        return polygons


    def visualize(self, image_path: str, image_id: int, category_id: int = 0,
                  show_bbox: bool = False, window_name: str = 'Visualization'):
        """
        Отображает изображение с наложенной маской (и опционально bounding boxes).

        Параметры:
            image_path (str): Путь к файлу изображения.
            image_id (int): Идентификатор изображения.
            category_id (int): Категория объектов для маски.
            show_bbox (bool): Если True, рисует bounding boxes вокруг объектов.
            window_name (str): Имя окна OpenCV.

        Выход:
            None
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return

        mask = self.create_mask(image_id, category_id, img.shape[0], img.shape[1])

        cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Window', 800, 600)
        cv2.imshow('Window', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        overlay = self.overlay_mask(img, mask)


        if show_bbox:
            annotations = self.get_annotations(image_id, category_id)
            for ann in annotations:
                seg = ann.get('segmentation')
                if not seg:
                    continue
                pts = np.array(seg[0], dtype=np.int32).reshape(-1, 2)
                x, y, w, h = cv2.boundingRect(pts)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def create_binary_mask(self, image_path: str, image_id: int,
                height: Optional[int] = None, width: Optional[int] = None, debug : bool = False) -> np.ndarray:
        """
        Создаёт бинарную маску для указанного изображения и категории объектов.
        Внутри полигонов применяется метод Оцу для получения бинарного текста.
        Вне полигонов фон остаётся белым (255).

        Параметры:
            image_path (str): Путь к файлу изображения.
            image_id (int): Идентификатор изображения.
            height (int, optional): Высота маски. Если не указана, берётся из JSON.
            width (int, optional): Ширина маски. Если не указана, берётся из JSON.

        Выход:
            np.ndarray: Маска (uint8) с бинаризованным текстом в областях полигонов.
        """
        # Определяем размеры маски
        if height is None or width is None:
            h, w = self.get_image_shape(image_id)
            height = height or h
            width = width or w

        # Загружаем исходное изображение и переводим в градации серого
        img = cv2.imread(image_path)  # предположим, есть такой метод
        #img = normalize_illumination(img, clip_limit=4, gamma=0.2)
        if img is None:
            raise ValueError(f"Изображение с id {image_id} не найдено")
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if debug:
                cv2.imwrite('debug_images/gray.jpg', gray)
        else:
            gray = img

        # Инициализируем маску белым фоном
        mask = np.full((height, width), 255, dtype=np.uint8)

        # Получаем аннотации для указанного image_id и category_id
        annotations = []
        annotations.extend(self.get_annotations(image_id, 0))
        annotations.extend(self.get_annotations(image_id, 1))
        annotations.extend(self.get_annotations(image_id, 2))

        for ann in annotations:
            seg = ann.get('segmentation')
            if not seg:
                continue

            # Берём первый полигон (в COCO их может быть несколько)
            polygon = seg[0]
            pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)

            # Вычисляем bounding box полигона
            x, y, w_box, h_box = cv2.boundingRect(pts)
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w_box)
            y2 = min(height, y + h_box)

            # Вырезаем область изображения и маски
            roi = gray[y1:y2, x1:x2]
            roi_mask = mask[y1:y2, x1:x2]

            # Применяем метод Оцу ко всей области roi
            _, binary_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #_, binary_roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)

            # Создаём маску полигона в локальных координатах roi
            poly_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
            local_pts = pts - np.array([x1, y1])
            cv2.fillPoly(poly_mask, [local_pts], 255)

            # Вставляем бинаризованные пиксели только внутрь полигона
            roi_mask[poly_mask == 255] = binary_roi[poly_mask == 255]

            if debug:
                cv2.imwrite('debug_images/binary_result.jpg', mask)
        return mask

    @staticmethod
    def iou(mask_pred: np.ndarray, mask_true: np.ndarray) -> float:
        """
        Вычисляет Intersection over Union (IoU) между двумя бинарными масками.

        Параметры:
            mask_pred (np.ndarray): Предсказанная маска (0/1 или 0/255).
            mask_true (np.ndarray): Истинная маска (0/1 или 0/255).

        Выход:
            float: IoU (значение от 0 до 1). Если объединение пусто, возвращает 0.
        """
        # Приводим к булевому типу
        pred = mask_pred.astype(bool)
        true = mask_true.astype(bool)

        intersection = np.logical_and(pred, true).sum()
        union = np.logical_or(pred, true).sum()

        if union == 0:
            return 0.0
        return float(intersection) / float(union)

    @staticmethod
    def f1_score(mask_pred: np.ndarray, mask_true: np.ndarray, beta: float = 1.0) -> float:
        """
        Вычисляет F-меру (по умолчанию F1) между двумя бинарными масками.

        Параметры:
            mask_pred (np.ndarray): Предсказанная маска (0/1 или 0/255).
            mask_true (np.ndarray): Истинная маска (0/1 или 0/255).
            beta (float): Коэффициент для F-beta. По умолчанию 1.0 (F1).

        Выход:
            float: F-мера (значение от 0 до 1). Если precision+recall=0, возвращает 0.
        """
        pred = mask_pred.astype(bool)
        true = mask_true.astype(bool)

        tp = np.logical_and(pred, true).sum()
        fp = np.logical_and(pred, ~true).sum()
        fn = np.logical_and(~pred, true).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0
        f = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        return float(f)
    
    
    
    
    

    def get_polygons_by_category(self, image_id: int, category_id: int) -> List[np.ndarray]:
        """Возвращает все полигоны указанной категории для изображения."""
        anns = self.get_annotations(image_id, category_id)
        polygons = []
        for ann in anns:
            seg = ann.get('segmentation')
            if seg and seg[0]:
                pts = np.array(seg[0], dtype=np.int32).reshape(-1, 2)
                polygons.append(pts)
        return polygons

    def get_polygons_by_category(self, image_id: int, category_id: int) -> List[np.ndarray]:
        """Возвращает все полигоны указанной категории для изображения."""
        anns = self.get_annotations(image_id, category_id)
        polygons = []
        for ann in anns:
            seg = ann.get('segmentation')
            if seg and seg[0]:
                pts = np.array(seg[0], dtype=np.int32).reshape(-1, 2)
                polygons.append(pts)
        return polygons

    def polygon_intersection_area(self, poly1: np.ndarray, poly2: np.ndarray) -> float:
        """Вычисляет площадь пересечения двух полигонов (в пикселях)."""
        all_pts = np.vstack([poly1, poly2])
        x, y, w, h = cv2.boundingRect(all_pts)
        mask1 = np.zeros((h, w), dtype=np.uint8)
        mask2 = np.zeros((h, w), dtype=np.uint8)
        poly1_loc = poly1 - [x, y]
        poly2_loc = poly2 - [x, y]
        cv2.fillPoly(mask1, [poly1_loc], 1)
        cv2.fillPoly(mask2, [poly2_loc], 1)
        intersection = cv2.bitwise_and(mask1, mask2)
        return float(np.sum(intersection))

    def assign_polygons_to_lines(self, image_id: int, line_category: int = 4,
                                 source_categories: List[int] = None) -> dict:
        """
        Для каждой строки возвращает список полигонов из source_categories,
        которые присвоены этой строке (по максимальной площади пересечения).
        Возвращает {line_index: list_of_polygons}
        """
        if source_categories is None:
            source_categories = [0, 1, 2]

        line_polygons = self.get_polygons_by_category(image_id, line_category)
        if not line_polygons:
            return {}

        # Собираем все source-полигоны
        source_polygons = []
        for cat in source_categories:
            source_polygons.extend(self.get_polygons_by_category(image_id, cat))

        # Для каждого source-полигона определяем лучшую строку
        best_line_for_source = {}
        for s_idx, s_poly in enumerate(source_polygons):
            best_line = -1
            best_area = 0
            for l_idx, l_poly in enumerate(line_polygons):
                area = self.polygon_intersection_area(s_poly, l_poly)
                if area > best_area:
                    best_area = area
                    best_line = l_idx
            if best_line != -1 and best_area > 0:
                best_line_for_source[s_idx] = best_line

        # Формируем словарь для строк
        lines_polygons = {i: [] for i in range(len(line_polygons))}
        for s_idx, l_idx in best_line_for_source.items():
            lines_polygons[l_idx].append(source_polygons[s_idx])

        return lines_polygons

    def create_line_masks_from_global_binarization(self, image_path: str, image_id: int,
                                               line_category: int = 4,
                                               source_categories: List[int] = None,
                                               debug: bool = False) -> List[np.ndarray]:
        if source_categories is None:
            source_categories = [0, 1, 2]

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить {image_path}")

        from u_net_binarization import binarize_image
        global_binary = binarize_image(img)

        if debug:
            debug_dir = "debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"debug_{image_id}_global_binary.jpg"), global_binary)

        line_polygons = self.get_polygons_by_category(image_id, line_category)
        if not line_polygons:
            return []

        assigned = self.assign_polygons_to_lines(image_id, line_category, source_categories)
        masks = []

        for idx, line_poly in enumerate(line_polygons):
            all_polys = [line_poly] + assigned.get(idx, [])
            if not all_polys:
                continue

            all_pts = np.vstack(all_polys)
            x, y, w, h = cv2.boundingRect(all_pts)

            h_img, w_img = global_binary.shape
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)
            roi_w = x2 - x1
            roi_h = y2 - y1

            poly_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            for poly in all_polys:
                local_poly = poly - np.array([x1, y1])
                cv2.fillPoly(poly_mask, [local_poly], 255)

            global_roi = global_binary[y1:y2, x1:x2]
            roi = np.where(poly_mask == 255, global_roi, 255)

            if debug:
                debug_img = img.copy()
                for poly in all_polys:
                    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                    cv2.polylines(debug_img, [poly], True, color, 2)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0,0,255), 3)
                cv2.imwrite(os.path.join(debug_dir, f"debug_{image_id}_line_{idx}_polys.jpg"), debug_img)
                cv2.imwrite(os.path.join(debug_dir, f"debug_{image_id}_line_{idx}_mask.jpg"), roi)
                cv2.imwrite(os.path.join(debug_dir, f"debug_{image_id}_line_{idx}_poly_mask.jpg"), poly_mask)

            masks.append(roi)

        return masks