import cv2
import numpy as np

def crop_line_rectangle(image: np.ndarray, line_pixels: set, debug: bool, padding: int = 20) -> np.ndarray:
        """
        Строит минимальный повёрнутый прямоугольник ТОЛЬКО по текстовым пикселям строки,
        поворачивает изображение так, чтобы строка стала горизонтальной,
        и возвращает чистое выпрямленное изображение строки.
        """
        if not line_pixels:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        # Точки только текстовых пикселей
        points = np.array([[x, y] for x, y in line_pixels], dtype=np.float32)

        if debug:
            xs, ys = zip(*line_pixels)
            min_y, max_y = max(0, min(ys)), min(image.shape[0]-1, max(ys))
            min_x, max_x = max(0, min(xs)), min(image.shape[1]-1, max(xs) )
            print('Прямоугольник', min_y, min_x, max_y, max_x)

        if len(points) < 5:  # слишком мало точек
            # fallback — обычный bounding box
            xs, ys = zip(*line_pixels)
            min_y, max_y = max(0, min(ys) - padding), min(image.shape[0]-1, max(ys) + padding)
            min_x, max_x = max(0, min(xs) - padding), min(image.shape[1]-1, max(xs) + padding)
            return image[min_y:max_y+1, min_x:max_x+1].copy()

        # Минимальный повёрнутый прямоугольник по текстовым пикселям
        rect = cv2.minAreaRect(points)
        (center_x, center_y), (width, height), angle = rect

        # rect = self.robust_min_area_rect(points)
        # (center_x, center_y), (width, height), angle = rect

        if width < height:
            angle += 90
            width, height = height, width


        if debug:
            box = cv2.boxPoints(rect)       # возвращает массив (4, 2) float32
            box = np.int32(box)              # преобразуем в целые координаты

            # Рисуем красные толстые линии (толщина = 3, цвет = (0,0,255) в BGR)
            cv2.drawContours(image, [box], 0, (0, 0, 255), thickness=3)

            # Показываем результат
            cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Result', 800, 600)
            cv2.imshow('Result', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Матрица поворота
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

        # Новые размеры холста
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((image.shape[1] * cos) + (image.shape[0] * sin))
        new_h = int((image.shape[1] * sin) + (image.shape[0] * cos))

        rotation_matrix[0, 2] += new_w / 2 - center_x
        rotation_matrix[1, 2] += new_h / 2 - center_y

        # Поворачиваем исходное изображение
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))  # белый фон

        # Вычисляем координаты вырезания
        crop_x = int(new_w / 2 - width / 2 - padding)
        crop_y = int(new_h / 2 - height / 2 - padding)
        crop_w = int(width + 2 * padding)
        crop_h = int(height + 2 * padding)

        # Защита границ
        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)
        crop_w = min(crop_w, new_w - crop_x)
        crop_h = min(crop_h, new_h - crop_y)

        straightened = rotated[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        return straightened