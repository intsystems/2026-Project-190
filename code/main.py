from hpp_method import LineSegmentation
from datasets import load_dataset
import cv2
from school_notebooks_RU import CocoMaskGenerator
import random
from typing import List, Tuple, Optional
import numpy as np

URLS = {
    "images": "/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/library/datasets/school_notebooks_RU/images_base",
    "train_data": "datasets/school_notebooks_RU/annotations_train.json",
    "test_data": "datasets/school_notebooks_RU/annotations_test.json",
    "val_data": "datasets/school_notebooks_RU/annotations_val.json"
}

if __name__ == "__main__":
    # Этот кусок показывает как мы бинаризуемы

    # generator = CocoMaskGenerator('datasets/school_notebooks_RU/annotations_train.json')
 
    # img_id = generator.get_image_id_by_filename('1_10.JPG')
    # category_id = 4
    # mask = generator.create_binary_mask('/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/library/datasets/school_notebooks_RU/images_base/1_10.JPG', img_id, debug = True)

    # cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Window', 800, 600)
    # cv2.imshow('Window', mask)
    # cv2.waitKey(0)                # ждём нажатия клавиши
    # cv2.destroyAllWindows()


    # этот кусок кода из чего состоит датаскт
    generator = CocoMaskGenerator('datasets/school_notebooks_RU/annotations_train.json')
    img_id = generator.get_image_id_by_filename('2_24.JPG')
    print('img_id', img_id)
    category_id = 3

    if img_id is not None:
        mask = generator.create_mask(img_id, category_id=category_id)

        # Визуализация (без bounding boxes)
        generator.visualize('datasets/school_notebooks_RU/images_base/2_24.JPG', img_id, category_id=category_id, show_bbox=True, stretch_x_percent=20.0, stretch_y_percent=50.0, padding_px=20)

        # Пример расчёта IoU между двумя масками (здесь просто сравнение с самой собой)
        iou_val = generator.iou(mask, mask)
        f1_val = generator.f1_score(mask, mask)
        print(f"IoU = {iou_val:.4f}, F1 = {f1_val:.4f}")
    else:
        print("Изображение не найдено.")


    # этот кусок демонстрирует работу первого мтеода

    # lineSegmentation = LineSegmentation()

    # # Чтение
    # image = cv2.imread(URLS['images'] + '/2013.jpg')  # BGR порядок каналов

    # # Отображение (в отдельном окне)
    # cv2.namedWindow('Window', cv2.WINDOW_NORMAL)

    # # (Необязательно) Устанавливаем желаемый размер окна, например 800x600
    # cv2.resizeWindow('Window', 800, 600)
    # cv2.imshow('Window', image)
    # cv2.waitKey(0)                # ждём нажатия клавиши
    # cv2.destroyAllWindows()

    # line_images = lineSegmentation.segment_lines(image.copy())
    # # После того как line_images получен:
    # # line_images = seg.segment_lines(image)

    # # Создаём копию изображения для визуализации (BGR)
    # vis_image = image.copy()

    # # Генерируем случайные цвета для каждой строки (устойчивые при каждом запуске)
    # random.seed(42)  # для воспроизводимости
    # colors = []
    # for _ in range(len(line_images)):
    #     # Случайный BGR цвет (от 0 до 255)
    #     color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #     colors.append(color)

    # # Накладываем маску: для каждого пикселя каждой строки
    # for idx, pixels in enumerate(line_images):
    #     color = colors[idx]
    #     #print('Цвет первого региона ', color)
    #     for (x, y) in pixels:
    #         # Проверяем границы (на случай, если координаты выходят за пределы)
    #         if 0 <= y < vis_image.shape[0] and 0 <= x < vis_image.shape[1]:
    #             vis_image[y, x] = color  # закрашиваем пиксель

    # # Отображение результата
    # cv2.namedWindow('Segmented lines', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Segmented lines', 800, 600)
    # cv2.imshow('Segmented lines', vis_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()