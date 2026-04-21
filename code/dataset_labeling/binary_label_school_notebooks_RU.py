import os
import cv2
from school_notebooks_RU import CocoMaskGenerator

URLS = {
    "images": "datasets/school_notebooks_RU/images_base",
    "train_data": "datasets/school_notebooks_RU/annotations_train.json",
    "test_data": "datasets/school_notebooks_RU/annotations_test.json",
    "val_data": "datasets/school_notebooks_RU/annotations_val.json"
}

OUTPUT_DIR = "datasets/school_notebooks_RU/images_binary"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
   for url in ["train_data", "test_data", "val_data"]:
        generator = CocoMaskGenerator(URLS[url]) # ты должен перебирать всё

        # Список всех файлов в папке с изображениями
        image_dir = URLS["images"]
        for filename in os.listdir(image_dir):
            print(filename)
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue  # пропускаем файлы, не являющиеся изображениями

            img_path = os.path.join(image_dir, filename)
            img_id = generator.get_image_id_by_filename(filename)

            if img_id is None:
                print(f"{filename} не найден в аннотациях, пропускаем")
                continue

            print(f"Обработка {filename} (id={img_id})")
            mask = generator.create_binary_mask(img_path, img_id)

            # Сохраняем маску с тем же именем, но с расширением .png
            out_name = os.path.splitext(filename)[0] + ".png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, mask)
            print(f"Сохранено {out_path}")

if __name__ == "__main__":
    main()