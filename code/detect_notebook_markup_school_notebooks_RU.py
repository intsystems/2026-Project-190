import os
import cv2
from school_notebooks_RU import CocoMaskGenerator

URLS = {
    "images": "datasets/school_notebooks_RU/images_base",
    "train_data": "datasets/school_notebooks_RU/annotations_train.json",
    "test_data": "datasets/school_notebooks_RU/annotations_test.json",
    "val_data": "datasets/school_notebooks_RU/annotations_val.json"
}
OUTPUT_BASE = "datasets/school_notebooks_RU/images_target"
os.makedirs(OUTPUT_BASE, exist_ok=True)

def process_json(json_path: str, split_name: str, debug: bool = False):
    generator = CocoMaskGenerator(json_path)
    image_dir = URLS["images"]

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # if filename != "2297.jpg":
        #     continue

        img_id = generator.get_image_id_by_filename(filename)
        if img_id is None:
            print(f"{filename} не найден в {split_name}")
            continue

        print(f"Обработка {filename} ({split_name})")
        img_path = os.path.join(image_dir, filename)
        line_masks = generator.create_line_masks_from_global_binarization(
            img_path, img_id,
            line_category=4,
            source_categories=[0,1,2],
            debug=debug
        )
        if not line_masks:
            print(f"  Нет строк для {filename}")
            continue

        base_name = os.path.splitext(filename)[0]
        out_dir = os.path.join(OUTPUT_BASE, base_name)
        os.makedirs(out_dir, exist_ok=True)

        for idx, mask in enumerate(line_masks):
            out_path = os.path.join(out_dir, f"{idx:04d}.png")
            cv2.imwrite(out_path, mask)
            print(f"  Сохранена строка {idx} -> {out_path}")


def main():
    debug_mode = False
    for split in ["train_data", "val_data", "test_data"]:
        print(f"\n=== Обработка {split} (debug={debug_mode}) ===")
        process_json(URLS[split], split, debug=debug_mode)

if __name__ == "__main__":
    main()