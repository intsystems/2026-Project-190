# School Notebooks Dataset

The images of school notebooks with handwritten notes in Russian.

The dataset annotation contain end-to-end markup for training detection and OCR models, as well as an end-to-end model for reading text from pages.

## Annotation format

The annotation is in COCO format. The `annotation.json` should have the following dictionaries:

- `annotation["categories"]` - a list of dicts with a categories info (categotiy names and indexes).
- `annotation["images"]` - a list of dictionaries with a description of images, each dictionary must contain fields:
   - `file_name` - name of the image file.
   - `id` for image id.
- `annotation["annotations"]` - a list of dictioraties with a murkup information. Each dictionary stores a description for one polygon from the dataset, and must contain the following fields:
   - `image_id` - the index of the image on which the polygon is located.
   - `category_id` - the polygon’s category index.
   - `attributes` - dict with some additional annotation information. In the `translation` subdict you can find text translation for the line.
   - `segmentation` - the coordinates of the polygon, a list of numbers - which are coordinate pairs x and y.

Ссылка на сайт от куда датасет был позаимствован: https://huggingface.co/datasets/ai-forever/school_notebooks_RU


id	name	Описание
0	pupil_text	Текст, написанный учеником (возможно, весь блок текста ученика)
1	pupil_comment	Комментарий ученика (например, пометки на полях)
2	teacher_comment	Комментарий учителя (исправления, замечания)
3	paper	Бумага, фон (вся страница)
4	text_line	Отдельная строка текста