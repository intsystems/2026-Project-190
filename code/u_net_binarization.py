import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Union, Tuple

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mp = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.mp(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNetTiny(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 8)
        
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        
        self.up1 = Up(64, 32)
        self.up2 = Up(32, 16)
        self.up3 = Up(16, 8)
        
        self.outc = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)

def load_unet_model(
    model_path: str = "models/u_net/unet_binarization_3_(6-architecture).pth",
    device: str = None,
) -> Tuple[UNetTiny, torch.device]:
    """
    Загружает U-Net один раз для многократного переиспользования.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    model = UNetTiny(in_channels=1, out_channels=1).to(torch_device)
    model.eval()

    state_dict = torch.load(model_path, map_location=torch_device, weights_only=True)
    for k, v in state_dict.items():
        if torch.isnan(v).any():
            print(f"NaN in {k}")
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model, torch_device


def _prepare_image_for_unet(
    image: Union[str, np.ndarray],
    target_size: Tuple[int, int],
    debug: bool,
) -> Tuple[np.ndarray, int, int, int, int, int, int]:
    """
    Приводит изображение к grayscale и формирует padded-версию для U-Net.
    """
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Не удалось открыть изображение: {image}")
        original_h, original_w = img.shape
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            img = image.squeeze()
        else:
            img = image
        original_h, original_w = img.shape[:2]
    else:
        raise TypeError("image должен быть str (путь) или np.ndarray")

    h, w = img.shape
    target_h, target_w = target_size
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    image_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if debug:
        cv2.imwrite("image_resized.png", image_resized)

    padded_image = np.full((target_h, target_w), 255, dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = image_resized
    if debug:
        cv2.imwrite("padded_image.png", padded_image)

    return padded_image, original_h, original_w, y_offset, x_offset, new_h, new_w


def binarize_image_with_loaded_model(
    image: Union[str, np.ndarray],
    model: UNetTiny,
    device: Union[str, torch.device],
    target_size: Tuple[int, int] = (3000, 3000),
    threshold: float = 0.5,
    debug: bool = False,
) -> np.ndarray:
    """
    Бинаризация изображения с уже загруженной U-Net моделью.
    """
    if isinstance(device, str):
        torch_device = torch.device(device)
    else:
        torch_device = device

    padded_image, original_h, original_w, y_offset, x_offset, new_h, new_w = _prepare_image_for_unet(
        image=image,
        target_size=target_size,
        debug=debug,
    )

    img_norm = padded_image.astype(np.float32) / 255.0
    img_norm = (img_norm - 0.5) / 0.5
    input_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(torch_device)

    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits)
        pred_padded = (prob > threshold).float() * 255

    pred_padded = pred_padded.squeeze().cpu().numpy().astype(np.uint8)
    if debug:
        cv2.imwrite("pred_padded.png", pred_padded)

    pred_resized = pred_padded[
        y_offset : y_offset + new_h,
        x_offset : x_offset + new_w
    ]

    final_mask = cv2.resize(
        pred_resized,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )
    return final_mask


def binarize_image(
    image: Union[str, np.ndarray],
    model_path: str = "models/u_net/unet_binarization_3_(6-architecture).pth",
    target_size: Tuple[int, int] = (3000, 3000),
    device: str = None,
    threshold: float = 0.5,
    debug: bool = False
) -> np.ndarray:
    """
    Функция бинаризации одного изображения с помощью обученной U-Net.

    Что делает:
    1. Загружает модель (поддерживает сохранение как с DataParallel, так и без).
    2. Приводит изображение к grayscale.
    3. Масштабирует с сохранением aspect ratio до target_size.
    4. Добавляет padding 255 (как в обучении).
    5. Пропускает через модель (logits → sigmoid).
    6. Обрезает padding.
    7. Масштабирует маску обратно к оригинальному размеру (INTER_NEAREST).
    8. Применяет порог и возвращает бинарную маску uint8.

    Входные параметры:
        image (Union[str, np.ndarray]):
            - str: путь к изображению (.jpg, .png и т.д.)
            - np.ndarray: изображение в формате (H, W) или (H, W, 3) — будет приведено к grayscale
        model_path (str): путь к файлу .pth с весами модели (по умолчанию "unet_binarization.pth")
        target_size (Tuple[int, int]): размер, на котором обучалась модель (по умолчанию (4000, 4000))
        device (str): 'cuda' или 'cpu'. Если None — автоматически выбирает CUDA если доступно.
        threshold (float): порог после sigmoid (по умолчанию 0.5)

    Выход:
        np.ndarray: бинарная маска размера оригинального изображения (H, W), dtype=uint8
                     0   — foreground (обычно текст/объекты)
                     255 — background (фон)

    Пример использования:
        mask = binarize_image("photo.jpg")
        cv2.imwrite("binary_mask.png", mask)
    """
    model, torch_device = load_unet_model(model_path=model_path, device=device)
    try:
        return binarize_image_with_loaded_model(
            image=image,
            model=model,
            device=torch_device,
            target_size=target_size,
            threshold=threshold,
            debug=debug,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Пример 1: из файла
    binary_mask = binarize_image("/home/sasha/Documents/CourseMIPT/MyFirstScientificWork/2026-Project-190/code/datasets/HWR200/hw_dataset/2/reuse5/ФотоТемное/2.jpg", target_size = (3000, 3000))
    cv2.imwrite("result_binary.png", binary_mask)
