# ----------------------------------------------------------------------
# Загрузка модели и получение предсказание (без дообучения)
# ----------------------------------------------------------------------

import matplotlib.pyplot as plt
import torch, torchvision
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем модель fasterrcnn с бекбоуном resnet50
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
model.eval().to(device)

# Открваем любое изображение
# image = Image.open("image.jpg")
# Возмем изображение из COCO
image = np.array(I)
img_tensor = transforms.ToTensor()(image).to(device)

# Делаем инференс модели
with torch.no_grad():
    outputs = model([img_tensor])

# Извлекаем предсказанные рамки, метки и скоры для первого изображения
pred_boxes = outputs[0]['boxes']
pred_scores = outputs[0]['scores']
pred_labels = outputs[0]['labels']
pred_masks = outputs[0]['masks']
print("Detected", len(pred_boxes), "objects")

# ----------------------------------------------------------------------
# Рисуем результаты
# ----------------------------------------------------------------------

def plot_boxes(torch_img, torch_boxes_xyxy, torch_masks=None):
    image = (255.0 * (torch_img - torch_img.min()) / (torch_img.max() - torch_img.min())).to(torch.uint8)
    image = image[:3, ...]
    
    # Если есть маски, сначала рисуем их
    if torch_masks is not None:
        # Преобразуем маски в бинарные (bool) и убираем лишнюю размерность
        binary_masks = torch_masks > 0.5
        if binary_masks.dim() == 4:  # (N, 1, H, W) -> (N, H, W)
            binary_masks = binary_masks.squeeze(1)
        # Рисуем маски поверх изображения
        image = draw_segmentation_masks(image, binary_masks, alpha=0.5)
    
    # Рисуем bounding boxes
    output_image = draw_bounding_boxes(image, torch_boxes_xyxy.long(), None)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(output_image.permute(1, 2, 0))

plot_boxes(img_tensor, pred_boxes, pred_masks)

# ----------------------------------------------------------------------
# быстроен дообучение
# ----------------------------------------------------------------------

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем предобученную модель
model = maskrcnn_resnet50_fpn(pretrained=True)

# -----------------------------
# 1. Меняем классификатор боксов
# -----------------------------
num_classes = 3  # 2 класса + фон

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# -----------------------------
# 2. Меняем голову сегментации
# -----------------------------
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256

model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask,
    hidden_layer,
    num_classes
)

model.to(device)

# -----------------------------
# 3. Оптимизатор
# -----------------------------
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# -----------------------------
# 4. Цикл обучения
# -----------------------------
num_epochs = 10
model.train()

for epoch in range(num_epochs):
    for images, targets in train_loader:
        # Mask R-CNN требует список изображений
        images = [img.to(device) for img in images]

        # И список словарей таргетов
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Получаем словарь лоссов
        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, loss: {total_loss.item():.4f}")

# -----------------------------
# 5. Сохранение модели
# -----------------------------
torch.save(model.state_dict(), "maskrcnn_resnet50_fpn.pth")
print("Модель сохранена!")

# ---------------------------------------------------------
#  Дата лоудер и его использование
# ---------------------------------------------------------

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MaskRCNNDataset(Dataset):
    """
    Универсальный датасет для Mask R-CNN.
    Поддерживает:
    - bounding boxes
    - бинарные маски PNG
    - классы
    """

    def __init__(self, root, transforms=None):
        """
        root: путь к датасету
            root/images/*.jpg
            root/masks/<img_id>/*.png
            root/labels/*.txt

        transforms: torchvision.transforms
        """
        self.root = root
        self.transforms = transforms

        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.label_dir = os.path.join(root, "labels")

        self.images = sorted(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        # -----------------------------
        # 1. Загружаем изображение
        # -----------------------------
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        # ID изображения без расширения
        img_id = os.path.splitext(img_name)[0]

        # -----------------------------
        # 2. Загружаем bounding boxes
        # -----------------------------
        label_path = os.path.join(self.label_dir, f"{img_id}.txt")
        boxes = []
        labels = []

        with open(label_path) as f:
            for line in f:
                cls, x1, y1, x2, y2 = map(float, line.split())
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # -----------------------------
        # 3. Загружаем маски
        # -----------------------------
        mask_folder = os.path.join(self.mask_dir, img_id)
        mask_files = sorted(os.listdir(mask_folder))

        masks = []
        for m in mask_files:
            mask_path = os.path.join(mask_folder, m)
            mask = Image.open(mask_path).convert("L")  # grayscale
            mask = np.array(mask)

            # Превращаем в бинарную маску 0/1
            mask = (mask > 0).astype(np.uint8)
            masks.append(mask)

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # -----------------------------
        # 4. Дополнительные поля
        # -----------------------------
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd
        }

        # -----------------------------
        # 5. Трансформации
        # -----------------------------
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)


# ---------------------------------------------------------
# collate_fn — обязателен для Mask R-CNN
# ---------------------------------------------------------
def collate_fn(batch):
    return tuple(zip(*batch))


from torch.utils.data import DataLoader
import torchvision.transforms as T

dataset = MaskRCNNDataset(
    root="dataset",
    transforms=T.ToTensor()
)

train_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

# dataset/
#     images/
#         0001.jpg
#         0002.jpg
#         ...
#     masks/
#         0001/
#             obj1.png
#             obj2.png
#         0002/
#             obj1.png
#     labels/
#         0001.txt
#         0002.txt

# Где:

# labels/0001.txt содержит строки:
# class x1 y1 x2 y2

# masks/0001/obj1.png — бинарная маска объекта (0/255)

torch.save(model.state_dict(), "maskrcnn_resnet50_fpn.pth")