# ----------------------------------------------------------------------
# Загрузка модели и получение предсказание (без дообучения)
# ----------------------------------------------------------------------

import matplotlib.pyplot as plt
import torch, torchvision
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем модель fasterrcnn с бекбоуном resnet50
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval().to(device)

# Открваем любое изображение
# image = Image.open("image.jpg")
# Возмем изображение из COCO
image = np.array(image)
img_tensor = transforms.ToTensor()(image).to(device)

# Делаем инференс модели
with torch.no_grad():
    outputs = model([img_tensor])

# Извлекаем предсказанные рамки, метки и скоры для первого изображения
pred_boxes = outputs[0]['boxes']
pred_scores = outputs[0]['scores']
pred_labels = outputs[0]['labels']
print("Detected", len(pred_boxes), "objects")

# ----------------------------------------------------------------------
# Рисуем результаты
# ----------------------------------------------------------------------

def plot_boxes(torch_img, torch_boxes_xyxy):
    image = (255.0 * (torch_img - torch_img.min()) / (torch_img.max() - torch_img.min())).to(torch.uint8)
    image = image[:3, ...]
    output_image = draw_bounding_boxes(image, torch_boxes_xyxy.long(), None)

    plt.figure(figsize=(10, 6))
    plt.imshow(output_image.permute(1, 2, 0))

plot_boxes(img_tensor, pred_boxes)

# ----------------------------------------------------------------------
# быстроен дообучение
# ----------------------------------------------------------------------

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Заменяем голову (классификатор) у pre-trained Faster R-CNN и меняем у неё количество классов на нужное нам
num_classes = 3  # например, 2 класса + фон
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).to(device)

# Оптимизатор
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

num_epochs = 10
train_loader = # YOUR BEST TRAIN LOADER
model.train()
for epoch in range(num_epochs):
    for images, targets in train_loader:
        images = images.to(device)
        # при необходимости можем перенести таргеты на GPU
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # обучаем модель
        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print(f"total loss: {total_loss}")


# ---------------------------------------------------------
#  Дата лоудер и его использование
# ---------------------------------------------------------

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class UniversalDetectionDataset(Dataset):
    """
    Универсальный датасет для задач детекции объектов.
    Поддерживает разные форматы аннотаций через адаптеры.
    Легко расширяется под новые форматы.
    """

    def __init__(self, root, annotation_format="custom", transforms=None):
        """
        root: корневая папка датасета
            root/images/*.jpg
            root/labels/*.txt / *.json / *.xml / ...

        annotation_format:
            - "custom" → txt: class x1 y1 x2 y2
            - "yolo"   → txt: class cx cy w h (нормализованные)
            - "coco"   → один JSON в корне
            - "none"   → без аннотаций (инференс)

        transforms: torchvision.transforms
        """

        self.root = root
        self.transforms = transforms
        self.annotation_format = annotation_format

        # Папки с изображениями и аннотациями
        self.img_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "labels")

        # Список файлов
        self.images = sorted(os.listdir(self.img_dir))
        self.annotations = (
            sorted(os.listdir(self.ann_dir))
            if os.path.exists(self.ann_dir)
            else None
        )

        # Для COCO — грузим JSON один раз
        if annotation_format == "coco":
            with open(os.path.join(root, "annotations.json"), "r") as f:
                self.coco = json.load(f)

    # ---------------------------------------------------------
    #  Основной метод: возвращает (image, target)
    # ---------------------------------------------------------
    def __getitem__(self, idx):
        # Загружаем изображение
        img_path = os.path.join(self.img_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        # Если аннотаций нет — режим инференса
        if self.annotation_format == "none":
            target = {"image_id": torch.tensor([idx])}
            if self.transforms:
                img = self.transforms(img)
            return img, target

        # Путь к аннотации
        ann_path = os.path.join(self.ann_dir, self.annotations[idx])

        # Вызываем нужный парсер
        if self.annotation_format == "custom":
            boxes, labels = self._parse_custom(ann_path)

        elif self.annotation_format == "yolo":
            boxes, labels = self._parse_yolo(ann_path, img.size)

        elif self.annotation_format == "coco":
            boxes, labels = self._parse_coco(idx)

        else:
            raise ValueError(f"Unknown annotation format: {self.annotation_format}")

        # Преобразуем в тензоры
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Обязательные поля для моделей детекции
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd
        }

        # Применяем трансформации
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)

    # ---------------------------------------------------------
    #  Ниже — адаптеры под разные форматы аннотаций
    # ---------------------------------------------------------

    def _parse_custom(self, ann_path):
        """
        Формат:
        class x1 y1 x2 y2
        """
        boxes, labels = [], []
        with open(ann_path) as f:
            for line in f:
                cls, x1, y1, x2, y2 = map(float, line.split())
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls))
        return boxes, labels

    def _parse_yolo(self, ann_path, img_size):
        """
        YOLO формат:
        class cx cy w h (нормализованные)
        """
        W, H = img_size
        boxes, labels = [], []

        with open(ann_path) as f:
            for line in f:
                cls, cx, cy, w, h = map(float, line.split())

                # Денормализация
                cx *= W
                cy *= H
                w *= W
                h *= H

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls))

        return boxes, labels

    def _parse_coco(self, idx):
        """
        COCO JSON (один общий файл)
        """
        boxes, labels = [], []
        img_id = idx  # можно заменить на реальный ID, если нужно

        for ann in self.coco["annotations"]:
            if ann["image_id"] == img_id:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])

        return boxes, labels


from torch.utils.data import DataLoader
import torchvision.transforms as T

dataset = UniversalDetectionDataset(
    root="dataset",
    annotation_format="yolo",   # custom / yolo / coco / none
    transforms=T.ToTensor()
)

train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

# dataset/
#     images/
#         0001.jpg
#         0002.jpg
#         ...
#     labels/
#         0001.txt
#         0002.txt

# 2) annotation_format — какой формат разметки лежит в папке labels/
# Ты выбираешь один из вариантов:

# "custom" → txt: class x1 y1 x2 y2

# "yolo" → txt: class cx cy w h (нормализованные)

# "coco" → один JSON-файл annotations.json

# "none" → без разметки (инференс)

# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# ----------------------------------------------------------------------
# Сохранение и загрузка
# ----------------------------------------------------------------------

torch.save(model.state_dict(), "faster_rcnn.pth")

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

num_classes = 3

model = fasterrcnn_resnet50_fpn(pretrained=False)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load("faster_rcnn.pth", map_location="cpu"))
model.to(device)
model.eval()