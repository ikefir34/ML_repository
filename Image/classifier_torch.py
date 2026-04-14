import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image

# Подключаем cuda если есть
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем предобученную модель (например, ResNet18)
model = models.resnet18(weights='IMAGENET1K_V1')

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

img = Image.open("Datasets/Image_dataset/test_image.jpg").convert('RGB') # Открываем и конвертируем в RGB
img_tensor = transform(img)                 # Применяем трансформации [3, 224, 224]
img_tensor = img_tensor.unsqueeze(0)        # Добавляем размерность батча -> [1, 3, 224, 224]
    
# Перевод в режим оценки
model.eval()
model.to(device)
img_tensor = img_tensor.to(device)
    
# Предсказание
with torch.no_grad():
    outputs = model(img_tensor)
    _, preds = torch.max(outputs, 1)
    # Получаем вероятности (опционально)
    probs = torch.nn.functional.softmax(outputs, dim=1)


# predicted_class = class_names[preds.item()] нужно получить список класоов
confidence = probs[0][preds[0]].item()

print(preds.item())
print(confidence)

# ----------------------------------------------------------------------
# До обучение
# ----------------------------------------------------------------------

# Замораживаем параметры,
for param in model.parameters():
    param.requires_grad = False

# Заменяем последний слой под количество  классов
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4) 
model = model.to(device)

# Функция для предобработки дфнных
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Данные должны лежать вот так в папках
# data/
# └── train/
#     ├── class_1/     
#     │   ├── image1.jpg
#     │   ├── image2.png
#     │   └── ...
#     └── class_2/        
#         ├── image_a.jpg
#         ├── image_b.jpg
#         └── ...
train_dataset = datasets.ImageFolder('Datasets/Image_dataset/train', transform=data_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Параметры для олбучения
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Короткий цикл обучения
print("начало обучения")
for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
print("конец обучения")

# ----------------------------------------------------------------------
# Сохранение и загрузка
# ----------------------------------------------------------------------

# сохраняем веса
torch.save(model.state_dict(), 'resnet18_finetuned.pth')

# 1. Создаем структуру модели (такую же, как при обучении)
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4) 

# Загружаем веса
model.load_state_dict(torch.load('resnet18_finetuned.pth'))

# ----------------------------------------------------------------------
# Полный цикл обучения с валидацией
# ----------------------------------------------------------------------

from torch.utils.data import DataLoader

# Данные и трансформации
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Загружаем датасеты (папки 'data/train' и 'data/val')
image_datasets = {x: datasets.ImageFolder(f'data/{x}', data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'val']}

# Параметры теже + num_epochs
num_epochs = 5

# Цикл обучения и валидации
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Включаем расчет градиентов только в фазе обучения
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    