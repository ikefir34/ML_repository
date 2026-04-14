import numpy as np
import tensorflow as tf

# ResNet50 - стандарт
# EfficientNetB0 - лучшая версия
# MobileNet - для слабых систем

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_input
from tensorflow.keras.applications.resnet50 import decode_predictions as resnet_decode_predictions

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_input
from tensorflow.keras.applications.efficientnet import decode_predictions as efficientnet_decode_predictions

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2  import preprocess_input as mobilenet_input
from tensorflow.keras.applications.mobilenet_v2  import decode_predictions as mobilenet_decode_predictions

from tensorflow.keras.utils import get_file, load_img, img_to_array

# загрузка модели с предобучеными весами на imagenet
model1 = ResNet50(weights='imagenet') # 224*224
model2 = EfficientNetB0(weights='imagenet') # 224*224
model3 = MobileNetV2(weights='imagenet') # 224*224

# предобработка
img = tf.keras.utils.load_img('Datasets/Image_dataset/test_image.jpg', target_size=(224, 224)) # target_size смотреть в документации если брать другие модели
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array_resnet = resnet_input(img_array)  
img_array_efficientnet = efficientnet_input(img_array)  
img_array_mobilenet = mobilenet_input(img_array)  

# пердсказание
predictions = model1.predict(img_array, verbose=0)[0]
class_idx = np.argmax(predictions)
confidence = predictions[class_idx]
print(f"{class_idx}: {confidence:.2%}")
# или
predictions = model2.predict(img_array)
results = efficientnet_decode_predictions(predictions, top=3)[0]
for i, (imagenet_id, label, score) in enumerate(results):
    print(f"{i+1}. {label}: {score:.2%}")

# ----------------------------------------------------------------------
# До обучение
# ----------------------------------------------------------------------

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Генераторы
# target_size входной размер
# 
# Пример датасета
# dataset/
# ├── cats/
# │   ├── cat1.jpg
# │   └── cat2.jpg
# └── dogs/
#     ├── dog1.jpg
#     └── dog2.jpg
# 
# Для  EfficientNetB0 и MobileNetV2 вместо rescale=1./255 пишем preprocess_input

train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory('Datasets\Image_dataset', target_size=(224,224), batch_size=32, subset='training')
val_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory('Datasets\Image_dataset', target_size=(224,224), batch_size=32, subset='validation')

# Модель
base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False
model = tf.keras.Sequential([base, GlobalAveragePooling2D(), Dense(train_gen.num_classes, activation='softmax')])

# Компиляция и обучение
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=10)

# ----------------------------------------------------------------------
# Сохранение и загрузка
# ----------------------------------------------------------------------

import os

# варианты файлов
# (в последних версиях очень хочет keras)
# my_model.h5
# my_model.keras

# Нужно сначала создать папку
os.makedirs('my_models', exist_ok=True)  # exist_ok=True - не ошибка, если папка есть
model.save('my_models/model.keras')

from tensorflow.keras.models import load_model
model = load_model('my_model.keras')

# ----------------------------------------------------------------------
# Создание датасета через библиотеку dataset
# ----------------------------------------------------------------------

from datasets import load_dataset

# Загружаем датасет
dataset = load_dataset("путь к папке с arrow и т. п.")

# Если данные хранятся в не в картинках а в чемто не понятном
# Вданном случае это формат ARROW
def transform_data(example):
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.resize(image, [224, 224])
    return image, example['label']
dataset = dataset.map(transform_data, batched=True)

# Превращаем в tf.data.Dataset
tf_dataset = dataset["train"].to_tf_dataset(
    columns=["image"],       # Название колонки с данными
    label_cols=["label"],    # Название колонки с ответами
    batch_size=32,
    shuffle=True
)