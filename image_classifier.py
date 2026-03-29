# ----------------------------------------------------------------------
# база
# ----------------------------------------------------------------------

import numpy as np
import tensorflow as tf

# ----------------------------------------------------------------------
# Загрузка модели через keras и получение предсказание (без дообучения)
# ----------------------------------------------------------------------

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import get_file, load_img, img_to_array

model = ResNet50(weights='imagenet')

# предобработка
img = tf.keras.utils.load_img('test.jpg', target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)  

# пердсказание
predictions = model.predict(img_array, verbose=0)[0]
class_idx = np.argmax(predictions)
confidence = predictions[class_idx]
# или
predictions = model.predict(img_array)
results = decode_predictions(predictions, top=3)[0]
for i, (imagenet_id, label, score) in enumerate(results):
    print(f"{i+1}. {label}: {score:.2%}")

# ----------------------------------------------------------------------
# До обучение
# ----------------------------------------------------------------------

from tensorflow.keras.applications import ResNet50 # Загружаем нужную модель
# from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.efficientnet import preprocess_input

# Генераторы (2 строки)
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
# Для  EfficientNetB0 вместо rescale=1./255 пишем preprocess_input

train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory('dataset/', target_size=(224,224), batch_size=32, subset='training')
val_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory('dataset/', target_size=(224,224), batch_size=32, subset='validation')

# Модель (3 строки)
base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
# base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False
model = tf.keras.Sequential([base, GlobalAveragePooling2D(), Dense(train_gen.num_classes, activation='softmax')])

# Компиляция и обучение (2 строки)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=10)

# ----------------------------------------------------------------------
# Сохранение и загрузка
# ----------------------------------------------------------------------

import os

model.save('my_model.h5') 
model.save('my_model.keras') 

# ✅ НУЖНО сначала создать папку
os.makedirs('my_models', exist_ok=True)  # exist_ok=True - не ошибка, если папка есть
model.save('my_models/model.h5')

from tensorflow.keras.models import load_model
model = load_model('my_model.h5')