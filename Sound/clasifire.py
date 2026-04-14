import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor, 
    WhisperForAudioClassification, 
    TrainingArguments, 
    Trainer
)
import numpy as np
import evaluate

# 1. ЗАГРУЗКА ВАШЕГО ДАТАСЕТА
# data/class_1/audio1.wav
# data/class_2/audio2.wav
# Укажите путь к папке с аудио. Структура папок автоматически станет метками (labels)
dataset = load_dataset("audiofolder", data_dir="path/to/your/data")
dataset = dataset["train"].train_test_split(test_size=0.2) # Делим на обучение и тест

# Получаем список меток
labels = dataset["train"].features["label"].names
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

# 2. ПОДГОТОВКА МОДЕЛИ И ПРОЦЕССОРА
model_id = "openai/whisper-base" # можно взять tiny, base, small
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)

model = WhisperForAudioClassification.from_pretrained(
    model_id,
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label
)

# 3. ПРЕПРОЦЕССИНГ (Приведение к 16кГц и создание признаков)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        max_length=480000, # 30 секунд
        truncation=True
    )
    return inputs

encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["audio"])

# 4. МЕТРИКИ (Accuracy)
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# 5. НАСТРОЙКИ ОБУЧЕНИЯ
training_args = TrainingArguments(
    output_dir="./whisper-speech-classifier",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=5,
    fp16=True, # Включите, если есть GPU NVIDIA
    push_to_hub=False,
    report_to="none",
    load_best_model_at_end=True,
)

# 6. ЗАПУСК
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

# Основные нюансы:
# Формат данных: Если у вас нет структуры папок, а есть .csv файл с путями и метками, используйте load_dataset("csv", data_files="train.csv").
# Длительность аудио: Whisper всегда работает с окнами по 30 секунд. Если ваши файлы короче, экстрактор дополнит их тишиной автоматически. Если длиннее — обрежет.
# Память (VRAM): Если модель не влезает в видеокарту
# Уменьшите per_device_train_batch_size до 1-2.
# Увеличьте gradient_accumulation_steps до 8-16.
# Используйте версию whisper-tiny.
# Заморозка весов: Для маленьких датасетов полезно "заморозить" энкодер, чтобы обучалась только голова:

# for param in model.model.parameters():
#     param.requires_grad = False

# ----------------------------------------------------------------------
# Сохранение и загрузка
# ----------------------------------------------------------------------

# Сохраняем модель, веса и конфиг в папку
trainer.save_model("./my_whisper_classifier")

# Важно: сохраните и feature_extractor, чтобы параметры обработки звука (16кГц) были те же
feature_extractor.save_pretrained("./my_whisper_classifier")

from transformers import WhisperForAudioClassification, WhisperFeatureExtractor
import torch
import librosa

# 1. Загружаем сохраненную модель и экстрактор
model_path = "./my_whisper_classifier"
model = WhisperForAudioClassification.from_pretrained(model_path)
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)

# 2. Подготовка аудио (обязательно 16000 Гц)
audio, sr = librosa.load("test_speech.wav", sr=16000)
inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

# 3. Предсказание
model.eval() # переводим в режим оценки
with torch.no_grad():
    logits = model(**inputs).logits

# Получаем ID самого вероятного класса
predicted_class_ids = torch.argmax(logits, dim=-1).item()
print(f"Результат: {model.config.id2label[predicted_class_ids]}")

# или

from transformers import pipeline

# Загружаем всё одной строчкой
classifier = pipeline("audio-classification", model="./my_whisper_classifier")

# Просто передаем путь к файлу
result = classifier("test_speech.wav")
print(result) 
# Вернет список словарей: [{'label': 'эмоция_1', 'score': 0.98}, ...]