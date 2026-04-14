import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    TrainingArguments, 
    Trainer
)
import evaluate

# 1. ЗАГРУЗКА ДАТАСЕТА
# Теперь нам нужны пары: аудио + текстовая транскрипция
# my_data/
# ├── audio1.wav
# ├── audio2.wav
# └── metadata.csv  # колонки: file_name, sentence
# Датасет должен иметь колонки: "audio" и "sentence" (или "text")
dataset = load_dataset("audiofolder", data_dir="path/to/data")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# csvpath,transcript
# /datasets/audio/voice_1.wav,привет мир
# /datasets/audio/voice_2.wav,вторая фраза

from datasets import load_dataset, Audio

# Загружаем структуру из CSV
dataset = load_dataset("csv", data_files="dataset.csv")

# Важно: указываем, какая колонка содержит пути к аудио, чтобы они превратились в массивы чисел
dataset = dataset.cast_column("path", Audio(sampling_rate=16000))
dataset = dataset.rename_column("path", "audio") # для совместимости с кодом выше

# Если у вас на каждый audio1.wav есть файл audio1.txt с тем же названием.
import os
from datasets import Dataset, Audio

def create_dataset(audio_dir, text_dir):
    data = {"audio": [], "sentence": []}
    
    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav"):
            base_name = os.path.splitext(filename)[0]
            audio_path = os.path.join(audio_dir, filename)
            text_path = os.path.join(text_dir, base_name + ".txt")
            
            if os.path.exists(text_path):
                with open(text_path, "r", encoding="utf-8") as f:
                    data["audio"].append(audio_path)
                    data["sentence"].append(f.read().strip())
    
    return Dataset.from_dict(data)

dataset = create_dataset("path/to/audio", "path/to/texts")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# В коде обучения обращаемся так:
# text = batch["transcript"] 

# 2. ПОДГОТОВКА ПРОЦЕССОРА И МОДЕЛИ
model_id = "openai/whisper-small"
# Processor объединяет FeatureExtractor (для звука) и Tokenizer (для текста)
processor = WhisperProcessor.from_pretrained(model_id, language="Russian", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_id)

# 3. ПРЕПРОЦЕССИНГ
def prepare_dataset(batch):
    audio = batch["audio"]
    # Извлекаем признаки из аудио
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    # Токенизируем текст (целевая переменная)
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"])

# 4. СБОРЩИК БАТЧЕЙ (Data Collator)
# Важно: он должен дополнять аудио и текст до нужной длины в рамках батча
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 5. МЕТРИКА (WER — Word Error Rate)
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# 6. ОБУЧЕНИЕ
training_args = TrainingArguments(
    output_dir="./whisper-asr-russian",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=500, # настройте под свой объем данных
    fp16=True,
    evaluation_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
