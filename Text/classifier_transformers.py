import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Загрузка данных (замените 'data.csv' на ваш файл)
# Файл должен иметь колонки 'text' и 'label_name' (например, 'spam'/'ham')
df = pd.read_csv('Datasets\Text_dataset\cyberbullying_tweets.csv')

# Превращаем текстовые метки в числа (0, 1, 2...)
df.loc[(df.cyberbullying_type == "religion"), "cyberbullying_type"] = int(0)
df.loc[(df.cyberbullying_type == "age"), "cyberbullying_type"] = int(1)
df.loc[(df.cyberbullying_type == "gender"), "cyberbullying_type"] = int(2)
df.loc[(df.cyberbullying_type == "ethnicity"), "cyberbullying_type"] = int(3)
df.loc[(df.cyberbullying_type == "not_cyberbullying"), "cyberbullying_type"] = int(4)
df.loc[(df.cyberbullying_type == "other_cyberbullying"), "cyberbullying_type"] = int(5)

# df = df.rename(columns={'cyberbullying_type': 'labels'}) обязательно
df = df.rename(columns={'cyberbullying_type': 'labels'})
df = df.rename(columns={'tweet_text': 'text'})


labels = sorted(df['labels'].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df['labels'] = df['labels'].map(label2id)

# Создаем объект Dataset и делим на обучение/тест
dataset = Dataset.from_pandas(df[['text', 'labels']]).train_test_split(test_size=0.2)

# мультиязычные
# "bert-base-multilingual-cased" — Преимущество: Понимает 104 языка одновременно. Полезна, если в данных перемешаны русский и английский.
# "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" — Преимущество: Очень легкая и быстрая мультиязычная модель.

# инглишь
# "distilbert-base-uncased" — Преимущество: Идеальный баланс скорости и точности. На 40% быстрее классического BERT.


# Подготовка модели и токенизатора
model_name = "cointegrated/rubert-tiny2" # Отличная легкая модель для русского языка
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_func(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_func, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(labels),
    id2label=id2label, # не обязательно
    label2id=label2id # не обязательно
)

# Настройка и запуск обучения
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

print("Пуск обучения")
trainer.train()

# ----------------------------------------------------------------------
# Сохранение и загрузка
# ----------------------------------------------------------------------

# # Сохранение результата
# model.save_pretrained("./my_classifier")
# tokenizer.save_pretrained("./my_classifier")

# from transformers import pipeline

# # Укажите путь к папке, куда вы сохранили модель
# model_path = "./my_classifier"

# # Создаем "пайплайн" для классификации текста
# # Он автоматически подгрузит и модель, и токенизатор из указанной папки
# classifier = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

# # Пример использования
# texts = [
#     "Этот продукт превзошел все мои ожидания!",
#     "Ужасный сервис, никогда больше не вернусь.",
#     "В целом нормально, но доставка долгая."
# ]

# results = classifier(texts)

# # Вывод результатов
# for text, res in zip(texts, results):
#     print(f"Текст: {text}")
#     print(f"Результат: {res['label']} (уверенность: {res['score']:.2f})/n")
