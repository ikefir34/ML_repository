import pandas as pd
import fasttext
from sklearn.model_selection import train_test_split
import re
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.preprocessing import remove_stopwords

# Подготовка датасета
data = pd.read_csv("/content/cyberbullying_tweets.csv")

data.loc[(data.cyberbullying_type == "religion"), "cyberbullying_type"] = int(0)
data.loc[(data.cyberbullying_type == "age"), "cyberbullying_type"] = int(1)
data.loc[(data.cyberbullying_type == "gender"), "cyberbullying_type"] = int(2)
data.loc[(data.cyberbullying_type == "ethnicity"), "cyberbullying_type"] = int(3)
data.loc[(data.cyberbullying_type == "not_cyberbullying"), "cyberbullying_type"] = int(4)
data.loc[(data.cyberbullying_type == "other_cyberbullying"), "cyberbullying_type"] = int(5)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\sa-zA-Z0-9@\[\]]',' ',text)
    text = re.sub(r'\w*\d+\w*', '', text)
    text = re.sub('\s{2,}', " ", text)
    return text

data['tweet_text'] = data['tweet_text'].apply(clean_text)
train, test = train_test_split(data, test_size = 0.2)

# Переводим в текстовые файлы
with open('train.txt', 'w') as f:
    for each_text, each_label in zip(train['tweet_text'], train['cyberbullying_type']):
        f.writelines(f'__label__{each_label} {each_text}\n')

with open('test.txt', 'w') as f:
    for each_text, each_label in zip(test['tweet_text'], test['cyberbullying_type']):
        f.writelines(f'__label__{each_label} {each_text}\n')

# ----------------------------------------------------------------------
# До обучение
# ----------------------------------------------------------------------

def print_results(sample_size, precision, recall):
    precision = round(precision, 2)
    recall = round(recall, 2)
    print(f'{sample_size=}')
    print(f'{precision=}')
    print(f'{recall=}')

model = fasttext.train_supervised('train.txt', epoch=1, autotuneValidationFile='test.txt', autotuneMetric="f1:__label__1")

print_results(*model.test('test.txt'))

pred = model.predict("^.^")

pred_int = int(pred[0][0][9])

# ----------------------------------------------------------------------
# Сохранение и загрузка
# ----------------------------------------------------------------------

model.save_model('optimized.model')
model1 = fasttext.load_model("optimized.model")