import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
import re

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, GRU, Dense, Dropout, SpatialDropout1D, Conv1D, MaxPooling1D

num_classes = 5

embed_num_dims = 200

max_seq_len = 500

class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']

def shuffle(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
      df.apply(np.random.shuffle, axis=axis)
    return df

data = pd.read_csv('data/dataset.csv', encoding='utf-8', sep=';')

data.sort_values(by='Emotion', axis=0, inplace=True)

data.set_index(keys=['Emotion'], drop=False,inplace=True)

emotions=data['Emotion'].unique().tolist()

joys = shuffle(data.loc[data.Emotion=='joy'])
fears = shuffle(data.loc[data.Emotion=='fear'])
angers = shuffle(data.loc[data.Emotion=='anger'])
sadnesss = shuffle(data.loc[data.Emotion=='sadness'])
neutrals = shuffle(data.loc[data.Emotion=='neutral'])

joys_train = joys.iloc[0:int(joys.shape[0]*0.8)]
joys_test = joys.iloc[int(joys.shape[0]*0.8)+1:joys.shape[0]]

fears_train = fears.iloc[0:int(fears.shape[0]*0.8)]
fears_test = fears.iloc[int(fears.shape[0]*0.8)+1:fears.shape[0]]

angers_train = angers.iloc[0:int(angers.shape[0]*0.8)]
angers_test = angers.iloc[int(angers.shape[0]*0.8)+1:angers.shape[0]]

sadnesss_train = sadnesss.iloc[0:int(sadnesss.shape[0]*0.8)]
sadnesss_test = sadnesss.iloc[int(sadnesss.shape[0]*0.8)+1:sadnesss.shape[0]]

neutrals_train = neutrals.iloc[0:int(neutrals.shape[0]*0.8)]
neutrals_test = neutrals.iloc[int(neutrals.shape[0]*0.8)+1:neutrals.shape[0]]

data_train = pd.concat([joys_train, fears_train, angers_train, sadnesss_train, neutrals_train])
data_test = pd.concat([joys_test, fears_test, angers_test, sadnesss_test, neutrals_test])

print(data_train.shape)
print(data_test.shape)

X_train = data_train.Text
X_test = data_test.Text

y_train = data_train.Emotion
y_test = data_test.Emotion

data = data_train.append(data_test, ignore_index=True)

print(data.Emotion.value_counts())
data.head(6)

print(data_train.Emotion.value_counts())
print(data_test.Emotion.value_counts())

def clean_text(data):
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    data = word_tokenize(data)
    return data

import nltk
nltk.download('punkt')

texts = [' '.join(clean_text(text)) for text in data.Text]

texts_train = [' '.join(clean_text(text)) for text in X_train]
texts_test = [' '.join(clean_text(text)) for text in X_test]

print(texts_train[22])

encoding = {
    'joy': 0,
    'fear': 1,
    'anger': 2,
    'sadness': 3,
    'neutral': 4
}

y_train = [encoding[x] for x in data_train.Emotion]
y_test = [encoding[x] for x in data_test.Emotion]

df_train = pd.DataFrame(list(zip(y_train, texts_train)), columns =['idx', 'text'])
df_test = pd.DataFrame(list(zip(y_test, texts_test)), columns =['idx', 'text'])

df_train

from fastai.text import *
data_lm = TextLMDataBunch.from_df('.', train_df=df_train, valid_df=df_test)

data_lm.show_batch()

data_clas = TextClasDataBunch.from_df(path="", train_df = df_train, valid_df = df_test, vocab=data_lm.train_ds.vocab, bs=32 )
data_clas.show_batch()

data_clas.save('data_clas.pkl')

learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

learn.lr_find()
learn.recorder.plot(show_grid=True, suggestion=True)

learn.fit_one_cycle(1, learn.recorder.min_grad_lr, moms=(0.9,0.7))

learn.save('fit_head')

learn.unfreeze()

learn.lr_find()
learn.recorder.plot(show_grid=True, suggestion=True)

learn.fit_one_cycle(10, learn.recorder.min_grad_lr, moms=(0.9,0.7))

learn.save('fine_tuned')
learn.save_encoder('fine_tuned_enc')

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('fine_tuned_enc')

learn.lr_find()
learn.recorder.plot(show_grid=True, suggestion=True)

learn.fit_one_cycle(1,learn.recorder.min_grad_lr, moms=(0.9,0.7))
learn.save('first')

learn.lr_find()
learn.recorder.plot(show_grid=True, suggestion=True)

learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(learn.recorder.min_grad_lr/(2.6**4),learn.recorder.min_grad_lr), moms=(0.9,0.7))
learn.save('second')

learn.lr_find()
learn.recorder.plot(show_grid=True, suggestion=True)

learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(learn.recorder.min_grad_lr/(2.6**4),learn.recorder.min_grad_lr), moms=(0.9,0.7))

learn.lr_find()
learn.recorder.plot(show_grid=True, suggestion=True)

!pip install git+https://github.com/lanpa/tensorboard-pytorch

!pip install git+https://github.com/Pendar2/fastai-tensorboard-callback

from fastai.collab import *
from fastai_tensorboard_callback import *

learn.unfreeze()
learn.fit_one_cycle(4, slice(learn.recorder.min_grad_lr/(2.6**4),learn.recorder.min_grad_lr), moms=(0.9,0.7), callbacks=[TensorboardLogger(learn, "run")])

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

y_true = df_test['text'].apply(lambda row: np.int64(learn.predict(row)[0]))

print(precision_recall_fscore_support(df_test['idx'], y_true, average='weighted'))

learn.save('final')