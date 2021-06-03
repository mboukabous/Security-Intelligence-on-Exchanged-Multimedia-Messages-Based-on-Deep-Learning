import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
import re

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

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

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequence_train = tokenizer.texts_to_sequences(texts_train)
sequence_test = tokenizer.texts_to_sequences(texts_test)

index_of_words = tokenizer.word_index

vocab_size = len(index_of_words) + 1

print('Number of unique words: {}'.format(len(index_of_words)))

X_train_pad = pad_sequences(sequence_train, maxlen = max_seq_len )
X_test_pad = pad_sequences(sequence_test, maxlen = max_seq_len )

X_train_pad

encoding = {
    'joy': 0,
    'fear': 1,
    'anger': 2,
    'sadness': 3,
    'neutral': 4
}

# Integer labels
y_train = [encoding[x] for x in data_train.Emotion]
y_test = [encoding[x] for x in data_test.Emotion]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train

GLOVE_DIM = 200

import urllib.request
import zipfile
import os

fname = 'glove.twitter.27B.' + str(GLOVE_DIM) + 'd.txt'

if not os.path.isfile(fname):
    print('Downloading word vectors...')
    urllib.request.urlretrieve('http://nlp.stanford.edu/data/glove.twitter.27B.zip',
                              'glove.twitter.27B.zip')
    print('Unzipping...')
    with zipfile.ZipFile('glove.twitter.27B.zip', 'r') as zip_ref:
        zip_ref.extractall('')
    print('done.')
    
    os.remove('glove.twitter.27B.zip')


glove_file = 'glove.twitter.27B.' + str(GLOVE_DIM) + 'd.txt'
emb_dict = {}
glove = open(glove_file)
for line in glove:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    emb_dict[word] = vector
glove.close()

NB_WORDS = vocab_size
MAX_LEN = max_seq_len

emb_matrix = np.zeros((NB_WORDS, GLOVE_DIM))
for w, i in tokenizer.word_index.items():
    if i < NB_WORDS:
        vect = emb_dict.get(w)
        if vect is not None:
          emb_matrix[i] = vect
    else:
        break

embedd_layer = Embedding(vocab_size,
                         GLOVE_DIM,
                         input_length = max_seq_len,
                         weights = [emb_matrix],
                         trainable=False)

lstm_size = 128

model = Sequential()
model.add(embedd_layer)

model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(lstm_size)))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
import os

basedir = "logs/"
logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tf.debugging.experimental.enable_dump_debug_info(logdir)

callbacks = [
ModelCheckpoint(filepath=basedir+'checkpoint1-{epoch:02d}.hdf5', verbose=2, save_best_only=True, monitor='accuracy',mode='max'),
CSVLogger(basedir+'model_1trainanalysis1.csv',separator=',', append=False),
EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=1, verbose=2, mode='auto'),
TensorBoard(log_dir=logdir,histogram_freq=1)]

from time import time
t1 = time()

batch_size = 32
epochs = 20

print(X_test_pad.shape)
print(y_test.shape)

print(X_train_pad.shape)
print(y_train.shape)

hist = model.fit(X_train_pad, y_train, 
                 batch_size=batch_size,
                 epochs=epochs,
                 callbacks = callbacks,
                 validation_data=(X_test_pad,y_test))

t2 = time()
t_delta = round(t2-t1,2)
print(t_delta)

# Accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

predictions = model.predict(X_test_pad)
predictions = np.argmax(predictions, axis=1)
predictions = [class_names[pred] for pred in predictions]

print(precision_recall_fscore_support(data_test.Emotion, predictions, average='micro'))
print(precision_recall_fscore_support(data_test.Emotion, predictions, average='macro'))
print(precision_recall_fscore_support(data_test.Emotion, predictions, average='weighted'))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    
    fig.set_size_inches(12.5, 7.5)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.grid(False)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

print("\nF1 Score: {:.2f}".format(f1_score(data_test.Emotion, predictions, average='micro') * 100))

plot_confusion_matrix(data_test.Emotion, predictions, classes=class_names, normalize=True, title='Normalized confusion matrix')
plt.show()

print('Message: {}\nPredicted: {}'.format(X_test[22], predictions[22]))

import time

message = ['Test!']

seq = tokenizer.texts_to_sequences(message)
padded = pad_sequences(seq, maxlen=max_seq_len)

start_time = time.time()
pred = model.predict(padded)

print('Message: ' + str(message))
print('predicted: {} ({:.2f} seconds)'.format(class_names[np.argmax(pred)], (time.time() - start_time)))

model.save('models/GloVe_BiLSTM.h5')