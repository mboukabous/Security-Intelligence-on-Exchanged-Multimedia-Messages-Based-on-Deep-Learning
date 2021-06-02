!pip3 install ktrain

import pandas as pd
import numpy as np

import ktrain
from ktrain import text

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

X_train = data_train.Text.tolist()
X_test = data_test.Text.tolist()

y_train = data_train.Emotion.tolist()
y_test = data_test.Emotion.tolist()

data = data_train.append(data_test, ignore_index=True)

class_names = ['joy', 'sadness', 'fear', 'anger', 'neutral']

print('size of training set: %s' % (len(data_train['Text'])))
print('size of validation set: %s' % (len(data_test['Text'])))
print(data.Emotion.value_counts())

data.head(10)

encoding = {
    'joy': 0,
    'sadness': 1,
    'fear': 2,
    'anger': 3,
    'neutral': 4
}

y_train = [encoding[x] for x in y_train]
y_test = [encoding[x] for x in y_test]

(x_train,  y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=X_train, y_train=y_train,
                                                                       x_test=X_test, y_test=y_test,
                                                                       class_names=class_names,
                                                                       preprocess_mode='bert',
                                                                       maxlen=350, 
                                                                       max_features=135000)

model = text.text_classifier('bert', train_data=(x_train, y_train), preproc=preproc)

learner = ktrain.get_learner(model, train_data=(x_train, y_train), 
                             val_data=(x_test, y_test),
                             batch_size=6)

import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
import os

basedir = "/content/logs/"
logdir = os.path.join("/content/logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tf.debugging.experimental.enable_dump_debug_info(logdir)

callbacks = [
ModelCheckpoint(filepath=basedir+'checkpoint1-{epoch:02d}.hdf5', verbose=2, save_best_only=True, monitor='accuracy',mode='max'),
CSVLogger(basedir+'model_1trainanalysis1.csv',separator=',', append=False),
EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=1, verbose=2, mode='auto'),
TensorBoard(log_dir=logdir,histogram_freq=1)]

learner.fit_onecycle(2e-5, 2, callbacks = callbacks)

learner.validate(val_data=(x_test, y_test), class_names=class_names)

predictor = ktrain.get_predictor(learner.model, preproc)
predictor.get_classes()

from sklearn.metrics import precision_recall_fscore_support

predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
predictions = [class_names[pred] for pred in predictions]

print(precision_recall_fscore_support(data_test.Emotion, predictions, average='weighted'))

import time 

message = 'Test Message'

start_time = time.time() 
prediction = predictor.predict(message)

print('predicted: {} ({:.2f})'.format(prediction, (time.time() - start_time)))

predictor.save("models/bert_model")
