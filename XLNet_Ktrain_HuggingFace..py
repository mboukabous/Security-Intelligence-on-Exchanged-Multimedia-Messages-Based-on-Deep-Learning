import pandas as pd
import numpy as np

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

x_train = data_train.Text.tolist()
x_test = data_test.Text.tolist()

y_train = data_train.Emotion.tolist()
y_test = data_test.Emotion.tolist()

data = data_train.append(data_test, ignore_index=True)

class_names = ['joy', 'sadness', 'fear', 'anger', 'neutral']

print('size of training set: %s' % (len(data_train['Text'])))
print('size of validation set: %s' % (len(data_test['Text'])))
print(data.Emotion.value_counts())

data.head(10)

import ktrain
from ktrain import text
MODEL_NAME = 'xlnet-base-cased'

t = text.Transformer(MODEL_NAME, maxlen=500, classes=class_names)
trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_test, y_test)
model = t.get_classifier()

learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)

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

learner.fit_onecycle(2e-5, 10, callbacks = callbacks)

learner.validate(val_data=val, class_names=t.get_classes())

predictor = ktrain.get_predictor(learner.model, preproc=t)
predictor.get_classes()

from sklearn.metrics import precision_recall_fscore_support
predictions = [predictor.predict(pred) for pred in x_test]
print(precision_recall_fscore_support(data_test.Emotion, predictions, average='weighted'))

predictor.save("models/xlnet_model")
