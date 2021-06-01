!pip install pytreebank torch transformers tqdm loguru click nltk numpy scikit-learn utils

import pandas as pd
import numpy as np

num_classes = 5

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
import re
from nltk.tokenize import word_tokenize

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

df_train = pd.DataFrame(list(zip(texts_train, y_train)), columns =['text', 'sentiment'])
df_test = pd.DataFrame(list(zip(texts_test, y_test)), columns =['text', 'sentiment'])

df_train

from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
import torch
from torch.utils.data import DataLoader
import utils
from math import ceil
from loguru import logger
import numpy as np
import os
import time
from datetime import timedelta
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from os.path import join
from os import path
import tensorflow

def switch(i):
  if (i == 0):
    return np.array([[1, 0, 0, 0, 0]])
  if (i == 1):
    return np.array([[0, 1, 0, 0, 0]])
  if (i == 2):
    return np.array([[0, 0, 1, 0, 0]])
  if (i == 3):
    return np.array([[0, 0, 0, 1, 0]])
  if (i == 4):
    return np.array([[0, 0, 0, 0, 1]])
  print("Error Class!")

def to_one_hot(y):
  y_result = switch(y[0])
  for i in range(1, len(y)):
    y_result = np.concatenate((y_result, switch(y[i])), axis=0)
  return y_result

def to_numpy_array(y):
  y_result = np.array([[y[0], y[1], y[2], y[3], y[4]]])
  for i in range(5, len(y), 5):
    y_init = np.array([[y[i], y[i+1], y[i+2], y[i+3], y[i+4]]])
    y_result = np.concatenate((y_result, y_init), axis=0)
  return y_result

def evaluation_metrics(Y_true, Y_pred, y_loss, split='test'):
    metrics = dict()
    metrics[split+'_accuracy'] = accuracy_score(Y_true, Y_pred)
    cce = tensorflow.keras.losses.CategoricalCrossentropy()
    metrics[split+'_loss_keras'] = np.float64(cce(to_one_hot(Y_true), to_numpy_array(y_loss)).numpy())
    metrics[split+'_precision'] = precision_score(Y_true, Y_pred, average='weighted')
    metrics[split+'_recall'] = recall_score(Y_true, Y_pred, average='weighted')
    metrics[split+'_f1_score'] = f1_score(Y_true, Y_pred, average='weighted')
    metrics[split+'_confusion_matrix'] = confusion_matrix(Y_true, Y_pred)

    return metrics


def save_model(model, name, prev_name):
    if prev_name is not None:
        if path.exists(prev_name):
            os.remove(prev_name)
    torch.save(model, name)


def root_and_binary_title(root, binary):
    if root:
        phrase_type = 'root'
    else:
        phrase_type = 'all'
    if binary:
        label = 'binary'
    else:
        label = 'fine'
    return phrase_type, label

def get_binary_label(sentiment):
    if sentiment < 2:
        return 0
    if sentiment > 2:
        return 1
    raise ValueError("Invalid sentiment")


def transformer_params(name):
    return {'batch_size': 6,
            'learning_rate': 1e-5}

class GPT2ForSequenceClassification(torch.nn.Module):
  def __init__(self, num_labels):
    super(GPT2ForSequenceClassification, self).__init__()
    self.model = GPT2Model.from_pretrained('gpt2',
                                       config=GPT2Config.from_pretrained('gpt2'))
    self.max_pool = torch.nn.MaxPool1d(3, 2)
    self.dropout = torch.nn.Dropout(p=0.1)
    self.layer_norm = torch.nn.LayerNorm(768)
    self.conv1d_1 = torch.nn.Conv1d(in_channels=768, out_channels=768, kernel_size=1, stride=1)
    self.fc_layer = torch.nn.Linear(in_features=768, out_features=768)
    self.tanh = torch.nn.Tanh()
    self.out_layer = torch.nn.Linear(in_features=768, out_features=num_labels)
    self.criterion = torch.nn.CrossEntropyLoss()

  def forward(self, input_ids, attention_mask, labels):
    gpt_last_layer = self.model(input_ids, attention_mask=attention_mask)[0]

    gpt_last_layer = gpt_last_layer.permute(0, 2, 1)

    max_pool_out = self.max_pool(gpt_last_layer)

    max_pool_out = self.dropout(max_pool_out)
    max_pool_out = max_pool_out.permute(0, 2, 1)

    layer_norm_out = self.layer_norm(max_pool_out)
    layer_norm_out = layer_norm_out.permute(0, 2, 1)

    conv1d_1_out = self.conv1d_1(layer_norm_out)
    conv1d_1_out = self.tanh(conv1d_1_out)

    global_max_pooling_out, _ = torch.max(conv1d_1_out, axis=2)
    global_max_pooling_out = self.dropout(global_max_pooling_out)

    fc_layer_out = self.fc_layer(global_max_pooling_out)
    fc_layer_out = self.tanh(fc_layer_out)

    fc_layer_out = self.dropout(fc_layer_out)
    logits = self.out_layer(fc_layer_out)
    
    loss = self.criterion(logits, labels)
                                   
    return logits, loss

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer.add_special_tokens({'pad_token': '.'})

def load_transformer(name, binary):
  num_classes = 5
  if binary:
    num_classes = 2
  model = GPT2ForSequenceClassification(num_classes)
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  tokenizer.add_special_tokens({'pad_token': '.'})

  return {'model': model,
          'tokenizer': tokenizer}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_step(model, inputs, labels, optimizer):
    optimizer.zero_grad()

    logits, loss = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)

    loss.backward()
    optimizer.step()

    return logits, loss

def eval_step(model, inputs, labels):
    logits, loss = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)

    return logits, loss

def train_epoch(model, tokenizer, train_dataset, optimizer, batch_size):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    correct_count = 0
    total_loss = 0

    model.train()
    with tqdm(total=ceil(len(train_dataset)/batch_size), desc='train', unit='batch') as pbar:
        for text, sentiment in train_loader:
            text = tokenizer(text, padding=True, return_tensors='pt').to(device)
            sentiment = sentiment.to(device)

            logits, loss = train_step(model, text, sentiment, optimizer)

            preds = torch.argmax(logits, axis=1)
            correct_count += (preds == sentiment).sum().item()
            total_loss += loss.item()
            pbar.update(1)

    return correct_count / len(train_dataset), total_loss / len(train_dataset)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0) 

def eval_epoch(model, tokenizer, eval_dataset, batch_size, split):
    eval_loader = DataLoader(dataset=eval_dataset,
                            batch_size=batch_size,
                            shuffle=True)

    correct_count = 0
    total_loss = 0
    y_pred = list()
    y_true = list()
    y_loss = list()
    model.eval()
    with torch.no_grad():
        with tqdm(total=ceil(len(eval_dataset)/batch_size), desc=split, unit='batch') as pbar:
            for text, sentiment in eval_loader:
                text = tokenizer(text, padding=True, return_tensors='pt').to(device)
                sentiment = sentiment.to(device)

                logits, loss = eval_step(model, text, sentiment)

                preds = torch.argmax(logits, axis=1)
                y_pred += preds.cpu().numpy().tolist()
                for i in range(len(logits)):
                  y_loss += softmax(logits[i].cpu().numpy().tolist()).tolist()
                y_true += sentiment.cpu().numpy().tolist()

                correct_count += (preds == sentiment).sum().item()
                total_loss += loss.item()
                pbar.update(1)

    metrics_score = evaluation_metrics(y_true, y_pred, y_loss, split=split)
    return correct_count / len(eval_dataset), total_loss / len(eval_dataset), metrics_score

class SSTDataset(object):
    def __init__(self, dataset):
        self.text = np.array(dataset.text)
        self.sentiment = np.array(dataset.sentiment)
        print(len(dataset))


    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        erreur = str(idx)+" : "+self.text[idx]+"\n"
        file = open(r"MyFile.txt","a+")
        file.write(erreur)
        file.close()

        text = "0"
        sentiment = np.int64(4)

        if(self.text[idx] != ""):
          text = self.text[idx]
          sentiment = self.sentiment[idx]

        return text, sentiment

from torch.utils.tensorboard import SummaryWriter       
writer = SummaryWriter("logs")

def train(name, root, binary, epochs=4, patience=1, save=False):
    try:
        transformer_container = load_transformer(name, binary)
    except ValueError:
        logger.error("Invalid transformer name!")
        os._exit(0)
    model = transformer_container['model']
    model = model.to(device)
    tokenizer = transformer_container['tokenizer']

    params_container = transformer_params(name)
    batch_size = params_container['batch_size']
    learning_rate = params_container['learning_rate']

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_acc = 0.0
    best_loss = np.inf
    stopping_step = 0
    best_model_name = None
    total_train_seconds = 0
    for epoch in range(epochs):
        
        start = time.time()
        train_acc, train_loss = train_epoch(model, tokenizer, SSTDataset(df_train), optimizer, batch_size)
        end = time.time()
        total_train_seconds += (end - start)
        logger.info(f"epoch: {epoch+1}, transformer: {name}, train_loss: {train_loss:.4f}, train_acc: {train_acc*100:.2f}")

        '''dev_acc, dev_loss, _ = eval_epoch(model, tokenizer, SSTDataset(df_test), batch_size, 'dev')
        logger.info(f"epoch: {epoch+1}, transformer: {name}, dev_loss: {dev_loss:.4f}, dev_acc: {dev_acc*100:.2f}")'''

        test_acc, test_loss, test_evaluation_metrics = eval_epoch(model, tokenizer, SSTDataset(df_test),
                                                                  batch_size, 'test')

        logger.info(f"epoch: {epoch+1}, transformer: {name}, test_loss: {test_loss:.4f}, test_acc: {test_acc*100:.2f}")
        logger.info(f"epoch: {epoch+1}, transformer: {name}, "
                    f"test_precision: {test_evaluation_metrics['test_precision']*100:.2f}, "
                    f"test_loss_keras: {test_evaluation_metrics['test_loss_keras']:.4f}, "
                    f"test_recall: {test_evaluation_metrics['test_recall']*100:.2f}, "
                    f"test_f1_score: {test_evaluation_metrics['test_f1_score']*100:.2f}, "
                    f"test_accuracy_score: {test_evaluation_metrics['test_accuracy']*100:.2f}")
        logger.info(f"epoch: {epoch+1}, transformer: {name}, test_confusion_matrix: \n"
                    f"{test_evaluation_metrics['test_confusion_matrix']}")

        logger.info(f"Total training time elapsed: {timedelta(seconds=total_train_seconds)}")
        logger.info(f"Mean time per train epoch: {timedelta(seconds=total_train_seconds/(epoch+1))}")

        writer.add_scalar('epoch_accuracy', test_acc, epoch)
        writer.add_scalar('epoch_loss', test_evaluation_metrics['test_loss_keras'], epoch)
        writer.flush()

        if save:
            if test_acc > best_acc:
                best_acc = test_acc
                phrase_type, label = root_and_binary_title(root, binary)
                model_name = "{}_{}_{}_{}.pickle".format(name, phrase_type, label, epoch)
                save_model(model, model_name, best_model_name)


        if test_loss < best_loss:
            best_loss = test_loss
            stopping_step = 0
        else:
            stopping_step += 1

        if stopping_step >= patience:
            logger.info("EarlyStopping!")
            os._exit(1)

train('gpt2', False, False, 2, 1, False)