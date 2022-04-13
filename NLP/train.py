# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:59:22 2022

@author: Riley
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import gensim.downloader
import os
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

def get_model(download=True):
    if download:
        gensim.downloader.download('word2vec-google-news-300') # downloads the model
    path = os.path.join(gensim.downloader.base_dir, 'word2vec-google-news-300', "GoogleNews-vectors-negative300.bin")
    keyedVector = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    return keyedVector

def tokenizer(arr,keyedVector):
    tokens = []
    for sentence in arr:
        sentence_tokens = []
        words = word_tokenize(sentence)
        for word in words:
            if word in keyedVector:
                encoding = keyedVector[word]
                sentence_tokens.append(encoding)
        tokens.append(sentence_tokens)
    tokens = [np.asarray(i).reshape(-1) for i in tokens]
    return tokens # List of arrays

keyedVector = get_model(download=False)

# Get data from csv and separate it
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Select model
ans = input('LSTM or Transformer: ')

# Define label vector
if ans == 'LSTM':
    y = df_train['target'].values.astype('float')
if ans == 'Transformer':
    y = np.repeat(df_train['target'].values.astype('float').reshape(-1,1),300,axis=1)

# Split train into train/val
df_train, df_val, y_train, y_val = train_test_split(df_train, y, test_size=0.2, random_state=42)

# Tokenize and pad text
text_train = tokenizer(df_train['text'].values.astype('str'), keyedVector)
text_val = tokenizer(df_val['text'].values.astype('str'), keyedVector)
text_test = tokenizer(df_test['text'].values.astype('str'), keyedVector)
text_train = [np.concatenate((i, np.zeros((9900-len(i))))) for i in text_train] # pad to length of max length
text_val = [np.concatenate((i, np.zeros((9900-len(i))))) for i in text_val]
text_test = [np.concatenate((i, np.zeros((9900-len(i))))) for i in text_test]

# Tokenize and pad keyword
keyword_train = tokenizer(df_train['keyword'].values.astype('str'), keyedVector)
keyword_val = tokenizer(df_val['keyword'].values.astype('str'), keyedVector)
keyword_test = tokenizer(df_test['keyword'].values.astype('str'), keyedVector)
keyword_train = [np.concatenate((i, np.zeros((900-len(i))))) for i in keyword_train] # pad to length of max length
keyword_val = [np.concatenate((i, np.zeros((900-len(i))))) for i in keyword_val]
keyword_test = [np.concatenate((i, np.zeros((900-len(i))))) for i in keyword_test]

# Tokenize and pad location
location_train = tokenizer(df_train['location'].values.astype('str'), keyedVector)
location_val = tokenizer(df_val['location'].values.astype('str'), keyedVector)
location_test = tokenizer(df_test['location'].values.astype('str'), keyedVector)
location_train = [np.concatenate((i, np.zeros((2400-len(i))))) for i in location_train] # pad to length of max length
location_val = [np.concatenate((i, np.zeros((2400-len(i))))) for i in location_val]
location_test = [np.concatenate((i, np.zeros((2400-len(i))))) for i in location_test]

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define Dataset
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, text, keyword, location, target=None):
        self.text = text
        self.keyword = keyword
        self.location = location
        self.target = target
    def __len__(self):
        return len(self.text)
    def __getitem__(self, idx):
        concat = np.concatenate((self.text[idx], self.keyword[idx], self.location[idx])).reshape(44,300) # 44 = (9900+900+2400)/300
        if self.target is not None:
            return torch.from_numpy(concat), torch.Tensor(np.array(self.target[idx]))
        return torch.from_numpy(concat)

# Define data loader
batch_size = 32
train_loader = torch.utils.data.DataLoader(TweetDataset(text_train, keyword_train, location_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(TweetDataset(text_val, keyword_val, location_val, y_val), batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(TweetDataset(text_test, keyword_test, location_test), batch_size=batch_size, shuffle=False)

# Create an LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Call model
if ans == 'LSTM':
    lstm_model()
if ans == 'Transformer':
    transformer()

def lstm_model():
    model = LSTM(input_size=300, hidden_size=200, num_layers=2, num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 20
    for epoch in range(num_epochs):
        correct_train = correct_val = 0
        for i, batch in enumerate(train_loader):
            model.train()
            x = batch[0].float().to(device)
            y = batch[1].float().to(device)
            y_pred = model(x).squeeze()
            loss = F.mse_loss(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct_train += (y_pred.round() == y).sum().item()
        for i, batch in enumerate(val_loader):
            model.eval()
            x = batch[0].float().to(device)
            y = batch[1].float().to(device)
            y_pred = model(x).squeeze()
            correct_val += (y_pred.round() == y).sum().item()
        print(f'Loss: {loss.item():.2f}, Train Accuracy: {correct_train/len(train_loader.dataset):.2f}, Val Accuracy: {correct_val/len(val_loader.dataset):.2f}')
        
    preds = np.empty((1,1))
    submission = pd.DataFrame(columns=['id','target'])
    for x in test_loader:
        model.eval()
        x = x.float().to(device)
        y_pred = model(x)
        y_pred = y_pred.round()
        y_pred = y_pred.cpu().detach().numpy()
        preds = np.concatenate((preds, y_pred))
    submission.id = df_test['id']
    submission.target = preds[1:].astype('int')
    submission.to_csv("submission.csv", index=False)

"""
Achieves ~80% accuracy; top 25%
"""

def transformer():  
    model = nn.Transformer(d_model=300, nhead=4, batch_first=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        correct_train = correct_val = 0
        for batch in train_loader:
            model.train()
            x = batch[0].float().to(device)
            y = batch[1].float().reshape(batch_size,1,300).to(device)
            y_pred = model(x,y) 
            loss = F.mse_loss(y_pred[:,0,0], y[:,0,0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct_train += (y_pred.round() == y).sum().item()
        for i, batch in enumerate(val_loader):
            model.eval()
            x = batch[0].float().to(device)
            y = batch[1].float().reshape(batch_size,1,300).to(device)
            y_pred = model(x,y)[:,0,0]
            correct_val += (y_pred.round() == y[:,0,0]).sum().item()
        print(f'Loss: {loss.item():.2f}, Train Accuracy: {correct_train/len(train_loader.dataset):.2f}, Val Accuracy: {correct_val/len(val_loader.dataset):.2f}')
    
"""
Achieves ~78% accuracy
"""