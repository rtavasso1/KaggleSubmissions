# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 21:52:44 2022

@author: Riley
"""
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import xgboost as xgb
from sklearn.metrics import accuracy_score



def encode(df,cols,type_='str'):
    enc = OrdinalEncoder()
    for col in cols:
        if type_ == 'bool':
            df[col] = df[col].fillna(df[col].median())
        if type_ == 'float':
            df[col] = df[col].fillna(df[col].median())
        if type_ == 'str':
            df[col] = df[col].fillna('o')
        df[col] = enc.fit_transform(df[col].values.reshape(-1,1))
        df = df.replace('o', df.median())
    return df

cols = ["HomePlanet", "CryoSleep", "Destination", "Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall","Spa", "VRDeck"]
cat_cols = ['HomePlanet', 'Destination',]
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
bool_cols = ['CryoSleep','VIP']

# Read and Combine Data
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
PassengerId = df_test['PassengerId'] # need this for submission
df_all = pd.concat([df_train, df_test], ignore_index=True)
df_all = df_all[cols]
y_train = df_train.Transported.astype('int').values.reshape(-1,1)

# Preprocess and Split Data
df_all = encode(df_all,cat_cols)
df_all = encode(df_all,num_cols, 'float')
df_all = encode(df_all,bool_cols, 'bool')
y_train = pd.DataFrame(OrdinalEncoder().fit_transform(y_train))
df_train, df_test = df_all.iloc[:len(df_train)], df_all.iloc[len(df_train):]
df_train, df_val, y_train, y_val = train_test_split(df_train, y_train, test_size=0.2, random_state=42)

ans = input('PyTorch or XGBoost: ')

if ans == 'PyTorch':
        
    """
    PyTorch Model
    """

    class TabularDataset(Dataset):
        def __init__(self, df, y=None, transform=None, train=True):
            self.df = df
            self.y = y
            self.len = len(df)
            self.transform = transform
        def __getitem__(self, index):
            if self.y is not None:
                sample = self.df.iloc[index,:], self.y.iloc[index]
            else:
                sample = self.df.iloc[index,:]
            if self.transform:
                sample = self.transform(sample)
            return sample
        def __len__(self):
            return self.len

    class Rescale(object):
        def __init__(self, scale):
            self.scale = scale
        def __call__(self, sample):
            if len(sample) == 2:
                x, y = sample
                return x/(self.scale), y
            else:
                x = sample
                return x/(self.scale)

    class RandomJitter(object):
        def __init__(self, sigma=0.01):
            self.sigma = sigma
        def __call__(self, sample):
            x, y = sample
            x = x.astype('float32')
            x = x + np.random.normal(0, self.sigma, x.shape)
            return x, y

    class ToNumpy(object):
        def __call__(self, sample):
            if len(sample) == 2:
                x, y = sample
                return x.values.astype('float32'), y.values.astype('float32')
            else:
                x = sample
                return x.values.astype('float32')

    class ToTensor(object):
        def __call__(self, sample):
            if len(sample) == 2:
                x, y = sample
                return torch.from_numpy(x), torch.from_numpy(y)
            else:
                x = sample
                return torch.from_numpy(x)

    batch_size = 32
    train_data = TabularDataset(df_train, y_train, transform=Compose([ToNumpy(), Rescale(1), RandomJitter(0.01), ToTensor()]))
    val_data = TabularDataset(df_val, y_val, transform=Compose([ToNumpy(), Rescale(1), ToTensor()]))
    test_data = TabularDataset(df_test, transform=Compose([ToNumpy(), Rescale(1), ToTensor()]))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    class TabNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.BatchNorm1 = nn.BatchNorm1d(df_train.shape[1])
            self.fc1 = nn.Linear(len(df_train.columns), 64)
            self.BatchNorm2 = nn.BatchNorm1d(64)
            self.fc2 = nn.Linear(64, 32)
            self.BatchNorm3 = nn.BatchNorm1d(32)
            self.fc3 = nn.Linear(32, 16)
            self.BatchNorm4 = nn.BatchNorm1d(16)
            self.fc4 = nn.Linear(16, 1)
        def forward(self, x):
            x = self.BatchNorm1(x)
            x = F.relu(self.fc1(x))
            x = self.BatchNorm2(x)
            x = F.relu(self.fc2(x))
            x = self.BatchNorm3(x)
            x = F.relu(self.fc3(x))
            x = self.BatchNorm4(x)
            x = torch.sigmoid(self.fc4(x))
            return x

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TabNet().to(device)
    optim = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    correct_train = correct_val = 0
    for epoch in range(5):
        correct_train = correct_val = 0
        for i, (x, y) in enumerate(train_loader):
            model.train()
            x, y = x.to(device), y.to(device)
            y_pred = model(x.float())
            loss = criterion(y_pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            correct_train += (y_pred.round() == y).sum().item()
        for j, (x, y) in enumerate(val_loader):
            model.eval()
            x, y = x.to(device), y.to(device)
            y_pred = model(x.float())
            correct_val += (y_pred.round() == y).sum().item()
        print(f"Epoch: {epoch}, Train Accuracy: {correct_train/((i+1)*batch_size)*100:.2f}%, Train Loss: {loss.item():.2f}, Val Accuracy: {correct_val/((j+1)*batch_size)*100:.2f}%")

    preds = np.empty((1,1))
    submission = pd.DataFrame(columns=['PassengerId', 'Transported'])
    for x in test_loader:
        model.eval()
        x = x.to(device)
        y_pred = model(x.float()).float()
        y_pred = y_pred.round()
        y_pred = y_pred.cpu().detach().numpy()
        preds = np.concatenate((preds, y_pred))
    submission.PassengerId = PassengerId
    submission.Transported = preds[1:].astype('bool')
    submission.to_csv("submission.csv", index=False)
    
    """
    Achieves ~78% Accuracy on Kaggle
    """


if ans == 'XGBoost':
    
    """
    XGBoost Model
    """
    # Create submission dataframe
    submission = pd.DataFrame(columns=['PassengerId', 'Transported'])
    
    # Set up DMatrix
    dtrain = xgb.DMatrix(df_train, label=y_train)
    dval = xgb.DMatrix(df_val, label=y_val)
    dtest = xgb.DMatrix(df_test)
    
    # Define Model Parameters
    param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic', 'eval_metric':'logloss'}
    num_round = 30
    
    # Train Model
    bst = xgb.train(param, dtrain, num_round)
    
    # Calculate Validation Accuracy
    val_preds = bst.predict(dval).round()
    val_acc = accuracy_score(y_val, val_preds)
    print(f'Validation Accuracy: {val_acc*100:.2f}%')
    
    # Make Predictions
    test_preds = bst.predict(dtest).round().astype('bool')
    submission['PassengerId'] = PassengerId
    submission['Transported'] = test_preds
    submission.to_csv('submission.csv', index=False)
    
    """
    Achieves ~79% Accuracy on Kaggle
    """
    
    
    
    
    
    
    