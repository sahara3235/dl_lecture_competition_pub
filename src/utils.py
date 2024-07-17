import os, sys,random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
import optuna

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.models import RNN

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_optimizer(trial, model):
  optimizer_names = ['Adam', 'MomentumSGD', 'rmsprop']
  optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)
  weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
  if optimizer_name == optimizer_names[0]: 
    adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
    optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
  elif optimizer_name == optimizer_names[1]:
    momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
    optimizer = optim.SGD(model.parameters(), lr=momentum_sgd_lr, momentum=0.9, weight_decay=weight_decay)
  else:
    optimizer = optim.RMSprop(model.parameters())
  return optimizer

def get_activation(trial):
    activation_names = ['ReLU', 'Sigmoid','Tanh']
    activation_name = trial.suggest_categorical('activation', activation_names)
    if activation_name == activation_names[0]:
        activation = F.relu
    elif activation_name == activation_names[1]:
        activation = F.sigmoid
    else:
       activation =F.tanh
    return activation

def objective(trial,EPOCH):
  device = "cuda" if torch.cuda.is_available() else "cpu"

  #畳み込み層の数
  num_layer = trial.suggest_int('num_layer', 3, 7)

  #FC層のユニット数
  mid_units = int(trial.suggest_discrete_uniform("mid_units", 100, 500, 100))

  #各畳込み層のフィルタ数
  num_filters = [int(trial.suggest_discrete_uniform("num_filter_"+str(i), 16, 128, 16)) for i in range(num_layer)]

  model = RNN(trial, num_layer, mid_units, num_filters).to(device)
  optimizer = get_optimizer(trial, model)

  for step in range(EPOCH):
    train(model, device, train_loader, optimizer)
    error_rate = test(model, device, test_loader)

  return error_rate

def train(model,train_loader,args: DictConfig,optimizer,train_loss,accuracy,train_acc):
    model.train()
    for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
        X, y = X.to(args.device), y.to(args.device)

        y_pred = model(X)
            
        loss = F.cross_entropy(y_pred, y)
        train_loss.append(loss.item())
            
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
            
        acc = accuracy(y_pred, y)
        train_acc.append(acc.item())

def test(model,val_loader,args: DictConfig,val_loss,val_acc,accuracy):
    model.eval()
    for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
        X, y = X.to(args.device), y.to(args.device)
            
        with torch.no_grad():
            y_pred = model(X)
            
        val_loss.append(F.cross_entropy(y_pred, y).item())
        val_acc.append(accuracy(y_pred, y).item())

