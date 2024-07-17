import os, sys
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
from src.utils import set_seed





@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )
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
    EPOCH=15
    def objective(trial):
        #畳み込み層の数
        #num_layer = trial.suggest_int('num_layer', 3, 7)

        #FC層のユニット数
        hid_dim = int(trial.suggest_discrete_uniform("hid_dim", 100, 500, 100))

        #各畳込み層のフィルタ数
        #num_filters = [int(trial.suggest_discrete_uniform("num_filter_"+str(i), 16, 128, 16)) for i in range(num_layer)]

        model = RNN(trial,num_classes=1854, seq_len=281, in_channels=271,hid_dim=hid_dim).to(args.device)
        optimizer = get_optimizer(trial, model)

        accuracy = Accuracy(task="multiclass", num_classes=1854, top_k=10).to(args.device)

        for step in range(EPOCH):
            train_loss, train_acc, val_loss, val_acc = [], [], [], []
            train(model=model,train_loader=train_loader, optimizer=optimizer,train_loss=train_loss,accuracy=accuracy,train_acc=train_acc)

            error_rate = test(model=model,val_loader=val_loader,val_loss=val_loss,val_acc=val_acc,accuracy=accuracy)
        return error_rate

    def train(model,train_loader,optimizer,train_loss,accuracy,train_acc):
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

    def test(model,val_loader,val_loss,val_acc,accuracy):
        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())
        return 1-np.mean(val_acc)

    TRIAL_SIZE = 30
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE)
    print(study.best_params)
    print(study.best_value)



if __name__ == "__main__":
    run()

