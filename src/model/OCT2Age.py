import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.nn as nn


class NN_OCT2Age(nn.Module):
    def __init__(self, input_dim, hidden_dims=[10], output_dim=1, dropout_rate=0.3):
        super(NN_OCT2Age, self).__init__()

        # 全ての層のサイズを保持するリスト
        all_layers = [input_dim] + hidden_dims + [output_dim]

        # 隠れ層と出力層を作成
        self.layers = nn.ModuleList()
        for i in range(len(all_layers) - 1):
            layer = nn.Linear(all_layers[i], all_layers[i + 1])

            # Heの初期化（ReLUの場合に適した初期化）
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

            self.layers.append(layer)

        # ドロップアウト層
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))  # 活性化関数を適用
            x = self.dropout(x)  # ドロップアウトを適用
        x = self.layers[-1](x)  # 出力層（活性化関数なし）
        return x


def train(model, train_loader, optimizer, criterion = nn.MSELoss(), device='mps'):
    model.train()  # Set the model to training mode
    total_loss = 0

    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Move X_batch to the specified device
        optimizer.zero_grad()  # Zero the gradient buffers

        Y_pred = model(X_batch)  # Forward pass
        Y_true = Y_batch.unsqueeze(1)

        loss = criterion(Y_pred, Y_true) 
        loss.backward()  # Backward pass

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
        optimizer.step()  # Update weights
        
        loss_per_batch = loss.item()
        total_loss += loss_per_batch

    average_loss = total_loss / len(train_loader)
    return average_loss


def valid(model, test_loader, criterion = nn.MSELoss(), device='mps'):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    Y_trues = []
    Y_preds = []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Move X_batch to the specified device
            Y_pred = model(X_batch)
            Y_true = Y_batch.unsqueeze(1)

            loss = criterion(Y_pred, Y_true)  
            loss_per_batch = loss.item()            
      
            total_loss += loss_per_batch
                
    average_loss = total_loss / len(test_loader)
    return average_loss
