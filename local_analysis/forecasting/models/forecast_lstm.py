import pandas as pd
from torch import tensor
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data as data
import torch

class ForecastLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)

        return x

def create_dataset(input_data: pd.Series, stride: int):
    x, y = [], []
    input_data = np.reshape(input_data.astype('float32').to_numpy(), (-1, 1))
    for i in range(len(input_data) - stride):
        curr_feature = input_data[i: i + stride]
        curr_target = input_data[i + 1: i + stride + 1]
        x.append(curr_feature)
        y.append(curr_target)

    return tensor(x), tensor(y)

def run_lstm_training(train: pd.Series, test: pd.Series, stride: int, num_epochs: int, batch_size: int, device: str):

    x_train, y_train = create_dataset(train, stride)
    x_test, y_test = create_dataset(test, stride)
    x_train.to(device)
    y_train.to(device)
    x_test.to(device)
    y_test.to(device)

    print('X_train shape: {} --- y_train shape: {}\nX_test shape: {} --- y_test shape: {}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    model = ForecastLSTM().to(device)
    optimizer = optim.Adam(model.parameters())
    loss_fun = nn.MSELoss()
    data_loader = data.DataLoader(data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    data_loader
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fun(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 != 0:
            continue

        model.eval()

        with torch.no_grad():
            y_pred = model(x_train.to(device))
            train_error = np.sqrt(loss_fun(y_pred.cpu(), y_train.cpu()))
            y_pred = model(x_test.to(device))
            test_error = np.sqrt(loss_fun(y_pred.cpu(), y_test.cpu()))

        print('Epoch: {} --- train RSME: {} --- test RSME: {}'.format(epoch, train_error, test_error))

    return model

