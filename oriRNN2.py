import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
from sklearn.preprocessing import Normalizer

user1 = pd.read_csv('../Data/User01_201508040439/Data.csv')
user1Output = pd.read_csv('OriOutput.csv')

user1.drop(user1.columns[[3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16]], axis=1, inplace=True)

cols = user1.columns.tolist()

user1Acc = np.asarray(user1[['ACCELEROMETER X (m/s²)', 'ACCELEROMETER Y (m/s²)', 'ACCELEROMETER Z (m/s²)']].values)
user1Mag = np.asarray(user1[['MAGNETIC FIELD X (μT)', 'MAGNETIC FIELD Y (μT)', 'MAGNETIC FIELD Z (μT)']].values)

def centAndScale(magData):

    x = [float(row[0]) for row in magData]
    y = [float(row[1]) for row in magData]
    z = [float(row[2]) for row in magData]

    offset_x = (max(x) + min(x)) / 2
    offset_y = (max(y) + min(y)) / 2
    offset_z = (max(z) + min(z)) / 2

    corrected_mag_data = []

    for row in magData:

        corrected_x = float(row[0]) - offset_x
        corrected_y = float(row[1]) - offset_y
        corrected_z = float(row[2]) - offset_z

        corrected_mag_data.append([corrected_x, corrected_y, corrected_z])

    norm_corr_mag_data = Normalizer().fit_transform(corrected_mag_data)

    return norm_corr_mag_data

norm_user1_mag = centAndScale(user1Mag)

userTensor1 = np.concatenate((user1Acc, norm_user1_mag), axis=1)

userTensor1Out = user1Output.values
X1 = torch.as_tensor(userTensor1).float()
y1 = torch.as_tensor(userTensor1Out).float()

train = torch.utils.data.TensorDataset(X1, y1)

batch_size = 100
n_iters = 10000
#num_epochs = 1
num_epochs = n_iters / (len(X1) / batch_size)
num_epochs = int(num_epochs)

print(num_epochs)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, drop_last=True, shuffle=False)

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, layer_size, output_size):

        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.layer_size = layer_size

        # self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=layer_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X, hidden):

        #hidden = self.init_hidden(X)

        out, hidden = self.rnn(X, hidden)

        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self):

        return torch.zeros(self.layer_size, batch_size, self.hidden_size).float()


input_dim = 6
hidden_dim = 6
layer_dim = 1
output_dim = 6

rnn = RNN(input_dim, hidden_dim, layer_dim, output_dim)

error = nn.MSELoss()

learning_rate = 0.05

optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

count = 0

for epoch in range(num_epochs):

    hidden = rnn.init_hidden()

    for X, y in train_loader:

        train = X.view(-1, 1, X.shape[1]).float()

        optimizer.zero_grad()

        outputs, hidden = rnn(train, hidden)

        loss = error(outputs, y)

        loss.backward(retain_graph=True)

        optimizer.step()

        count += 1

        # if count % 250 == 0:
        #
        #     correct = 0
        #     total = 0
        #
        #     for X, y in train_loader:
        #
        #         train = X.view(-1, 1, X.shape[1]).float()
        #
        #         outputs = rnn(train)



    print('Output:', outputs[0])
    print('True:', y[0])
    print('Loss:', loss.data.item())