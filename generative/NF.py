import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from torch import nn
from torch.utils.data import DataLoader, Dataset

device = torch.device("cpu")

class Dataset2D(Dataset):
    def __init__(self, data):
        mean = [0, 0]
        cov = [[1, 0.8], [0.8, 1]]
        self.size = len(data)
        self.origX = data 
        self.X = torch.tensor(self.origX).float()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.X[idx, :]
    
def backbone(input_width, network_width=10):
    return nn.Sequential(
            nn.Linear(input_width, network_width),
            nn.ReLU(),
            nn.Linear(network_width, network_width),
            nn.ReLU(),
            nn.Linear(network_width, network_width),
            nn.ReLU(),
            nn.Linear(network_width, input_width),
            nn.Tanh(),
    )

class NormalizingFlow2D(nn.Module):
    def __init__(self, num_coupling, width):
        super(NormalizingFlow2D, self).__init__()
        self.num_coupling = num_coupling
        self.s = nn.ModuleList([backbone(1, width) for x in range(num_coupling)])
        self.t = nn.ModuleList([backbone(1, width) for x in range(num_coupling)])
        
        # Learnable scaling parameters for outputs of S
        self.s_scale = torch.nn.Parameter(torch.randn(num_coupling))
        self.s_scale.requires_grad = True

    def forward(self, x):
        if model.training:
            s_vals = []
            y1, y2 = x[:, :1], x[:, 1:]
            for i in range(self.num_coupling):
                # Alternating which var gets transformed
                if i % 2 == 0:
                    x1, x2 = y1, y2
                    y1 = x1
                    s = self.s_scale[i] * self.s[i](x1)
                    y2 = torch.exp(s) * x2 + self.t[i](x1)                    
                else:
                    x1, x2 = y1, y2
                    y2 = x2
                    s = self.s_scale[i] * self.s[i](x2)
                    y1 = torch.exp(s) * x1 + self.t[i](x2)
                s_vals.append(s)
          
            # Return outputs and vars needed for determinant
            return torch.cat([y1, y2], 1), torch.cat(s_vals)
        else:
            # Assume x is sampled from random Gaussians
            x1, x2 = x[:, :1], x[:, 1:]
             
            for i in reversed(range(self.num_coupling)):
                # Alternating which var gets transformed
                if i % 2 == 0:
                    y1, y2 = x1, x2
                    x1 = y1
                    s = self.s_scale[i] * self.s[i](y1)
                    x2 = (y2 - self.t[i](y1)) * torch.exp(-s)
                else:
                    y1, y2 = x1, x2
                    x2 = y2
                    s = self.s_scale[i] * self.s[i](y2)
                    x1 = (y1 - self.t[i](y2)) * torch.exp(-s)

            return torch.cat([x1, x2], 1)
        
def train_loop(dataloader, model, loss_fn, optimizer, report_iters=10, batch_size=1, device='cpu'):
    size = len(dataloader)
    for batch, X in enumerate(dataloader):
        # Transfer to GPU
        X = X.to(device)
        
        # Compute prediction and loss
        y, s = model(X)
        loss = loss_fn(y, s, batch_size)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % report_iters == 0:
            loss, current = loss.item(), batch
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn, batch_size=1, device='cpu'):
    size = len(dataloader)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            y, s = model(X)
            test_loss += loss_fn(y, s, batch_size)

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

def loss_fn(y, s, batch_size):
    # -log(zero-mean gaussian) + log determinant
    # -log p_x = log(pz(f(x))) + log(det(\partial f/\partial x))
    # -log p_x = 0.5 * y**2 + s1 + s2
    logpx = -torch.sum(0.5 * y**2)
    det = torch.sum(s)
    
    ret = -(logpx + det)
    return torch.div(ret, batch_size)

if __name__ == '__main__':
    # Hyperparameters
    learning_rate = 0.001
    batch_size = 1000
    epochs = 300
    
    # load data
    # DATASET_NAME = 'moons'
    # DATASET_NAME = 'arrow'
    DATASET_NAME = 'phone'
    FOLDER_NAME = '../data/'
    if DATASET_NAME == 'moons':
        FOLDER_NAME += 'moons.csv'
    else:
        FOLDER_NAME += 'encoded/' + DATASET_NAME + '.csv'

    df = pd.read_csv(FOLDER_NAME)
    df = df.drop('label', axis=1)

    mean = [0, 0]
    cov = [[1, 0.8], [0.8, 1]]

    training_data = Dataset2D(df.to_numpy())
    test_data = Dataset2D(np.random.multivariate_normal(mean, cov, 1000))
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Model
    model = NormalizingFlow2D(16, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    print("Done!")

    # Generate synthetic data
    model.eval()
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    with torch.no_grad():
        X = torch.Tensor(np.random.multivariate_normal(mean, cov, 10000)).to(device)
        Y = model(X)
    samples = Y.cpu().numpy().T
    data = training_data.origX.T

    # Save synthetic data
    samples_df = pd.DataFrame(samples.T, columns=['x1', 'x2'])
    samples_df.to_csv('../data/generated/NF/' + DATASET_NAME + '.csv', header=True, index=False)