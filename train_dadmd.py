"""
Train the DA-DMD model using synthetic data and save the model.

"""

from model.dadmd import DA_DMD
from utils import standardize_minmax
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.MSELoss()


def train_model(model, train_loader, num_epochs=50, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            _, reconstructed = model(X_batch)
            loss = loss_fn(reconstructed, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss / len(train_loader):.6f}")
    return model


if __name__ == '__main__':

    hankel_dim = 40

    cars = standardize_minmax(np.load('synthetic_data/2_cars_2000.npy'))
    raman = standardize_minmax(np.load('synthetic_data/2_raman_2000.npy'))

    X_noisy = torch.tensor(cars[:1700, :], dtype=torch.float32).to(device)
    X_clean = torch.tensor(raman[:1700, :], dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_noisy, X_clean)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = DA_DMD(input_dim=1000, num_modes=hankel_dim, Ndelay=hankel_dim).to(device)
    trained_model = train_model(model, train_loader, num_epochs=50)

    ## save the model
    torch.save(trained_model, 'models/pretrained_dadmd.pt')
