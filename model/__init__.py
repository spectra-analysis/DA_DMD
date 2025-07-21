"""
Deep Learning-Assisted Dynamic Mode Decomposition (DA-DMD) model

Author: Adithya Ashok C.V.
Date: 2025-07-17
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pydmd import DMD


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel attention on DMD modes.

    Args:
        num_modes (int): Number of input channels (DMD modes).
        reduction_ratio (int): Ratio for channel reduction in bottleneck. Default is 2.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, num_modes, spectrum_length)

    Returns:
        torch.Tensor: Channel-attention scaled tensor of the same shape.
    """
    def __init__(self, num_modes, reduction_ratio=2):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(num_modes, num_modes // reduction_ratio)
        self.fc2 = nn.Linear(num_modes // reduction_ratio, num_modes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        squeeze = torch.mean(x, dim=2)  # shape: (batch_size, num_modes)
        excitation = F.relu(self.fc1(squeeze))
        excitation = self.sigmoid(self.fc2(excitation)).unsqueeze(2)  # shape: (batch_size, num_modes, 1)
        return x * excitation


class SE_CNN(nn.Module):
    """
    Deep learning network for reconstructing Raman spectra from DMD modes.

    Args:
        spectrum_length (int): Length of input CARS spectrum (default: 1000).
        num_modes (int): Number of DMD modes used as input channels (default: 12).

    Inputs:
        x (torch.Tensor): Tensor of shape (batch_size, num_modes, spectrum_length)

    Returns:
        torch.Tensor: Reconstructed Raman spectra of shape (batch_size, spectrum_length)
    """
    def __init__(self, spectrum_length=1000, num_modes=12):
        super(SE_CNN, self).__init__()

        self.se_block = SEBlock(num_modes)

        self.conv1 = nn.Conv1d(num_modes, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)

        self.skip_conv = nn.Conv1d(num_modes, 512, kernel_size=1)

        self.res1 = nn.Conv1d(512, 512, kernel_size=1)
        self.res2 = nn.Conv1d(512, 512, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.Conv1d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, 3, padding=1)
        )

        self.upsample = nn.Upsample(size=spectrum_length, mode='linear', align_corners=True)

    def forward(self, x):
        """
        Forward pass through the SE and CNN Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_modes, spectrum_length)

        Returns:
            torch.Tensor: Reconstructed spectra of shape (batch_size, spectrum_length)
        """
        x = self.se_block(x)
        skip = self.skip_conv(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = F.relu(self.res1(x + skip))
        x = F.relu(self.res2(x))

        x = self.decoder(x)
        x = self.upsample(x)

        return x.squeeze(1)


class DA_DMD(nn.Module):
    """
    Deep Learning-Assisted Dynamic Mode Decomposition (DA-DMD) model.

    Combines Hankel embedding, DMD, and DL to perform NRB removal in CARS spectra to reconstruct Raman spectra.

    Args:
        input_dim (int): Length of the input spectrum (default: 1000).
        num_modes (int): Number of DMD modes.
        Ndelay (int): Number of delay embeddings for Hankel matrix.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

    Returns:
        latent_dmd (torch.Tensor): DMD latent features of shape (batch_size, num_modes, input_dim)
        reconstructed (torch.Tensor): Reconstructed spectra of shape (batch_size, input_dim)
    """
    def __init__(self, input_dim=1000, num_modes=12, Ndelay=12):
        super(DA_DMD, self).__init__()
        self.num_modes = num_modes
        self.Ndelay = Ndelay
        self.input_dim = input_dim
        self.se_cnn = SE_CNN(spectrum_length=input_dim, num_modes=num_modes)

    def get_hankel_matrix(self, data):
        """
        Converts 1D spectrum into a Hankel matrix with delay embeddings.

        Args:
            data (torch.Tensor): Input tensor of shape (batch_size, spectrum_length)

        Returns:
            torch.Tensor: Hankel matrix of shape (batch_size, Ndelay, spectrum_length-Ndelay+1)
        """
        n = data.shape[1]
        return torch.stack([data[:, i:n - self.Ndelay + 1 + i] for i in range(self.Ndelay)], dim=1).to(data.device)

    def apply_dmd(self, hankel_latent):
        """
        Applies Dynamic Mode Decomposition (DMD) to Hankel-embedded features.

        Args:
            hankel_latent (torch.Tensor): Tensor of shape (batch_size, Ndelay, spectrum_length-Ndelay+1)

        Returns:
            torch.Tensor: DMD-processed features of shape (batch_size, num_modes, spectrum_length)
        """
        batch_size, num_delay, time_steps = hankel_latent.shape
        X = hankel_latent.cpu().detach().numpy().transpose(0, 2, 1)
        dmd_results = []

        for i in range(batch_size):
            dmd = DMD(svd_rank=self.num_modes)
            try:
                dmd.fit(X[i])
                cl_modes = 5 * np.abs(np.real(dmd.modes @ dmd.dynamics))
                intp = np.array([
                    np.interp(np.arange(self.input_dim), np.arange(cl_modes.shape[0]), cl_modes[:, m])
                    for m in range(cl_modes.shape[1])
                ])
            except Exception:
                intp = np.zeros((self.num_modes, self.input_dim))
            dmd_results.append(intp)

        return torch.tensor(np.array(dmd_results), dtype=torch.float32).to(hankel_latent.device)

    def forward(self, x):
        """
        Full forward pass through DA-DMD model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, spectrum_length)

        Returns:
            latent_dmd (torch.Tensor): Latent DMD features (batch_size, num_modes, spectrum_length)
            reconstructed (torch.Tensor): Reconstructed spectra (batch_size, spectrum_length)
        """
        hankel_latent = self.get_hankel_matrix(x)
        latent_dmd = self.apply_dmd(hankel_latent)
        reconstructed = self.se_cnn(latent_dmd)
        return latent_dmd, reconstructed
