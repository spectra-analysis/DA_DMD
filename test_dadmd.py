"""
Test the DA-DMD model.

"""

import torch
import torch.nn as nn
import numpy as np
from model.dadmd import DA_DMD
from utils import standardize_minmax, normalize
from sklearn.metrics import mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def denoise_spectrum(model, raw_spectrum):
    """
    Pass a raw Raman spectrum through the DA-DMD model to denoise it.

    Args:
        model (torch.nn.Module): Trained DA-DMD model
        raw_spectrum (np.ndarray): 1D input spectrum of shape (spectrum_length,)

    Returns:
        np.ndarray: Denoised output spectrum of shape (spectrum_length,)
    """
    model.eval()
    spectrum_tensor = torch.tensor(raw_spectrum, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        _, denoised_spectrum = model(spectrum_tensor)

    return denoised_spectrum.cpu().numpy().squeeze()


if __name__ == '__main__':
    # Load and preprocess data
    cars = standardize_minmax(np.load('synthetic_data/2_cars_2000.npy'))
    raman = standardize_minmax(np.load('synthetic_data/2_raman_2000.npy'))

    X_noisy = cars[1700:, :]   # test input
    X_clean = raman[1700:, :]  # ground truth

    # Load model and pretrained weights
    trained_model = torch.load('pretrained_models/dmd_cnn_2.1_40xskip.pt')

    # Run denoising on all test samples
    test_losses = []
    for i in range(len(X_noisy)):
        noisy_input = normalize(X_noisy[i]).copy()
        clean_target = torch.tensor(X_clean[i], dtype=torch.float32).to(device)

        output = denoise_spectrum(model, noisy_input)
        output_tensor = torch.tensor(output, dtype=torch.float32).to(device)

        loss = mean_squared_error(output_tensor, clean_target).item()
        test_losses.append(loss)

        print(f"Sample {i+1}/{len(X_noisy)} - MSE Loss: {loss:.6f}")

    print(f"Average Test Loss: {np.mean(test_losses):.6f}")
