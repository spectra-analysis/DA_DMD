import numpy as np

def standardize_minmax(x):
    h = np.max(x, axis=1, keepdims=True)
    l = np.min(x, axis=1, keepdims=True)
    return (x - l) / (h - l + 1e-8)

def normalize(ref):
    h = max(ref)
    l = min(ref)
    return (ref - l) / (h - l + 1e-8)