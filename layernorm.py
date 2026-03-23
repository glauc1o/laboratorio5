import numpy as np

def layer_norm(X, eps=1e-6):

    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)

    X_norm = (X - mean) / np.sqrt(var + eps)

    return X_norm