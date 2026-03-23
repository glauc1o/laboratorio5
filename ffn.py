import numpy as np

def ffn(X, d_ff=128):
    d_model = X.shape[-1]

    W1 = np.random.randn(d_model, d_ff)
    W2 = np.random.randn(d_ff, d_model)

    return np.maximum(0, X @ W1) @ W2