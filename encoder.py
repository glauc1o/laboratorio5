import numpy as np
from attention import scaled_dot_product_attention
from layernorm import layer_norm
from ffn import ffn

def encoder_block(X, WQ, WK, WV):
    Q = X @ WQ
    K = X @ WK
    V = X @ WV

    Z = scaled_dot_product_attention(Q, K, V)

    X = layer_norm(X + Z)

    ffn_out = ffn(X)
    X = layer_norm(X + ffn_out)

    return X


def encoder(X, num_layers=2):
    d_model = X.shape[-1]

    for _ in range(num_layers):
        WQ = np.random.randn(d_model, d_model)
        WK = np.random.randn(d_model, d_model)
        WV = np.random.randn(d_model, d_model)

        X = encoder_block(X, WQ, WK, WV)

    return X