import numpy as np

def softmax(x):
    x_stable = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_stable)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]

    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores)
    output = np.matmul(weights, V)

    return output