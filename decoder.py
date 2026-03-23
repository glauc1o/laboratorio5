import numpy as np
from attention import scaled_dot_product_attention, softmax
from layernorm import layer_norm
from ffn import ffn

def create_causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return np.where(mask == 1, -np.inf, 0)


def decoder_block(Y, Z):
    d_model = Y.shape[-1]

    WQ = np.random.randn(d_model, d_model)
    WK = np.random.randn(d_model, d_model)
    WV = np.random.randn(d_model, d_model)

    Q = Y @ WQ
    K = Y @ WK
    V = Y @ WV

    mask = create_causal_mask(Y.shape[1])
    mask = np.expand_dims(mask, axis=0)

    attn1 = scaled_dot_product_attention(Q, K, V, mask)
    Y = layer_norm(Y + attn1)

    WQ2 = np.random.randn(d_model, d_model)
    WK2 = np.random.randn(d_model, d_model)
    WV2 = np.random.randn(d_model, d_model)

    Q = Y @ WQ2
    K = Z @ WK2
    V = Z @ WV2

    attn2 = scaled_dot_product_attention(Q, K, V)
    Y = layer_norm(Y + attn2)

    ffn_out = ffn(Y)
    Y = layer_norm(Y + ffn_out)

    return Y


def decoder(Y, Z, vocab_size, num_layers=2):
    for _ in range(num_layers):
        Y = decoder_block(Y, Z)

    d_model = Y.shape[-1]
    W_out = np.random.randn(d_model, vocab_size)

    logits = Y @ W_out

    return softmax(logits)