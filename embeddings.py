import numpy as np

def create_embedding_table(vocab_size, d_model=64):
    return np.random.randn(vocab_size, d_model)

def get_embeddings(ids, embedding_table):
    return embedding_table[ids]
