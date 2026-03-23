import torch
from transformers import AutoTokenizer

_TOKENIZER_NAME = "bert-base-multilingual-cased"
_tokenizer = None
CONFIG_MAX_LEN = 64


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        print(f"[tokenizer_utils.py] Carregando tokenizador: {_TOKENIZER_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
    return _tokenizer


def tokenizar_pares(pares: list[dict], max_len: int = 64):
    tok = get_tokenizer()
    START_ID = tok.cls_token_id   
    EOS_ID   = tok.sep_token_id   
    PAD_ID   = tok.pad_token_id   

    src_lista, tgt_in_lista, tgt_out_lista = [], [], []

    for par in pares:
        src_enc = tok(
            par["src"],
            max_length=max_len,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]

  
        tgt_ids = tok(
            par["tgt"],
            max_length=max_len - 1,   
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        tgt_in_ids  = [START_ID] + tgt_ids  

        tgt_out_ids = tgt_ids + [EOS_ID]

        src_lista.append(    _pad(src_enc,     max_len, PAD_ID))
        tgt_in_lista.append( _pad(tgt_in_ids,  max_len, PAD_ID))
        tgt_out_lista.append(_pad(tgt_out_ids, max_len, PAD_ID))

    src_ids  = torch.tensor(src_lista,     dtype=torch.long)
    tgt_in   = torch.tensor(tgt_in_lista,  dtype=torch.long)
    tgt_out  = torch.tensor(tgt_out_lista, dtype=torch.long)

    print(f"[tokenizer_utils.py] Tensores gerados: src{src_ids.shape}, "
          f"tgt_in{tgt_in.shape}, tgt_out{tgt_out.shape}")
    return src_ids, tgt_in, tgt_out, PAD_ID


def _pad(ids: list, length: int, pad_id: int) -> list:
    ids = ids[:length]
    ids += [pad_id] * (length - len(ids))
    return ids