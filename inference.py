import torch
from transformers import AutoTokenizer
from main import Transformer          
from dados import carregar_dataset
from tokenizer_utils import tokenizar_pares, get_tokenizer, _pad, CONFIG_MAX_LEN

CONFIG = {
    "d_model":  128,
    "n_heads":  4,
    "n_layers": 2,
    "d_ff":     256,
    "dropout":  0.0,   
    "max_len":  64,
}

MAX_GEN = 50          


def traduzir(frase_src: str, model, tok, device, max_len: int = MAX_GEN) -> str:
    model.eval()
    START_ID = tok.cls_token_id   
    EOS_ID   = tok.sep_token_id   
    PAD_ID   = tok.pad_token_id   
    src_ids = tok(
        frase_src,
        max_length=CONFIG["max_len"],
        truncation=True,
        add_special_tokens=False,
    )["input_ids"]
    src_ids = _pad(src_ids, CONFIG["max_len"], PAD_ID)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    tokens_gerados = [START_ID]

    with torch.no_grad():
        for _ in range(max_len):
            tgt_seq = _pad(tokens_gerados, CONFIG["max_len"], PAD_ID)
            tgt_tensor = torch.tensor([tgt_seq], dtype=torch.long).to(device)

            logits = model(src_tensor, tgt_tensor)
            pos = min(len(tokens_gerados) - 1, CONFIG["max_len"] - 1)
            next_token_logits = logits[0, pos, :]         # [vocab_size]
            next_token_id     = next_token_logits.argmax(-1).item()

            if next_token_id == EOS_ID:
                break

            tokens_gerados.append(next_token_id)

    tokens_saida = tokens_gerados[1:]
    traducao = tok.decode(tokens_saida, skip_special_tokens=True)
    return traducao


def prova_de_fogo(pesos_path: str = "transformer_lab5.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok    = get_tokenizer()

    model = Transformer(
        src_vocab_size = tok.vocab_size,
        tgt_vocab_size = tok.vocab_size,
        d_model        = CONFIG["d_model"],
        n_heads        = CONFIG["n_heads"],
        n_layers       = CONFIG["n_layers"],
        d_ff           = CONFIG["d_ff"],
        max_seq_len    = CONFIG["max_len"],
        dropout        = CONFIG["dropout"],
    ).to(device)

    model.load_state_dict(torch.load(pesos_path, map_location=device))
    print(f"[inference.py] Pesos carregados de '{pesos_path}'")

    pares = carregar_dataset(1000)
    frases_teste = pares[:5]

    print("\n" + "=" * 60)
    print("PROVA DE FOGO — O modelo deve reproduzir frases do treino")
    print("=" * 60)

    for i, par in enumerate(frases_teste):
        src        = par["src"]
        esperado   = par["tgt"]
        gerado     = traduzir(src, model, tok, device)

        print(f"\n[{i+1}] Entrada (EN) : {src}")
        print(f"     Esperado (DE): {esperado}")
        print(f"     Gerado   (DE): {gerado}")
        match = gerado.strip().lower() == esperado.strip().lower()
        print(f"     Match exato  : {'✓ SIM' if match else '≈ aproximado (normal)'}")

    print("\n[inference.py] Se as saídas são parecidas com o esperado → gradientes OK!")


if __name__ == "__main__":
    prova_de_fogo()