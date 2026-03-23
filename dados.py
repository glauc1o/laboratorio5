from datasets import load_dataset


def carregar_dataset(tamanho: int = 1000):
    
    try:
        ds = load_dataset("bentrevett/multi30k", split="train")
        pares = [
            {"src": ex["en"], "tgt": ex["de"]}
            for ex in ds.select(range(min(tamanho, len(ds))))
        ]
    except Exception:
        ds = load_dataset(
            "Helsinki-NLP/opus_books",
            "en-de",
            split="train",
        )
        pares = [
            {"src": ex["translation"]["en"], "tgt": ex["translation"]["de"]}
            for ex in ds.select(range(min(tamanho, len(ds))))
        ]

    print(f"[dados.py] {len(pares)} pares carregados.")
    print(f"  Exemplo src: {pares[0]['src']}")
    print(f"  Exemplo tgt: {pares[0]['tgt']}")
    return pares


if __name__ == "__main__":
    pares = carregar_dataset(1000)