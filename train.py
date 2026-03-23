import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from main import Transformer

from dados import carregar_dataset
from tokenizer_utils import tokenizar_pares, get_tokenizer

CONFIG = {
    "d_model":    128,
    "n_heads":    4,
    "n_layers":   2,
    "d_ff":       256,
    "dropout":    0.1,
    "max_len":    64,
    "n_amostras": 1000,
    "epochs":     15,
    "batch_size": 32,
    "lr":         1e-3,
}
def treinar():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train.py] Dispositivo: {device}")

    pares = carregar_dataset(CONFIG["n_amostras"])
    src_ids, tgt_in, tgt_out, PAD_ID = tokenizar_pares(
        pares, max_len=CONFIG["max_len"]
    )

    dataset    = TensorDataset(src_ids, tgt_in, tgt_out)
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
    )

    tok    = get_tokenizer()
    vocab_size = tok.vocab_size

    model = Transformer(
        src_vocab_size = vocab_size,
        tgt_vocab_size = vocab_size,
        d_model        = CONFIG["d_model"],
        n_heads        = CONFIG["n_heads"],
        n_layers       = CONFIG["n_layers"],
        d_ff           = CONFIG["d_ff"],
        max_seq_len    = CONFIG["max_len"],
        dropout        = CONFIG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train.py] Parâmetros treináveis: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    print("\n[train.py] Iniciando treinamento...\n")
    historico_loss = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        loss_total = 0.0
        n_batches  = 0

        for src_batch, tgt_in_batch, tgt_out_batch in dataloader:
            src_batch     = src_batch.to(device)
            tgt_in_batch  = tgt_in_batch.to(device)
            tgt_out_batch = tgt_out_batch.to(device)

            logits = model(src_batch, tgt_in_batch)

            logits_flat  = logits.reshape(-1, vocab_size)
            targets_flat = tgt_out_batch.reshape(-1)

            loss = criterion(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            loss_total += loss.item()
            n_batches  += 1

        loss_media = loss_total / n_batches
        historico_loss.append(loss_media)
        print(f"  Época {epoch:02d}/{CONFIG['epochs']} | Loss: {loss_media:.4f}")

    queda = historico_loss[0] - historico_loss[-1]
    print(f"\n[train.py] Loss inicial: {historico_loss[0]:.4f}")
    print(f"[train.py] Loss final  : {historico_loss[-1]:.4f}")
    print(f"[train.py] Queda total : {queda:.4f} ({'✓ convergiu' if queda > 0 else '✗ não convergiu'})")

    torch.save(model.state_dict(), "transformer_lab5.pt")
    print("[train.py] Pesos salvos em transformer_lab5.pt")

    return model, historico_loss


if __name__ == "__main__":
    treinar()