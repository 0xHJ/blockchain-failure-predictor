# scripts/train_tiny.py
import argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
from bfp.data import LogDataConfig, LogDataset
from bfp.model import FailurePredictorModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train_logs.csv")
    ap.add_argument("--sequence-length", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--ckpt", default="weights/best_model_ddp.pt")
    args = ap.parse_args()

    ds = LogDataset(LogDataConfig(path=args.csv, sequence_length=args.sequence_length))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    x0, y0 = next(iter(loader))
    input_dim = x0.shape[-1]

    model = FailurePredictorModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    crit = nn.BCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    for _ in range(args.epochs):
        for x, y in loader:
            opt.zero_grad()
            p = model(x)
            loss = crit(p, y.float())
            loss.backward()
            opt.step()

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }, args.ckpt)
    print(f"Saved checkpoint to {args.ckpt}")

if __name__ == "__main__":
    main()
