import argparse
import pandas as pd
import torch
from pathlib import Path
from .model import FailurePredictorModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input-csv", type=str, required=True)
    p.add_argument("--sequence-length", type=int, default=60)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    input_dim = ckpt["input_dim"]

    model = FailurePredictorModel(
        input_dim=input_dim,
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    df = pd.read_csv(args.input_csv)
    feature_cols = [c for c in df.columns if c not in ("timestamp", "label")]
    df = df.tail(args.sequence_length)
    x = df[feature_cols].to_numpy(dtype="float32")
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(x).item()

    print(f"Predicted failure probability: {prob:.4f}")

if __name__ == "__main__":
    main()
