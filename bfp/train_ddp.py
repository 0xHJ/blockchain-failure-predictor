from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from .data import LogDataConfig, LogDataset
from .model import FailurePredictorModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed training for Blockchain Failure Predictor.")
    p.add_argument("--train-csv", type=str, default="data/train_logs.csv")
    p.add_argument("--valid-csv", type=str, default="data/valid_logs.csv")
    p.add_argument("--sequence-length", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--output-dir", type=str, default="checkpoints")
    p.add_argument("--num-workers", type=int, default=8)
    return p.parse_args()


def setup_ddp(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_dataloaders_ddp(
    train_path: Path,
    valid_path: Path | None,
    sequence_length: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, DataLoader | None, int]:
    train_ds = LogDataset(LogDataConfig(path=train_path, sequence_length=sequence_length))
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = None
    if valid_path is not None and valid_path.exists():
        valid_ds = LogDataset(LogDataConfig(path=valid_path, sequence_length=sequence_length))
        valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False)
        valid_loader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[-1]

    return train_loader, valid_loader, input_dim


def train_one_epoch(
    model: DDP,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            probs = model(x)
            loss = criterion(probs, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        preds = (probs > 0.5).float()
        total_correct += (preds == y).sum().item()
        total_examples += x.size(0)

    loss_tensor = torch.tensor([total_loss, total_examples], device=device)
    acc_tensor = torch.tensor([total_correct, total_examples], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)

    global_loss = loss_tensor[0].item() / loss_tensor[1].item()
    global_acc = acc_tensor[0].item() / acc_tensor[1].item()
    return global_loss, global_acc


@torch.no_grad()
def evaluate(
    model: DDP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            probs = model(x)
            loss = criterion(probs, y)

        total_loss += loss.item() * x.size(0)
        preds = (probs > 0.5).float()
        total_correct += (preds == y).sum().item()
        total_examples += x.size(0)

    loss_tensor = torch.tensor([total_loss, total_examples], device=device)
    acc_tensor = torch.tensor([total_correct, total_examples], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)

    global_loss = loss_tensor[0].item() / loss_tensor[1].item()
    global_acc = acc_tensor[0].item() / acc_tensor[1].item()
    return global_loss, global_acc


def ddp_worker(rank: int, world_size: int, args: argparse.Namespace):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    train_path = Path(args.train_csv)
    valid_path = Path(args.valid_csv) if args.valid_csv else None

    train_loader, valid_loader, input_dim = create_dataloaders_ddp(
        train_path=train_path,
        valid_path=valid_path,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rank=rank,
        world_size=world_size,
    )

    model = FailurePredictorModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    ddp_model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    criterion = nn.BCELoss()
    optimizer = AdamW(
        ddp_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler()

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            ddp_model, train_loader, criterion, optimizer, scaler, device
        )

        if valid_loader is not None:
            val_loss, val_acc = evaluate(ddp_model, valid_loader, criterion, device)
        else:
            val_loss, val_acc = float("nan"), float("nan")

        if rank == 0:
            print(
                f"[Rank 0][Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if valid_loader is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = output_dir / "best_model_ddp.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "input_dim": input_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                    },
                    ckpt_path,
                )
                print(f"[Rank 0] saved best model to {ckpt_path}")

    cleanup_ddp()


def main():
    args = parse_args()
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA device available")

    mp.spawn(ddp_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
