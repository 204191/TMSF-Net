import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from cache_dataset import Dataset
from losses import build_loss
from model import Model
from utils import resolve_config


def build_loaders(cfg):
    train_set = Dataset(cache_dir=cfg["data"]["cache_dir"], mode="train")
    val_set = Dataset(cache_dir=cfg["data"]["cache_dir"], mode="val")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["val_batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
    )
    return train_loader, val_loader


def move_batch(batch, device):
    inputs, target = batch
    inputs = {k: v.to(device) for k, v in inputs.items()}
    target = target.to(device)
    return inputs, target


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        inputs, target = move_batch(batch, device)

        optimizer.zero_grad()
        output = model(**inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        inputs, target = move_batch(batch, device)
        output = model(**inputs)
        total_loss += criterion(output, target).item()

    return total_loss / len(loader)


def run(cfg):
    device = torch.device(cfg["training"]["device"])

    train_loader, val_loader = build_loaders(cfg)

    model = Model(**cfg["model"]).to(device)
    criterion = build_loss(cfg["loss"]).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    os.makedirs(cfg["output"]["save_dir"], exist_ok=True)

    best_val = float("inf")

    for epoch in range(cfg["training"]["epochs"]):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(cfg["output"]["save_dir"], cfg["output"]["model_name"]),
            )

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(resolve_config(args.config))


if __name__ == "__main__":
    main()
