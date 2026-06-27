#!/usr/bin/env python3
"""
Training script for cv-nets.

Usage
-----
    cvnets-train --config config/train.yaml
    cvnets-train --config config/train.yaml --epochs 50 --lr 0.001 --amp
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cvnets.core.registry import MODEL_REGISTRY
from cvnets.data import build_dataloader
from cvnets.data.datasets import CIFAR10, ImageFolderDataset
from cvnets.loss_fn import build_loss_fn
from cvnets.models.factory import ModelFactory
from cvnets.optim import build_optimizer
from cvnets.scheduler import build_scheduler
from cvnets.trainer import Trainer
from cvnets.utils import (
    double_dash_line,
    info,
    print_header,
    safe_load_yaml,
    set_seed,
)
from cvnets.utils.logger import LoggerError, error


def _build_model(cfg: Dict[str, Any], device: str) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "")
    import cvnets.models.zoo  # noqa: F401
    model = None
    if model_name and MODEL_REGISTRY.contains(model_name):
        info(f"Building model from MODEL_REGISTRY: {model_name!r}")
        zoo_kwargs = model_cfg.get("zoo_kwargs", {})
        model = MODEL_REGISTRY.build(model_name, **zoo_kwargs)
    else:
        info("Building model via ModelFactory from config layers")
        model = ModelFactory.build(cfg)
    if model is None:
        error("Could not build model.")
    model = model.to(torch.device(device))
    info(f"Model moved to {device}")
    return model


def _build_dataset(cfg: Dict[str, Any]) -> Dataset:
    data_cfg = cfg.get("data", {})
    dataset_type = data_cfg.get("dataset", "cifar10").lower()
    root = data_cfg.get("root", "./data")
    train_split = data_cfg.get("train", True)
    transform_cfg = data_cfg.get("transform", None)
    download = data_cfg.get("download", False)
    if dataset_type == "cifar10":
        info(f"Building CIFAR10 dataset (root={root}, train={train_split})")
        return CIFAR10(root=root, train=train_split, download=download, transform=transform_cfg)
    elif dataset_type == "image_folder":
        info(f"Building ImageFolderDataset (root={root})")
        return ImageFolderDataset(root=root, transform=transform_cfg)
    else:
        error(f"Unknown dataset type {dataset_type!r}. Supported: 'cifar10', 'image_folder'.")


def _build_criterion(cfg: Dict[str, Any]) -> nn.Module:
    train_cfg = cfg.get("train", {})
    loss_type = train_cfg.get("loss", "cross_entropy")
    loss_kwargs = train_cfg.get("loss_kwargs", {})
    try:
        criterion = build_loss_fn(loss_type, **loss_kwargs)
        info(f"Built loss function: {loss_type}")
        return criterion
    except ValueError:
        info(f"Loss function {loss_type!r} not in registry; falling back to nn.CrossEntropyLoss()")
        return nn.CrossEntropyLoss()


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="cv-nets training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--amp", action="store_true", default=None, help="Enable AMP training")
    parser.add_argument("--workers", type=int, default=None, help="Number of DataLoader workers")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    return parser.parse_args(argv)


def _merge_args_with_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if "epochs" not in train_cfg:
        train_cfg["epochs"] = 10
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if "lr" not in train_cfg:
        train_cfg["lr"] = 0.001
    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size
    if "batch_size" not in data_cfg:
        data_cfg["batch_size"] = 64
    if args.workers is not None:
        data_cfg["workers"] = args.workers
    if "workers" not in data_cfg:
        data_cfg["workers"] = 0
    if args.device is not None:
        train_cfg["device"] = args.device
    if "device" not in train_cfg:
        train_cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if args.amp is not None:
        train_cfg["amp"] = args.amp
    if "amp" not in train_cfg:
        train_cfg["amp"] = False
    if args.checkpoint_dir is not None:
        train_cfg["checkpoint_dir"] = args.checkpoint_dir
    if "checkpoint_dir" not in train_cfg:
        train_cfg["checkpoint_dir"] = "./checkpoints"
    if args.resume is not None:
        train_cfg["resume"] = args.resume
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if "seed" not in train_cfg:
        train_cfg["seed"] = 42
    cfg["train"] = train_cfg
    cfg["data"] = data_cfg
    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        print_header("cv-nets Training Script")
        cfg = safe_load_yaml(args.config)
        cfg = _merge_args_with_config(args, cfg)
        train_cfg = cfg.get("train", {})
        seed = train_cfg.get("seed", 42)
        set_seed(seed)
        device = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        info(f"Using device: {device}")
        double_dash_line()
        model = _build_model(cfg, device)
        info(f"Model created: {model.__class__.__name__}")
        double_dash_line()
        dataset = _build_dataset(cfg)
        batch_size = cfg.get("data", {}).get("batch_size", 64)
        num_workers = cfg.get("data", {}).get("workers", 0)
        train_loader = build_dataloader(
            dataset=dataset, batch_size=batch_size,
            shuffle=cfg.get("data", {}).get("shuffle", True),
            num_workers=num_workers,
            pin_memory=cfg.get("data", {}).get("pin_memory", device != "cpu"),
            drop_last=cfg.get("data", {}).get("drop_last", False),
        )
        double_dash_line()
        criterion = _build_criterion(cfg)
        double_dash_line()
        optim_type = train_cfg.get("optimizer", "adamw")
        lr = train_cfg.get("lr", 0.001)
        weight_decay = train_cfg.get("weight_decay", 0.0)
        optimizer = build_optimizer(model.parameters(), optim_type=optim_type, lr=lr, weight_decay=weight_decay, verbose=True)
        double_dash_line()
        scheduler = None
        sched_type = train_cfg.get("scheduler", None)
        if sched_type:
            scheduler_kwargs = train_cfg.get("scheduler_kwargs", {})
            scheduler = build_scheduler(optimizer=optimizer, sched_type=sched_type, **scheduler_kwargs)
            info(f"Scheduler: {sched_type}")
        double_dash_line()
        use_amp = train_cfg.get("amp", False)
        epochs = train_cfg.get("epochs", 10)
        clip_grad_norm = train_cfg.get("clip_grad_norm", 0.0)
        grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
        trainer = Trainer(
            model=model, train_loader=train_loader, optimizer=optimizer,
            criterion=criterion, val_loader=None, num_epochs=epochs,
            device=device, use_amp=use_amp, scheduler=scheduler,
            clip_grad_norm=clip_grad_norm, grad_accum_steps=grad_accum_steps,
        )
        info("Trainer initialized. Starting training...")
        double_dash_line()
        metrics = trainer.fit()
        print_header("Training Complete")
        info(f"Final metrics: {metrics}")
        double_dash_line()
        return 0
    except (Exception, LoggerError) as exc:
        try:
            error(f"Training failed: {exc}")
        except LoggerError:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
