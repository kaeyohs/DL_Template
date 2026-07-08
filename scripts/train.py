from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy

from DL_Template.data.datamodule import build_dataloaders
from DL_Template.models.model import build_model
from DL_Template.training.trainer import evaluate, train_one_epoch
from DL_Template.utils import get_device, get_logger, seed_everything

log = get_logger(__name__)


def build_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer):
    name = cfg.train.scheduler.name
    if name in (None, "none"):
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.train.scheduler.step_size,
            gamma=cfg.train.scheduler.gamma,
        )
    raise ValueError(f"Unknown scheduler: {name}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device()
    log.info("Using device: %s", device)

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = build_scheduler(cfg, optimizer)
    loss_fn = torch.nn.CrossEntropyLoss()

    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    train_metric = MulticlassAccuracy(num_classes=cfg.model.output_dim).to(device)
    val_metric = MulticlassAccuracy(num_classes=cfg.model.output_dim).to(device)

    checkpoint_dir = Path(cfg.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.train.log_dir)

    start_epoch = 0
    best_val_loss = float("inf")
    last_ckpt = checkpoint_dir / "last.pt"
    if cfg.train.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        log.info("Resumed from %s at epoch %d", last_ckpt, start_epoch)

    for epoch in range(start_epoch, cfg.train.epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scaler=scaler,
            grad_clip=cfg.train.grad_clip,
            metric=train_metric,
        )
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device, metric=val_metric)
        if scheduler is not None:
            scheduler.step()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        log.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f",
            epoch,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": best_val_loss,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        }
        torch.save(ckpt, last_ckpt)
        if is_best:
            torch.save(ckpt, checkpoint_dir / "best.pt")

    writer.close()


if __name__ == "__main__":
    main()
