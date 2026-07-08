from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassAccuracy

from DL_Template.data.datamodule import build_dataloaders
from DL_Template.models.model import build_model
from DL_Template.training.trainer import evaluate
from DL_Template.utils import get_device, get_logger

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = get_device()

    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    checkpoint_path = Path(cfg.train.checkpoint_dir) / "best.pt"
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    loss_fn = torch.nn.CrossEntropyLoss()
    metric = MulticlassAccuracy(num_classes=cfg.model.output_dim).to(device)
    test_loss, test_acc = evaluate(model, val_loader, loss_fn, device, metric=metric)
    log.info("test_loss=%.4f test_acc=%.4f", test_loss, test_acc)


if __name__ == "__main__":
    main()
