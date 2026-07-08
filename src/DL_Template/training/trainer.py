import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float | None = None,
    metric: Metric | None = None,
) -> tuple[float, float | None]:
    """Run one training epoch. Returns (mean loss, metric value or None).

    When ``scaler`` is provided, forward/backward run under autocast mixed
    precision. ``grad_clip`` applies max-norm gradient clipping (after unscaling
    when AMP is active). ``metric`` is a torchmetrics metric updated per batch.
    """
    model.train()
    if metric is not None:
        metric.reset()

    use_amp = scaler is not None
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        if metric is not None:
            metric.update(outputs, targets)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_metric = metric.compute().item() if metric is not None else None
    return epoch_loss, epoch_metric


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    metric: Metric | None = None,
) -> tuple[float, float | None]:
    """Evaluate the model. Returns (mean loss, metric value or None)."""
    model.eval()
    if metric is not None:
        metric.reset()

    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        if metric is not None:
            metric.update(outputs, targets)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_metric = metric.compute().item() if metric is not None else None
    return epoch_loss, epoch_metric
