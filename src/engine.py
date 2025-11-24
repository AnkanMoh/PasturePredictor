import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .config import CFG

def train_one_epoch(model, loader, optimizer, scheduler, device, global_step):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = nn.MSELoss()(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        global_step += 1

        total_loss += loss.item() * images.size(0)
        pbar.set_description(f"Train loss {loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, global_step

def valid_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(loader, leave=False)
    with torch.inference_mode():
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = nn.MSELoss()(outputs, targets)
            total_loss += loss.item() * images.size(0)
            pbar.set_description(f"Valid loss {loss.item():.4f}")
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
