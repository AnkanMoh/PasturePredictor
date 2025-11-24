import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .config import CFG
from .utils import set_seed, ensure_dir
from .data import load_and_pivot, add_folds, BiomassDataset, train_transform, valid_transform
from .model import BiomassModel
from .engine import train_one_epoch, valid_one_epoch, get_scheduler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(CFG.data_dir))
    parser.add_argument("--output-dir", type=str, default=str(CFG.output_dir))
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--folds", type=int, default=CFG.n_folds)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    use_pretrained = not args.no_pretrained

    set_seed(CFG.seed)

    train_wide, _, _ = load_and_pivot(data_dir)
    train_wide = add_folds(train_wide, n_folds=args.folds, seed=CFG.seed)

    all_fold_history = []
    fold_scores = []

    for fold in range(args.folds):
        print(f"\n========== FOLD {fold} ==========")

        train_df_fold = train_wide[train_wide["fold"] != fold].reset_index(drop=True)
        valid_df_fold = train_wide[train_wide["fold"] == fold].reset_index(drop=True)

        train_dataset = BiomassDataset(train_df_fold, data_dir, train_transform, is_train=True)
        valid_dataset = BiomassDataset(valid_df_fold, data_dir, valid_transform, is_train=True)

        train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, shuffle=True,
                                  num_workers=CFG.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, shuffle=False,
                                  num_workers=CFG.num_workers)

        model = BiomassModel(pretrained=use_pretrained).to(CFG.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

        num_training_steps = len(train_loader) * args.epochs
        num_warmup_steps = int(CFG.warmup_ratio * num_training_steps)
        scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)

        best_loss = np.inf
        best_path = output_dir / f"model_fold{fold}.pth"
        no_improve = 0
        global_step = 0

        fold_history = {"train_loss": [], "valid_loss": []}

        for epoch in range(args.epochs):
            print(f"Fold {fold} | Epoch {epoch+1}/{args.epochs}")
            train_loss, global_step = train_one_epoch(
                model, train_loader, optimizer, scheduler, CFG.device, global_step
            )
            valid_loss = valid_one_epoch(model, valid_loader, CFG.device)

            fold_history["train_loss"].append(train_loss)
            fold_history["valid_loss"].append(valid_loss)

            print(f"Epoch {epoch+1} train_loss={train_loss:.5f} | valid_loss={valid_loss:.5f}")

            if valid_loss < best_loss - 1e-6:
                best_loss = valid_loss
                no_improve = 0
                torch.save(model.state_dict(), best_path)
                print(f"  -> New best at epoch {epoch+1}, saved.")
            else:
                no_improve += 1
                print(f"  -> No improvement for {no_improve} epoch(s).")

            if no_improve >= CFG.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        fold_scores.append(best_loss)
        all_fold_history.append(fold_history)

    print("\nFold scores:", fold_scores)
    print("CV mean:", np.mean(fold_scores), "CV std:", np.std(fold_scores))

    plt.figure(figsize=(8, 5))
    for fold, hist in enumerate(all_fold_history):
        epochs_range = range(1, len(hist["valid_loss"]) + 1)
        plt.plot(epochs_range, hist["valid_loss"], label=f"Fold {fold}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss (log-space MSE)")
    plt.title("CV per epoch (per fold)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "cv_per_epoch.png")

if __name__ == "__main__":
    main()
