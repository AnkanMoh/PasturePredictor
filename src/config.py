from pathlib import Path
import torch

class CFG:
    model_name = "convnext_small"
    img_size = 448
    n_targets = 5

    n_folds = 5
    epochs = 20
    train_bs = 16
    valid_bs = 32

    lr = 1e-4
    weight_decay = 1e-2

    warmup_ratio = 0.05
    patience = 5

    num_workers = 0
    seed = 42

    use_pretrained = True   # set False on Kaggle offline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path("data/csiro-biomass")
    output_dir = Path("outputs")
