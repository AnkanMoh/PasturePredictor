from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

from .config import CFG

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((CFG.img_size, CFG.img_size)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

valid_transform = T.Compose([
    T.Resize((CFG.img_size, CFG.img_size)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def load_and_pivot(data_dir: Path):
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    train_df["target"] = train_df["target"].fillna(0)
    train_df.loc[train_df["target"] < 0, "target"] = 0

    id_cols = [
        "sample_id", "image_path", "Sampling_Date", "State",
        "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"
    ]

    train_wide = (
        train_df
        .pivot_table(index=id_cols, columns="target_name", values="target")
        .reset_index()
    )

    target_cols = ["Dry_Green_g","Dry_Dead_g","Dry_Clover_g","GDM_g","Dry_Total_g"]

    for col in target_cols:
        train_wide[col] = train_wide[col].fillna(0)
        train_wide.loc[train_wide[col] < 0, col] = 0

    train_wide = train_wide[id_cols + target_cols]
    return train_wide, test_df, target_cols

def add_folds(train_wide: pd.DataFrame, n_folds: int = CFG.n_folds, seed: int = CFG.seed):
    train_wide = train_wide.copy()
    train_wide["fold"] = -1
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed
    )
    for fold, (_, val_idx) in enumerate(skf.split(train_wide, train_wide["State"])):
        train_wide.loc[val_idx, "fold"] = fold
    return train_wide

class BiomassDataset(Dataset):
    def __init__(self, df, data_dir: Path, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train

        if is_train:
            arr = self.df[[
                "Dry_Green_g","Dry_Dead_g","Dry_Clover_g","GDM_g","Dry_Total_g"
            ]].astype(np.float32).values
            arr[arr < 0] = 0
            self.targets = np.log1p(arr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.is_train:
            targets = torch.as_tensor(self.targets[idx], dtype=torch.float32)
            return img, targets
        return img, row["image_path"]

class BiomassTestDataset(Dataset):
    def __init__(self, df, data_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, row["image_path"]
