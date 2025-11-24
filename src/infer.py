import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import CFG
from .utils import ensure_dir
from .data import load_and_pivot, BiomassTestDataset, valid_transform
from .model import BiomassModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(CFG.data_dir))
    parser.add_argument("--output-dir", type=str, default=str(CFG.output_dir))
    parser.add_argument("--folds", type=int, default=CFG.n_folds)
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    _, test_df, _ = load_and_pivot(data_dir)

    use_pretrained = not args.no_pretrained

    tta_transforms = [valid_transform]
    if args.use_tta:
        from .data import IMAGENET_MEAN, IMAGENET_STD
        import torchvision.transforms as T
        tta_transforms = [
            valid_transform,
            T.Compose([
                T.Resize((CFG.img_size, CFG.img_size)),
                T.RandomHorizontalFlip(p=1.0),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            T.Compose([
                T.Resize((CFG.img_size, CFG.img_size)),
                T.RandomVerticalFlip(p=1.0),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
        ]

    test_images = test_df["image_path"].unique()
    test_img_df = pd.DataFrame({"image_path": test_images})

    test_dataset = BiomassTestDataset(test_img_df, data_dir, valid_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=CFG.num_workers)

    image_preds = {}

    for img, img_path in tqdm(test_loader, desc="Test inference"):
        img_path = img_path[0]
        preds_tta = []

        for t in tta_transforms:
            from PIL import Image
            img_file = Image.open(data_dir / img_path).convert("RGB")
            t_img = t(img_file).unsqueeze(0).to(CFG.device)

            fold_preds = []
            for fold in range(args.folds):
                model = BiomassModel(pretrained=use_pretrained).to(CFG.device)
                weight_path = output_dir / f"model_fold{fold}.pth"
                state_dict = torch.load(weight_path, map_location=CFG.device)
                model.load_state_dict(state_dict)
                model.eval()
                with torch.inference_mode():
                    out = model(t_img)
                    fold_preds.append(out.squeeze(0).cpu().numpy())

            fold_preds = np.stack(fold_preds, axis=0).mean(axis=0)
            preds_tta.append(fold_preds)

        preds_tta = np.stack(preds_tta, axis=0).mean(axis=0)
        preds_lin = np.expm1(preds_tta)
        image_preds[img_path] = preds_lin

    target_idx = {
        "Dry_Green_g": 0,
        "Dry_Dead_g": 1,
        "Dry_Clover_g": 2,
        "GDM_g": 3,
        "Dry_Total_g": 4,
    }

    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
    preds = []
    for _, row in test_df.iterrows():
        preds.append(image_preds[row["image_path"]][target_idx[row["target_name"]]])

    sample_sub["target"] = preds
    sample_sub.to_csv(output_dir / "submission.csv", index=False)
    print("Saved:", output_dir / "submission.csv")

if __name__ == "__main__":
    main()
