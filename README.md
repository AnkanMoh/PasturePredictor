# PasturePredictor 

PasturePredictor is a deep learning pipeline for the **CSIRO Biomass Prediction** task (Kaggle-style), where the goal is to predict five biomass components from pasture images:

- Dry_Green_g
- Dry_Dead_g
- Dry_Clover_g
- GDM_g
- Dry_Total_g

The project uses a **ConvNeXt-based** multi-target regression model with:

- Image augmentations
- log1p / expm1 target transform
- K-Fold cross-validation
- Warmup + cosine learning rate schedule
- Early stopping and best-epoch saving
- Optional Test-Time Augmentation (TTA)

> ⚠️ **Important for Kaggle offline competitions**  
> Kaggle often disables internet, and newer `timm` versions try to download pretrained weights from Hugging Face.  
> In that case, set `USE_PRETRAINED = False` in `config.py` (or pass `--no-pretrained` to `train.py`) so the model does **not** try to download weights.

---

## Dataset Layout

Expected directory structure (local):

```text
data/
└── csiro-biomass/
    ├── train.csv
    ├── test.csv
    ├── sample_submission.csv
    ├── train/
    │   ├── IDxxxxxxx.jpg
    │   └── ...
    └── test/
        ├── IDyyyyyyy.jpg
        └── ...
