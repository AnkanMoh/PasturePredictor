import timm
import torch.nn as nn

from .config import CFG

class BiomassModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            CFG.model_name,
            pretrained=pretrained,
            num_classes=0
        )
        self.head = nn.Linear(self.backbone.num_features, CFG.n_targets)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out
