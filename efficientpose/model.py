from pathlib import Path

import timm
import torch
import torch.nn as nn


class PoseRegressionNet(nn.Module):
    def __init__(
        self,
        backbone_name="efficientnet_b2",
        pretrained=True,
        local_weight_path="pretrained/efficientnet_b2_ra-bcdf34b7.pth",
    ):
        super().__init__()

        local_weight_path = Path(local_weight_path)

        if pretrained and local_weight_path.exists():
            self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
            state_dict = torch.load(local_weight_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=False)
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)

        self.feature_dim = self.backbone.num_features

        self.common_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
        )
        self.rot_head = nn.Linear(512, 6)
        self.trans_head = nn.Linear(512, 3)

    def forward(self, x):
        feat = self.backbone.forward_features(x)
        common = self.common_head(feat)
        return self.rot_head(common), self.trans_head(common)
