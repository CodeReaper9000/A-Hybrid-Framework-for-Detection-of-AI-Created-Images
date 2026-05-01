import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import clip


class FusionModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # ResNet
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

        # CLIP
        self.clip_model, _ = clip.load("ViT-B/32")

        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Projection layers
        self.resnet_proj = nn.Linear(2048, 512)
        self.clip_proj = nn.Linear(512, 512)

        # Fusion head
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, resnet_img, clip_img):
        r_feat = self.resnet(resnet_img)
        r_feat = self.resnet_proj(r_feat)

        c_feat = self.clip_model.encode_image(clip_img).float()

        # 🔥 Clamp to prevent explosion
        c_feat = torch.clamp(c_feat, -10, 10)

        # 🔥 Safe normalize
        c_norm = c_feat.norm(dim=-1, keepdim=True) + 1e-6
        c_feat = c_feat / c_norm

        c_feat = self.clip_proj(c_feat)

        fused = torch.cat((r_feat, c_feat), dim=1)

        return self.fc(fused)