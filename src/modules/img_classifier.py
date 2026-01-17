import torch
import torch.nn as nn
# from analysis_modules import *

class IMG_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = NetworkProcesses.StreamA_EfficientNetNetwork()  # 512
        self.hfri = NetworkProcesses.HFRINetwork()  # 128
        self.hfrfs = NetworkProcesses.HFRFSNetwork()  # 128

        fusion_size = 512 + 128 + 128

        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, images):
        # Input: [Batch, 3, 256, 256]

        visual_features = self.efficientnet(images)
        hfri_features = self.hfri(images)
        hfrfs_features = self.hfrfs(images)

        fused_features = torch.cat([visual_features, hfri_features, hfrfs_features], dim=1)

        logits = self.classifier(fused_features)

        return logits