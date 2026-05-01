import torch
import torch.nn as nn
from torchvision import models

class MultimodalASDModel(nn.Module):
    def __init__(self):
        super(MultimodalASDModel, self).__init__()
        
        # 1. Image Branch: ResNet50
        self.resnet = models.resnet50(weights=None)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity() 
        
        self.image_bottleneck = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 2. Behavioral Branch: MLP (AQ-10-Child scores)
        self.behavioral_mlp = nn.Sequential(
            nn.Linear(10, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 3. Auxiliary Classifiers (The "Unexpected Keys" in your error)
        # These must be present for the weights to load correctly
        self.image_aux = nn.Linear(128, 2)
        self.behave_aux = nn.Linear(64, 2)
        
        # 4. Final Fusion Classifier
        # Your error shows 'fusion_layer.3', meaning it has 4 parts (0,1,2,3)
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 64, 64), # 0
            nn.ReLU(),               # 1
            nn.Dropout(0.4),         # 2
            nn.Linear(64, 2)         # 3
        )

    def forward(self, img, beh):
        img_feats = self.image_bottleneck(self.resnet(img))
        beh_feats = self.behavioral_mlp(beh)
        
        # Auxiliary outputs
        img_out = self.image_aux(img_feats)
        beh_out = self.behave_aux(beh_feats)
        
        # Combined output
        combined = torch.cat((img_feats, beh_feats), dim=1)
        final_out = self.fusion_layer(combined)
        
        return final_out, img_out, beh_out