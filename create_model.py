import torch
import torch.nn as nn
from torchvision import models

def create_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight = models.MobileNet_V2_Weights.DEFAULT
    print(device)
    model = models.mobilenet_v2(weight).to(device)
    # Freeze parameters
    for param in model.features.parameters():
        param.requires_grad = False
    # Change class size
    model.classifier = torch.nn.Sequential(
                                           torch.nn.Dropout(p=0.2, inplace=False),
                                           torch.nn.Linear(in_features=1280, out_features=2, bias=True)
                                            )
    return model
