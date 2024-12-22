
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class TrashMobileNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_classes=6):
        super(TrashMobileNet, self).__init__()
        self.model = mobilenet_v3_large(weights='DEFAULT')
        for param in self.model.parameters():
            param.requires_grad = False
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, num_classes)
        for param in self.model.classifier[-1].parameters():
            param.requires_grad = True
            
    def forward(self, x):
        x = self.model(x)
        return x