import torch.nn as nn
import torchvision.models as models

def get_backbone(name='resnet18', pretrained=True, num_classes=10):
    if name == 'resnet18':
        m = models.resnet18(pretrained=pretrained)
        # replace last fc
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m
    if name == 'mobilenet_v2':
        m = models.mobilenet_v2(pretrained=pretrained)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    # fallback: small custom CNN
    class SmallCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(8*8*64,128), nn.ReLU(), nn.Linear(128,num_classes)
            )
        def forward(self,x): return self.net(x)
    return SmallCNN(num_classes=num_classes)

