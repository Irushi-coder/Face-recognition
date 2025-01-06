import torch
import torch.nn as nn
import torchvision

# Define YOLOv8 Backbone
class YOLOv8Backbone(nn.Module):
    def __init__(self):
        super(YOLOv8Backbone, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

    def forward(self, x):
        return self.backbone(x)

# Define YOLOv8 Neck
class YOLOv8Neck(nn.Module):
    def __init__(self):
        super(YOLOv8Neck, self).__init__()
        self.neck = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.neck(x)

# Define YOLOv8 Head
class YOLOv8Head(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv8Head, self).__init__()
        self.cls_head = nn.Conv2d(512, num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(512, 4, kernel_size=1)

    def forward(self, x):
        cls_pred = self.cls_head(x)
        reg_pred = self.reg_head(x)
        return cls_pred, reg_pred

# Define YOLOv8 Model
class YOLOv8(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv8, self).__init__()
        self.backbone = YOLOv8Backbone()
        self.neck = YOLOv8Neck()
        self.head = YOLOv8Head(num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        cls_pred, reg_pred = self.head(x)
        return cls_pred, reg_pred

# Instantiate Model
num_classes = 80
model = YOLOv8(num_classes)

# Example input
x = torch.randn(1, 3, 640, 640)
cls_pred, reg_pred = model(x)
print(f'Class Prediction Shape: {cls_pred.shape}')
print(f'Regression Prediction Shape: {reg_pred.shape}')
