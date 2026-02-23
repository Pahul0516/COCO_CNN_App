import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

class CocoDetectionModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = self._load_model()
        self.categories = self._load_categories()
        self.transform = self.weights.transforms()

    def _load_model(self):
        model = fasterrcnn_resnet50_fpn(weights=self.weights)
        model.to(self.device)
        model.eval()
        return model

    def _load_categories(self):
        return self.weights.meta["categories"]