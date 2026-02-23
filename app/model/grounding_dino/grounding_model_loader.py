import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os


class GroundingDinoModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model_id = "IDEA-Research/grounding-dino-base"

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = self._load_model()
        self.labels = self._load_labels()

    def _load_model(self):
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        )
        model.to(self.device)
        model.eval()
        return model

    def _load_labels(self):
        labels_path = os.path.join(
            os.path.dirname(__file__),
            "./../../../resources/lvis_labels.txt",
        )

        with open(labels_path, "r") as f:
            labels = [line.strip() for line in f if line.strip()]

        return labels