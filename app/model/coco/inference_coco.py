import torch
from app.utils.image_utils import load_image_from_bytes
from app.model.coco.coco_model_loader import CocoDetectionModel


class ObjectDetector:
    def __init__(self, threshold: float = 0.9):
        self.model_wrapper = CocoDetectionModel()
        self.model = self.model_wrapper.model
        self.categories = self.model_wrapper.categories
        self.transform = self.model_wrapper.transform
        self.device = self.model_wrapper.device
        self.threshold = threshold

    def predict_from_bytes(self, image_bytes: bytes):
        image = load_image_from_bytes(image_bytes)
        image_tensor = self.transform(image)

        with torch.no_grad():
            predictions = self.model([image_tensor])[0]

        results = []

        for box, score, label in zip(
            predictions["boxes"],
            predictions["scores"],
            predictions["labels"],
        ):
            if score >= self.threshold:
                class_name = self.categories[label.item()]
                results.append(
                    {
                        "label": class_name,
                        "score": float(score),
                        "box": box.tolist(),
                    }
                )

        return results