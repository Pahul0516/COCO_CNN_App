import torch
import torchvision.ops as ops
from app.utils.image_utils import load_image_from_bytes
from app.model.grounding_dino.grounding_model_loader import GroundingDinoModel


class GroundingDinoDetector:
    def __init__(
        self,
        box_threshold: float = 0.25,
        text_threshold: float = 0.10,
        chunk_size: int = 50,
        nms_iou: float = 0.5,
    ):
        self.wrapper = GroundingDinoModel()
        self.model = self.wrapper.model
        self.processor = self.wrapper.processor
        self.labels = self.wrapper.labels
        self.device = self.wrapper.device

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.chunk_size = chunk_size
        self.nms_iou = nms_iou

    def _run_chunked(self, image):
        all_boxes = []
        all_scores = []
        all_text_labels = []

        for i in range(0, len(self.labels), self.chunk_size):
            chunk = self.labels[i:i + self.chunk_size]
            prompt = ". ".join(label.lower() for label in chunk) + "."

            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs.input_ids,
                target_sizes=[image.size[::-1]],
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )[0]

            if len(results["boxes"]) == 0:
                continue

            all_boxes.append(results["boxes"])
            all_scores.append(results["scores"])
            all_text_labels.extend(results["text_labels"])

        if not all_boxes:
            return [], [], []

        all_boxes = torch.cat(all_boxes)
        all_scores = torch.cat(all_scores)

        return all_boxes, all_scores, all_text_labels

    def predict_from_bytes(self, image_bytes: bytes):
        image = load_image_from_bytes(image_bytes)

        boxes, scores, text_labels = self._run_chunked(image)

        if len(boxes) == 0:
            return []

        keep = ops.nms(boxes, scores, self.nms_iou)

        boxes = boxes[keep]
        scores = scores[keep]
        text_labels = [text_labels[i] for i in keep]

        results = []

        for box, score, label in zip(boxes, scores, text_labels):
            if score >= self.box_threshold:
                results.append(
                    {
                        "label": label,
                        "score": float(score),
                        "box": box.tolist(),
                    }
                )

        return results