from app.model.coco.inference_coco import ObjectDetector
from app.model.grounding_dino.inference_grounding import GroundingDinoDetector

# load image as bytes
with open("C:\\Users\\berin\\PycharmProjects\\COCO_CNN_App\\test2.jpeg", "rb") as f:
    image_bytes = f.read()

# -----------------------------
# COCO detection
# -----------------------------
coco_detector = ObjectDetector(threshold=0.9)
coco_results = coco_detector.predict_from_bytes(image_bytes)

print("\nCOCO objects:")
for r in coco_results:
    print(r)

# -----------------------------
# Grounding DINO detection
# -----------------------------
gd_detector = GroundingDinoDetector()
gd_results = gd_detector.predict_from_bytes(image_bytes)

print("\nGrounding DINO objects:")
for r in gd_results:
    print(r)