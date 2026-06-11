from unittest.mock import patch

# consumer.py instantiates ObjectDetector at module level, which would load
# the real FasterRCNN model. Patch before any test file is collected.
_frcnn_patch = patch('app.model.coco.coco_model_loader.fasterrcnn_resnet50_fpn')
_weights_patch = patch('app.model.coco.coco_model_loader.FasterRCNN_ResNet50_FPN_Weights')

_frcnn_patch.start()
_weights_patch.start()
