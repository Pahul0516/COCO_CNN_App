import torch
from unittest.mock import patch, MagicMock

from app.model.coco.coco_model_loader import CocoDetectionModel


@patch('app.model.coco.coco_model_loader.FasterRCNN_ResNet50_FPN_Weights')
@patch('app.model.coco.coco_model_loader.fasterrcnn_resnet50_fpn')
def test_device_is_cpu(mock_frcnn, mock_weights):
    loader = CocoDetectionModel()
    assert loader.device == torch.device("cpu")


@patch('app.model.coco.coco_model_loader.FasterRCNN_ResNet50_FPN_Weights')
@patch('app.model.coco.coco_model_loader.fasterrcnn_resnet50_fpn')
def test_load_model_called_with_default_weights(mock_frcnn, mock_weights):
    CocoDetectionModel()
    mock_frcnn.assert_called_once_with(weights=mock_weights.DEFAULT)


@patch('app.model.coco.coco_model_loader.FasterRCNN_ResNet50_FPN_Weights')
@patch('app.model.coco.coco_model_loader.fasterrcnn_resnet50_fpn')
def test_model_moved_to_device_and_set_to_eval(mock_frcnn, mock_weights):
    mock_model_instance = MagicMock()
    mock_frcnn.return_value = mock_model_instance

    loader = CocoDetectionModel()

    mock_model_instance.to.assert_called_once_with(loader.device)
    mock_model_instance.eval.assert_called_once()


@patch('app.model.coco.coco_model_loader.FasterRCNN_ResNet50_FPN_Weights')
@patch('app.model.coco.coco_model_loader.fasterrcnn_resnet50_fpn')
def test_categories_come_from_weights_meta(mock_frcnn, mock_weights):
    mock_weights.DEFAULT.meta = {"categories": ["__background__", "person", "cat"]}

    loader = CocoDetectionModel()

    assert loader.categories == ["__background__", "person", "cat"]


@patch('app.model.coco.coco_model_loader.FasterRCNN_ResNet50_FPN_Weights')
@patch('app.model.coco.coco_model_loader.fasterrcnn_resnet50_fpn')
def test_transform_comes_from_weights(mock_frcnn, mock_weights):
    mock_transform = MagicMock()
    mock_weights.DEFAULT.transforms.return_value = mock_transform

    loader = CocoDetectionModel()

    assert loader.transform is mock_transform
