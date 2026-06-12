import torch
from unittest.mock import patch, MagicMock


@patch('app.model.grounding_dino.grounding_model_loader.AutoModelForZeroShotObjectDetection')
@patch('app.model.grounding_dino.grounding_model_loader.AutoProcessor')
def test_processor_loaded_with_correct_model_id(mock_proc, mock_model):
    from app.model.grounding_dino.grounding_model_loader import GroundingDinoModel
    GroundingDinoModel()
    mock_proc.from_pretrained.assert_called_once_with("IDEA-Research/grounding-dino-base")


@patch('app.model.grounding_dino.grounding_model_loader.AutoModelForZeroShotObjectDetection')
@patch('app.model.grounding_dino.grounding_model_loader.AutoProcessor')
def test_model_loaded_with_correct_model_id(mock_proc, mock_model):
    from app.model.grounding_dino.grounding_model_loader import GroundingDinoModel
    GroundingDinoModel()
    mock_model.from_pretrained.assert_called_once_with("IDEA-Research/grounding-dino-base")


@patch('app.model.grounding_dino.grounding_model_loader.AutoModelForZeroShotObjectDetection')
@patch('app.model.grounding_dino.grounding_model_loader.AutoProcessor')
def test_model_moved_to_device_and_set_to_eval(mock_proc, mock_model):
    from app.model.grounding_dino.grounding_model_loader import GroundingDinoModel

    mock_instance = MagicMock()
    mock_model.from_pretrained.return_value = mock_instance

    loader = GroundingDinoModel()

    mock_instance.to.assert_called_once_with(loader.device)
    mock_instance.eval.assert_called_once()


@patch('app.model.grounding_dino.grounding_model_loader.AutoModelForZeroShotObjectDetection')
@patch('app.model.grounding_dino.grounding_model_loader.AutoProcessor')
def test_labels_loaded_from_lvis_file(mock_proc, mock_model):
    from app.model.grounding_dino.grounding_model_loader import GroundingDinoModel

    loader = GroundingDinoModel()

    assert isinstance(loader.labels, list)
    assert len(loader.labels) > 0
    assert all(isinstance(lbl, str) for lbl in loader.labels)


@patch('app.model.grounding_dino.grounding_model_loader.AutoModelForZeroShotObjectDetection')
@patch('app.model.grounding_dino.grounding_model_loader.AutoProcessor')
def test_device_is_cpu(mock_proc, mock_model):
    from app.model.grounding_dino.grounding_model_loader import GroundingDinoModel

    loader = GroundingDinoModel()

    assert loader.device == torch.device("cpu")
