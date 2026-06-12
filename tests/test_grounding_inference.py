import pytest
import torch
from unittest.mock import patch, MagicMock


def _make_detector(**kwargs):
    with patch('app.model.grounding_dino.inference_grounding.GroundingDinoModel') as mock_cls:
        mock_wrapper = MagicMock()
        mock_cls.return_value = mock_wrapper
        mock_wrapper.labels = ["cat", "dog", "person"]
        mock_wrapper.device = torch.device("cpu")
        from app.model.grounding_dino.inference_grounding import GroundingDinoDetector
        return GroundingDinoDetector(**kwargs)


@patch('app.model.grounding_dino.inference_grounding.load_image_from_bytes')
def test_predict_returns_empty_when_no_detections(mock_load):
    detector = _make_detector()
    mock_load.return_value = MagicMock()

    with patch.object(detector, '_run_chunked', return_value=([], [], [])):
        results = detector.predict_from_bytes(b"fake")

    assert results == []


@patch('app.model.grounding_dino.inference_grounding.load_image_from_bytes')
def test_predict_returns_results_above_threshold(mock_load):
    detector = _make_detector(box_threshold=0.25)
    mock_load.return_value = MagicMock()

    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    scores = torch.tensor([0.8])

    with patch.object(detector, '_run_chunked', return_value=(boxes, scores, ["cat"])):
        results = detector.predict_from_bytes(b"fake")

    assert len(results) == 1
    assert results[0]["label"] == "cat"
    assert results[0]["score"] == pytest.approx(0.8, abs=1e-4)
    assert "box" in results[0]


@patch('app.model.grounding_dino.inference_grounding.load_image_from_bytes')
def test_predict_filters_detections_below_threshold(mock_load):
    detector = _make_detector(box_threshold=0.5)
    mock_load.return_value = MagicMock()

    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    scores = torch.tensor([0.3])  # below box_threshold of 0.5

    with patch.object(detector, '_run_chunked', return_value=(boxes, scores, ["cat"])):
        results = detector.predict_from_bytes(b"fake")

    assert results == []


@patch('app.model.grounding_dino.inference_grounding.load_image_from_bytes')
def test_predict_applies_nms_to_overlapping_boxes(mock_load):
    detector = _make_detector(box_threshold=0.1, nms_iou=0.5)
    mock_load.return_value = MagicMock()

    # Two heavily overlapping boxes — NMS should suppress the lower-scored one
    boxes = torch.tensor([
        [0.0, 0.0, 100.0, 100.0],
        [1.0, 1.0, 99.0, 99.0],
    ])
    scores = torch.tensor([0.9, 0.8])

    with patch.object(detector, '_run_chunked', return_value=(boxes, scores, ["cat", "cat"])):
        results = detector.predict_from_bytes(b"fake")

    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(0.9, abs=1e-4)


@patch('app.model.grounding_dino.inference_grounding.load_image_from_bytes')
def test_predict_loads_image_from_bytes(mock_load):
    detector = _make_detector()
    mock_load.return_value = MagicMock()

    with patch.object(detector, '_run_chunked', return_value=([], [], [])):
        detector.predict_from_bytes(b"image_data")

    mock_load.assert_called_once_with(b"image_data")


@patch('app.model.grounding_dino.inference_grounding.load_image_from_bytes')
def test_predict_result_has_correct_structure(mock_load):
    detector = _make_detector(box_threshold=0.1)
    mock_load.return_value = MagicMock()

    boxes = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
    scores = torch.tensor([0.75])

    with patch.object(detector, '_run_chunked', return_value=(boxes, scores, ["dog"])):
        results = detector.predict_from_bytes(b"fake")

    assert len(results) == 1
    result = results[0]
    assert result["label"] == "dog"
    assert isinstance(result["score"], float)
    assert isinstance(result["box"], list)
    assert len(result["box"]) == 4
