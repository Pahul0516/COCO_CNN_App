from unittest.mock import patch, MagicMock

from app.model.coco.inference_coco import ObjectDetector

CATEGORIES = ["__background__", "person", "cat", "dog"]


def _setup_model_class_mock(mock_model_class):
    mock_model_class.return_value.categories = CATEGORIES


def _make_prediction(label_idx: int, score: float, box: list):
    label = MagicMock()
    label.item.return_value = label_idx
    bx = MagicMock()
    bx.tolist.return_value = box
    return label, score, bx


@patch('app.model.coco.inference_coco.CocoDetectionModel')
@patch('app.model.coco.inference_coco.load_image_from_bytes')
def test_filters_detections_below_threshold(mock_load, mock_model_class):
    _setup_model_class_mock(mock_model_class)
    detector = ObjectDetector(threshold=0.9)

    lbl1, sc1, box1 = _make_prediction(1, 0.95, [0, 0, 10, 10])   # above threshold
    lbl2, sc2, box2 = _make_prediction(2, 0.80, [5, 5, 20, 20])   # below threshold

    detector.model.return_value = [{"boxes": [box1, box2], "scores": [sc1, sc2], "labels": [lbl1, lbl2]}]
    mock_load.return_value = MagicMock()

    results = detector.predict_from_bytes(b"fake")

    assert len(results) == 1
    assert results[0]["label"] == "person"


@patch('app.model.coco.inference_coco.CocoDetectionModel')
@patch('app.model.coco.inference_coco.load_image_from_bytes')
def test_result_has_label_score_and_box(mock_load, mock_model_class):
    _setup_model_class_mock(mock_model_class)
    detector = ObjectDetector(threshold=0.5)

    lbl, sc, box = _make_prediction(1, 0.95, [0.0, 1.0, 100.0, 200.0])
    detector.model.return_value = [{"boxes": [box], "scores": [sc], "labels": [lbl]}]
    mock_load.return_value = MagicMock()

    results = detector.predict_from_bytes(b"img")

    assert results == [{"label": "person", "score": 0.95, "box": [0.0, 1.0, 100.0, 200.0]}]


@patch('app.model.coco.inference_coco.CocoDetectionModel')
@patch('app.model.coco.inference_coco.load_image_from_bytes')
def test_returns_empty_when_all_below_threshold(mock_load, mock_model_class):
    _setup_model_class_mock(mock_model_class)
    detector = ObjectDetector(threshold=0.99)

    lbl, sc, box = _make_prediction(1, 0.50, [0, 0, 10, 10])
    detector.model.return_value = [{"boxes": [box], "scores": [sc], "labels": [lbl]}]
    mock_load.return_value = MagicMock()

    assert detector.predict_from_bytes(b"img") == []


@patch('app.model.coco.inference_coco.CocoDetectionModel')
@patch('app.model.coco.inference_coco.load_image_from_bytes')
def test_returns_empty_when_no_detections(mock_load, mock_model_class):
    _setup_model_class_mock(mock_model_class)
    detector = ObjectDetector(threshold=0.9)
    detector.model.return_value = [{"boxes": [], "scores": [], "labels": []}]
    mock_load.return_value = MagicMock()

    assert detector.predict_from_bytes(b"img") == []


@patch('app.model.coco.inference_coco.CocoDetectionModel')
@patch('app.model.coco.inference_coco.load_image_from_bytes')
def test_loads_image_from_provided_bytes(mock_load, mock_model_class):
    _setup_model_class_mock(mock_model_class)
    detector = ObjectDetector()
    detector.model.return_value = [{"boxes": [], "scores": [], "labels": []}]
    mock_load.return_value = MagicMock()

    detector.predict_from_bytes(b"raw_image_data")

    mock_load.assert_called_once_with(b"raw_image_data")
