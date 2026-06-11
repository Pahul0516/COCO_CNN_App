import json
import base64
from unittest.mock import patch, MagicMock

from app.messiging.consumer import callback, start_consumer


def _make_body(file_name: str = "test.jpg", data: bytes = b"\x89PNG") -> bytes:
    payload = {"fileName": file_name, "data": base64.b64encode(data).decode()}
    return json.dumps(payload).encode()


def test_callback_calls_predict_and_sends_label():
    mock_ch = MagicMock()
    mock_method = MagicMock()
    mock_method.delivery_tag = "tag1"

    with patch('app.messiging.consumer.coco_detector') as mock_detector, \
         patch('app.messiging.consumer.send_label') as mock_send:

        mock_detector.predict_from_bytes.return_value = [
            {"label": "cat", "score": 0.95, "box": [0, 0, 10, 10]},
            {"label": "dog", "score": 0.91, "box": [5, 5, 15, 15]},
        ]

        callback(mock_ch, mock_method, None, _make_body("photo.jpg"))

        mock_detector.predict_from_bytes.assert_called_once()
        mock_send.assert_called_once_with(
            "coco_response_queue",
            {"fileName": "photo.jpg", "labels": ["cat", "dog"]},
        )


def test_callback_acks_on_success():
    mock_ch = MagicMock()
    mock_method = MagicMock()
    mock_method.delivery_tag = "abc"

    with patch('app.messiging.consumer.coco_detector') as mock_detector, \
         patch('app.messiging.consumer.send_label'):

        mock_detector.predict_from_bytes.return_value = []
        callback(mock_ch, mock_method, None, _make_body())

    mock_ch.basic_ack.assert_called_once_with(delivery_tag="abc")


def test_callback_nacks_on_invalid_json():
    mock_ch = MagicMock()
    mock_method = MagicMock()
    mock_method.delivery_tag = "err"

    callback(mock_ch, mock_method, None, b"not json")

    mock_ch.basic_nack.assert_called_once_with(delivery_tag="err", requeue=False)
    mock_ch.basic_ack.assert_not_called()


def test_callback_nacks_when_predict_raises():
    mock_ch = MagicMock()
    mock_method = MagicMock()
    mock_method.delivery_tag = "tag2"

    with patch('app.messiging.consumer.coco_detector') as mock_detector, \
         patch('app.messiging.consumer.send_label'):

        mock_detector.predict_from_bytes.side_effect = RuntimeError("model error")
        callback(mock_ch, mock_method, None, _make_body())

    mock_ch.basic_nack.assert_called_once_with(delivery_tag="tag2", requeue=False)


def test_start_consumer_declares_exchange_queue_and_binding():
    mock_channel = MagicMock()
    mock_conn = MagicMock()
    mock_conn.channel.return_value = mock_channel
    mock_channel.start_consuming.side_effect = KeyboardInterrupt

    with patch('app.messiging.consumer.create_connection', return_value=mock_conn):
        try:
            start_consumer()
        except KeyboardInterrupt:
            pass

    mock_channel.exchange_declare.assert_called_once_with(
        exchange="photo_exchange",
        exchange_type="fanout",
        durable=True,
    )
    mock_channel.queue_declare.assert_called_once_with(
        queue="coco_service_queue",
        durable=True,
    )
    mock_channel.queue_bind.assert_called_once_with(
        exchange="photo_exchange",
        queue="coco_service_queue",
    )
    mock_channel.basic_consume.assert_called_once()
    mock_channel.start_consuming.assert_called_once()
