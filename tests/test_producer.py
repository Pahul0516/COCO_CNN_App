import json
from unittest.mock import patch, MagicMock, call

from app.messiging.producer import send_label


def _make_channel_and_connection():
    mock_channel = MagicMock()
    mock_conn = MagicMock()
    mock_conn.channel.return_value = mock_channel
    return mock_conn, mock_channel


def test_declares_queue_as_durable():
    mock_conn, mock_channel = _make_channel_and_connection()
    with patch('app.messiging.producer.create_connection', return_value=mock_conn):
        send_label("test_queue", {"labels": ["cat"]})

    mock_channel.queue_declare.assert_called_once_with(queue="test_queue", durable=True)


def test_publishes_json_body_to_correct_queue():
    mock_conn, mock_channel = _make_channel_and_connection()
    message = {"fileName": "img.jpg", "labels": ["dog"]}

    with patch('app.messiging.producer.create_connection', return_value=mock_conn):
        send_label("my_queue", message)

    _, kwargs = mock_channel.basic_publish.call_args
    assert kwargs['exchange'] == ""
    assert kwargs['routing_key'] == "my_queue"
    assert json.loads(kwargs['body'].decode()) == message


def test_closes_connection_after_publish():
    mock_conn, _ = _make_channel_and_connection()
    with patch('app.messiging.producer.create_connection', return_value=mock_conn):
        send_label("q", {})

    mock_conn.close.assert_called_once()


def test_publishes_empty_message():
    mock_conn, mock_channel = _make_channel_and_connection()
    with patch('app.messiging.producer.create_connection', return_value=mock_conn):
        send_label("q", {})

    _, kwargs = mock_channel.basic_publish.call_args
    assert json.loads(kwargs['body'].decode()) == {}
