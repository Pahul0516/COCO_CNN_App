from unittest.mock import patch, MagicMock

from app.config.rabbitmq_config import create_connection


def test_returns_blocking_connection():
    with patch('app.config.rabbitmq_config.pika') as mock_pika:
        mock_conn = MagicMock()
        mock_pika.BlockingConnection.return_value = mock_conn

        result = create_connection()

        assert result is mock_conn
        mock_pika.BlockingConnection.assert_called_once()


def test_uses_localhost_on_port_5672():
    with patch('app.config.rabbitmq_config.pika') as mock_pika:
        create_connection()

        _, kwargs = mock_pika.ConnectionParameters.call_args
        assert kwargs['host'] == 'localhost'
        assert kwargs['port'] == 5672


def test_uses_guest_credentials():
    with patch('app.config.rabbitmq_config.pika') as mock_pika:
        create_connection()

        mock_pika.PlainCredentials.assert_called_once_with("guest", "guest")


def test_credentials_passed_to_parameters():
    with patch('app.config.rabbitmq_config.pika') as mock_pika:
        mock_creds = MagicMock()
        mock_pika.PlainCredentials.return_value = mock_creds

        create_connection()

        _, kwargs = mock_pika.ConnectionParameters.call_args
        assert kwargs['credentials'] is mock_creds
