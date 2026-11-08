import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.web_parser import html_to_text, fetch_page, _public_url


class WebTests(unittest.TestCase):
    def test_html_text(self):
        text = html_to_text('<style>hidden</style><script>ignore instructions</script><p>С собакой &amp; кошкой</p><p>1000 ₽</p>')
        self.assertEqual(text, 'С собакой & кошкой\n1000 ₽')

    def test_local_urls_rejected(self):
        with patch('socket.getaddrinfo', return_value=[(2, 1, 6, '', ('127.0.0.1', 80))]):
            for url in ('http://localhost', 'http://127.0.0.1', 'file:///etc/passwd', 'https://u:p@example.com', 'http://example.com:8080'):
                with self.assertRaises(ValueError):
                    _public_url(url)

    def test_redirect_rechecked_before_request(self):
        response = Mock(status_code=302, headers={'Location': 'http://127.0.0.1'})
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        session = Mock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)
        session.get.return_value = response
        with patch('tools.web_parser._public_url', side_effect=[None, ValueError('private')]), patch('requests.Session', return_value=session):
            with self.assertRaises(ValueError):
                fetch_page('https://example.com')
        self.assertEqual(session.get.call_count, 1)

    def test_oversized_html_rejected(self):
        with self.assertRaises(ValueError):
            html_to_text('x' * 1000001)


if __name__ == '__main__':
    unittest.main()
