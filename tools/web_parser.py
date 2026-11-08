"""Извлечение текста публичной HTML-страницы. JS не исполняется, формы не отправляются."""
import ipaddress
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import requests

MAX_BYTES = 1_000_000


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript', 'template'):
            self.hidden += 1
        if tag in ('p', 'div', 'br', 'li', 'h1', 'h2', 'h3', 'tr') and not self.hidden:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript', 'template'):
            self.hidden = max(0, self.hidden - 1)
        if tag in ('p', 'div', 'li', 'tr') and not self.hidden:
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def html_to_text(html):
    if len(html.encode('utf-8')) > MAX_BYTES:
        raise ValueError('HTML слишком большой')
    parser = _Text()
    parser.feed(html)
    return '\n'.join(' '.join(line.split()) for line in ''.join(parser.parts).splitlines()
                     if line.strip())


def _public_url(url):
    parsed = urlsplit(url)
    if parsed.scheme not in ('https', 'http') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Нужен публичный HTTP(S) URL без логина и пароля')
    if parsed.port not in (None, 80, 443):
        raise ValueError('Нестандартный порт запрещён')
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80),
                                   type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError('Локальные и служебные адреса запрещены')


def fetch_page(url):
    """Только явно переданный URL, до трёх редиректов и 1 МБ; не сканер сайтов.

    DNS проверяется до запроса; для недоверенного многопользовательского сервиса
    нужна отдельная сетевая изоляция против DNS rebinding.
    """
    with requests.Session() as session:
        for _ in range(4):
            _public_url(url)
            with session.get(url, timeout=(5, 15), allow_redirects=False, stream=True,
                             headers={'User-Agent': 'HotelInsightAI/1.0'}) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    if not response.headers.get('Location'):
                        raise ValueError('Редирект без Location')
                    url = urljoin(url, response.headers['Location'])
                    continue
                response.raise_for_status()
                if 'text/html' not in response.headers.get('Content-Type', '').lower():
                    raise ValueError('Поддерживаются только HTML-страницы')
                content = bytearray()
                for block in response.iter_content(16384):
                    content.extend(block)
                    if len(content) > MAX_BYTES:
                        raise ValueError('Страница больше 1 МБ')
                encoding = response.encoding
                if not encoding or encoding.lower() == 'iso-8859-1':
                    encoding = 'utf-8'
                text = html_to_text(content.decode(encoding, errors='replace'))
                if not text:
                    raise ValueError('Нет текста: возможно, страница требует JavaScript')
                return {'url': url, 'text': text}
    raise ValueError('Слишком много редиректов')
