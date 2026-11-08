"""Единая точка входа в Gemini. Все агенты ходят в модель только через этот файл."""
import os
import json
import time
from typing import TypeVar

from dotenv import load_dotenv
import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
T = TypeVar("T", bound=BaseModel)

_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Клиент создаётся лениво: импорт модуля не падает, если ключа ещё нет."""
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Не найден GEMINI_API_KEY. Создайте .env (см. .env.example).")
        _client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=20_000,
            retry_options=types.HttpRetryOptions(attempts=1)))
    return _client


def _generate(contents: str, config: types.GenerateContentConfig, retries: int = 1):
    for attempt in range(retries + 1):
        try:
            return get_client().models.generate_content(model=MODEL, contents=contents, config=config)
        except (errors.APIError, httpx.TransportError) as exc:
            retryable = not isinstance(exc, errors.APIError) or exc.code == 429 or exc.code >= 500
            if not retryable or attempt == retries:
                raise
            time.sleep(1.5 * (attempt + 1))  # Повторяем только временные ошибки API


def ask_gemini(prompt: str, system: str | None = None, temperature: float = 0.5) -> str:
    """Обычный текстовый ответ."""
    cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
    return _generate(prompt, cfg).text


def ask_gemini_structured(prompt: str, schema: type[T], system: str | None = None,
                          temperature: float = 0.0) -> T:
    """Ответ строго по pydantic-схеме: на выходе уже готовый объект, а не текст."""
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json",
        response_json_schema=schema.model_json_schema(),
    )
    resp = _generate(prompt, cfg)
    body = getattr(resp, 'parsed', None)
    if body is None:
        text = getattr(resp, 'text', None)
        if not isinstance(text, str) or not text.strip():
            raise StructuredOutputError('empty_response')
        try:
            body = json.loads(text)
        except (ValueError, TypeError):
            raise StructuredOutputError('invalid_json') from None
    try:
        return schema.model_validate(body)
    except ValidationError:
        # Оркестратор может восстановить валидные критерии. Другие потребители
        # продолжают получать ошибку; невалидные факты не принимаются автоматически.
        raise StructuredOutputError('invalid_schema', payload=body) from None


class StructuredOutputError(ValueError):
    def __init__(self, kind, payload=None):
        self.kind = kind
        self.payload = payload
        super().__init__(kind)



if __name__ == "__main__":
    print(ask_gemini("Привет! Ты готов искать отели? Ответь одной фразой."))
