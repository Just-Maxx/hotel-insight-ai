"""Реестр характеристик (features/registry.yaml): чтение и добавление новых без правки кода.

Новая характеристика «от отельера» = id + название + описание + вопрос. После add_feature():
  - оркестратор видит её в промпте (разбор запроса пользователя);
  - hotelier_agent задаёт про неё вопрос и раскладывает свободный ответ в поле;
  - поиск ранжирует по ней, как только в БД есть значение со score.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "features" / "registry.yaml"
ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
ANSWER_TYPES = ("boolean", "text")


def load_registry(path: Path | None = None) -> dict:
    return yaml.safe_load((path or REGISTRY_PATH).read_text(encoding="utf-8"))["features"]


def add_feature(feature_id: str, title: str, description: str, question: str, *,
                answer_type: str = "boolean", scale: str | None = None, ttl_days: int = 365,
                path: Path | None = None) -> dict:
    """Дописывает блок в registry.yaml (комментарии файла сохраняются). Возвращает добавленный блок."""
    path = path or REGISTRY_PATH
    if not ID_RE.match(feature_id or ""):
        raise ValueError("id: латиница в нижнем регистре, цифры и _, 3-41 символ, начинается с буквы")
    if feature_id in load_registry(path):
        raise ValueError(f"Характеристика «{feature_id}» уже есть")
    if answer_type not in ANSWER_TYPES:
        raise ValueError(f"answer_type: {' | '.join(ANSWER_TYPES)}")
    for name, val in (("title", title), ("description", description), ("question", question)):
        if not isinstance(val, str) or not val.strip() or len(val) > 500:
            raise ValueError(f"{name}: непустая строка до 500 символов")
    if not 1 <= int(ttl_days) <= 3650:
        raise ValueError("ttl_days: 1..3650")
    block = {"title": title.strip(), "description": description.strip(), "kind": "hotelier",
             "question": question.strip(), "answer_type": answer_type, "ttl_days": int(ttl_days)}
    if scale:
        block["scale"] = scale.strip()
    q = lambda v: json.dumps(v, ensure_ascii=False)  # JSON-строка — валидный YAML-скаляр
    lines = [f"  {feature_id}:"] + [f"    {k}: {q(v) if isinstance(v, str) else v}" for k, v in block.items()]
    text = path.read_text(encoding="utf-8")
    path.write_text(text.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return block
