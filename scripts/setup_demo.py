"""Одна команда: пересобрать демо-данные без сети (синтетический городок) -> facts.db -> сценарии отельеров.

    python scripts/setup_demo.py
Для реальных данных (нужна сеть): scripts/build_demo.py, затем scripts/import_hotels.py (см. README).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_demo_town import build
from import_hotels import import_records
from seed_hotelier_demo import seed
from storage import db


def main():
    out = ROOT / "data" / "demo"
    path = build(out)
    db.reset()
    st = import_records(json.loads(path.read_text(encoding="utf-8")), "Сочи")
    print("Импорт:", st)
    print("Сценарии отельеров:", seed())
    print("Готово:", db.DB_PATH)


if __name__ == "__main__":
    main()
