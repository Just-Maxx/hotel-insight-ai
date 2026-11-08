"""SQLite: отели и их характеристики. У каждого значения — источник, уверенность, дата."""
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("HOTELS_DB", Path(__file__).resolve().parents[1] / "data" / "demo" / "facts.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS hotels (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, city TEXT, lat REAL, lon REAL, floors INTEGER
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hotel_id TEXT NOT NULL REFERENCES hotels(id),
    feature TEXT NOT NULL,
    value_json TEXT,          -- «сырое» значение (по фасадам/этажам/месяцам)
    score REAL,               -- нормированная пригодность 0..1 — по ней ранжируем
    confidence REAL NOT NULL, -- 0..1
    summary TEXT,             -- человекочитаемо, для объяснений
    source TEXT NOT NULL,     -- откуда: 'geo:visibility', 'hotelier', ...
    fetched_at TEXT NOT NULL, -- ISO-дата
    ttl_days INTEGER DEFAULT 180
);
CREATE INDEX IF NOT EXISTS idx_facts_hotel ON facts(hotel_id, feature);
CREATE TABLE IF NOT EXISTS questions (   -- очередь вопросов отельерам (кабинет отельера)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hotel_id TEXT NOT NULL REFERENCES hotels(id),
    feature TEXT NOT NULL,
    question TEXT NOT NULL,
    reason TEXT,              -- почему спрашиваем: 'нет данных', 'данные устарели', 'противоречие'
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | answered
    created_at TEXT NOT NULL,
    answered_at TEXT,
    answer_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_q_hotel ON questions(hotel_id, status);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def reset() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()


def upsert_hotel(con, hotel_id: str, name: str, city: str, lat: float, lon: float, floors: int) -> None:
    con.execute("INSERT OR REPLACE INTO hotels VALUES (?,?,?,?,?,?)", (hotel_id, name, city, lat, lon, floors))
    con.commit()


def add_fact(con, hotel_id: str, feature: str, *, value, score: float | None, confidence: float,
             summary: str, source: str, ttl_days: int = 180, fetched_at: str | None = None) -> None:
    con.execute(
        "INSERT INTO facts (hotel_id, feature, value_json, score, confidence, summary, source, fetched_at, ttl_days)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (hotel_id, feature, json.dumps(value, ensure_ascii=False, default=str), score, confidence,
         summary, source, fetched_at or datetime.now(timezone.utc).isoformat(), ttl_days))
    con.commit()


def list_hotels(con) -> list[dict]:
    return [dict(r) for r in con.execute("SELECT * FROM hotels ORDER BY name")]


def get_facts(con, hotel_id: str) -> list[dict]:
    return [dict(r) for r in con.execute("SELECT * FROM facts WHERE hotel_id=?", (hotel_id,))]


def get_hotel(con, hotel_id: str) -> dict | None:
    r = con.execute("SELECT * FROM hotels WHERE id=?", (hotel_id,)).fetchone()
    return dict(r) if r else None


def delete_facts(con, hotel_id: str, feature: str, source: str | None = None) -> int:
    """Удаляет факты (все или одного источника) — чтобы повторный ответ отельера заменял прежний."""
    if source is None:
        cur = con.execute("DELETE FROM facts WHERE hotel_id=? AND feature=?", (hotel_id, feature))
    else:
        cur = con.execute("DELETE FROM facts WHERE hotel_id=? AND feature=? AND source=?",
                          (hotel_id, feature, source))
    con.commit()
    return cur.rowcount


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_question(con, hotel_id: str, feature: str, question: str, reason: str) -> int | None:
    """Ставит вопрос в очередь; дубль незакрытого вопроса по (отель, характеристика) не создаём."""
    dup = con.execute("SELECT id FROM questions WHERE hotel_id=? AND feature=? AND status='pending'",
                      (hotel_id, feature)).fetchone()
    if dup:
        return None
    cur = con.execute("INSERT INTO questions (hotel_id, feature, question, reason, created_at) VALUES (?,?,?,?,?)",
                      (hotel_id, feature, question, reason, _now()))
    con.commit()
    return cur.lastrowid


def list_questions(con, status: str | None = "pending", hotel_id: str | None = None) -> list[dict]:
    sql, args = "SELECT * FROM questions WHERE 1=1", []
    if status:
        sql += " AND status=?"; args.append(status)
    if hotel_id:
        sql += " AND hotel_id=?"; args.append(hotel_id)
    return [dict(r) for r in con.execute(sql + " ORDER BY id", args)]


def close_question(con, question_id: int, answer_text: str) -> None:
    con.execute("UPDATE questions SET status='answered', answered_at=?, answer_text=? WHERE id=?",
                (_now(), answer_text, question_id))
    con.commit()
