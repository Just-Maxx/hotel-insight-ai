"""Интеграция новых модулей без сети и без ключа LLM: офлайн-планировщик, резолвер, мост отельера,
расширяемость реестра, актуализация, сравнение, метрики и смоук интерфейса."""
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORK = Path(tempfile.mkdtemp())
os.environ["HOTELS_DB"] = str(WORK / "facts.db")
os.environ.pop("GEMINI_API_KEY", None)
shutil.copy(ROOT / "data" / "demo" / "facts.db", WORK / "facts.db")

import agents.hotelier_agent  # noqa: E402,F401
import agents.search_agent  # noqa: E402,F401
from agents.hotelier_agent import clarify, extract_fact  # noqa: E402
from agents.offline import plan_offline  # noqa: E402
from agents.orchestrator import run  # noqa: E402
from agents.search_agent import compare_hotels, rank_hotels  # noqa: E402
from core.registry import add_feature, load_registry  # noqa: E402
from simulator.hotelier_sim import HotelierSimulator  # noqa: E402
from storage import db, refresh  # noqa: E402
from storage.hotelier_bridge import confirm_fact, dispute_fact, store_session  # noqa: E402
from storage.resolver import resolve  # noqa: E402


class OfflineSearch(unittest.TestCase):
    def test_plan_features_and_importance(self):
        p = plan_offline("Хочу тихий номер, желательно с видом на закат, недалеко от моря в Сочи")
        got = {r.feature: r.importance for r in p.requirements}
        self.assertEqual(got, {"quiet": "must", "sunset_view": "nice", "walk_access": "must"})
        self.assertEqual(p.location, "Сочи")

    def test_near_sea_is_not_sea_view(self):
        p = plan_offline("недалеко от моря")
        self.assertEqual([r.feature for r in p.requirements], ["walk_access"])

    def test_chitchat_and_clarification(self):
        self.assertEqual(plan_offline("привет, как дела").intent, "chitchat")
        self.assertTrue(plan_offline("Хочу отель").clarifying_question)

    def test_run_end_to_end(self):
        out = run("Нужна круглосуточная аптека рядом ночью")
        self.assertEqual(out["plan"].planner, "offline")
        self.assertEqual(out["results"][0]["output"]["ranked"][0]["hotel_id"], "w5")
        self.assertIn("Ночной город", out["answer"])

    def test_pets_gap_goes_to_hotelier(self):
        plan = plan_offline("Еду с собакой")
        res = rank_hotels(plan, top_k=8)
        by = {r["hotel_id"]: r for r in res["ranked"]}
        self.assertTrue(by["w4"]["gaps"] or by["w4"]["matched"])
        self.assertEqual(by["w3"]["misses"][0]["feature"], "pets_allowed")  # w3: с животными нельзя


class Resolver(unittest.TestCase):
    def test_conflict_detected_and_flagged(self):
        r = resolve(db.get_facts(db.connect(), "w3"))["sea_view"]
        self.assertTrue(r["conflict"] and r["needs_review"] and r["resolution"])
        self.assertLess(r["confidence"], r["fact"]["confidence"])

    def test_agreement_raises_confidence(self):
        r = resolve(db.get_facts(db.connect(), "w1"))["sea_view"]
        self.assertTrue(r["agreement"])
        self.assertGreater(r["confidence"], 0.9)

    def test_confirm_and_dispute_validation(self):
        con = db.connect()
        with self.assertRaises(ValueError):
            dispute_fact(con, "w5", "quiet", "может быть", "x")
        with self.assertRaises(ValueError):
            dispute_fact(con, "w5", "quiet", "да", "  ")
        with self.assertRaises(ValueError):
            confirm_fact(con, "w5", "pets_allowed_unknown_feature")


class Hotelier(unittest.TestCase):
    def test_store_session_replaces_old_answer(self):
        con = db.connect()
        for _ in range(2):
            s = clarify("w6", ["pets_allowed"], HotelierSimulator("w6"), source="hotelier_simulator", use_llm=False)
            rep = store_session(con, s)
        self.assertTrue(rep[0]["saved"])
        n = sum(1 for f in db.get_facts(con, "w6") if f["feature"] == "pets_allowed")
        self.assertEqual(n, 1)

    def test_unknown_answer_not_saved(self):
        con = db.connect()
        s = clarify("w8", ["quiet"], lambda q: "Не знаю.", source="hotelier_human", use_llm=False, max_rounds=1)
        self.assertFalse(store_session(con, s)[0]["saved"])


class Extensibility(unittest.TestCase):
    def test_add_feature_and_boolean_extraction(self):
        reg = WORK / "registry.yaml"
        shutil.copy(load_registry.__globals__["REGISTRY_PATH"], reg)
        add_feature("soundproof_windows", "Шумоизоляция окон", "Двойные стеклопакеты", "Есть ли шумоизоляция?", path=reg)
        self.assertIn("soundproof_windows", load_registry(reg))
        with self.assertRaises(ValueError):
            add_feature("soundproof_windows", "x", "y", "z", path=reg)  # дубль
        with self.assertRaises(ValueError):
            add_feature("Bad Id", "x", "y", "z", path=reg)

    def test_boolean_feature_offline(self):
        reg = load_registry.__globals__["REGISTRY_PATH"]
        backup = reg.read_text(encoding="utf-8")
        try:
            add_feature("balcony_all", "Балконы", "Балкон в каждом номере", "Есть ли балкон в каждом номере?")
            self.assertEqual(extract_fact("balcony_all", "Да, в каждом", use_llm=False).suitability, 1.0)
            self.assertEqual(extract_fact("balcony_all", "Нет, только в люксах", use_llm=False).suitability, 0.0)
            self.assertEqual(extract_fact("balcony_all", "Зависит от номера", use_llm=False).status, "unknown")
        finally:
            reg.write_text(backup, encoding="utf-8")


class FreshnessAndCompare(unittest.TestCase):
    def test_stale_detected_and_hotelier_queue(self):
        con = db.connect()
        old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        con.execute("UPDATE facts SET fetched_at=? WHERE hotel_id='w1' AND feature='pets_allowed'", (old,))
        con.commit()
        rep = refresh.queue_refresh(con)
        feats = {(i["hotel_id"], i["feature"]): i["action"] for i in rep["stale"]}
        self.assertEqual(feats[("w1", "pets_allowed")], "ask_hotelier")
        self.assertEqual(feats[("w7", "night_amenities")], "recompute")
        self.assertTrue(any(q["hotel_id"] == "w1" and q["feature"] == "pets_allowed" for q in db.list_questions(con)))

    def test_compare(self):
        cmp = compare_hotels(["w1", "w3"], ["sea_view", "kids_infrastructure"])
        rows = {r["feature"]: r for r in cmp["rows"]}
        self.assertEqual(rows["sea_view"]["best"], ["w1"])
        self.assertEqual(rows["kids_infrastructure"]["best"], ["w3"])
        with self.assertRaises(ValueError):
            compare_hotels(["w1", "nope"])


class EvalAndApp(unittest.TestCase):
    def test_eval_runs(self):
        sys.path.insert(0, str(ROOT / "eval"))
        from run_eval import evaluate
        r = evaluate(db.connect())
        self.assertGreaterEqual(r["overall_accuracy"], 0.8)
        self.assertEqual(r["search"]["hit@3"], 1.0)

    def test_app_smoke(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
        self.assertFalse(at.exception)
        at.text_input(key="q").set_value("Хочу тихий номер с видом на закат недалеко от моря в Сочи").run()
        [b for b in at.button if b.label == "Найти"][0].click().run()
        self.assertFalse(at.exception)


if __name__ == "__main__":
    unittest.main()
