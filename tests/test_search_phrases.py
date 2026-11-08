"""Разбор живых формулировок про море/дорогу пешком и ранжирование по ним (без сети и без ключа LLM)."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORK = Path(tempfile.mkdtemp())
os.environ["HOTELS_DB"] = str(WORK / "facts.db")
os.environ.pop("GEMINI_API_KEY", None)
shutil.copy(ROOT / "data" / "demo" / "facts.db", WORK / "facts.db")

import agents.search_agent  # noqa: E402,F401
from agents.offline import plan_offline  # noqa: E402
from agents.orchestrator import run  # noqa: E402
from agents.search_agent import compare_simple, walk_minutes_score  # noqa: E402
from app.actions import make_feature_id  # noqa: E402


def reqs(text):
    return [(r.feature, r.target, r.max_minutes, r.importance) for r in plan_offline(text).requirements]


class PhraseParsing(unittest.TestCase):
    def test_sea_in_walking_distance_variants(self):
        for q in ["Хочу море в пешей доступности", "море в пешеходной доступности", "Хочу море в шаговой доступности",
                  "Нужно море недалеко, чтобы дойти пешком", "хочу жить рядом с морем", "пляж рядом"]:
            self.assertEqual(reqs(q), [("walk_access", "beach", 15, "must")], q)

    def test_explicit_minutes_and_very_close(self):
        self.assertEqual(reqs("до моря 5 минут пешком")[0][:3], ("walk_access", "beach", 5))
        self.assertEqual(reqs("пляж в 10 минутах ходьбы")[0][:3], ("walk_access", "beach", 10))
        self.assertEqual(reqs("хочу пляж в двадцати минутах пешком")[0][:3], ("walk_access", "beach", 20))
        self.assertEqual(reqs("жить у самого моря")[0][:3], ("walk_access", "beach", 5))

    def test_sea_view_is_not_walking(self):
        self.assertEqual([r[0] for r in reqs("номер с видом на море")], ["sea_view"])
        self.assertEqual([r[0] for r in reqs("море из окна")], ["sea_view"])
        self.assertEqual(sorted(r[0] for r in reqs("с видом на закат недалеко от моря")), ["sunset_view", "walk_access"])
        self.assertEqual(sorted(r[0] for r in reqs("вид на море и пляж рядом")), ["sea_view", "walk_access"])

    def test_stops_and_station_need_proximity(self):
        self.assertEqual([(r[0], r[1]) for r in reqs("остановка рядом и вокзал пешком")],
                         [("walk_access", "bus_stop"), ("walk_access", "station")])
        self.assertEqual(reqs("приеду на поезде, хочу тишину"), [("quiet", None, None, "must")])

    def test_no_false_walk_requirement(self):
        self.assertEqual(sorted(r[0] for r in reqs("нужна детская площадка рядом и тишина")), ["kids_infrastructure", "quiet"])
        self.assertEqual([r[0] for r in reqs("нужна аптека рядом ночью")], ["night_amenities"])

    def test_importance(self):
        self.assertEqual(reqs("тихо, а пляж желательно рядом")[1][3], "nice")

    def test_unknown_wish_asks_instead_of_chitchat(self):
        self.assertEqual(plan_offline("привет, как дела").intent, "chitchat")
        self.assertTrue(plan_offline("хочу отдохнуть у бассейна").clarifying_question)


class WalkRanking(unittest.TestCase):
    def test_score_curve(self):
        self.assertEqual(walk_minutes_score(5, 15), 1.0)
        self.assertAlmostEqual(walk_minutes_score(15, 15), 0.6)
        self.assertEqual(walk_minutes_score(30, 15), 0.0)
        self.assertEqual(walk_minutes_score(None, 15), 0.0)

    def test_sea_walk_query_ranks_by_beach_only(self):
        out = run("Хочу море в пешей доступности")
        self.assertIn(out["plan"].intent, "search_hotels")
        ranked = out["results"][0]["output"]["ranked"]
        self.assertEqual([r["hotel_id"] for r in ranked], ["w1", "w5", "w2"])  # 2, 12 и 14 минут до пляжа
        self.assertTrue(all(m["ok"] and m["target"] == "beach" for r in ranked for m in r["matched"]))

    def test_far_hotel_is_a_miss_with_reason(self):
        out = run("до моря 5 минут пешком")
        ranked = out["results"][0]["output"]["ranked"]
        self.assertEqual(ranked[0]["hotel_id"], "w1")
        self.assertTrue(ranked[1]["misses"])  # Ночной город: 12 мин при пороге 5
        self.assertIn("мин пешком", ranked[1]["misses"][0]["summary"])


class CompareAndIds(unittest.TestCase):
    def test_compare_simple_levels_only(self):
        c = compare_simple(["w1", "w3"])
        for row in c["rows"]:
            for cell in row["cells"].values():
                self.assertIn(cell["level"], {"good", "mid", "bad", "unknown"})
        labels = [r["label"] for r in c["rows"]]
        self.assertIn("Пляж в 15 минутах пешком", labels)
        self.assertNotIn("walk_access", labels)
        self.assertEqual(c["rows_total"], len(c["rows"]))

    def test_feature_id_from_russian_title(self):
        self.assertEqual(make_feature_id("Шумоизоляция окон"), "shumoizolyatsiya_okon")
        self.assertRegex(make_feature_id("123"), r"^[a-z][a-z0-9_]{2,40}$")
        self.assertRegex(make_feature_id("!!"), r"^[a-z][a-z0-9_]{2,40}$")


if __name__ == "__main__":
    unittest.main()
