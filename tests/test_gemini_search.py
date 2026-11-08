"""Контракт Gemini/сбои API: без сети, на изолированной БД."""
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import httpx
from google.genai import errors
from core.llm_client import StructuredOutputError
from agents import orchestrator as o
from agents.search_agent import rank_hotels
from storage import db
from streamlit.testing.v1 import AppTest


class GeminiSearchTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name)
        p = patch.object(db, 'DB_PATH', self.work/'facts.db');p.start();self.addCleanup(p.stop)
        p = patch.object(o, 'llm_available', return_value=True);p.start();self.addCleanup(p.stop)
        self.llm = patch.object(o, 'ask_gemini_structured');self.parse=self.llm.start();self.addCleanup(self.llm.stop)
        p = patch.object(o, 'ask_gemini', return_value='Объяснение Gemini из результатов БД.');p.start();self.addCleanup(p.stop)
        self.con = db.connect();self.addCleanup(self.con.close)
        for i in range(3):
            hid=f'g{i}'
            db.upsert_hotel(self.con,hid,f'Отель {i}','Сочи',43.4,39.9,3)
            db.add_fact(self.con,hid,'quiet',value={},score=.8,confidence=.8,summary='Тихо',source='geo:test')
            db.add_fact(self.con,hid,'walk_access',value={'beach': {'minutes': 5+10*i,'distance_m': 300+500*i}},
                        score=.7,confidence=.8,summary='Пешая доступность',source='geo:test')

    def plan(self):
        return o.Plan(intent='search_hotels', requirements=[o.Requirement(feature='walk_access', detail='Море пешком',
                      importance='must',target='beach',max_minutes=15)])

    def results(self,out):
        return next(r['output']['ranked'] for r in out['results'] if r['agent']=='search_agent')

    def test_online_primary_and_missing_task_repaired(self):
        self.parse.return_value=self.plan()
        out=o.run('Хочу море в пешей доступности')
        self.assertEqual(out['plan'].planner,'llm')
        self.assertTrue(out['plan'].llm_attempted)
        self.assertTrue(self.results(out))
        self.assertEqual(self.results(out)[0]['hotel_id'],'g0')
        self.parse.assert_called_once()

    def test_unseen_paraphrase_still_uses_gemini(self):
        self.parse.return_value=o.Plan(intent='search_hotels',requirements=[o.Requirement(feature='quiet', detail='Без грохота',importance='must')])
        out=o.run('Хочется засыпать без грохота')
        self.assertEqual(out['plan'].planner,'llm')
        self.assertEqual(self.results(out)[0]['matched'][0]['feature'],'quiet')

    def test_unknown_feature_not_fabricated(self):
        self.parse.return_value=o.Plan(intent='search_hotels',requirements=[o.Requirement(feature='teleport',detail='Телепорт к пляжу',importance='must')])
        out=o.run('Хочу телепорт к пляжу')
        self.assertTrue(self.results(out))
        self.assertTrue(out['ask_user'])
        self.assertTrue(out['default_listing'])
        self.assertEqual(out['plan'].requirements[0].feature,'other')

    def test_supported_and_unknown_together(self):
        plan=self.plan();plan.requirements.append(o.Requirement(feature='other',detail='Ретрит с телепортом',importance='nice'))
        self.parse.return_value=plan
        out=o.run('Море пешком и ретрит с телепортом')
        self.assertFalse(out['default_listing'])
        self.assertEqual(len(out['plan'].requirements),2)
        self.assertTrue(self.results(out))

    def test_empty_plan_shows_catalog_and_question(self):
        self.parse.return_value=o.Plan(intent='search_hotels')
        out=o.run('Хочу необычный ретрит')
        self.assertTrue(self.results(out))
        self.assertTrue(out['ask_user'])
        self.assertTrue(out['default_listing'])

    def test_clarification_does_not_hide_results(self):
        plan=self.plan();plan.clarifying_question='В каком городе?';self.parse.return_value=plan
        out=o.run('Хочу море в пешей доступности')
        self.assertTrue(out['ask_user']);self.assertTrue(self.results(out))

    def test_api_failures_are_explicit_fallbacks(self):
        failures=[(errors.APIError(429,{'error':{'message':'quota'}}),'http_429'),
                  (errors.APIError(503,{'error':{'message':'busy'}}),'http_503'),
                  (httpx.ReadTimeout('timeout'),'timeout'),
                  (StructuredOutputError('empty_response'),'empty_response'),
                  (StructuredOutputError('invalid_json'),'invalid_json'),
                  (StructuredOutputError('invalid_schema'),'invalid_schema')]
        for exc,reason in failures:
            with self.subTest(reason=reason):
                self.parse.side_effect=exc
                out=o.run('Хочу море в пешей доступности')
                self.assertEqual(out['plan'].planner,'offline')
                self.assertEqual(out['plan'].fallback_reason,reason)
                self.assertTrue(out['plan'].notices)
                self.assertTrue(self.results(out))

    def test_unknown_phrase_and_api_error_shows_catalog(self):
        self.parse.side_effect=TimeoutError()
        out=o.run('Эко-ретрит с квантовыми снами')
        self.assertTrue(self.results(out));self.assertTrue(out['ask_user'])

    def test_partial_structured_output_salvaged(self):
        self.parse.side_effect=StructuredOutputError('invalid_schema',payload={
            'intent':'search_hotels','requirements':[{'feature':'walk_access','target':'beach','max_minutes':10,
                'detail':'пляж','importance':'must'}],'tasks':[{'agent':'bad_agent','tools':['poi']}]})
        out=o.run('Пляж в десяти минутах')
        self.assertEqual(out['plan'].planner,'llm')
        self.assertEqual(out['plan'].requirements[0].max_minutes,10)
        self.assertTrue(out['plan'].notices);self.assertTrue(self.results(out))

    def test_empty_and_long_inputs_are_explained_without_api(self):
        for query in ('',' '*3,'a'*4001):
            out=o.run(query)
            self.assertTrue(out['ask_user']);self.assertFalse(out['results'])
        self.parse.assert_not_called()

    def test_explanation_failure_is_visible(self):
        self.parse.return_value=self.plan()
        with patch.object(o,'ask_gemini',return_value=' '):
            out=o.run('Море пешком')
        self.assertTrue(out['answer'])
        self.assertTrue(any(d['status']=='fallback' for d in out['diagnostics']))

    def test_missing_city_is_not_substituted(self):
        plan=self.plan();plan.location='Казань';self.parse.return_value=plan
        out=o.run('Пляж в Казани')
        self.assertFalse(self.results(out));self.assertIn('Казань',out['answer'])

    def test_search_failure_is_explained(self):
        self.parse.return_value=self.plan()
        with patch('agents.search_agent.rank_hotels',side_effect=RuntimeError('db failed')):
            out=o.run('Море пешком')
        self.assertTrue(out['answer']);self.assertEqual(out['diagnostics'][0]['status'],'error')

    def test_missing_target_data_is_gap_not_aggregate_match(self):
        db.delete_facts(self.con,'g0','walk_access')
        db.add_fact(self.con,'g0','walk_access',value={'claims':[]},score=1,confidence=.9,summary='Отельер: все рядом',source='hotelier_human')
        result=rank_hotels(self.plan(),top_k=3,con=self.con)
        hotel=next(r for r in result['ranked'] if r['hotel_id']=='g0')
        self.assertTrue(hotel['gaps']);self.assertFalse(hotel['matched'])

    def test_ui_fallback_warning_and_results(self):
        self.parse.side_effect=errors.APIError(429,{'error':{'message':'quota'}})
        app=AppTest.from_file(str(ROOT/'app/streamlit_app.py')).run(timeout=15)
        app.text_input(key='q').set_value('Хочу море в пешей доступности')
        next(b for b in app.button if b.label=='Найти').click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertTrue(any('429' in w.value for w in app.warning))
        self.assertTrue(app.session_state['out']['results'])

    def test_ui_unknown_and_empty_queries(self):
        self.parse.return_value=o.Plan(intent='search_hotels')
        app=AppTest.from_file(str(ROOT/'app/streamlit_app.py')).run(timeout=15)
        app.text_input(key='q').set_value('Неизвестный ретрит')
        next(b for b in app.button if b.label=='Найти').click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertTrue(any(m.value=='Не оценено' for m in app.metric))
        app.text_input(key='q').set_value('')
        next(b for b in app.button if b.label=='Найти').click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertTrue(any('Введите пожелания' in w.value for w in app.warning))

    def test_no_key_is_visible_and_does_not_call_gemini(self):
        with patch.object(o, 'llm_available', return_value=False):
            out=o.run('Море пешком')
        self.assertEqual(out['plan'].fallback_reason,'missing_key')
        self.assertTrue(out['plan'].notices);self.parse.assert_not_called()

    def test_ui_single_hotel_and_updated_explanation(self):
        self.parse.return_value=self.plan()
        app=AppTest.from_file(str(ROOT/'app/streamlit_app.py')).run(timeout=15)
        app.text_input(key='q').set_value('Море пешком')
        next(b for b in app.button if b.label=='Найти').click().run(timeout=15)
        app.slider[0].set_value(1).run(timeout=15)
        self.assertFalse(app.exception)
        self.assertFalse(any(m.value=='Объяснение Gemini из результатов БД.' for m in app.markdown))
        self.con.execute("DELETE FROM hotels WHERE id IN ('g1','g2')");self.con.commit()
        single=AppTest.from_file(str(ROOT/'app/streamlit_app.py')).run(timeout=15)
        self.assertFalse(single.exception);self.assertEqual(len(single.slider),0)

    def test_internal_exception_text_not_exposed(self):
        self.parse.side_effect=RuntimeError('secret_api_key_1234')
        out=o.run('Хочу море в пешей доступности')
        self.assertNotIn('secret_api_key_1234',str(out))
        self.assertTrue(out['plan'].notices)


if __name__=='__main__':
    unittest.main()
