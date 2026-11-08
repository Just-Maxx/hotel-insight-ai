"""Оркестратор: Gemini разбирает запрос и раздаёт инструкции агентам."""
import json
import logging
import re

import httpx
import os
from pathlib import Path
from typing import Callable, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from core.labels import DEFAULT_WALK_MINUTES
from core.llm_client import ask_gemini, ask_gemini_structured, StructuredOutputError
from core.prompts import EXPLAIN_SYSTEM, ORCHESTRATOR_SYSTEM

from core.registry import REGISTRY_PATH, load_registry  # noqa: F401  (реэкспорт для старых импортов)

# --------------------------------------------------------------------------
# Схема плана — это «контракт» между мозгом и агентами
# --------------------------------------------------------------------------
class Requirement(BaseModel):
    feature: str = Field(description="id из списка доступных характеристик, либо 'other'")
    detail: str = Field(description="Уточнение своими словами, напр. 'вид на закат зимой'")
    importance: Literal["must", "nice"]
    target: Optional[Literal["beach", "bus_stop", "station"]] = Field(
        default=None, description="Только для walk_access: до чего считать дорогу пешком — пляж/море (beach), "
                                  "остановка (bus_stop), ж/д станция (station). Иначе null.")
    max_minutes: Optional[int] = Field(
        default=None, ge=1, le=120, description="Только вместе с target: максимум минут пешком. "
                                  "«Пешая доступность» без числа = 15, «у самого моря» = 5, иначе — число из запроса.")


class Task(BaseModel):
    hotel_id: str | None = None
    hotelier_reply: str | None = None
    agent: Literal["search_agent", "hotelier_agent"]
    instruction: str = Field(description="Самодостаточная инструкция для агента")
    tools: list[Literal["visibility", "noise", "walking"]] = Field(default_factory=list)


class GeminiPlan(BaseModel):
    intent: Literal["search_hotels", "ask_hotelier", "chitchat"]
    location: Optional[str] = None
    requirements: list[Requirement] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    clarifying_question: Optional[str] = None


class Plan(GeminiPlan):
    """Статус работы присваивает код, а не Gemini."""
    planner: Literal["llm", "offline"] = "llm"  # заполняет код, а не модель
    notices: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None
    llm_attempted: bool = False


# --------------------------------------------------------------------------
# Реестр агентов: имя -> функция(instruction, plan) -> результат
# --------------------------------------------------------------------------
AgentFn = Callable[[Task, Plan], dict]
AGENTS: dict[str, AgentFn] = {}


def register(name: str):
    def deco(fn: AgentFn) -> AgentFn:
        AGENTS[name] = fn
        return fn
    return deco


def llm_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


LOGGER = logging.getLogger(__name__)
MAX_QUERY_LENGTH = 4000


def failure_reason(exc):
    """Только категория/код, без ключей, URL прокси, исходного текста и traceback."""
    if isinstance(exc, StructuredOutputError):
        return exc.kind
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return 'timeout'
    code = getattr(exc, 'code', None)
    if isinstance(code, int):
        return f'http_{code}'
    return 'invalid_schema' if isinstance(exc, ValueError) else 'service_error'


def failure_message(reason):
    return {
        'missing_key': 'Ключ Gemini не настроен.',
        'offline_selected': 'Выбран резервный режим без Gemini.',
        'timeout': 'Gemini не ответил вовремя.',
        'http_429': 'Gemini ограничил запросы или исчерпана квота API (429).',
        'http_400': 'Gemini отклонил запрос или схему ответа (400).',
        'http_401': 'Gemini не принял API-ключ (401).',
        'http_403': 'Нет доступа к Gemini (403).',
        'http_404': 'Указанная модель Gemini недоступна (404).',
        'http_503': 'Gemini временно недоступен (503).',
        'empty_response': 'Gemini вернул пустой ответ.',
        'invalid_json': 'Gemini вернул невалидный JSON.',
        'invalid_schema': 'Ответ Gemini не соответствует схеме.',
    }.get(reason, 'Не удалось получить корректный ответ Gemini.')


def _fallback(user_text, reason, attempted):
    from agents.offline import plan_offline
    LOGGER.warning('Search planning fallback: %s', reason)
    try:
        plan = plan_offline(user_text)
    except Exception:
        plan = Plan(intent='search_hotels', clarifying_question='Уточните город и пожелания к отелю.')
    # Неизвестное слово не должно ошибочно считаться обычной беседой.
    if plan.intent == 'chitchat' and not re.fullmatch(r'(?i)\s*(привет|здравствуй(?:те)?|спасибо|hello|hi)[!.,?\s]*', user_text):
        plan.intent = 'search_hotels'
        plan.clarifying_question = 'Не удалось уверенно понять пожелание. Что важно в отеле?'
    plan.planner = 'offline'
    plan.llm_attempted = attempted
    plan.fallback_reason = reason
    plan.notices = [failure_message(reason) + ' Использован резервный разбор; проверьте понятые условия.']
    return plan


def _repair_payload(payload):
    """Восстановление частичного structured output, без угадывания смысла параметров."""
    if not isinstance(payload, dict) or not isinstance(payload.get('requirements'), list):
        return None
    requirements = []
    for raw in payload['requirements'][:30]:
        if not isinstance(raw, dict) or not isinstance(raw.get('feature'), str) or not raw['feature'].strip():
            continue
        target = raw.get('target') if raw.get('target') in ('beach', 'bus_stop', 'station') else None
        minutes = raw.get('max_minutes')
        if type(minutes) is not int or not 1 <= minutes <= 120:
            minutes = None
        requirements.append(Requirement(feature=raw['feature'].strip(),
            detail=raw.get('detail') if isinstance(raw.get('detail'), str) else raw['feature'],
            importance=raw.get('importance') if raw.get('importance') in ('must', 'nice') else 'nice',
            target=target, max_minutes=minutes))
    if not requirements:
        return None
    return Plan(intent=payload.get('intent') if payload.get('intent') in ('search_hotels', 'ask_hotelier') else 'search_hotels',
        location=payload.get('location') if isinstance(payload.get('location'), str) else None,
        clarifying_question=payload.get('clarifying_question') if isinstance(payload.get('clarifying_question'), str) else None,
        requirements=requirements,
        notices=['Часть ответа Gemini восстановлена. Проверьте понятые условия запроса.'])


def _normalize_plan(plan):
    registry = load_registry()
    unique = {}
    for r in plan.requirements:
        if r.feature not in registry:
            r.feature = 'other'
        if r.feature != 'walk_access':
            r.target = r.max_minutes = None
        elif r.target and not (r.max_minutes and 1 <= r.max_minutes <= 120):
            r.max_minutes = DEFAULT_WALK_MINUTES
        elif not r.target:
            r.max_minutes = None
        key = (r.feature, r.target, r.detail if r.feature == 'other' else '')
        unique[key] = r
    plan.requirements = list(unique.values())
    # Нельзя доверять агентам/ответам, выдуманным моделью. Параметры остаются от Gemini,
    # но исполнение поиска гарантирует код, даже если LLM забыла создать задачу.
    hotelier_tasks = [task for task in plan.tasks if task.agent == 'hotelier_agent']
    for task in hotelier_tasks:
        task.hotel_id = task.hotelier_reply = None
    if plan.intent != 'chitchat':
        plan.tasks = [Task(agent='search_agent', instruction='Подбери отели по структурированным требованиям плана.')] + hotelier_tasks
        if not any(r.feature != 'other' for r in plan.requirements):
            plan.clarifying_question = plan.clarifying_question or 'Какие характеристики отеля для вас важны?'
            plan.notices.append('Нет поддерживаемых критериев: ниже общий список, соответствие пожеланию не подтверждено.')
    return plan


def analyze(user_text: str, use_llm: bool | None = None) -> Plan:
    """Gemini — основной парсер; деградация всегда явно отмечена в плане."""
    if not isinstance(user_text, str) or not user_text.strip():
        return Plan(intent='search_hotels', planner='offline', clarifying_question='Введите пожелания к отелю.')
    if len(user_text) > MAX_QUERY_LENGTH:
        return Plan(intent='search_hotels', planner='offline', clarifying_question='Сократите запрос до 4000 символов.')
    user_text = user_text.strip()
    selected = llm_available() if use_llm is None else use_llm
    if not selected or not llm_available():
        return _normalize_plan(_fallback(user_text, 'missing_key' if not llm_available() else 'offline_selected', False))
    registry = load_registry()
    listing = '\n'.join(f"- {k}: {v['title']} — {v['description']}" for k, v in registry.items())
    from storage import db
    con = db.connect()
    try:
        cities = sorted({h['city'] for h in db.list_hotels(con) if h['city']})
    finally:
        con.close()
    system = ORCHESTRATOR_SYSTEM.replace('{features}', listing) + '\nГорода в текущей БД: ' + ', '.join(cities)
    repaired = False
    try:
        try:
            response = ask_gemini_structured(user_text, schema=GeminiPlan, system=system)
            plan = Plan.model_validate(response.model_dump() if isinstance(response, BaseModel) else response)
        except StructuredOutputError as exc:
            plan = _repair_payload(exc.payload)
            if plan is None:
                raise
            repaired = True
    except Exception as exc:
        return _normalize_plan(_fallback(user_text, failure_reason(exc), True))
    plan.planner = 'llm'
    plan.llm_attempted = True
    plan.fallback_reason = None
    if not repaired:
        plan.notices = []
    if plan.intent == 'chitchat' and re.search(r'(?i)хочу|отел|номер|гостиниц|подбер|ищу', user_text):
        plan.intent = 'search_hotels'
        plan.notices.append('Пожелание распознано неуверенно: уточните условия, доступные отели всё равно показаны.')
    return _normalize_plan(plan)


def explain(user_text: str, search_output: dict, use_llm: bool | None = None, diagnostics=None) -> str:
    from agents.offline import explain_offline
    if use_llm is None:
        use_llm = llm_available()
    if use_llm:
        try:
            text = ask_gemini(f"Запрос пользователя: {user_text}\n\nРезультаты поиска:\n" +
                              json.dumps(search_output, ensure_ascii=False), system=EXPLAIN_SYSTEM, temperature=0.3)
            if not isinstance(text, str) or not text.strip():
                raise StructuredOutputError('empty_response')
            if diagnostics is not None:
                diagnostics.append({'stage': 'explanation', 'status': 'llm'})
            return text.strip()
        except Exception as exc:
            reason = failure_reason(exc)
            LOGGER.warning('Search explanation fallback: %s', reason)
            if diagnostics is not None:
                diagnostics.append({'stage': 'explanation', 'status': 'fallback', 'reason': reason,
                                    'message': failure_message(reason) + ' Пояснение составлено по шаблону из фактов БД.'})
    else:
        if diagnostics is not None:
            diagnostics.append({'stage': 'explanation', 'status': 'template'})
    return explain_offline(search_output)


def run(user_text: str, *, hotelier_context: dict | None = None, use_llm: bool | None = None, top_k: int = 3) -> dict:
    """LLM-план -> проверяемый поиск по БД -> LLM-пояснение; сбои не дают пустой экран."""
    diagnostics = []
    try:
        plan = analyze(user_text, use_llm)
    except Exception as exc:
        reason = failure_reason(exc)
        return {'plan': Plan(intent='search_hotels', planner='offline'), 'results': [], 'ask_user': None,
                'answer': 'Не удалось прочитать данные для поиска. Проверьте подключение к БД и повторите запрос.',
                'diagnostics': [{'stage': 'planning', 'status': 'error', 'reason': reason}], 'default_listing': True}
    out = {'plan': plan, 'results': [], 'ask_user': plan.clarifying_question, 'answer': None,
           'diagnostics': diagnostics, 'default_listing': not any(r.feature != 'other' for r in plan.requirements)}
    if plan.intent == 'chitchat' or not isinstance(user_text, str) or not user_text.strip() or len(user_text) > MAX_QUERY_LENGTH:
        return out
    import agents.search_agent  # Регистрирует агент и при прямом вызове run(), без main.py.
    for task in plan.tasks:
        if task.agent == 'hotelier_agent' and hotelier_context is not None:
            task = task.model_copy(update={'hotel_id': hotelier_context.get('hotel_id'),
                                          'hotelier_reply': hotelier_context.get('answer')})
        agent = AGENTS.get(task.agent)
        try:
            if agent is None:
                raise RuntimeError('agent_unavailable')
            if task.agent == 'search_agent':
                # Единый источник выдачи для CLI, UI и пояснения; top_k передаёт интерфейс.
                output = agents.search_agent.rank_hotels(plan, top_k=top_k)
            else:
                output = agent(task, plan)
            out['results'].append({'agent': task.agent, 'output': output})
        except Exception:
            LOGGER.warning('Search execution failed: %s', task.agent)
            out['results'].append({'agent': task.agent, 'error': 'Не удалось выполнить задачу. Попробуйте ещё раз.'})
            diagnostics.append({'stage': task.agent, 'status': 'error',
                                'message': 'Не удалось получить данные от агента. Проверьте БД и повторите запрос.'})
    for item in out['results']:
        if item['agent'] == 'search_agent' and 'output' in item:
            result = item['output']
            if result.get('ranked'):
                out['answer'] = explain(user_text, result, plan.planner == 'llm' and not out['default_listing'], diagnostics)
            else:
                out['answer'] = result.get('note') or 'В базе пока нет подходящих отелей. Измените город или условия поиска.'
    if not out['answer']:
        out['answer'] = 'Результаты получить не удалось. Проверьте данные и попробуйте ещё раз.'
    return out
