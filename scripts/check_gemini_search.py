"""Живая проверка Gemini: fallback не засчитывается как успешный онлайн-тест."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.orchestrator import llm_available, run

QUERIES = ['Хочу море в пешей доступности', 'Хочется засыпать без грохота и видеть морской горизонт из окна',
           'Хочу тихий отель с телепортом к пляжу']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', action='append', help='Можно указать несколько раз; по умолчанию три сценария')
    args = parser.parse_args()
    if not llm_available():
        print('Ключ GEMINI_API_KEY не задан. Онлайн-тест не выполнялся.')
        return 1
    failed = False
    for query in args.query or QUERIES:
        print('\nЗапрос:', query, flush=True)
        out = run(query, use_llm=True)
        plan = out['plan']
        print('Разбор:', plan.planner, '| API вызван:', plan.llm_attempted, flush=True)
        for req in plan.requirements:
            print('  ', req.feature, req.target or '', req.max_minutes or '', req.detail, flush=True)
        for notice in plan.notices:
            print('Предупреждение:', notice, flush=True)
        ranked = next((r.get('output', {}).get('ranked', []) for r in out['results'] if r['agent']=='search_agent'), [])
        print('Отелей:', len(ranked), '| Уточнение:', out['ask_user'], flush=True)
        for diagnostic in out['diagnostics']:
            print('Диагностика:', diagnostic, flush=True)
        success = plan.planner == 'llm' and bool(ranked) and not any(
            d['status'] in ('fallback', 'error') for d in out['diagnostics'])
        print('ONLINE PASS' if success else 'ONLINE НЕ ПОДТВЕРЖДЁН (резервная выдача не считается онлайн-успехом)', flush=True)
        failed |= not success
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
