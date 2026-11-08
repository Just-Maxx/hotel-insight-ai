"""Демо коммуникаций: офлайн, Gemini, ввод человека или явно указанный сайт."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.hotelier_agent import clarify, QUESTIONS
from simulator.hotelier_sim import CARDS, HotelierSimulator
from tools.web_parser import fetch_page


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hotel', default='demo_dogs', help='ID карточки или реального отеля для --human/--url')
    parser.add_argument('--features', nargs='+', choices=QUESTIONS, default=['pets_allowed'])
    parser.add_argument('--llm', action='store_true', help='Gemini для извлечения фактов')
    parser.add_argument('--llm-simulator', action='store_true', help='Gemini также отвечает за вымышленного отельера')
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument('--human', action='store_true')
    transport.add_argument('--url', help='Публичная HTML-страница с условиями отеля')
    parser.add_argument('--max-rounds', type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    if (args.human or args.url or args.llm_simulator) and not args.llm:
        parser.error('Для свободного текста нужен --llm')
    if args.llm_simulator and (args.human or args.url):
        parser.error('--llm-simulator применим только к симулятору')
    try:
        source_url = None
        if args.url:
            page = fetch_page(args.url)
            source_url = page['url']
            respondent = lambda q: page['text']
            source = 'hotel_website'
        elif args.human:
            respondent = lambda q: input(q + '\nОтвет: ')
            source = 'hotelier_human'
        else:
            if args.hotel not in CARDS:
                parser.error('Неизвестная демо-карточка: ' + ', '.join(CARDS))
            respondent = HotelierSimulator(args.hotel, use_llm=args.llm_simulator)
            source = 'hotelier_simulator'
        result = clarify(args.hotel, args.features, respondent, source=source, use_llm=args.llm,
            max_rounds=1 if args.url else args.max_rounds, source_url=source_url)
        print(result.model_dump_json(indent=2))
        print('Утверждения источника не верифицированы. База данных не изменена.')
        return 1 if result.status == 'error' else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'Демо не завершено: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
