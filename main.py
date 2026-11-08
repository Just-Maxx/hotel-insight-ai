import sys

import agents.hotelier_agent  # noqa: F401  (импорт нужен, чтобы агент зарегистрировался)
import agents.search_agent    # noqa: F401
from agents.orchestrator import run


def main():
    text = " ".join(sys.argv[1:]) or input("Запрос: ")
    out = run(text)
    print("\n=== ПЛАН ===")
    print(out["plan"].model_dump_json(indent=2, ensure_ascii=False))
    for notice in out['plan'].notices:
        print("\nПредупреждение:", notice)
    for diagnostic in out.get('diagnostics', []):
        if diagnostic.get('message'):
            print("\nПредупреждение:", diagnostic['message'])
    if out["ask_user"]:
        print("\nУточняющий вопрос:", out["ask_user"])
    if out.get("answer"):
        print("\n=== ОТВЕТ ===\n" + out["answer"])
    else:
        for r in out["results"]:
            print("\n---", r["agent"], "->", r.get("output") or r.get("error"))


if __name__ == "__main__":
    main()
