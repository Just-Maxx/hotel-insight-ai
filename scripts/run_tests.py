"""Все офлайн-тесты в отдельной копии с синтетической БД; рабочие данные не меняются."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='hotel-insight-tests-') as folder:
        work = Path(folder)
        for name in ('agents', 'app', 'core', 'schemas', 'scripts', 'simulator', 'storage',
                     'tools', 'tests', 'eval', 'features'):
            shutil.copytree(ROOT/name, work/name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(ROOT/'main.py', work/'main.py')
        env = os.environ.copy()
        env['GEMINI_API_KEY'] = ''
        env['HOTELS_DB'] = str(work/'data/demo/facts.db')
        print('Подготовка синтетической БД в отдельной временной копии…', flush=True)
        setup = subprocess.run([sys.executable, 'scripts/setup_demo.py'], cwd=work,
                               env=env, capture_output=True, text=True)
        if setup.returncode:
            print((setup.stdout+setup.stderr)[-5000:])
            return setup.returncode
        failures = []
        tests = sorted((work/'tests').glob('test_*.py'))
        if not tests:
            print('Тестовые файлы не найдены')
            return 1
        for test in tests:
            result = subprocess.run([sys.executable, str(test)], cwd=work,
                                    env=env, capture_output=True, text=True)
            print(f'{test.name}: {"PASS" if result.returncode == 0 else "FAIL"}', flush=True)
            if result.returncode:
                failures.append(test.name)
                print((result.stdout+result.stderr)[-5000:])
        print(f'Пройдено: {len(tests)-len(failures)}/{len(tests)} тестовых файлов. Рабочая БД не изменена.')
        return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
