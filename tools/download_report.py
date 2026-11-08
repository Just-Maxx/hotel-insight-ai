"""Отчёт текущего запуска: пробелы сохраняются сразу после каждого сбоя."""
import hashlib
import json
from pathlib import Path


class DownloadReport:
    def __init__(self, path, bbox):
        self.path = Path(path)
        self.data = {"bbox": list(bbox), "status": "running", "missing": []}
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(self.path)

    def callback(self, query_fn):
        def record(bbox, error):
            query = query_fn(bbox)
            self.data['missing'].append({"layer": query_fn.__name__, "bbox": list(bbox),
                "cache_key": hashlib.sha1(query.encode()).hexdigest()[:16], "error": str(error)})
            self.save()
        return record

    def missing(self, *layers):
        return [m for m in self.data['missing'] if m['layer'] in layers]

    def finish(self):
        self.data['status'] = 'partial' if self.data['missing'] else 'complete'
        self.save()
