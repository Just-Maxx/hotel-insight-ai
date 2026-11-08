"""Сбой плитки не теряет соседние данные и не превращается в оценку отеля."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.osm import fetch_adaptive, OverpassQueryError, OverpassUnavailable
from tools.download_report import DownloadReport
from tools.pipeline import mark_incomplete
from scripts.import_hotels import score_of


class PartialTests(unittest.TestCase):
    def test_partial_children_preserved_but_parent_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / 'raw'
            report = DownloadReport(Path(tmp) / 'report.json', (0, 0, .02, .02))
            query = lambda bbox: str(bbox)
            fetch = Mock(side_effect=[OverpassQueryError('timeout'),
                {'elements': [{'type': 'way', 'id': 1}]}, OverpassUnavailable('offline'),
                {'elements': [{'type': 'way', 'id': 2}]}, {'elements': []}])
            result = fetch_adaptive(query, (0, 0, .02, .02), cache, fetch=fetch,
                                    on_missing=report.callback(query))
            self.assertEqual([e['id'] for e in result], [1, 2])
            self.assertEqual(fetch.call_count, 5)
            self.assertEqual(len(report.data['missing']), 1)
            self.assertFalse(list(cache.glob('*.json')))
            report.finish()
            self.assertEqual(json.loads(report.path.read_text())['status'], 'partial')

    def test_next_run_replaces_missing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.json'
            report = DownloadReport(path, (0, 0, 1, 1))
            report.callback(str)((0, 0, 1, 1), RuntimeError('offline'))
            report.finish()
            report = DownloadReport(path, (0, 0, 1, 1))
            report.finish()
            self.assertEqual(json.loads(path.read_text())['status'], 'complete')
            self.assertEqual(report.data['missing'], [])

    def test_missing_data_not_scored_even_with_existing_score(self):
        feature = {'value': {'west': {'1': .9}}, 'confidence': .9,
                   'meta': {'score': .9, 'summary': 'sea view'}}
        mark_incomplete(feature, [{'layer': 'buildings_query'}])
        self.assertIsNone(feature['value'])
        self.assertEqual(feature['confidence'], 0)
        self.assertIsNone(score_of('sea_view', feature, 3))

    def test_pipeline_continues_after_walk_failure(self):
        import numpy as np
        from tools import osm_extra as ox
        from tools.osm import Grid, synthetic_square
        from tools.geo.visibility import Scene
        from tools.pipeline import load_extra_layers, compute_hotel_features
        with tempfile.TemporaryDirectory() as tmp:
            grid = Grid(43.4, 39.9, 43.401, 39.901, res=10)
            scene = Scene(np.zeros((grid.ny, grid.nx)), np.zeros((grid.ny, grid.nx)),
                          10, 0, 0, np.zeros((grid.ny, grid.nx), bool))
            report = DownloadReport(Path(tmp) / 'report.json', grid.bbox)
            seen = []
            def fetch(query, cache):
                seen.append(query)
                if 'out body qt' in query:
                    raise OverpassUnavailable('offline')
                return {'elements': []}
            with patch.object(ox, 'fetch_overpass', side_effect=fetch):
                layers = load_extra_layers(scene, grid, Path(tmp) / 'raw', report=report)
            self.assertEqual(len(seen), 4)  # POI и пляжи загружались после сбоя пешей сети.
            hotel = {'poly': synthetic_square(50, 50), 'levels': 1, 'height_source': 'default',
                     'footprint_source': 'synthetic', 'x': 50, 'y': 50, 'lat': 43.4}
            features = compute_hotel_features(scene, grid, hotel, layers)
            for name in ('walk_access', 'night_amenities', 'kids_infrastructure'):
                self.assertTrue(features[name]['meta']['incomplete'])
                self.assertIsNone(score_of(name, features[name], 1))
            self.assertNotIn('incomplete', features['quiet']['meta'])
            self.assertNotIn('incomplete', features['sea_view']['meta'])


if __name__ == '__main__':
    unittest.main()
