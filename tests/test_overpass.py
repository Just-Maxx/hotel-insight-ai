"""Регрессии загрузчика, без реальной сети и ожиданий."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests
from tools import osm


class OverpassTests(unittest.TestCase):
    def setUp(self):
        osm._preferred_url = None
        osm._cooldown_until.clear()
        osm._failure_counts.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name)
        self.data = {"elements": [{"type": "way", "id": 1}]}

    def response(self, data):
        r = Mock(status_code=200, content=b'{}')
        r.json.return_value = data
        return r

    @patch.object(osm.time, 'sleep')
    @patch('requests.post')
    def test_failover_and_preferred_mirror(self, post, sleep):
        post.side_effect = [requests.ConnectionError('reset'), self.response(self.data), self.response(self.data)]
        osm.fetch_overpass('[timeout:240];first', self.cache)
        osm.fetch_overpass('second', self.cache)
        self.assertEqual(post.call_args_list[1].args[0], post.call_args_list[2].args[0])
        self.assertIn('[timeout:45]', post.call_args_list[0].kwargs['data']['data'])
        sleep.assert_not_called()

    @patch('requests.post')
    def test_cache_works_without_network(self, post):
        post.return_value = self.response(self.data)
        osm.fetch_overpass('cached', self.cache)
        post.reset_mock()
        self.assertEqual(osm.fetch_overpass('cached', self.cache), self.data)
        post.assert_not_called()

    @patch.object(osm.time, 'sleep')
    @patch('requests.post')
    def test_partial_response_not_cached(self, post, sleep):
        post.return_value = self.response({'elements': [], 'remark': 'runtime error: Query timed out'})
        with self.assertRaises(osm.OverpassQueryError):
            osm.fetch_overpass('partial', self.cache)
        self.assertEqual(list(self.cache.glob('*.json')), [])

    @patch('requests.post')
    def test_corrupt_cache_reloaded(self, post):
        post.return_value = self.response(self.data)
        osm.fetch_overpass('broken', self.cache)
        next(self.cache.glob('*.json')).write_text('{')
        self.assertEqual(osm.fetch_overpass('broken', self.cache), self.data)
        self.assertEqual(post.call_count, 2)

    def test_connection_failure_does_not_subdivide(self):
        fetch = Mock(side_effect=osm.OverpassUnavailable('offline'))
        with self.assertRaises(osm.OverpassUnavailable):
            osm.fetch_adaptive(lambda b: str(b), (0, 0, 1, 1), self.cache, fetch=fetch)
        self.assertEqual(fetch.call_count, 1)

    def test_query_failure_subdivides(self):
        fetch = Mock(side_effect=[osm.OverpassQueryError('timeout')] + [self.data] * 4)
        result = osm.fetch_adaptive(lambda b: str(b), (0, 0, .02, .02), self.cache, fetch=fetch)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(list(self.cache.glob("*.json"))), 1)
        self.assertEqual(fetch.call_count, 5)

    @patch.object(osm.time, 'sleep')
    @patch('requests.post')
    def test_repeated_gateway_errors_are_splittable(self, post, sleep):
        response = Mock(status_code=502, headers={})
        response.raise_for_status.side_effect = requests.HTTPError('502', response=response)
        post.return_value = response
        with self.assertRaises(osm.OverpassQueryError):
            osm.fetch_overpass('gateway', self.cache)

    @patch.object(osm.time, 'sleep')
    @patch('requests.post')
    def test_dead_mirror_not_retried_on_next_tile(self, post, sleep):
        post.side_effect = [requests.ConnectionError('reset'), self.response(self.data),
                            requests.ConnectionError('reset'), self.response(self.data)]
        osm.fetch_overpass('tile1', self.cache)
        osm.fetch_overpass('tile2', self.cache)
        urls = [call.args[0] for call in post.call_args_list]
        self.assertEqual(urls.count(osm.OVERPASS_URLS[0]), 1)



if __name__ == '__main__':
    unittest.main()
