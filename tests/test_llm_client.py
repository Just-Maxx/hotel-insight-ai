import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from google.genai import errors, types
from core.llm_client import _generate, ask_gemini_structured
from schemas.hotelier import PolicyExtraction


class RetryTests(unittest.TestCase):
    @patch('core.llm_client.time.sleep')
    @patch('core.llm_client.get_client')
    def test_temporary_failure_retried(self, get_client, sleep):
        method = get_client.return_value.models.generate_content
        method.side_effect = [errors.APIError(503, {'error': {'message': 'busy'}}), Mock(text='ok')]
        self.assertEqual(_generate('test', types.GenerateContentConfig()).text, 'ok')
        self.assertEqual(method.call_count, 2)

    @patch('core.llm_client.time.sleep')
    @patch('core.llm_client.get_client')
    def test_bad_key_not_retried(self, get_client, sleep):
        method = get_client.return_value.models.generate_content
        method.side_effect = errors.APIError(403, {'error': {'message': 'denied'}})
        with self.assertRaises(errors.APIError):
            _generate('test', types.GenerateContentConfig())
        self.assertEqual(method.call_count, 1)
        sleep.assert_not_called()

    @patch('core.llm_client._generate')
    def test_json_schema_preserves_local_strict_validation(self, generate):
        generate.return_value = Mock(parsed={'policy': {'allowed': False}, 'evidence': ['Нет']})
        result = ask_gemini_structured('Нет', PolicyExtraction)
        self.assertIsInstance(result, PolicyExtraction)
        config = generate.call_args.args[1]
        self.assertIsNone(config.response_schema)
        self.assertEqual(config.response_json_schema['additionalProperties'], False)

    @patch('core.llm_client._generate')
    def test_unknown_fields_from_provider_rejected(self, generate):
        from core.llm_client import StructuredOutputError
        generate.return_value = Mock(parsed={'invented_field': True})
        with self.assertRaises(StructuredOutputError):
            ask_gemini_structured('Нет', PolicyExtraction)

    @patch('core.llm_client._generate')
    def test_empty_and_invalid_json_are_classified(self, generate):
        from core.llm_client import StructuredOutputError
        for text,kind in [(None,'empty_response'), ('  ','empty_response'), ('not JSON','invalid_json')]:
            generate.return_value=Mock(parsed=None,text=text)
            with self.assertRaises(StructuredOutputError) as ctx:
                ask_gemini_structured('test',PolicyExtraction)
            self.assertEqual(ctx.exception.kind,kind)

    @patch('core.llm_client._generate')
    def test_invalid_schema_retains_payload_for_search_repair(self, generate):
        from core.llm_client import StructuredOutputError
        generate.return_value=Mock(parsed={'requirements':[]})
        with self.assertRaises(StructuredOutputError) as ctx:
            ask_gemini_structured('test',PolicyExtraction)
        self.assertEqual(ctx.exception.kind,'invalid_schema')
        self.assertEqual(ctx.exception.payload,{'requirements':[]})


if __name__ == '__main__':
    unittest.main()
