"""TEST ONLY. Synthetic outputs never enter experiment inference artifacts."""
import json
import os
import sqlite3
import tempfile
import unittest
from collections import Counter
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from . import encoder, qwen_backend, run, runtime


def synthetic(raw='{"alpha":0.75}'):
    return qwen_backend.Generation(raw, 100, 8, 0.01)


class ResponseTests(unittest.TestCase):
    def test_valid_alpha_and_raw_token_metadata(self):
        result = runtime.record_call('TEST ONLY', lambda _: synthetic())
        self.assertEqual(result['raw_response'], '{"alpha":0.75}')
        self.assertEqual((result['parsed_alpha'], result['applied_alpha']), (0.75, 0.75))
        self.assertEqual((result['status'], result['validation_status']), ('completed', 'valid'))
        self.assertEqual((result['input_tokens'], result['generated_tokens']), (100, 8))
        self.assertFalse(result['fallback'])

    def test_malformed_output_is_preserved_with_equal_fallback(self):
        result = runtime.record_call('TEST ONLY', lambda _: synthetic('Here: {"alpha":0.75}'))
        self.assertEqual(result['raw_response'], 'Here: {"alpha":0.75}')
        self.assertIsNone(result['parsed_alpha'])
        self.assertEqual(result['applied_alpha'], 0.50)
        self.assertEqual((result['status'], result['validation_status']), ('failed', 'invalid'))
        self.assertTrue(result['fallback'])
        self.assertTrue(result['failure_reason'])

    def test_recoverable_error_and_fatal_error(self):
        def recoverable(_):
            raise qwen_backend.GenerationError('TEST ONLY timeout', 60, 0.25)
        result = runtime.record_call('TEST ONLY', recoverable)
        self.assertEqual(result['validation_status'], 'inference_error')
        self.assertEqual(result['applied_alpha'], 0.50)
        self.assertIsNone(result['raw_response'])
        self.assertEqual(result['input_tokens'], 60)
        self.assertIsNone(result['generated_tokens'])

        def fatal(_):
            raise RuntimeError('TEST ONLY backend unavailable')
        with self.assertRaisesRegex(RuntimeError, 'backend unavailable'):
            runtime.record_call('TEST ONLY', fatal)

    def test_all_inference_errors_cannot_be_exported(self):
        with self.assertRaisesRegex(ValueError, 'No model outputs'):
            runtime.starter_responses('test', {'a': 'u'}, {}, {
                'a': {'validation_status': 'inference_error', 'applied_alpha': 0.5}})

    def test_export_100_users_including_single_window_and_fallback(self):
        histories = {str(i): (Counter(A=4), Counter(B=1) if i < 35 else Counter())
                     for i in range(100)}
        mapping = {f'case-{i}': str(i) for i in range(35)}
        responses = {case: {'validation_status': 'valid', 'applied_alpha': 0.75}
                     for case in mapping}
        responses['case-0'] = {'validation_status': 'invalid', 'applied_alpha': 0.50}
        rows = runtime.starter_responses('test-protocol', mapping, histories, responses)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'TEST_ONLY.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            loaded = encoder.starter.read_responses(path)
        self.assertEqual(set(loaded), set(histories))
        self.assertEqual(loaded['0']['history_weights'], {'A': 0.50, 'B': 0.50})
        self.assertEqual(loaded['1']['history_weights'], {'A': 0.75, 'B': 0.25})
        self.assertEqual(loaded['35']['history_weights'], {'A': 1.0})
        eligible = list('ABCDEFGHIJ')
        for row in loaded.values():
            self.assertEqual(row['stage'], 'encoder')
            self.assertEqual(row['protocol_hash'], 'test-protocol')
            self.assertAlmostEqual(sum(row['history_weights'].values()), 1.0)
            context = {'history': [{'id': i} for i in row['history_weights']],
                       'eligible_asset_ids': eligible}
            ranking = encoder.starter.encode_ranking(
                row['history_weights'], context, {i: [i] for i in eligible},
                dict.fromkeys(eligible, 0))
            encoder.starter.validate_ranking(ranking, eligible)
        del responses['case-1']
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            runtime.starter_responses('test-protocol', mapping, histories, responses)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'TEST_ONLY.sqlite3'
        self.prompts = {'a': 'TEST ONLY one', 'b': 'TEST ONLY two'}

    def test_resume_preserves_completed_and_failed_calls(self):
        def interrupted(prompt):
            if prompt.endswith('two'):
                raise KeyboardInterrupt()
            return synthetic('malformed')
        with closing(runtime.database(self.path)) as db:
            runtime.bind_configuration(db, {'source': 'test'})
            with self.assertRaises(KeyboardInterrupt):
                runtime.process_cases(db, self.prompts, interrupted)
            self.assertEqual(set(runtime.records(db, self.prompts)), {'a'})
            self.assertEqual(runtime.records(db, self.prompts)['a']['status'], 'failed')
            self.assertEqual(db.execute('SELECT case_id FROM in_flight').fetchone(), ('b',))

        def remaining(prompt):
            self.assertEqual(prompt, 'TEST ONLY two')
            return synthetic()
        with closing(runtime.database(self.path)) as db:
            runtime.bind_configuration(db, {'source': 'test'})
            runtime.process_cases(db, self.prompts, remaining)
            self.assertEqual(set(runtime.records(db, self.prompts)), {'a', 'b'})
            self.assertIsNone(db.execute('SELECT case_id FROM in_flight').fetchone())
            runtime.process_cases(db, self.prompts, lambda _: self.fail('Repeated call'))

    def test_duplicate_and_incompatible_checkpoint_are_rejected(self):
        with closing(runtime.database(self.path)) as db:
            runtime.bind_configuration(db, {'device': 'test-a'})
            with self.assertRaisesRegex(ValueError, 'Incompatible'):
                runtime.bind_configuration(db, {'device': 'test-b'})
            runtime.process_cases(db, self.prompts, lambda _: synthetic())
            with self.assertRaises(sqlite3.IntegrityError), db:
                db.execute('INSERT INTO responses VALUES (?, ?)', ('a', '{}'))
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                runtime.records(db, {'a': 'different prompt', 'b': self.prompts['b']})


class ProvenanceTests(unittest.TestCase):
    def test_historical_only_payload(self):
        assets = {'SECRET-ASSET': {'assetCategory': 'Stock', 'timestamp': '2021-04-02',
                                   'closePrice': '19.99', 'assetName': 'TEST PRIVATE'}}
        recent, older = Counter({'SECRET-ASSET': 2}), Counter({'SECRET-ASSET': 1})
        text = encoder.model_prompt(recent, older, assets)
        runtime.validate_prompt(text, recent, older, assets, {'SECRET-ASSET', 'SECRET-USER'})
        for injected in ('secret-asset', 'secret-user', '2021-06-01'):
            with self.assertRaises(ValueError):
                runtime.validate_prompt(text.replace('"stock"', json.dumps(injected)), recent,
                                        older, assets, {'SECRET-ASSET', 'SECRET-USER'})

    def test_freeze_detects_mutated_configuration_and_prompt(self):
        run.check_freeze()
        with tempfile.TemporaryDirectory() as tmp:
            temporary = Path(tmp)
            for name in ('freeze.sha256', 'encoder.py', 'model_config.json', 'prompt.txt'):
                (temporary / name).write_bytes((runtime.ROOT / name).read_bytes())
            with patch.object(run, 'ROOT', temporary):
                for name in ('prompt.txt', 'model_config.json'):
                    original = (temporary / name).read_bytes()
                    (temporary / name).write_bytes(original + b' ')
                    with self.assertRaisesRegex(ValueError, 'Frozen design changed'):
                        run.check_freeze()
                    (temporary / name).write_bytes(original)

    def test_paired_calculation_uses_user_differences(self):
        comparison = runtime.paired({'a': .5, 'b': .1, 'c': .0},
                                    {'a': .0, 'b': .3, 'c': .0}, ['a', 'b', 'c'])
        self.assertAlmostEqual(comparison['mean_difference'], .1)
        self.assertEqual([comparison[k] for k in ('improved', 'worsened', 'tied')], [1, 1, 1])


@unittest.skipUnless(os.environ.get('ENCODER_PREPARED'), 'Set ENCODER_PREPARED for real prompt checks')
class RealPreparation(unittest.TestCase):
    def test_35_real_historical_prompts_and_65_skips(self):
        directory = Path(os.environ['ENCODER_PREPARED']).resolve()
        manifest, prompts, mapping, histories = runtime.load_preparation(directory)
        self.assertEqual((manifest['prepared'], manifest['skipped'], len(histories)), (35, 65, 100))
        self.assertEqual(len(mapping), len(prompts))
        self.assertEqual(runtime.progress(directory)['not_yet_attempted'], 35)
