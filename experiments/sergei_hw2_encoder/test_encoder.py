import csv
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from starter.test_starter import write_fixture
from . import encoder, run


class HistoricalAdapter(unittest.TestCase):
    def test_window_boundaries_and_cutoff(self):
        events = [('2021-03-02', 'older', 'Buy'), ('2021-03-03', 'recent', 'Buy'),
                  ('2021-05-31', 'recent', 'Buy'), ('2021-06-01', 'cutoff', 'Buy'),
                  ('2021-07-01', 'future', 'Buy'), ('2021-03-03', 'sold', 'Sell')]
        self.assertEqual(encoder.split_history(events),
                         (Counter(recent=2), Counter(older=1)))

    def test_filters_and_repeat_buys(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_fixture(root)
            with (root / 'transactions.csv').open('a', newline='') as stream:
                csv.writer(stream).writerow(['U1', 'A', '20', 'Buy', '2021-04-02',
                                             '100', '1', 'Branch', 'XATH'])
            assets, windows = encoder.load_histories(root)
            self.assertTrue({'M', 'N'}.isdisjoint(assets))
            self.assertEqual(set(windows), {'U1', 'U2'})
            self.assertEqual(windows['U1'], (Counter(A=2, B=1, C=1), Counter()))
            self.assertEqual(windows['U2'], (Counter(A=1, D=1), Counter()))

    def test_future_mutations_leave_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as base, tempfile.TemporaryDirectory() as changed:
            base, changed = Path(base), Path(changed)
            write_fixture(base)
            write_fixture(changed, 'mutated')
            with (changed / 'asset_information.csv').open('a', newline='') as stream:
                csv.writer(stream).writerow(['A', 'Future Name', 'A', 'Future Type',
                                             '', '', '', '', '2021-06-01'])
            baseline = encoder.load_histories(base)
            self.assertEqual(encoder.load_histories(changed), baseline)
            self.assertNotEqual(encoder.starter.load_data(base, encoder.CUTOFF, 30)[3],
                                encoder.starter.load_data(changed, encoder.CUTOFF, 30)[3])


class Weighting(unittest.TestCase):
    def test_overlap_and_normalization(self):
        weights = encoder.history_weights(Counter(A=2, B=2), Counter(A=1, C=3), 0.25)
        self.assertEqual(weights, {'A': 0.3125, 'B': 0.125, 'C': 0.5625})
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_deterministic_alpha_and_ties(self):
        for recent, older, expected in [(1, 3, 0.25), (2, 2, 0.50), (3, 1, 0.75),
                                         (3, 5, 0.50), (5, 3, 0.50)]:
            with self.subTest(recent=recent, older=older):
                self.assertEqual(encoder.purchase_share_alpha(recent, older), expected)

    def test_either_single_window_skips_inference(self):
        def unexpected(_):
            self.fail('Single-window user reached inference')
        counts = Counter(A=3, B=1)
        for recent, older in [(counts, Counter()), (Counter(), counts)]:
            self.assertEqual(encoder.llm_weights(recent, older, {}, unexpected),
                             ({'A': 0.75, 'B': 0.25}, None))


class ModelContract(unittest.TestCase):
    def test_strict_json(self):
        for value in ('0.25', '0.50', '0.75', '5e-1'):
            self.assertEqual(encoder.parse_alpha('{"alpha":' + value + '}'), float(value))
        for response in ('{"alpha":', '{"alpha":0.25,"alpha":0.75}',
                         '{"alpha":0.25,"extra":null}', 'Answer: {"alpha":0.50}',
                         '{"alpha":0.50} trailing', '{"alpha":true}', '{"alpha":NaN}',
                         '{"alpha":Infinity}', '{"alpha":-Infinity}', '{"alpha":1e999}',
                         '{"alpha":0.4}', '{"alpha":"0.50"}', '[{"alpha":0.25}]',
                         '{"alpha":0.25000000000000000001}', '{"wrong":0.50}'):
            with self.subTest(response=response), self.assertRaises(ValueError):
                encoder.parse_alpha(response)

    def test_equal_mixture_fallback_with_reason(self):
        def unavailable(_):
            raise encoder.RecoverableInferenceError('test timeout')

        assets = {'A': {}, 'B': {}}
        for callback, reason in [(lambda _: '{}', 'invalid response'),
                                 (unavailable, 'inference failure: test timeout')]:
            weights, failure = encoder.llm_weights(Counter(A=5), Counter(B=1), assets, callback)
            self.assertEqual(weights, {'A': 0.5, 'B': 0.5})
            self.assertIn(reason, failure)
        self.assertEqual(encoder.llm_weights(Counter(A=5), Counter(B=1), assets,
                                            lambda _: '{"alpha":0.75}'),
                         ({'A': 0.75, 'B': 0.25}, None))

    def test_prompt_contains_only_aggregates(self):
        asset = {'assetCategory': ' Stock ', 'sector': 'Not_Available',
                 'marketID': 'XATH', 'ISIN': 'private-asset', 'timestamp': '2021-05-10',
                 'assetName': 'private-name'}
        prompt = encoder.model_prompt(Counter({'private-asset': 2}), Counter(),
                                      {'private-asset': asset})
        payload = json.loads(prompt.split('Historical aggregates:\n')[1])
        self.assertEqual(set(payload), {'recent', 'older'})
        recent = payload['recent']
        self.assertEqual(set(recent), {'buy_events', 'distinct_assets', 'histograms', 'missing'})
        self.assertEqual((recent['buy_events'], recent['distinct_assets']), (2, 1))
        self.assertEqual(recent['histograms']['assetCategory'], {'stock': 2})
        self.assertEqual(recent['histograms']['marketID'], {'xath': 2})
        self.assertEqual(recent['missing']['sector'], 2)
        for field in encoder.FEATURES:
            self.assertEqual(sum(recent['histograms'][field].values()) + recent['missing'][field], 2)
        for private in ('private-asset', 'private-name', '2021-05-10'):
            self.assertNotIn(private, prompt)

    def test_frozen_contract(self):
        run.check_freeze()


@unittest.skipUnless(os.environ.get('ENCODER_RUN') and os.environ.get('ENCODER_REFERENCE'),
                     'Set ENCODER_RUN and ENCODER_REFERENCE for actual-data checks')
class JuneArtifacts(unittest.TestCase):
    def setUp(self):
        self.output = Path(os.environ['ENCODER_RUN'])
        self.reference = Path(os.environ['ENCODER_REFERENCE'])

    def test_baseline_parity_and_matching_protocol(self):
        run.verify_outputs(self.output, self.reference)

    def test_all_users_get_historical_only_weights_and_unique_top_ten(self):
        protocol = run.read_json(self.output / 'protocol.json')
        users = set(protocol['users'])
        contexts = encoder.starter.read_responses(self.output / 'contexts.jsonl')
        responses = encoder.starter.read_responses(self.output / 'deterministic_responses.jsonl')
        self.assertEqual(set(responses), users)
        self.assertEqual(len(users), 100)
        for uid, response in responses.items():
            self.assertEqual(response['stage'], 'encoder')
            self.assertEqual(response['protocol_hash'], protocol['protocol_hash'])
            weights = response['history_weights']
            self.assertEqual(set(weights), {asset['id'] for asset in contexts[uid]['history']})
            self.assertAlmostEqual(sum(weights.values()), 1.0)
        for method in run.METHODS | {'custom'}:
            rows = [row for row in run.read_csv(self.output / 'predictions.csv')
                    if row['method'] == method]
            self.assertEqual(len(rows), 1000)
            self.assertEqual({row['user_id'] for row in rows}, users)
            counts = Counter((row['user_id'], row['ISIN']) for row in rows)
            self.assertTrue(all(count == 1 for count in counts.values()))
            self.assertTrue({row['ISIN'] for row in rows} <= set(protocol['eligible_assets']))
