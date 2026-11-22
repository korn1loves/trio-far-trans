"""Encoder arithmetic and injected-generator tests; not LLM experiments."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoder import (tfidf, combine_views, history_scores, source_text,
                     paraphrase_prompt, generate_views, predict, _prose_error,
                     _target_word_count, _templates_by_length, _unmatched_fields)


class _FakeGenerator:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, prompt, max_new_tokens):
        self.calls.append(prompt)
        return {'text': next(self.responses)}


class EncoderTests(unittest.TestCase):
    def assertSparse(self, actual, expected, tolerance=1e-12):
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key], delta=tolerance)

    def test_all_five_ledger_vectors(self):
        spec = Path(__file__).resolve().parents[2] / 'docs/research/math-ledger/far-trans-llm-baselines.md'
        if spec.is_file():
            vectors = json.loads(spec.read_text().split('```json\n')[1].split('```')[0])
        else:
            # Standalone distribution: the same independently specified E701–E703
            # vectors, without depending on the course repository's docs tree.
            vectors = [
                dict(id='TV1', equation='E701', input={'documents': ['alpha', 'alpha']},
                     expected=[{'alpha': 1.0}, {'alpha': 1.0}], tolerance={'atol': 1e-12}),
                dict(id='TV2', equation='E701', input={'documents': ['', '']},
                     expected=[{}, {}], tolerance={'atol': 0}),
                dict(id='TV3', equation='E702', input={'original': {'x': 1.0}, 'generated': {'y': 1.0}},
                     expected={'original:x': 0.7071067811865476, 'generated:y': 0.7071067811865476}),
                dict(id='TV4', equation='E703', input={'vectors': {'A': {'x': 1.0}, 'B': {'y': 1.0}},
                     'counts': {'A': 2, 'B': 1}}, expected={'A': 0.8944271909999159, 'B': 0.4472135954999579}),
                dict(id='TV5', equation='E703', input={'vectors': {'A': {'x': 1.0}}, 'counts': {}},
                     expected={'A': 0.0}),
            ]
        self.assertEqual(len(vectors), 5)
        for tv in vectors:
            with self.subTest(vector=tv['id']):
                data = tv['input']
                if tv['equation'] == 'E701':
                    actual = list(tfidf(dict(enumerate(data['documents']))).values())
                    for a, e in zip(actual, tv['expected']):
                        self.assertSparse(a, e, tv['tolerance']['atol'])
                elif tv['equation'] == 'E702':
                    actual = combine_views({'A': data['original']}, {'A': data['generated']})['A']
                    self.assertSparse(actual, tv['expected'])
                else:
                    self.assertSparse(history_scores(data['counts'], data['vectors']), tv['expected'])

    def test_nontrivial_df_and_raw_counts(self):
        # beta appears once in two docs: IDF = 1 + ln(3/2); alpha IDF = 1.
        idf = 1.4054651081081644
        norm = math.sqrt(4 + idf * idf)
        actual = tfidf({'A': 'ALPHA alpha beta', 'B': 'alpha'})
        self.assertSparse(actual['A'], {'alpha': 2 / norm, 'beta': idf / norm})
        self.assertSparse(actual['B'], {'alpha': 1})
        self.assertEqual(tfidf({}), {})
        self.assertEqual(tfidf({'A': ''}), {'A': {}})

    def test_namespaces_and_missing_view(self):
        self.assertSparse(combine_views({'A': {'x': 4}}, {'A': {'x': 2}})['A'],
                          {'original:x': 1 / math.sqrt(2), 'generated:x': 1 / math.sqrt(2)})
        self.assertEqual(combine_views({'A': {}}, {'B': {'x': 1}}),
                         {'A': {}, 'B': {'generated:x': 1}})

    def test_count_validation(self):
        for count in (-1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                history_scores({'A': count}, {'A': {'x': 1}})
        self.assertEqual(history_scores({'missing': 2}, {'A': {}}), {'A': 0})

    def test_safe_canonical_source(self):
        text = source_text({'assetName': 'Alpha', 'sector': 'Energy', 'ISIN': 'SECRET',
                            'timestamp': '2099-01-01', 'future_returns': 'LEAK'})
        self.assertEqual(set(json.loads(text)), {'name', 'category', 'subcategory', 'sector', 'industry', 'market'})
        self.assertNotIn('SECRET', text)
        self.assertNotIn('LEAK', text)
        self.assertNotIn('2099', text)
        self.assertEqual(json.loads(text)['industry'], 'unknown')
        prompt = paraphrase_prompt(text, 28)
        self.assertIn('Aim for 28 whitespace-separated words', prompt)
        self.assertIn('its sector is unknown, and its industry is unknown', prompt)
        self.assertIn('Target asset as-of fields:', prompt)
        self.assertEqual(text, source_text({'sector': 'Energy', 'assetName': 'Alpha'}))

    def test_generation_once_per_asset_and_matches_whole_values(self):
        complete = source_text({
            'assetName': 'PPC SA', 'assetCategory': 'Stock', 'assetSubCategory': 'Large Cap',
            'sector': 'Utilities', 'industry': 'Utilities - Renewable', 'marketID': 'XATH',
        })
        missing = source_text({
            'assetName': 'Viohalco', 'assetCategory': 'Stock', 'assetSubCategory': 'Large Cap',
            'marketID': 'XATH',
        })
        prose = (
            'The name is PPC SA, the category is Stock, and the subcategory is Large Cap. '
            'The sector is Utilities, the industry is Utilities - Renewable, and the market is XATH.'
        )
        missing_prose = (prose.replace('PPC SA', 'Viohalco')
                         .replace('sector is Utilities', 'sector is unknown')
                         .replace('industry is Utilities - Renewable', 'industry is unknown'))
        documents = {'B': complete, 'A': missing}
        fake = _FakeGenerator([missing_prose, prose])
        generated, templates, failures, audit = generate_views(documents, fake)
        self.assertEqual(fake.calls, [paraphrase_prompt(missing, 29), paraphrase_prompt(complete, 32)])
        self.assertEqual(failures, [])
        self.assertEqual(generated, {'A': missing_prose, 'B': prose})
        for asset_id in documents:
            self.assertEqual(len(templates[asset_id].split()), len(generated[asset_id].split()))
            self.assertEqual(audit[asset_id]['actual_word_count'], len(templates[asset_id].split()))
            self.assertNotEqual(audit[asset_id]['requested_word_count'], audit[asset_id]['actual_word_count'])
            self.assertEqual(_unmatched_fields(templates[asset_id], json.loads(documents[asset_id])), [])
            self.assertIn('Large Cap', templates[asset_id])
        self.assertIn('industry is Utilities - Renewable', templates['B'])
        self.assertIn('industry is unknown', templates['A'])
        self.assertEqual(
            generate_views(documents, _FakeGenerator([missing_prose, prose]))[1], templates,
        )

    def test_prose_format_and_word_count_boundaries(self):
        for raw, reason in (
            ('', 'empty'),
            ('{"name": "Alpha"}', 'json_output'),
            ('["Alpha"]', 'json_output'),
            ('{"name":', 'json_output'),
            ('\x60\x60\x60json\n{"name": "Alpha"}\n\x60\x60\x60', 'non_prose'),
            ('- The name is Alpha.', 'non_prose'),
            ('Alpha', None),
            (' '.join(['word'] * 40), None),
            (' '.join(['word'] * 41), 'word_limit'),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(_prose_error(raw, len(raw.split())), reason)

    def test_rejected_pairs_share_canonical_fallback_and_log_reasons(self):
        canonical = source_text({
            'assetName': 'Alpha', 'assetCategory': 'Bond', 'assetSubCategory': 'Corporate',
            'sector': 'Corporate', 'marketID': 'XATH',
        })
        prose = (
            'The name is Alpha, its category is Bond, and its subcategory is Corporate. '
            'Its sector is Corporate, its industry is unknown, and its market is XATH.'
        )
        documents = dict.fromkeys(('empty', 'fields', 'json', 'length'), canonical)
        responses = [
            '',
            prose.replace('Its sector is Corporate, ', ''),
            '{"name": "Alpha"}',
            prose.replace('The name is Alpha', 'Name is Alpha')
                 .replace('its category', 'category').replace('and its subcategory', 'subcategory')
                 .replace('Its sector', 'Sector').replace('its industry', 'industry')
                 .replace('and its market', 'market'),
        ]
        generated, templates, failures, audit = generate_views(
            documents, _FakeGenerator(responses), 'fallback',
        )
        self.assertEqual(generated, templates)
        self.assertEqual(templates, documents)
        self.assertEqual([failure['reason'] for failure in failures], [
            'empty', 'missing_or_ambiguous_fields', 'json_output', 'no_exact_faithful_template',
        ])
        self.assertEqual(failures[1]['unmatched_fields'], ['sector'])
        self.assertTrue(all(failure['fallback'] for failure in failures))
        self.assertTrue(all(row['exact_text_duplicate'] is None for row in audit.values()))
        self.assertTrue(all(row['identical_tfidf_token_counts'] is None for row in audit.values()))
        self.assertNotIn(failures[3]['word_count'], failures[3]['attainable_template_lengths'])
        with self.assertRaisesRegex(ValueError, 'json_output'):
            generate_views({'A': canonical}, _FakeGenerator(['{}']), 'error')

    def test_duplicate_checks_distinguish_text_from_token_counts(self):
        canonical = source_text({'assetName': 'Northstar Paper Co', 'assetCategory': 'Stock',
                                 'assetSubCategory': 'Small Cap', 'marketID': 'XDEM'})
        facts = json.loads(canonical)
        candidates = _templates_by_length(facts)
        self.assertEqual(_target_word_count(candidates), 31)
        template = candidates[31]
        first, second = template.split('. ', 1)
        reordered = f'{second} {first}.'
        different = template.replace('is listed as', 'is recorded as')
        docs = dict.fromkeys(('duplicate', 'reordered', 'tokens_differ'), canonical)
        _, _, failures, audit = generate_views(
            docs, _FakeGenerator([template, reordered, different]),
        )
        self.assertEqual(failures, [])
        self.assertEqual([
            (row['exact_text_duplicate'], row['identical_tfidf_token_counts'])
            for row in audit.values()
        ], [(True, True), (False, True), (False, False)])

    def test_no_template_within_cap_falls_back_without_a_model_call(self):
        canonical = source_text({'assetName': 'Very ' * 41})
        documents = {'A': canonical}
        fake = _FakeGenerator([])
        generated, templates, failures, audit = generate_views(documents, fake, 'fallback')
        self.assertEqual(fake.calls, [])
        self.assertEqual(generated, documents)
        self.assertEqual(templates, documents)
        self.assertEqual(failures[0]['reason'], 'no_template_within_word_limit')
        self.assertIsNone(audit['A']['requested_word_count'])
        self.assertIsNone(audit['A']['actual_word_count'])

    def test_user_coverage_cold_ties_same_full_universe(self):
        ids = [f'{i:02d}' for i in range(12)]
        contexts = {u: {'eligible_asset_ids': ids} for u in ('warm', 'cold')}
        vectors = tfidf({i: 'alpha' for i in ids})
        combined = combine_views(vectors, vectors)
        predictions = predict(contexts, {'warm': {'00': 3}}, vectors,
                              combined, {i: int(i) for i in ids}, template=combined)
        self.assertEqual(set(predictions), set(contexts))
        for methods in predictions.values():
            for ranking in methods.values():
                self.assertEqual(len(ranking), 10)
                self.assertEqual(len(set(ranking)), 10)
                self.assertTrue(set(ranking) <= set(ids))
        self.assertEqual(predictions['warm']['original_text_cbf'], ids[:10])
        self.assertEqual(predictions['cold']['llmrec_encoder'], list(reversed(ids))[:10])
        for methods in predictions.values():
            self.assertEqual(methods['template_text_cbf'], methods['llmrec_encoder'])

    def test_adapter_excludes_future_metadata(self):
        import tempfile
        starter_path = Path(__file__).resolve().parents[1] / 'starter/starter.py'
        spec = importlib.util.spec_from_file_location('encoder_test_starter', starter_path)
        starter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(starter)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'assets.csv'
            path.write_text('ISIN,timestamp,assetName\nA,2020-01-01,Past\nA,2022-01-01,Future\n')
            records = starter.latest_before(path, 'ISIN', '2021-06-01')
            self.assertIn('Past', source_text(records['A']))
            self.assertNotIn('Future', source_text(records['A']))


if __name__ == '__main__':
    unittest.main()
