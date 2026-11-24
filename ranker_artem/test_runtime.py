"""Runtime contracts using injected doubles only; never Qwen experiment results."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

if __package__:
    from . import adapter, ranker, runtime
    from .test_ranker import IDS, make_input
else:
    import adapter
    import ranker
    import runtime
    from test_ranker import IDS, make_input


class FakeGenerator:
    metadata = {'backend': 'fake_test_not_model', 'version': 1}

    def __init__(self, actions=()):
        self.actions = iter(actions)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        action = next(self.actions, json.dumps(list(IDS[40:50])))
        if isinstance(action, BaseException):
            raise action
        if isinstance(action, dict):
            return action
        return {'text': action, 'rendered_prompt': 'TEST CHAT\n' + prompt,
                'input_tokens': 10, 'output_tokens': 20, 'seconds': 0.0,
                'model_generate_called': False}


class RuntimeOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='runtime_contract_', dir=adapter.OUTPUT_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        baseline = self.root / 'synthetic_baseline'
        baseline.mkdir()
        users = [f'SECRET_USER_{i}' for i in range(100)]
        inputs = {uid: make_input() for uid in users}
        baselines = {uid: {method: ranker.control_a(inputs[uid]) for method in adapter.BASELINE_METHODS}
                     for uid in users}
        truth = {uid: {IDS[0]} for uid in users[:11]}
        _predictions, rows, metrics = ranker.evaluate(users, inputs, baselines, truth, IDS[:60])
        protocol = {'protocol_hash': 'SYNTHETIC_TEST_PROTOCOL', 'users': users,
                    'cutoff': adapter.CUTOFF, 'eligible_assets': list(IDS[:60])}
        adapter.starter.json_file(baseline / 'protocol.json', protocol)
        self.bundle = adapter.ExperimentData(
            users, inputs, baselines, truth, protocol, {},
            {'methods': {method: metrics[method] for method in adapter.BASELINE_METHODS}},
            {(row['user_id'], row['method']): (row['relevant_count'], row['ndcg_at_10'])
             for row in rows if row['method'] in adapter.BASELINE_METHODS},
            baseline, {'protocol.json': adapter.sha256_file(baseline / 'protocol.json')})
        self.out = self.root / 'run'

    def run_fake(self, fake=None, **kwargs):
        return runtime.run(self.bundle, self.out, generator=fake or FakeGenerator(), **kwargs)

    def read(self, name):
        return runtime.read_json(self.out / name)

    def test_prepare_never_constructs_backend_or_writes_metrics(self):
        with patch.object(runtime, 'LocalGenerator', side_effect=AssertionError('must not load')):
            state = runtime.run(self.bundle, self.out, prepare_only=True)
        self.assertEqual(state['llm_status'], 'NOT RUN')
        self.assertEqual(state['status'], 'PREPARED')
        self.assertFalse((self.out / 'metrics.json').exists())
        self.assertFalse((self.out / 'predictions.csv').exists())
        prompts = [json.loads(line) for line in (self.out / 'prompts.jsonl').read_text().splitlines()]
        self.assertEqual(len(prompts), 100)
        for request in prompts:
            self.assertEqual(len(request['candidate_ids']), 50)
            self.assertNotIn('SECRET_USER', request['prompt'])
            self.assertNotIn('truth', request['prompt'])

    def test_fake_success_has_all_users_and_seven_methods_but_test_only_status(self):
        fake = FakeGenerator()
        before = copy.deepcopy(self.bundle.baselines)
        result = self.run_fake(fake)
        self.assertEqual(len(fake.prompts), 100)
        self.assertEqual(result['status'], 'TEST ONLY')
        self.assertEqual(set(result['method_status'].values()), {'TEST ONLY'})
        self.assertEqual(len(result['methods']), 7)
        for metrics in result['methods'].values():
            self.assertEqual((metrics['users'], metrics['nonempty_truth_users'], metrics['empty_truth_users']), (100, 11, 89))
        self.assertEqual(self.bundle.baselines, before)
        with (self.out / 'predictions.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 7000)
        with (self.out / 'per_user.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 700)
        responses = [json.loads(line) for line in (self.out / 'responses.jsonl').read_text().splitlines()]
        self.assertEqual([r['request']['user_id'] for r in responses], self.bundle.users)
        self.assertTrue(all(r['ranking'] == list(IDS[40:50]) for r in responses))
        self.assertEqual(self.read('experiment_manifest.json')['actual_model_calls'], 0)

    def test_invalid_json_duplicate_and_noncandidate_all_fallback_without_dropping_users(self):
        invalid = ['invalid', json.dumps([IDS[0]] * 10), json.dumps(list(IDS[:9]) + [IDS[60]])]
        result = self.run_fake(FakeGenerator(invalid))
        self.assertEqual(len(result['custom_failures']), 3)
        self.assertEqual(self.read('run_state.json')['invalid_response_count'], 3)
        for uid in self.bundle.users[:3]:
            record = runtime.read_json(runtime.cache_path(self.out, uid))['record']
            self.assertTrue(record['fallback'])
            self.assertEqual(record['ranking'], ranker.control_a(self.bundle.inputs[uid]))
            self.assertTrue(record['failure_reason'])
        self.assertEqual(result['methods'][ranker.METHOD_C]['users'], 100)

    def test_recoverable_inference_errors_preserve_evidence_and_fallback(self):
        failure = runtime.GenerationFailure('test OOM', {'input_tokens': 12, 'rendered_prompt': 'TEST',
                                                        'model_generate_called': False})
        self.run_fake(FakeGenerator([failure, RuntimeError('test runtime'), TimeoutError('test timeout')]))
        state = self.read('run_state.json')
        self.assertEqual(state['inference_error_count'], 3)
        self.assertEqual(state['fallback_count'], 3)
        first = runtime.read_json(runtime.cache_path(self.out, self.bundle.users[0]))['record']
        self.assertEqual(first['generation']['input_tokens'], 12)
        self.assertIn('test OOM', first['failure_reason'])

    def test_generator_contract_budget_error_is_logged_as_failure(self):
        bad = FakeGenerator().generate('test')
        bad['output_tokens'] = 513
        self.run_fake(FakeGenerator([bad]))
        record = runtime.read_json(runtime.cache_path(self.out, self.bundle.users[0]))['record']
        self.assertTrue(record['fallback'])
        self.assertIn('token budget', record['failure_reason'])

    def test_interrupt_resume_only_generates_missing_users_and_keeps_first_response(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_fake(FakeGenerator(['invalid', KeyboardInterrupt()]))
        self.assertEqual(self.read('run_state.json')['completed_users'], 1)
        self.assertFalse((self.out / 'metrics.json').exists())
        first_path = runtime.cache_path(self.out, self.bundle.users[0])
        original = first_path.read_bytes()
        fake = FakeGenerator()
        result = self.run_fake(fake, resume=True)
        self.assertEqual(len(fake.prompts), 99)
        self.assertEqual(first_path.read_bytes(), original)
        self.assertEqual(len(result['custom_failures']), 1)

    def test_completed_resume_replays_without_new_calls_or_backend_construction(self):
        result = self.run_fake()
        fake = FakeGenerator()
        with patch.object(runtime, 'LocalGenerator', side_effect=AssertionError('must not load')):
            replay = self.run_fake(fake, resume=True)
        self.assertEqual(fake.prompts, [])
        self.assertEqual(replay['methods'], result['methods'])

    def test_real_prepare_can_resume_with_same_settings_without_loading_in_prepare(self):
        first = runtime.run(self.bundle, self.out, prepare_only=True)
        with patch.object(runtime, 'LocalGenerator', side_effect=AssertionError('must not load')):
            second = runtime.run(self.bundle, self.out, prepare_only=True, resume=True)
        self.assertEqual(first, second)

    def test_fake_and_real_jobs_cannot_be_mixed_on_resume(self):
        self.run_fake(prepare_only=True)
        with patch.object(runtime, 'LocalGenerator') as loader, self.assertRaisesRegex(ValueError, 'Resume settings'):
            runtime.run(self.bundle, self.out, resume=True)
        loader.assert_not_called()

    def test_changed_candidate_features_or_prompt_reject_resume_before_calls(self):
        self.run_fake(prepare_only=True)
        uid = self.bundle.users[0]
        self.bundle.inputs[uid] = make_input({IDS[0]: {'name': 'Changed as-of name'}})
        fake = FakeGenerator()
        with self.assertRaisesRegex(ValueError, 'Resume settings'):
            self.run_fake(fake, resume=True)
        self.assertEqual(fake.prompts, [])

    def test_changed_precision_or_code_rejects_resume(self):
        self.run_fake(prepare_only=True)
        with self.assertRaisesRegex(ValueError, 'Resume settings'):
            self.run_fake(resume=True, precision='float16')
        with patch.object(runtime, 'GENERATION', dict(runtime.GENERATION, max_new_tokens=511)):
            with self.assertRaisesRegex(ValueError, 'Resume settings'):
                self.run_fake(resume=True)

    def test_changed_backend_metadata_rejects_pending_resume(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_fake(FakeGenerator([json.dumps(list(IDS[:10])), KeyboardInterrupt()]))
        fake = FakeGenerator()
        fake.metadata = {'backend': 'different_test_backend'}
        with self.assertRaisesRegex(ValueError, 'backend/hardware/dependency mismatch'):
            self.run_fake(fake, resume=True)
        self.assertEqual(fake.prompts, [])

    def test_corrupted_response_cache_rejected_without_regeneration(self):
        self.run_fake()
        path = runtime.cache_path(self.out, self.bundle.users[0])
        envelope = runtime.read_json(path)
        envelope['record']['raw_response'] = 'tampered'
        runtime.atomic_json(path, envelope)
        fake = FakeGenerator()
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.run_fake(fake, resume=True)
        self.assertEqual(fake.prompts, [])

    def test_prepared_artifact_corruption_rejected(self):
        self.run_fake(prepare_only=True)
        (self.out / 'prompts.jsonl').write_text('tampered', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Prepared artifact changed'):
            self.run_fake(resume=True)

    def test_startup_failure_is_incomplete_not_100_fabricated_fallbacks(self):
        with patch.object(runtime, 'LocalGenerator', side_effect=RuntimeError('not installed')):
            with self.assertRaisesRegex(RuntimeError, 'not installed'):
                runtime.run(self.bundle, self.out)
        self.assertEqual(self.read('run_state.json')['status'], 'INCOMPLETE')
        self.assertEqual(self.read('run_state.json')['completed_users'], 0)
        self.assertFalse((self.out / 'metrics.json').exists())
        self.assertEqual(list((self.out / 'responses').glob('*.json')), [])

    def test_unexpected_programming_error_is_not_swallowed_as_model_failure(self):
        with self.assertRaisesRegex(KeyError, 'programming'):
            self.run_fake(FakeGenerator([KeyError('programming')]))
        self.assertFalse((self.out / 'metrics.json').exists())
        self.assertEqual(self.read('run_state.json')['completed_users'], 0)

    def test_persisted_hashes_bind_protocol_candidates_settings_and_responses(self):
        self.run_fake()
        state = self.read('experiment_manifest.json')
        self.assertEqual(adapter.canonical_hash(state['experiment']), state['experiment_sha256'])
        self.assertEqual(state['experiment']['candidate_policy'], 'latest_buy_top50_popularity_fill')
        self.assertEqual(state['experiment']['canonical_ordered_candidate_sha256'],
                         adapter.candidate_list_hash(self.bundle.users, self.bundle.inputs))
        for name, digest in {**state['prepared_artifact_sha256'], **state['output_sha256']}.items():
            self.assertEqual(adapter.sha256_file(self.out / name), digest)
        self.assertEqual((self.out / 'protocol.json').read_bytes(),
                         (self.bundle.baseline_dir / 'protocol.json').read_bytes())

    def test_existing_ab_directory_cannot_be_overwritten(self):
        with self.assertRaises(ValueError):
            runtime.run(self.bundle, self.bundle.baseline_dir, generator=FakeGenerator())
        with self.assertRaises(ValueError):
            runtime.run(self.bundle, adapter.ROOT / 'not_allowed', generator=FakeGenerator())
        before = sorted(self.bundle.baseline_dir.iterdir())
        with self.assertRaisesRegex(ValueError, 'existing A/B outputs are protected'):
            runtime.run(self.bundle, self.bundle.baseline_dir, generator=FakeGenerator(), resume=True)
        self.assertEqual(sorted(self.bundle.baseline_dir.iterdir()), before)

    def test_complete_output_corruption_is_rejected_without_loading_or_regeneration(self):
        self.run_fake()
        (self.out / 'metrics.json').write_text('{}', encoding='utf-8')
        fake = FakeGenerator()
        with self.assertRaisesRegex(ValueError, 'Completed output changed'):
            self.run_fake(fake, resume=True)
        self.assertEqual(fake.prompts, [])

    def test_backend_metadata_checksum_is_checked_on_completed_resume(self):
        self.run_fake()
        state = self.read('run_state.json')
        state['backend_metadata']['version'] = 2
        runtime.atomic_json(self.out / 'run_state.json', state)
        with self.assertRaisesRegex(ValueError, 'Backend metadata checksum'):
            self.run_fake(resume=True)

    def test_concurrent_run_is_rejected_by_os_lock(self):
        self.run_fake(prepare_only=True)
        with runtime.exclusive_run(self.out), self.assertRaisesRegex(ValueError, 'Another process'):
            self.run_fake(resume=True)

    def test_cli_help_and_preparation_import_work_when_ml_packages_are_blocked(self):
        script = (
            "import sys; sys.modules['torch']=None; sys.modules['transformers']=None; "
            "from ranker_artem import runtime; runtime.main(['--help'])"
        )
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=adapter.ROOT,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('prepare', result.stdout)


class FakeBatch(dict):
    def __init__(self, length):
        super().__init__(input_ids=SimpleNamespace(shape=(1, length)))

    def to(self, device):
        return self


class FakeOutput:
    def __init__(self, input_length, generated=(101, 102)):
        self.tokens = [999] * input_length + list(generated)

    def __getitem__(self, key):
        row, token_slice = key
        assert row == 0
        return self.tokens[token_slice]


class FakeGenerationConfig:
    def __init__(self, **kwargs):
        self.values = kwargs

    def to_dict(self):
        return self.values


class LocalGeneratorContractTests(unittest.TestCase):
    """Exercise the real wrapper with stub library objects, no torch/weights."""
    def setUp(self):
        self.tokenizer = Mock()
        self.tokenizer.init_kwargs = {'_commit_hash': runtime.REVISION}
        self.tokenizer.chat_template = 'TEST_TEMPLATE'
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 1
        self.tokenizer.apply_chat_template.return_value = 'TEST_RENDERED_CHAT'
        self.tokenizer.return_value = FakeBatch(4)
        self.tokenizer.decode.return_value = '["test-only"]'
        self.model = Mock()
        self.model.config = SimpleNamespace(_commit_hash=runtime.REVISION)
        self.model.generation_config = SimpleNamespace(eos_token_id=[1, 2])
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model
        self.model.generate.return_value = FakeOutput(4)
        self.torch = SimpleNamespace(
            __version__='TEST_TORCH', float32='TEST_FP32', float16='TEST_FP16', bfloat16='TEST_BF16',
            manual_seed=Mock(), inference_mode=Mock(side_effect=nullcontext),
            cuda=SimpleNamespace(is_available=Mock(return_value=False)))
        self.transformers = SimpleNamespace(
            __version__='TEST_TRANSFORMERS', AutoTokenizer=Mock(), AutoModelForCausalLM=Mock(),
            GenerationConfig=FakeGenerationConfig)
        self.transformers.AutoTokenizer.from_pretrained.return_value = self.tokenizer
        self.transformers.AutoModelForCausalLM.from_pretrained.return_value = self.model
        self.modules = patch.dict(sys.modules, {'torch': self.torch, 'transformers': self.transformers})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def cuda_stubs(self, bf16_supported=True):
        self.torch.cuda = SimpleNamespace(
            is_available=Mock(return_value=True), is_bf16_supported=Mock(return_value=bf16_supported),
            manual_seed_all=Mock(), synchronize=Mock(), reset_peak_memory_stats=Mock(), empty_cache=Mock(),
            max_memory_allocated=Mock(return_value=4096), current_device=Mock(return_value=0),
            get_device_properties=Mock(return_value=SimpleNamespace(name='TEST_GPU', total_memory=8192)),
            get_device_capability=Mock(return_value=(8, 0)))
        self.torch.backends = SimpleNamespace(cudnn=SimpleNamespace(benchmark=True, deterministic=False))
        self.torch.version = SimpleNamespace(cuda='TEST_CUDA')

    def test_model_and_tokenizer_are_both_pinned_and_cached_only_by_default(self):
        backend = runtime.LocalGenerator(device='cpu')
        for loader in (self.transformers.AutoTokenizer, self.transformers.AutoModelForCausalLM):
            args, kwargs = loader.from_pretrained.call_args
            self.assertEqual(args, (runtime.MODEL_ID,))
            self.assertEqual(kwargs['revision'], runtime.REVISION)
            self.assertTrue(kwargs['local_files_only'])
            self.assertFalse(kwargs['trust_remote_code'])
        self.model.eval.assert_called_once()
        self.assertEqual(backend.metadata['precision'], 'float32')
        self.assertEqual(backend.metadata['model_revision'], runtime.REVISION)

    def test_decodes_only_generated_tokens_and_applies_non_thinking_chat(self):
        backend = runtime.LocalGenerator(device='cpu')
        response = backend.generate('user prompt')
        self.tokenizer.apply_chat_template.assert_called_once_with(
            [{'role': 'user', 'content': 'user prompt'}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        self.tokenizer.assert_called_once_with('TEST_RENDERED_CHAT', return_tensors='pt',
                                               add_special_tokens=False, truncation=False)
        self.tokenizer.decode.assert_called_once_with([101, 102], skip_special_tokens=True)
        self.assertEqual((response['input_tokens'], response['output_tokens']), (4, 2))
        self.assertTrue(response['model_generate_called'])
        self.assertGreaterEqual(response['seconds'], 0)
        self.torch.inference_mode.assert_called_once()

    def test_generation_configuration_is_greedy_one_beam_with_fixed_budget(self):
        backend = runtime.LocalGenerator(device='cpu')
        backend.generate('prompt')
        config = self.model.generate.call_args.kwargs['generation_config'].to_dict()
        self.assertEqual(config['max_new_tokens'], 512)
        self.assertFalse(config['do_sample'])
        self.assertEqual(config['num_beams'], 1)
        self.assertEqual(config['num_return_sequences'], 1)
        self.assertEqual(config['eos_token_id'], [1, 2])
        self.assertEqual(self.torch.manual_seed.call_args.args, (42,))
        self.assertEqual(backend.metadata['batch_size'], 1)
        self.assertEqual(backend.metadata['context_limit'], 8192)

    def test_context_overflow_records_token_count_and_never_truncates_or_calls_generate(self):
        self.tokenizer.return_value = FakeBatch(7681)
        backend = runtime.LocalGenerator(device='cpu')
        with self.assertRaises(runtime.GenerationFailure) as raised:
            backend.generate('prompt')
        self.assertEqual(raised.exception.evidence['input_tokens'], 7681)
        self.assertFalse(raised.exception.evidence['model_generate_called'])
        self.model.generate.assert_not_called()

    def test_exact_context_boundary_is_allowed(self):
        self.tokenizer.return_value = FakeBatch(7680)
        self.model.generate.return_value = FakeOutput(7680)
        backend = runtime.LocalGenerator(device='cpu')
        self.assertEqual(backend.generate('prompt')['input_tokens'], 7680)

    def test_download_requires_explicit_constructor_flag(self):
        runtime.LocalGenerator(device='cpu', allow_download=True)
        self.assertFalse(self.transformers.AutoTokenizer.from_pretrained.call_args.kwargs['local_files_only'])
        self.assertFalse(self.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs['local_files_only'])

    def test_cuda_unavailable_fails_without_cpu_fallback_or_model_load(self):
        with self.assertRaisesRegex(ValueError, 'CUDA requested'):
            runtime.LocalGenerator()
        self.transformers.AutoTokenizer.from_pretrained.assert_not_called()
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_cuda_auto_bfloat16_records_hardware_precision_timing_and_memory(self):
        self.cuda_stubs()
        backend = runtime.LocalGenerator()
        response = backend.generate('prompt')
        self.assertEqual(backend.metadata['precision'], 'bfloat16')
        self.assertEqual(backend.metadata['hardware']['gpu_name'], 'TEST_GPU')
        self.assertEqual(response['peak_cuda_memory_allocated_bytes'], 4096)
        self.torch.cuda.manual_seed_all.assert_called_with(42)
        self.torch.cuda.synchronize.assert_called()
        self.assertFalse(self.torch.backends.cudnn.benchmark)
        self.assertTrue(self.torch.backends.cudnn.deterministic)

    def test_cuda_auto_float16_and_explicit_unsupported_bfloat16(self):
        self.cuda_stubs(bf16_supported=False)
        self.assertEqual(runtime.LocalGenerator().metadata['precision'], 'float16')
        with self.assertRaisesRegex(ValueError, 'does not support bfloat16'):
            runtime.LocalGenerator(precision='bfloat16')

    def test_cuda_oom_records_failure_and_releases_cache_without_retry(self):
        self.cuda_stubs()
        self.model.generate.side_effect = RuntimeError('TEST OOM')
        backend = runtime.LocalGenerator()
        with self.assertRaises(runtime.GenerationFailure):
            backend.generate('prompt')
        self.torch.cuda.empty_cache.assert_called_once()
        self.model.generate.assert_called_once()

    def test_revision_mismatch_is_rejected(self):
        self.model.config._commit_hash = 'wrong'
        with self.assertRaisesRegex(ValueError, 'pinned revision'):
            runtime.LocalGenerator(device='cpu')

    def test_tokenizer_revision_mismatch_is_rejected(self):
        self.tokenizer.init_kwargs['_commit_hash'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'Tokenizer revision'):
            runtime.LocalGenerator(device='cpu')

    def test_model_runtime_error_carries_rendered_prompt_and_call_evidence(self):
        self.model.generate.side_effect = RuntimeError('TEST model failure')
        backend = runtime.LocalGenerator(device='cpu')
        with self.assertRaises(runtime.GenerationFailure) as raised:
            backend.generate('prompt')
        self.assertEqual(raised.exception.evidence['rendered_prompt'], 'TEST_RENDERED_CHAT')
        self.assertTrue(raised.exception.evidence['model_generate_called'])


if __name__ == '__main__':
    unittest.main()
