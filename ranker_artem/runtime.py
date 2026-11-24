"""Pinned Qwen inference; importing, preparing, and injected tests need no ML packages.

Adapted from the instructor's f6850861 runtime. Retrieval, strict parsing, and
fallback remain Artem's fixed latest_buy experiment, never reference CBF retrieval.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time

if __package__:
    from . import adapter, ranker
else:
    import adapter
    import ranker


MODEL_ID = 'Qwen/Qwen3-1.7B'
REVISION = ranker.MODEL_REVISION
CONTEXT_LIMIT = 8192
SEED = 42
GENERATION = {
    'max_new_tokens': 512, 'do_sample': False, 'num_beams': 1,
    'num_return_sequences': 1, 'use_cache': True, 'temperature': 1.0,
    'top_p': 1.0, 'top_k': 0, 'repetition_penalty': 1.0,
}


class GenerationFailure(RuntimeError):
    """Recoverable per-user failure, carrying any evidence collected before failure."""
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


class LocalGenerator:
    """The only code path that imports Transformers/torch or loads model files."""
    def __init__(self, *, device='cuda', precision='auto', allow_download=False):
        adapter.require(device in ('cuda', 'cpu'), 'Choose cuda or cpu explicitly')
        adapter.require(precision in ('auto', 'bfloat16', 'float16', 'float32'), 'Invalid precision')
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable; no automatic CPU fallback')
        if precision == 'auto':
            precision = ('bfloat16' if torch.cuda.is_bf16_supported() else 'float16') if device == 'cuda' else 'float32'
        if device == 'cpu' and precision != 'float32':
            raise ValueError('CPU mode requires float32')
        if device == 'cuda' and precision == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise ValueError('Requested GPU does not support bfloat16; request float16 explicitly')
        self.torch, self.device = torch, device
        self._seed()
        if device == 'cuda':
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        kwargs = {'revision': REVISION, 'local_files_only': not allow_download,
                  'trust_remote_code': False}
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=getattr(torch, precision), **kwargs)
        self.model.to(device).eval()
        resolved_model = getattr(self.model.config, '_commit_hash', None)
        resolved_tokenizer = self.tokenizer.init_kwargs.get('_commit_hash')
        adapter.require(resolved_model == REVISION, 'Loaded model did not resolve to pinned revision')
        adapter.require(resolved_tokenizer in (None, REVISION), 'Tokenizer revision mismatch')
        eos = self.model.generation_config.eos_token_id
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        self.generation_config = GenerationConfig(**GENERATION, eos_token_id=eos, pad_token_id=pad)
        self.load_seconds = time.perf_counter() - started
        hardware = {'platform': platform.platform(), 'machine': platform.machine(),
                    'processor': platform.processor(), 'python': platform.python_version(),
                    'cpu_count': os.cpu_count(), 'device': device}
        if device == 'cuda':
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            hardware.update(gpu_name=props.name, gpu_total_memory_bytes=props.total_memory,
                            cuda_version=torch.version.cuda,
                            compute_capability=list(torch.cuda.get_device_capability()))
        self.metadata = {
            'backend': 'transformers_qwen', 'model': MODEL_ID,
            'model_revision': REVISION, 'resolved_model_revision': resolved_model,
            'tokenizer_revision': REVISION, 'tokenizer_commit_reported_by_library': resolved_tokenizer,
            'tokenizer_revision_resolution': 'from_pretrained called with immutable revision',
            'tokenizer_class': type(self.tokenizer).__name__,
            'chat_template_sha256': adapter.canonical_hash(self.tokenizer.chat_template),
            'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
            'optional_dependency_versions': {
                name: self._installed_version(name) for name in
                ('tokenizers', 'huggingface-hub', 'safetensors', 'numpy')},
            'hardware': hardware, 'precision': precision, 'seed': SEED,
            'enable_thinking': False, 'batch_size': 1, 'context_limit': CONTEXT_LIMIT,
            'effective_generation_config': self.generation_config.to_dict(),
            'eval_mode': True, 'inference_mode': True,
            'cudnn_deterministic': device == 'cuda',
            'deterministic_algorithms_enforced': False,
        }

    @staticmethod
    def _installed_version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def _seed(self):
        random.seed(SEED)
        self.torch.manual_seed(SEED)
        if self.device == 'cuda':
            self.torch.cuda.manual_seed_all(SEED)

    def generate(self, prompt):
        evidence = {'rendered_prompt': None, 'input_tokens': None, 'output_tokens': None,
                    'seconds': None, 'model_generate_called': False,
                    'peak_cuda_memory_allocated_bytes': None}
        started = None
        try:
            rendered = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            evidence['rendered_prompt'] = rendered
            inputs = self.tokenizer(rendered, return_tensors='pt', add_special_tokens=False,
                                    truncation=False)
            n = inputs['input_ids'].shape[1]
            evidence['input_tokens'] = n
            if n + GENERATION['max_new_tokens'] > CONTEXT_LIMIT:
                raise ValueError('Prompt plus 512 reserved output tokens exceeds 8192; candidates are never truncated')
            inputs = inputs.to(self.device)
            # Per-request reset makes cache/resume independent of earlier random state.
            self._seed()
            if self.device == 'cuda':
                self.torch.cuda.synchronize()
                self.torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with self.torch.inference_mode():
                evidence['model_generate_called'] = True
                outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            if self.device == 'cuda':
                self.torch.cuda.synchronize()
                evidence['peak_cuda_memory_allocated_bytes'] = self.torch.cuda.max_memory_allocated()
            evidence['seconds'] = time.perf_counter() - started
            tokens = outputs[0, n:]
            evidence['output_tokens'] = len(tokens)
            adapter.require(n + len(tokens) <= CONTEXT_LIMIT and len(tokens) <= 512,
                            'Backend exceeded fixed generation budget')
            # Decode generated tokens only, never the input prompt.
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            return dict(evidence, text=text)
        except (ValueError, RuntimeError, TimeoutError) as error:
            if started is not None and evidence['seconds'] is None:
                evidence['seconds'] = time.perf_counter() - started
            if self.device == 'cuda':
                self.torch.cuda.empty_cache()
            raise GenerationFailure(f'{type(error).__name__}: {error}', evidence) from error


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


@contextmanager
def exclusive_run(output_dir):
    # macOS/Linux standard library; the OS releases this lock on process death.
    import fcntl
    with (output_dir / 'run.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another process owns this run directory') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def job_specification(bundle, prompts, device, precision, test_only):
    files = sorted(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'README.md']
    return {
        'schema_version': 1, 'author': 'Artem Bondarenko', 'team': 'Trio',
        'execution_kind': 'TEST ONLY' if test_only else 'REAL MODEL',
        'protocol_hash': bundle.protocol['protocol_hash'],
        'candidate_policy': adapter.CANDIDATE_POLICY, 'candidate_count': 50,
        'canonical_ordered_candidate_sha256': adapter.candidate_list_hash(bundle.users, bundle.inputs),
        'prompt_sha256_in_user_order': [adapter.canonical_hash(prompts[uid]) for uid in bundle.users],
        'model': MODEL_ID, 'model_revision': REVISION, 'tokenizer_revision': REVISION,
        'seed': SEED, 'enable_thinking': False, 'batch_size': 1,
        'total_token_budget': CONTEXT_LIMIT, 'generation': GENERATION,
        'device_requested': device, 'precision_requested': precision,
        'control_b': {'weights': ranker.WEIGHTS, 'half_life_days': ranker.HALF_LIFE_DAYS,
                      'tie_breaking': ranker.TIE_POLICY},
        'validation': 'native JSON array of exactly ten distinct original candidate strings',
        'fallback': 'canonical latest_buy first ten; no regeneration or ID repair',
        'code_sha256': {path.name: adapter.sha256_file(path) for path in files},
        'starter_py_sha256': adapter.sha256_file(adapter.STARTER_PATH),
        'baseline_artifact_sha256': bundle.baseline_artifact_sha256,
    }


def request_descriptor(uid, item, prompt, experiment_hash):
    return {'user_id': uid, 'experiment_sha256': experiment_hash,
            'candidate_ids': list(item.candidate_ids), 'prompt': prompt,
            'prompt_sha256': adapter.canonical_hash(prompt)}


def validate_generation(evidence, test_only):
    adapter.require(type(evidence) is dict and type(evidence.get('text')) is str,
                    'Generator must return a dictionary with text')
    adapter.require(type(evidence.get('rendered_prompt')) is str, 'Missing rendered chat prompt')
    for key in ('input_tokens', 'output_tokens'):
        adapter.require(type(evidence.get(key)) is int and evidence[key] >= 0,
                        f'Invalid {key}')
    adapter.require(evidence['input_tokens'] + evidence['output_tokens'] <= CONTEXT_LIMIT
                    and evidence['output_tokens'] <= 512, 'Generator exceeded token budget')
    adapter.require(type(evidence.get('seconds')) in (int, float)
                    and math.isfinite(evidence['seconds']) and evidence['seconds'] >= 0,
                    'Invalid generation time')
    adapter.require(type(evidence.get('model_generate_called')) is bool, 'Missing model call evidence')
    adapter.require(test_only or evidence['model_generate_called'], 'Real backend did not call generate')


def completed_record(request, item, generator, metadata_hash, test_only):
    started = time.perf_counter()
    evidence, inference_error = {}, None
    try:
        evidence = generator.generate(request['prompt'])
        validate_generation(evidence, test_only)
    except (ValueError, RuntimeError, TimeoutError) as error:
        if isinstance(error, GenerationFailure):
            evidence = error.evidence
        elif type(evidence) is not dict:
            evidence = {}
        inference_error = f'{type(error).__name__}: {error}'
    result = ranker.response_or_fallback(
        item, evidence.get('text'), inference_error=inference_error)
    return {'request': request, 'backend_metadata_sha256': metadata_hash,
            'status': 'TEST ONLY' if test_only else 'MODEL RESPONSE',
            'raw_response': evidence.get('text'), 'generation': evidence,
            'request_seconds': time.perf_counter() - started,
            'inference_error': inference_error, **result}


def cache_path(output_dir, uid):
    return output_dir / 'responses' / (adapter.canonical_hash(uid) + '.json')


def cached_record(output_dir, request, item, metadata_hash):
    path = cache_path(output_dir, request['user_id'])
    if not path.exists():
        return None
    envelope = read_json(path)
    record = envelope['record']
    adapter.require(adapter.canonical_hash(record) == envelope['record_sha256'], 'Response cache checksum mismatch')
    adapter.require(record['request'] == request, 'Cached response request mismatch')
    adapter.require(record['backend_metadata_sha256'] == metadata_hash, 'Cached backend mismatch')
    expected = ranker.response_or_fallback(item, record['raw_response'], inference_error=record['inference_error'])
    adapter.require(all(record[key] == value for key, value in expected.items()), 'Cached validation/fallback mismatch')
    return record


def initialize_run(bundle, output_dir, job, prompts, resume):
    job_hash = adapter.canonical_hash(job)
    state_path = output_dir / 'run_state.json'
    if resume:
        state = read_json(state_path)
        adapter.require(state['experiment'] == job and state['experiment_sha256'] == job_hash,
                        'Resume settings, code, protocol, prompts, or candidates mismatch')
        for name, digest in state['prepared_artifact_sha256'].items():
            adapter.require(adapter.sha256_file(output_dir / name) == digest,
                            f'Prepared artifact changed: {name}')
        return state
    # No metric is produced by preparation, and no tokenizer/model is instantiated.
    (output_dir / 'protocol.json').write_bytes((bundle.baseline_dir / 'protocol.json').read_bytes())
    with (output_dir / 'canonical_candidates.jsonl').open('w', encoding='utf-8') as stream:
        for uid in bundle.users:
            stream.write(json.dumps(adapter.candidate_record(uid, bundle.inputs[uid]), ensure_ascii=False) + '\n')
    with (output_dir / 'prompts.jsonl').open('w', encoding='utf-8') as stream:
        for uid in bundle.users:
            stream.write(json.dumps(request_descriptor(uid, bundle.inputs[uid], prompts[uid], job_hash), ensure_ascii=False) + '\n')
    state = {'status': 'PREPARED', 'llm_status': 'NOT RUN', 'completed_users': 0,
             'experiment': job, 'experiment_sha256': job_hash,
             'prepared_artifact_sha256': {name: adapter.sha256_file(output_dir / name) for name in
                                         ('protocol.json', 'canonical_candidates.jsonl', 'prompts.jsonl')},
             'baseline_protocol': bundle.protocol, 'backend_metadata': None,
             'backend_metadata_sha256': None, 'metrics': None, 'last_error': None}
    (output_dir / 'responses').mkdir()
    atomic_json(state_path, state)
    return state


def finalize(bundle, output_dir, state, records, predictions, per_user, metrics, test_only):
    status = 'TEST ONLY' if test_only else 'MEASURED'
    fallback_count = sum(record['fallback'] for record in records)
    c_rows = []
    for uid, record in zip(bundle.users, records):
        ranking = adapter.starter.validate_ranking(record['ranking'], bundle.inputs[uid].candidate_ids)
        predictions[uid][ranker.METHOD_C] = ranking
        relevant = bundle.truth.get(uid, set()) & set(bundle.protocol['eligible_assets'])
        c_rows.append({'user_id': uid, 'method': ranker.METHOD_C, 'relevant_count': len(relevant),
                       'ndcg_at_10': adapter.starter.ndcg10(ranking, relevant)})
    per_user += c_rows
    metrics[ranker.METHOD_C] = adapter.starter.aggregate(c_rows)
    ranker.verify_saved_results(bundle, predictions, per_user, metrics)
    with (output_dir / 'responses.jsonl').open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    ranker.write_csv(output_dir / 'predictions.csv', ['method', 'user_id', 'rank', 'ISIN'],
                     ({'method': method, 'user_id': uid, 'rank': rank, 'ISIN': isin}
                      for uid in bundle.users for method, ranking in predictions[uid].items()
                      for rank, isin in enumerate(ranking, 1)))
    ranker.write_csv(output_dir / 'per_user.csv', ['user_id', 'method', 'relevant_count', 'ndcg_at_10'], per_user)
    failures = [{'user_id': record['request']['user_id'], 'error': record['failure_reason']}
                for record in records if record['fallback']]
    result = {'status': status, 'protocol_hash': bundle.protocol['protocol_hash'],
              'experiment_sha256': state['experiment_sha256'], 'methods': metrics,
              'method_status': {method: status for method in metrics},
              'custom_failures': failures, 'excluded_rows': bundle.excluded_rows,
              'note': ('TEST ONLY: injected generator, not an LLM experiment.' if test_only else
                       'Qwen plus canonical latest-buy fallback; all selected users retained.')}
    atomic_json(output_dir / 'metrics.json', result)
    state.update(status='TEST ONLY' if test_only else 'COMPLETE', llm_status=status,
                 completed_users=len(records), last_error=None,
                 fallback_count=fallback_count,
                 invalid_response_count=sum(record['fallback'] and record['inference_error'] is None for record in records),
                 inference_error_count=sum(record['inference_error'] is not None for record in records),
                 actual_model_calls=(0 if test_only else sum(record['generation'].get('model_generate_called', False) for record in records)),
                 input_tokens=sum(record['generation'].get('input_tokens') or 0 for record in records),
                 output_tokens=sum(record['generation'].get('output_tokens') or 0 for record in records),
                 token_count_missing_users=sum(record['generation'].get('input_tokens') is None or
                                               record['generation'].get('output_tokens') is None for record in records),
                 summed_generation_seconds=sum(record['generation'].get('seconds') or 0 for record in records),
                 timing_scope='per-response original generation times; cached replay is not new inference',
                 metrics=metrics, output_sha256={name: adapter.sha256_file(output_dir / name) for name in
                                                ('responses.jsonl', 'predictions.csv', 'per_user.csv', 'metrics.json')})
    atomic_json(output_dir / 'run_state.json', state)
    atomic_json(output_dir / 'experiment_manifest.json', state)
    return result


def run(bundle, output_dir, *, prepare_only=False, resume=False, generator=None,
        device='cuda', precision='auto', allow_download=False):
    """Injected generators are always TEST ONLY, never measured Qwen results."""
    output_dir = Path(output_dir).resolve()
    root = adapter.OUTPUT_ROOT.resolve()
    adapter.require(root.is_dir() and output_dir != root and output_dir.is_relative_to(root),
                    'Output must be a child of existing starter/outputs/')
    adapter.require(output_dir.is_dir() if resume else not output_dir.exists(),
                    'Resume needs an existing run; new runs require a fresh directory')
    if resume:
        adapter.require((output_dir / 'run_state.json').is_file(),
                        'Resume directory is not a runtime run; existing A/B outputs are protected')
    adapter.require(device in ('cuda', 'cpu') and precision in ('auto', 'bfloat16', 'float16', 'float32'),
                    'Invalid device/precision')
    test_only = generator is not None
    prompts = {uid: ranker.build_prompt(bundle.inputs[uid]) for uid in bundle.users}
    predictions, per_user, metrics = ranker.evaluate(
        bundle.users, bundle.inputs, bundle.baselines, bundle.truth, bundle.protocol['eligible_assets'])
    ranker.verify_saved_results(bundle, predictions, per_user, metrics)
    job = job_specification(bundle, prompts, device, precision, test_only)
    if not resume:
        output_dir.mkdir(parents=True, exist_ok=False)
    with exclusive_run(output_dir):
        state = initialize_run(bundle, output_dir, job, prompts, resume)
        if prepare_only:
            adapter.require(state['status'] == 'PREPARED', 'Preparation cannot relabel an inference run as NOT RUN')
            return state
        if state['backend_metadata'] is not None:
            adapter.require(adapter.canonical_hash(state['backend_metadata']) == state['backend_metadata_sha256'],
                            'Backend metadata checksum mismatch')
        records = {}
        # Validate all durable records BEFORE loading a model or making new calls.
        for uid in bundle.users:
            request = request_descriptor(uid, bundle.inputs[uid], prompts[uid], state['experiment_sha256'])
            record = cached_record(output_dir, request, bundle.inputs[uid], state['backend_metadata_sha256'])
            if record is not None:
                records[uid] = record
        pending = [uid for uid in bundle.users if uid not in records]
        if state['status'] in ('COMPLETE', 'TEST ONLY'):
            adapter.require(not pending, 'Completed run is missing durable responses')
            for name, digest in state['output_sha256'].items():
                adapter.require(adapter.sha256_file(output_dir / name) == digest,
                                f'Completed output changed: {name}')
            return read_json(output_dir / 'metrics.json')
        state.update(status='INCOMPLETE', llm_status='TEST ONLY' if test_only else 'INCOMPLETE',
                     completed_users=len(records), metrics=None)
        atomic_json(output_dir / 'run_state.json', state)
        try:
            if pending:
                if generator is None:
                    generator = LocalGenerator(device=device, precision=precision, allow_download=allow_download)
                metadata = copy.deepcopy(generator.metadata)
                metadata_hash = adapter.canonical_hash(metadata)
                if state['backend_metadata'] is not None:
                    adapter.require(state['backend_metadata'] == metadata, 'Resume backend/hardware/dependency mismatch')
                state.update(backend_metadata=metadata, backend_metadata_sha256=metadata_hash)
                atomic_json(output_dir / 'run_state.json', state)
                with (output_dir / 'invocations.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'status': 'TEST ONLY' if test_only else 'INFERENCE',
                                             'resumed_users': len(records), 'pending_users': len(pending),
                                             'allow_download_requested': allow_download,
                                             'model_load_seconds': getattr(generator, 'load_seconds', None),
                                             'backend_metadata_sha256': metadata_hash}) + '\n')
                for uid in pending:
                    request = request_descriptor(uid, bundle.inputs[uid], prompts[uid], state['experiment_sha256'])
                    record = completed_record(request, bundle.inputs[uid], generator, metadata_hash, test_only)
                    atomic_json(cache_path(output_dir, uid),
                                {'record': record, 'record_sha256': adapter.canonical_hash(record)})
                    records[uid] = record
                    state['completed_users'] = len(records)
                    atomic_json(output_dir / 'run_state.json', state)
            return finalize(bundle, output_dir, state, [records[uid] for uid in bundle.users],
                            predictions, per_user, metrics, test_only)
        except BaseException as error:
            # Interruptions/startup/programming/disk errors are incomplete runs, not
            # synthetic all-user fallbacks. Previously completed requests survive.
            state.update(status='INCOMPLETE', completed_users=len(records), metrics=None,
                         last_error=f'{type(error).__name__}: {error}')
            atomic_json(output_dir / 'run_state.json', state)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'infer'))
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, default=adapter.DEFAULT_BASELINE)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--precision', choices=('auto', 'bfloat16', 'float16', 'float32'), default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-download', action='store_true', help='Explicitly permit the pinned model download during infer')
    args = parser.parse_args(argv)
    if args.command == 'prepare' and args.allow_download:
        parser.error('--allow-download is only valid with infer')
    bundle = adapter.load_experiment(args.data_dir, args.baseline_dir)
    result = run(bundle, args.output_dir, prepare_only=args.command == 'prepare', resume=args.resume,
                 device=args.device, precision=args.precision, allow_download=args.allow_download)
    if args.command == 'prepare':
        print(f'PREPARED: {len(bundle.users)} users; Qwen NOT RUN; tokenizer/model not loaded')
        print(f'experiment_sha256 {result["experiment_sha256"]}')
    else:
        adapter.starter.print_summary(result)
        print(f'experiment_sha256 {result["experiment_sha256"]}')


if __name__ == '__main__':
    main()
