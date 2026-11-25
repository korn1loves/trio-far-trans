"""Prepare historical prompts, checkpoint Qwen inference, and evaluate paired rankings."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from contextlib import closing, contextmanager
from pathlib import Path
from time import perf_counter

from . import encoder, qwen_backend, run

ROOT = run.ROOT
MARKER = 'Historical aggregates:\n'
BACKENDS = {'cpu': {'float32'}, 'mps': {'float16', 'float32'},
            'cuda': {'float16', 'bfloat16', 'float32'}}


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def serialize(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def atomic_write(path, content):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_prompt(prompt, recent, older, assets, identifiers):
    prefix = encoder.PROMPT.read_text(encoding='utf-8')
    run.require(prompt.startswith(prefix) and prefix.endswith(MARKER),
                'Unexpected prompt prefix')
    payload = json.loads(prompt[len(prefix):])
    run.require(set(payload) == {'recent', 'older'}, 'Unexpected prompt windows')
    for name, counts in [('recent', recent), ('older', older)]:
        window = payload[name]
        run.require(set(window) == {'buy_events', 'distinct_assets', 'histograms', 'missing'},
                    'Unexpected historical aggregate fields')
        run.require((window['buy_events'], window['distinct_assets'])
                    == (sum(counts.values()), len(counts)), 'Aggregate counts differ')
        for field_set in (window['histograms'], window['missing']):
            run.require(set(field_set) == set(encoder.FEATURES), 'Unexpected prompt attributes')
        for field in encoder.FEATURES:
            histogram = window['histograms'][field]
            run.require(sum(histogram.values()) + window['missing'][field] == sum(counts.values()),
                        'Event-weighted histogram does not account for every event')
    # Check literal IDs (case-folded by the starter) and date-shaped values.
    tail = prompt[len(prefix):].casefold()
    run.require(not re.search(r'\b\d{4}-\d{2}-\d{2}\b', tail), 'Date in prompt payload')
    run.require(not any(identifier.casefold() in tail for identifier in identifiers if identifier),
                'Customer ID or ISIN in prompt payload')
    run.require(payload == {'recent': encoder.summarize_window(recent, assets),
                            'older': encoder.summarize_window(older, assets)},
                'Prompt differs from historical aggregates')


def prepare(data_dir, reference, directory):
    started = perf_counter()
    protocol = run.read_json(reference / 'protocol.json')
    run.verify_protocol(protocol, protocol)
    unhashed = {key: value for key, value in protocol.items() if key != 'protocol_hash'}
    run.require(hashlib.sha256(json.dumps(unhashed, sort_keys=True).encode()).hexdigest()
                == protocol['protocol_hash'], 'Invalid reference protocol hash')
    source = {name: digest((data_dir / name).read_bytes())
              for name in protocol['source_sha256']}
    run.require(source == protocol['source_sha256'], 'Source CSV hashes differ')
    assets, histories = encoder.load_histories(data_dir)
    run.require(list(histories) == protocol['users'] and sorted(assets) == protocol['eligible_assets'],
                'Historical cohort or universe differs')
    customers = {row['customerID'] for row in encoder.starter.rows(
        data_dir / 'customer_information.csv')}
    isins = {row['ISIN'] for row in encoder.starter.rows(data_dir / 'asset_information.csv')}
    identifiers = customers | isins
    prompts, mapping = {}, {}
    for uid, (recent, older) in histories.items():
        if recent and older:
            prompt = encoder.model_prompt(recent, older, assets)
            validate_prompt(prompt, recent, older, assets, identifiers)
            case = f'case-{len(prompts) + 1:03d}'
            prompts[case] = prompt
            mapping[case] = uid
    run.require(len(prompts) == run.audit_windows(data_dir, assets, histories) == 35,
                'Expected 35 two-window prompts')
    files = {'prompts.json': serialize(prompts), 'user_mapping.json': serialize(mapping),
             'histories.json': serialize(histories)}
    manifest = {'protocol_hash': protocol['protocol_hash'], 'sources': source,
                'cutoff': encoder.CUTOFF, 'recent_start': encoder.RECENT_START,
                'prepared': 35, 'skipped': 65, 'users': 100,
                'model_config': run.read_json(ROOT / 'model_config.json'),
                'freeze': digest((ROOT / 'freeze.sha256').read_bytes()),
                'prompt_template_sha256': digest(encoder.PROMPT.read_bytes()),
                'prompt_sha256': {case: digest(prompt) for case, prompt in prompts.items()},
                'files': {name: digest(content) for name, content in files.items()}}
    directory.mkdir(parents=True, exist_ok=False)
    for name, contents in files.items():
        atomic_write(directory / name, contents + '\n')
    manifest['prepare_seconds'] = perf_counter() - started
    atomic_write(directory / 'manifest.json', serialize(manifest) + '\n')
    return manifest


def load_preparation(directory):
    manifest = run.read_json(directory / 'manifest.json')
    run.require(manifest['freeze'] == digest((ROOT / 'freeze.sha256').read_bytes())
                and manifest['model_config'] == run.read_json(ROOT / 'model_config.json')
                and manifest['prompt_template_sha256'] == digest(encoder.PROMPT.read_bytes()),
                'Prepared run uses a different frozen contract')
    loaded = []
    for name in ('prompts.json', 'user_mapping.json', 'histories.json'):
        content = (directory / name).read_text(encoding='utf-8').removesuffix('\n')
        run.require(digest(content) == manifest['files'][name], f'Prepared artifact changed: {name}')
        loaded.append(json.loads(content))
    prompts, mapping, histories = loaded
    run.require({case: digest(prompt) for case, prompt in prompts.items()}
                == manifest['prompt_sha256'], 'Prompt fingerprint mismatch')
    two_window = {uid for uid, (recent, older) in histories.items() if recent and older}
    run.require(set(prompts) == set(mapping) and len(mapping) == 35
                and set(mapping.values()) == two_window and len(two_window) == 35
                and len(histories) == 100, 'Prepared user coverage differs')
    return manifest, prompts, mapping, histories


@contextmanager
def exclusive(directory):
    with (directory / 'runtime.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def database(path):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS configuration (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS responses (case_id TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS in_flight (id INTEGER PRIMARY KEY CHECK(id=1), case_id TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS timing (load_seconds REAL, invocation_seconds REAL)')
    return db


def bind_configuration(db, config):
    recorded = db.execute('SELECT value FROM configuration WHERE id=1').fetchone()
    if recorded is None:
        with db:
            db.execute('INSERT INTO configuration VALUES (1, ?)', (serialize(config),))
    else:
        run.require(recorded[0] == serialize(config), 'Incompatible runtime configuration')


def record_call(prompt, generate):
    started = perf_counter()
    raw = parsed = reason = input_tokens = generated_tokens = generation_seconds = None
    try:
        result = generate(prompt)
        raw, input_tokens = result.raw, result.input_tokens
        generated_tokens, generation_seconds = result.generated_tokens, result.generation_seconds
    except encoder.RecoverableInferenceError as error:
        validation, reason = 'inference_error', str(error) or type(error).__name__
        input_tokens = getattr(error, 'input_tokens', None)
        generation_seconds = getattr(error, 'generation_seconds', None)
    else:
        try:
            parsed = encoder.parse_alpha(raw)
            validation = 'valid'
        except ValueError as error:
            validation, reason = 'invalid', str(error)
    return {'status': 'completed' if validation == 'valid' else 'failed',
            'raw_response': raw, 'parsed_alpha': parsed,
            'applied_alpha': parsed if parsed is not None else 0.50,
            'validation_status': validation, 'failure_reason': reason,
            'fallback': parsed is None, 'input_tokens': input_tokens,
            'generated_tokens': generated_tokens, 'generation_seconds': generation_seconds,
            'inference_latency_seconds': perf_counter() - started}


def records(db, prompts):
    result = {case: json.loads(value) for case, value in
              db.execute('SELECT case_id, value FROM responses')}
    for case, row in result.items():
        run.require(case in prompts and row['prompt_sha256'] == digest(prompts[case]),
                    'Unexpected response case or prompt fingerprint')
        validation = row['validation_status']
        run.require(validation in {'valid', 'invalid', 'inference_error'}
                    and row['status'] == ('completed' if validation == 'valid' else 'failed')
                    and row['fallback'] == (validation != 'valid'),
                    'Invalid checkpoint status')
        if validation == 'valid':
            run.require(encoder.parse_alpha(row['raw_response']) == row['parsed_alpha']
                        == row['applied_alpha'], 'Checkpoint alpha mismatch')
        else:
            run.require(row['parsed_alpha'] is None and row['applied_alpha'] == 0.50
                        and bool(row['failure_reason']), 'Invalid checkpoint fallback')
    return result


def process_cases(db, prompts, generate):
    previous = records(db, prompts)
    pending = db.execute('SELECT case_id FROM in_flight WHERE id=1').fetchone()
    if pending:
        run.require(pending[0] in prompts and pending[0] not in previous,
                    'Unexpected in-flight case')
    for case, prompt in prompts.items():
        if case not in previous:
            with db:
                db.execute('INSERT OR REPLACE INTO in_flight VALUES (1, ?)', (case,))
            response = record_call(prompt, generate)
            response['prompt_sha256'] = digest(prompt)
            with db:
                db.execute('INSERT INTO responses VALUES (?, ?)', (case, serialize(response)))
                db.execute('DELETE FROM in_flight WHERE id=1')


def starter_responses(protocol_hash, mapping, histories, responses):
    run.require(set(responses) == set(mapping), 'Incomplete inference checkpoint')
    run.require(any(row['validation_status'] != 'inference_error' for row in responses.values()),
                'No model outputs were generated')
    alphas = {mapping[case]: row['applied_alpha'] for case, row in responses.items()}
    return [{'user_id': uid, 'stage': 'encoder', 'protocol_hash': protocol_hash,
             'history_weights': encoder.history_weights(recent, older, alphas.get(uid, 0.50))}
            for uid, (recent, older) in histories.items()]


def preparation_fingerprint(manifest):
    return digest(serialize(manifest))


def infer(directory, device, dtype, allow_download):
    started = perf_counter()
    with exclusive(directory), closing(database(directory / 'responses.sqlite3')) as db:
        manifest, prompts, mapping, histories = load_preparation(directory)
        run.require(dtype in BACKENDS[device], f'Unsupported dtype {dtype} for {device}')
        config = {'source': 'qwen', 'preparation': preparation_fingerprint(manifest),
                  'device': device, 'dtype': dtype, 'platform': platform.platform(),
                  'python': platform.python_version(),
                  'versions': {package: importlib.metadata.version(package) for package in
                               ('torch', 'transformers', 'huggingface-hub', 'tokenizers', 'safetensors')},
                  'runtime_sha256': {name: digest((ROOT / name).read_bytes())
                                     for name in ('runtime.py', 'qwen_backend.py')}}
        bind_configuration(db, config)
        load_seconds = 0.0
        try:
            if len(records(db, prompts)) < len(prompts):
                loading = perf_counter()
                generate = qwen_backend.load(manifest['model_config'], device, dtype, allow_download)
                load_seconds = perf_counter() - loading
                process_cases(db, prompts, generate)
            responses = records(db, prompts)
            exported = starter_responses(manifest['protocol_hash'], mapping, histories, responses)
            atomic_write(directory / 'llm_responses.jsonl',
                         ''.join(serialize(row) + '\n' for row in exported))
            if all(row['fallback'] for row in responses.values()):
                raise ValueError('Every inference case fell back. Checkpoint retained and evaluation blocked.')
        finally:
            with db:
                db.execute('INSERT INTO timing VALUES (?, ?)',
                           (load_seconds, perf_counter() - started))


def per_user(path, method):
    return {row['user_id']: float(row['ndcg_at_10']) for row in run.read_csv(path / 'per_user.csv')
            if row['method'] == method}


def paired(llm, comparator, users):
    differences = [llm[uid] - comparator[uid] for uid in users]
    return {'users': len(users),
            'llm_mean': sum(llm[uid] for uid in users) / len(users),
            'comparator_mean': sum(comparator[uid] for uid in users) / len(users),
            'mean_difference': sum(differences) / len(users),
            'improved': sum(delta > 0 for delta in differences),
            'worsened': sum(delta < 0 for delta in differences),
            'tied': sum(delta == 0 for delta in differences)}


def evaluate(directory, data_dir, reference, phase2, output_name):
    started = perf_counter()
    manifest, prompts, mapping, histories = load_preparation(directory)
    with exclusive(directory), closing(sqlite3.connect(
            (directory / 'responses.sqlite3').as_uri() + '?mode=ro', uri=True)) as db:
        config = json.loads(db.execute('SELECT value FROM configuration WHERE id=1').fetchone()[0])
        run.require(config['source'] == 'qwen' and config['preparation']
                    == preparation_fingerprint(manifest), 'A real Qwen checkpoint is required')
        responses = records(db, prompts)
        expected = starter_responses(manifest['protocol_hash'], mapping, histories, responses)
        run.require(any(not row['fallback'] for row in responses.values()),
                    'All cases fell back. Evaluation is blocked.')
        weights_path = directory / 'llm_responses.jsonl'
        expected_text = ''.join(serialize(row) + '\n' for row in expected)
        run.require(weights_path.read_text(encoding='utf-8') == expected_text,
                    'Saved response file differs from checkpoint')
        timings = db.execute('SELECT load_seconds, invocation_seconds FROM timing').fetchall()
    protocol = run.read_json(reference / 'protocol.json')
    run.require(protocol['protocol_hash'] == manifest['protocol_hash'], 'Reference protocol differs')
    run.verify_outputs(phase2, reference)
    control_weights = encoder.starter.read_responses(phase2 / 'deterministic_responses.jsonl')
    run.require(set(control_weights) == set(histories), 'Deterministic coverage differs')
    for uid, (recent, older) in histories.items():
        alpha = encoder.purchase_share_alpha(sum(recent.values()), sum(older.values()))
        run.require(control_weights[uid]['history_weights'] == encoder.history_weights(recent, older, alpha),
                    'Deterministic weights differ from the frozen control')
    output = directory / output_name
    output.mkdir(exist_ok=False)
    subprocess.run([sys.executable, str(ROOT.parents[1] / 'starter' / 'starter.py'), 'run',
                    '--data-dir', str(data_dir), '--output-dir', str(output),
                    '--cutoff', encoder.CUTOFF, '--horizon-days', str(encoder.HORIZON),
                    '--cohort', 'warm', '--max-users', str(encoder.COHORT_SIZE),
                    '--candidate-count', '50', '--stage', 'encoder',
                    '--responses', str(weights_path), '--on-invalid', 'error'], check=True)
    metrics = run.verify_outputs(output, reference)
    users = protocol['users']
    nonempty = {row['user_id'] for row in run.read_csv(output / 'per_user.csv')
                if row['method'] == 'custom' and int(row['relevant_count'])}
    groups = {'all': users, 'nonempty_truth': [uid for uid in users if uid in nonempty],
              'two_window': list(mapping.values())}
    llm = per_user(output, 'custom')
    comparators = {'deterministic': per_user(phase2, 'custom'), 'cbf': per_user(output, 'cbf')}
    summary = {'protocol_hash': manifest['protocol_hash'], 'llm_metrics': metrics['methods']['custom'],
               'paired': {method: {group: paired(llm, scores, cohort)
                                   for group, cohort in groups.items()}
                          for method, scores in comparators.items()},
               'response_status': dict(Counter(row['validation_status'] for row in responses.values())),
               'selected_alpha': dict(Counter(row['parsed_alpha'] for row in responses.values()
                                              if row['parsed_alpha'] is not None)),
               'applied_alpha': dict(Counter(row['applied_alpha'] for row in responses.values())),
               'fallback_count': sum(row['fallback'] for row in responses.values()),
               'model_load_seconds': sum(row[0] for row in timings),
               'generation_seconds': sum(row['generation_seconds'] or 0 for row in responses.values()),
               'inference_invocation_seconds': sum(row[1] for row in timings),
               'preparation_seconds': manifest['prepare_seconds'],
               'evaluation_seconds': perf_counter() - started,
               'timing_scope': 'Wall time. Sum of preparation, recorded inference invocations and '
                               'evaluation, excluding idle gaps, dependency installation, abrupt '
                               'terminations and summary writing. Load includes any allowed download.'}
    summary['end_to_end_seconds'] = (summary['preparation_seconds']
                                     + summary['inference_invocation_seconds']
                                     + summary['evaluation_seconds'])
    atomic_write(output / 'llm_summary.json', json.dumps(summary, indent=2) + '\n')


def directory_name(value):
    if Path(value).name != value or value in ('', '.', '..'):
        raise argparse.ArgumentTypeError('Use a single directory name')
    return value


def progress(directory):
    _, prompts, _, _ = load_preparation(directory)
    path = directory / 'responses.sqlite3'
    responses = {}
    pending = None
    if path.exists():
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
            db.execute('BEGIN')
            responses = records(db, prompts)
            pending = db.execute('SELECT case_id FROM in_flight WHERE id=1').fetchone()
    counts = Counter(row['status'] for row in responses.values())
    return {'completed': counts['completed'], 'failed': counts['failed'],
            'in_flight_or_interrupted': int(pending is not None),
            'not_yet_attempted': len(prompts) - len(responses) - int(pending is not None),
            'single_window_skipped': 65}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'infer', 'status', 'evaluate'):
        command = commands.add_parser(name)
        command.add_argument('--run-name', type=directory_name, required=True)
        if name in ('prepare', 'evaluate'):
            command.add_argument('--data-dir', type=Path, required=True)
            command.add_argument('--reference', type=Path, required=True)
        if name == 'infer':
            command.add_argument('--device', choices=BACKENDS, required=True)
            command.add_argument('--dtype', choices=sorted(set.union(*BACKENDS.values())), required=True)
            command.add_argument('--execute-inference', action='store_true', required=True)
            command.add_argument('--allow-download', action='store_true')
        if name == 'evaluate':
            command.add_argument('--phase2', type=Path, required=True)
            command.add_argument('--evaluation-name', type=directory_name, default='evaluation')
    args = parser.parse_args()
    run.check_freeze()
    directory = ROOT / 'outputs' / args.run_name
    try:
        if args.command == 'prepare':
            result = prepare(args.data_dir, args.reference, directory)
            print(serialize({key: result[key] for key in ('prepared', 'skipped', 'users')}))
        elif args.command == 'infer':
            infer(directory, args.device, args.dtype, args.allow_download)
        elif args.command == 'status':
            print(serialize(progress(directory)))
        else:
            evaluate(directory, args.data_dir, args.reference, args.phase2, args.evaluation_name)
    except (ValueError, OSError, RuntimeError, ImportError, sqlite3.Error) as error:
        parser.exit(1, f'Runtime stopped: {error}\n')


if __name__ == '__main__':
    main()
