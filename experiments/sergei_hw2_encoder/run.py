"""Run the June classical comparison in a fresh ignored outputs directory."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

from . import encoder

ROOT = Path(__file__).resolve().parent
METHODS = {'cbf', 'popularity', 'latest_buy', 'latest_buy_or_sell'}
TIMING_SCOPE = ('Wall-clock seconds. Adapter includes starter data loading, historical '
                'split, independent raw audit and response export. Starter includes '
                'process launch, data loading, all five rankings, evaluation and '
                'artifact writes. Setup and post-run parity checks are outside both scopes.')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_freeze():
    for line in (ROOT / 'freeze.sha256').read_text().splitlines():
        digest, filename = line.split()
        require(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest,
                f'Frozen design changed: {filename}')


def audit_windows(data_dir, assets, windows):
    """Independently derive both windows from raw rows with explicit date predicates."""
    observed = {uid: (Counter(), Counter()) for uid in windows}
    seen = set()
    with (data_dir / 'transactions.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            uid, isin, stamp = row['customerID'], row['ISIN'], row['timestamp'][:10]
            if (uid not in observed or isin not in assets or row['transactionType'] != 'Buy'
                    or stamp >= encoder.CUTOFF or int(row['transactionID']) < 0):
                continue
            key = tuple(row.values())
            if key in seen:
                continue
            seen.add(key)
            index = 0 if encoder.RECENT_START <= stamp < encoder.CUTOFF else 1
            observed[uid][index][isin] += 1
    require(observed == windows, 'Independent raw transaction audit disagrees with adapter')
    return sum(bool(recent and older) for recent, older in observed.values())


def verify_protocol(protocol, reference):
    require(protocol == reference, 'Full starter protocol differs from June reference')
    require((protocol['cutoff'], protocol['end_exclusive'], protocol['cohort_selection'],
             protocol['exclude_seen'], protocol['candidate_count'], protocol['k'],
             len(protocol['users']), len(protocol['eligible_assets']))
            == ('2021-06-01', '2021-07-01', 'warm', False, 50, 10, 100, 767),
            'Unexpected June protocol settings')


def verify_outputs(output, original):
    protocol = read_json(output / 'protocol.json')
    verify_protocol(protocol, read_json(original / 'protocol.json'))
    metrics = read_json(output / 'metrics.json')
    reference = read_json(original / 'metrics.json')
    require(metrics['protocol_hash'] == protocol['protocol_hash'] == reference['protocol_hash'],
            'Inconsistent protocol hash')
    require(metrics['excluded_rows'] == reference['excluded_rows'], 'Exclusions differ')
    require(not metrics['custom_failures'], 'Starter reported Encoder failures')
    require(set(metrics['methods']) == METHODS | {'custom'}, 'Unexpected methods')
    for method in METHODS:
        require(metrics['methods'][method] == reference['methods'][method],
                f'Aggregate metrics differ: {method}')
    for filename in ('per_user.csv', 'predictions.csv'):
        require([row for row in read_csv(output / filename) if row['method'] in METHODS]
                == read_csv(original / filename), f'Baseline parity failed: {filename}')

    expected_users = set(protocol['users'])
    eligible = set(protocol['eligible_assets'])
    for method, scores in metrics['methods'].items():
        require((scores['users'], scores['nonempty_truth_users'], scores['empty_truth_users'])
                == (100, 11, 89), f'Unexpected truth coverage: {method}')
    per_user = [row for row in read_csv(output / 'per_user.csv') if row['method'] == 'custom']
    require(len(per_user) == 100 and {row['user_id'] for row in per_user} == expected_users,
            'Encoder user coverage differs')
    truth_counts = {row['user_id']: row['relevant_count'] for row in
                    read_csv(original / 'per_user.csv') if row['method'] == 'cbf'}
    require(all(row['relevant_count'] == truth_counts[row['user_id']] for row in per_user),
            'Encoder truth coverage differs from reference')
    rankings = {uid: [] for uid in expected_users}
    for row in read_csv(output / 'predictions.csv'):
        if row['method'] == 'custom':
            ranking = rankings[row['user_id']]
            require(int(row['rank']) == len(ranking) + 1, 'Encoder ranks are not sequential')
            ranking.append(row['ISIN'])
    for ranking in rankings.values():
        encoder.starter.validate_ranking(ranking, eligible)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in ('', '.', '..'):
        parser.error('--run-name must be a single directory name')
    check_freeze()
    reference = read_json(args.reference / 'protocol.json')
    output = ROOT / 'outputs' / args.run_name
    output.mkdir(parents=True, exist_ok=False)

    started = perf_counter()
    assets, windows = encoder.load_histories(args.data_dir)
    require(list(windows) == reference['users'] and sorted(assets) == reference['eligible_assets'],
            'Historical cohort or eligible assets differ from June reference')
    both = audit_windows(args.data_dir, assets, windows)
    require(both == 35, 'Expected 35 two-window users')
    responses = output / 'deterministic_responses.jsonl'
    alpha_counts = Counter()
    with responses.open('w', encoding='utf-8') as stream:
        for uid, (recent, older) in windows.items():
            alpha = encoder.purchase_share_alpha(sum(recent.values()), sum(older.values()))
            if recent and older:
                alpha_counts[alpha] += 1
            weights = encoder.history_weights(recent, older, alpha)
            stream.write(json.dumps({'user_id': uid, 'stage': 'encoder',
                                     'protocol_hash': reference['protocol_hash'],
                                     'history_weights': weights}, sort_keys=True) + '\n')
    adapter_seconds = perf_counter() - started

    started = perf_counter()
    subprocess.run([sys.executable, str(ROOT.parents[1] / 'starter' / 'starter.py'), 'run',
                    '--data-dir', str(args.data_dir), '--output-dir', str(output),
                    '--cutoff', encoder.CUTOFF, '--horizon-days', str(encoder.HORIZON),
                    '--cohort', 'warm', '--max-users', str(encoder.COHORT_SIZE),
                    '--candidate-count', '50', '--stage', 'encoder',
                    '--responses', str(responses), '--on-invalid', 'error'], check=True)
    starter_seconds = perf_counter() - started

    metrics = verify_outputs(output, args.reference)
    methods = metrics['methods']
    methods['deterministic_purchase_share'] = methods.pop('custom')
    summary = {'methods': methods, 'llm': {'status': 'NOT RUN', 'calls': 0, 'failures': None},
               'protocol_hash': metrics['protocol_hash'], 'baseline_parity': True,
               'eligible_assets': len(assets), 'two_window_users': both,
               'single_window_users': len(windows) - both,
               'two_window_alpha_counts': dict(alpha_counts),
               'deterministic_failures': len(metrics['custom_failures']),
               'excluded_rows': metrics['excluded_rows'], 'timing_scope': TIMING_SCOPE,
               'adapter_seconds': adapter_seconds, 'starter_seconds': starter_seconds,
               'combined_timed_seconds': adapter_seconds + starter_seconds,
               'python_version': sys.version.split()[0]}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'Verified outputs: {output}')


if __name__ == '__main__':
    main()
