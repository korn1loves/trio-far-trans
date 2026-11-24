"""Artem Bondarenko, Team Trio: frozen latest-buy experiment data adapter.

All data policy and classical baseline algorithms belong to the unmodified starter.
RankerInput is the only object supplied to ranking/prompt functions; it has no truth,
customer identifier, profile, or access to the dataset.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STARTER_PATH = ROOT / 'starter' / 'starter.py'
OUTPUT_ROOT = ROOT / 'starter' / 'outputs'
DEFAULT_BASELINE = OUTPUT_ROOT / 'session2_baseline'
CUTOFF = '2021-06-01'
CANDIDATE_COUNT = 50
BASELINE_METHODS = ('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell')
EXPECTED_PROTOCOL_HASH = '50efd4aeb415ba6f0c66dd0793a7c2033cdc32a42b86b58eebbca3e81eaf3417'
CANDIDATE_POLICY = 'latest_buy_top50_popularity_fill'

_spec = importlib.util.spec_from_file_location('provided_far_trans_starter', STARTER_PATH)
starter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(starter)


def require(condition, message):
    """A runtime assertion that remains active under python -O."""
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    """SHA256 of UTF-8 JSON: sorted keys, no whitespace, ensure_ascii=False."""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class Candidate:
    isin: str
    name: str
    metadata_timestamp: str
    personal_buy_count: int
    last_buy_date: str | None
    global_distinct_buyers: int


@dataclass(frozen=True)
class RankerInput:
    cutoff: str
    candidates: tuple[Candidate, ...]

    def __post_init__(self):
        date.fromisoformat(self.cutoff)
        require(type(self.candidates) is tuple, 'Candidates must be an immutable tuple')
        require(len(self.candidates) == CANDIDATE_COUNT, 'Require exactly 50 candidates')
        ids = []
        for candidate in self.candidates:
            require(type(candidate) is Candidate, 'Require Candidate records')
            require(type(candidate.isin) is str and bool(candidate.isin), 'Require original string IDs')
            require(type(candidate.name) is str, 'Asset name must be text')
            require(date.fromisoformat(candidate.metadata_timestamp[:10]) < date.fromisoformat(self.cutoff),
                    'Candidate metadata must precede cutoff')
            for count in (candidate.personal_buy_count, candidate.global_distinct_buyers):
                require(type(count) is int and count >= 0, 'Counts must be nonnegative integers')
            if candidate.personal_buy_count:
                require(candidate.last_buy_date is not None, 'Positive Buy count has no Buy date')
                require(date.fromisoformat(candidate.last_buy_date) < date.fromisoformat(self.cutoff),
                        'Latest Buy date must precede cutoff')
            else:
                require(candidate.last_buy_date is None, 'Zero Buy count must have no Buy date')
            ids.append(candidate.isin)
        require(len(set(ids)) == CANDIDATE_COUNT, 'Candidate IDs must be unique')

    @property
    def candidate_ids(self):
        return tuple(candidate.isin for candidate in self.candidates)


def build_inputs(contexts, assets, history, pop, events, cutoff=CUTOFF):
    """Build canonical immutable inputs from the starter's historical outputs.

    Reject unsanitized events rather than quietly trusting recency_rankings, which
    expects its caller to have applied the temporal policy already.
    """
    for row in assets.values():
        require(row['timestamp'][:10] < cutoff, 'Future asset metadata supplied')
    for user_events in events.values():
        for timestamp, isin, _kind in user_events:
            date.fromisoformat(timestamp)
            require(timestamp < cutoff, 'Future event supplied to candidate adapter')
            require(isin in assets, 'Event asset is outside eligible universe')
    top50 = starter.recency_rankings(contexts, pop, events, k=CANDIDATE_COUNT)
    top10 = starter.recency_rankings(contexts, pop, events, k=10)
    inputs = {}
    for uid, context in contexts.items():
        require(context['cutoff'] == cutoff, 'Context cutoff mismatch')
        allowed = set(context['eligible_asset_ids'])
        require(allowed <= set(assets) and len(allowed) >= CANDIDATE_COUNT,
                'Require at least 50 eligible assets with as-of metadata')
        counts, latest = Counter(), {}
        for timestamp, isin, kind in events.get(uid, ()):
            if kind == 'Buy':
                counts[isin] += 1
                latest[isin] = max(timestamp, latest.get(isin, timestamp))
        require(dict(counts) == dict(history.get(uid, {})), 'Buy counts and event dates disagree')
        ids = top50[uid]['latest_buy']
        require(len(ids) == len(set(ids)) == CANDIDATE_COUNT and set(ids) <= allowed,
                'Canonical candidates must be 50 distinct eligible IDs')
        require(ids[:10] == top10[uid]['latest_buy'], 'Latest-buy Top-10 parity failed')
        records = tuple(Candidate(
            isin=isin,
            name=('unknown' if assets[isin].get('assetName') in (None, '', 'Not_Available', 'nan')
                  else assets[isin]['assetName']),
            metadata_timestamp=assets[isin]['timestamp'],
            personal_buy_count=counts[isin],
            last_buy_date=latest.get(isin),
            global_distinct_buyers=pop[isin],
        ) for isin in ids)
        inputs[uid] = RankerInput(cutoff, records)
    return inputs


def candidate_list_hash(users, inputs):
    """Bind cohort order and every user's ordered original IDs (not just sets)."""
    return canonical_hash([[uid, list(inputs[uid].candidate_ids)] for uid in users])


@dataclass
class ExperimentData:
    users: list[str]
    inputs: dict[str, RankerInput]
    baselines: dict
    truth: dict
    protocol: dict
    excluded_rows: dict
    saved_metrics: dict
    saved_per_user: dict
    baseline_dir: Path
    baseline_artifact_sha256: dict


def load_experiment(data_dir, baseline_dir=DEFAULT_BASELINE):
    """Load only the approved real-data protocol; verify against saved artifacts."""
    data_dir, baseline_dir = Path(data_dir), Path(baseline_dir)
    protocol = json.loads((baseline_dir / 'protocol.json').read_text(encoding='utf-8'))
    unhashed = dict(protocol)
    digest = unhashed.pop('protocol_hash')
    require(hashlib.sha256(json.dumps(unhashed, sort_keys=True).encode()).hexdigest() == digest,
            'Saved baseline protocol hash is invalid')
    require(digest == EXPECTED_PROTOCOL_HASH, 'Not the approved Session 1 baseline protocol')
    require(protocol['cutoff'] == CUTOFF and protocol['end_exclusive'] == '2021-07-01'
            and protocol['candidate_count'] == 50 and protocol['k'] == 10
            and protocol['cohort_selection'] == 'warm' and not protocol['exclude_seen'],
            'Baseline settings disagree with fixed experiment')
    for name, expected in protocol['source_sha256'].items():
        require(sha256_file(data_dir / name) == expected, f'Source hash mismatch: {name}')
    assets, profiles, history, truth, end, excluded = starter.load_data(data_dir, CUTOFF, 30)
    users = starter.select_users(profiles, history, 'warm', 100, None)
    require(users == protocol['users'] and len(users) == 100, 'Frozen 100-user cohort mismatch')
    require(sorted(assets) == protocol['eligible_assets'] and len(assets) == 767,
            'Frozen 767-asset universe mismatch')
    require(end == protocol['end_exclusive'], 'Evaluation window mismatch')
    features = {isin: starter.tokens(row) for isin, row in assets.items()}
    pop = starter.popularity_scores(history, assets)
    contexts, baselines = starter.contexts_for(
        users, assets, profiles, history, features, pop, CUTOFF, 50, False)
    events = starter.scan_recent_events(data_dir, CUTOFF, assets)
    recent = starter.recency_rankings(contexts, pop, events, k=10)
    inputs = build_inputs(contexts, assets, history, pop, events)
    saved_predictions = {}
    for row in starter.rows(baseline_dir / 'predictions.csv'):
        key = (row['user_id'], row['method'])
        saved_predictions.setdefault(key, []).append((int(row['rank']), row['ISIN']))
    require(set(saved_predictions) == {(uid, method) for uid in users for method in BASELINE_METHODS},
            'Saved baseline prediction coverage mismatch')
    for uid in users:
        baselines[uid].update(recent[uid])
        for method in BASELINE_METHODS:
            starter.validate_ranking(baselines[uid][method], contexts[uid]['eligible_asset_ids'])
            saved = sorted(saved_predictions[(uid, method)])
            require([rank for rank, _isin in saved] == list(range(1, 11)), 'Invalid saved ranks')
            require(baselines[uid][method] == [isin for _rank, isin in saved],
                    f'Saved baseline ranking mismatch: {uid} {method}')
        require(list(inputs[uid].candidate_ids[:10]) == baselines[uid]['latest_buy'],
                'Canonical first ten differ from saved latest-buy baseline')
    saved_metrics = json.loads((baseline_dir / 'metrics.json').read_text(encoding='utf-8'))
    require(saved_metrics['protocol_hash'] == digest and saved_metrics['excluded_rows'] == excluded,
            'Saved metrics provenance mismatch')
    saved_per_user = {}
    for row in starter.rows(baseline_dir / 'per_user.csv'):
        key = (row['user_id'], row['method'])
        require(key not in saved_per_user, 'Duplicate saved per-user score')
        saved_per_user[key] = (int(row['relevant_count']), float(row['ndcg_at_10']))
    require(set(saved_per_user) == set(saved_predictions), 'Saved per-user coverage mismatch')
    require(sum(bool(truth.get(uid)) for uid in users) == 11, 'Expected 11 nonempty truth users')
    artifact_names = ('protocol.json', 'metrics.json', 'predictions.csv', 'per_user.csv', 'contexts.jsonl')
    return ExperimentData(
        users, inputs, baselines, truth, protocol, excluded, saved_metrics, saved_per_user,
        baseline_dir, {name: sha256_file(baseline_dir / name) for name in artifact_names})


def candidate_record(uid, ranker_input):
    """Local audit artifact; user linkage is never part of build_prompt's input."""
    return {'user_id': uid, 'candidate_ids': list(ranker_input.candidate_ids),
            'candidates': [asdict(candidate) for candidate in ranker_input.candidates]}
