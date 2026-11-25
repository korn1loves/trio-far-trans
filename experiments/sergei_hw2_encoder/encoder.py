"""Historical input and shared weighting for the FAR-Trans Encoder."""
import json
from collections import Counter
from decimal import Decimal, DecimalException
from pathlib import Path

from starter import starter

CUTOFF = '2021-06-01'
RECENT_START = '2021-03-03'
HORIZON = 30
COHORT_SIZE = 100
FEATURES = ('assetCategory', 'assetSubCategory', 'sector', 'industry', 'marketID')
ALLOWED = (0.25, 0.50, 0.75)
PROMPT = Path(__file__).with_name('prompt.txt')


def split_history(events):
    recent, older = Counter(), Counter()
    for timestamp, isin, kind in events:
        if timestamp < CUTOFF and kind == 'Buy':
            window = recent if timestamp >= RECENT_START else older
            window[isin] += 1
    return recent, older


def load_histories(data_dir):
    assets, profiles, history, _, _, _ = starter.load_data(data_dir, CUTOFF, HORIZON)
    users = starter.select_users(profiles, history, 'warm', COHORT_SIZE, None)
    events = starter.scan_recent_events(data_dir, CUTOFF, assets)
    windows = {uid: split_history(events[uid]) for uid in users}
    for uid, (recent, older) in windows.items():
        if recent + older != history[uid]:
            raise ValueError('Window counts disagree with starter Buy history')
    return assets, windows


def purchase_share_alpha(recent_count, older_count):
    # Compare the exact share to 3/8 and 5/8, the midpoints of the grid.
    total = recent_count + older_count
    if 8 * recent_count < 3 * total:
        return 0.25
    if 8 * recent_count > 5 * total:
        return 0.75
    return 0.50


def history_weights(recent, older, alpha):
    weights = {}
    windows = [(recent, alpha), (older, 1 - alpha)] if recent and older else [
        (recent or older, 1.0)]
    for counts, mixture in windows:
        total = sum(counts.values())
        for isin, count in counts.items():
            weights[isin] = weights.get(isin, 0.0) + mixture * (count / total)
    return weights


def summarize_window(counts, assets):
    histograms = {field: Counter() for field in FEATURES}
    missing = dict.fromkeys(FEATURES, 0)
    for isin, count in counts.items():
        # Use the scorer's metadata normalization and missing-value rules.
        attributes = dict(token.split('=', 1) for token in starter.tokens(assets[isin]))
        for field in FEATURES:
            if field in attributes:
                histograms[field][attributes[field]] += count
            else:
                missing[field] += count
    return {'buy_events': sum(counts.values()), 'distinct_assets': len(counts),
            'histograms': histograms, 'missing': missing}


def model_prompt(recent, older, assets):
    payload = {'recent': summarize_window(recent, assets),
               'older': summarize_window(older, assets)}
    return PROMPT.read_text(encoding='utf-8') + json.dumps(payload, sort_keys=True)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def reject_constant(_):
    raise ValueError('non-finite JSON number')


def parse_alpha(response):
    try:
        obj = json.loads(response, object_pairs_hook=unique_object, parse_float=Decimal,
                         parse_int=Decimal, parse_constant=reject_constant)
    except (json.JSONDecodeError, DecimalException, RecursionError, UnicodeError, TypeError) as error:
        raise ValueError('malformed JSON') from error
    if not isinstance(obj, dict) or set(obj) != {'alpha'}:
        raise ValueError('expected one JSON field: alpha')
    alpha = obj['alpha']
    if not isinstance(alpha, Decimal) or alpha not in ALLOWED:
        raise ValueError('alpha must be 0.25, 0.50, or 0.75')
    return float(alpha)


class RecoverableInferenceError(Exception):
    """A future backend should translate recoverable call failures to this type."""


def llm_weights(recent, older, assets, infer):
    """Return (weights, failure_reason); keep the callback out of single-window cases."""
    if not recent or not older:
        return history_weights(recent, older, 0.50), None
    prompt = model_prompt(recent, older, assets)
    try:
        response = infer(prompt)
    except RecoverableInferenceError as error:
        return history_weights(recent, older, 0.50), f'inference failure: {error}'
    try:
        alpha = parse_alpha(response)
    except ValueError as error:
        return history_weights(recent, older, 0.50), f'invalid response: {error}'
    return history_weights(recent, older, alpha), None
