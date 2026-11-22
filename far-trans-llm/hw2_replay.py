"""Prepare the frozen HW2 Encoder inputs or replay saved Qwen responses locally."""

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import encoder
from data_adapter import load, save_results

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_HASH = "50efd4aeb415ba6f0c66dd0793a7c2033cdc32a42b86b58eebbca3e81eaf3417"
MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
FROZEN_SOURCES = {
    "far-trans-llm/encoder.py": "83733787c65e00cf01b38358db7c918be5a53a3b0e0e020f6ece6a4d177a941c",
    "far-trans-llm/runtime.py": "b807167090355b342bcab54bf39fed6c5a40bf7b74ffd7f1027768490333e5fd",
    "far-trans-llm/data_adapter.py": "83c489e7dbc4e38055a1934844cf4d71bdec333b06d1c116447c745ce9fc7931",
    "far-trans-llm/requirements.txt": "afbb798c7ce286cfde753e5b3ad8b18b67c1758ab66ec1bd6ba70d3beca393cc",
    "starter/starter.py": "c543ff1278cc232bdc815786f56086cb27d15a0b10c7a0335b011554c55eac2a",
}
REFERENCE_HASHES = {
    "protocol.json": "b3cc0f3ac43ddac889a7852e92246271768007b862669f7aef64c1892c6fb8fd",
    "predictions.csv": "ca17710fe93e9223222847d747505dbe07b1337d7e6c7a1ebf7b66c25ba1d858",
    "metrics.json": "35cf281a46ace985652aae05d4b2bc972bca45a3c32016a31d4f294173c2fbbb",
}
ASSET_FIELDS = ("assetName", "assetCategory", "assetSubCategory", "sector", "industry", "marketID")
CONTROLS = ("original_text_cbf", "template_text_cbf")
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 42


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reference(directory):
    for name, expected in REFERENCE_HASHES.items():
        require(sha256(directory / name) == expected, f"Saved starter reference hash mismatch: {name}")
    protocol = read_json(directory / "protocol.json")
    recorded_hash = protocol.pop("protocol_hash")
    structural_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    require(recorded_hash == structural_hash == PROTOCOL_HASH, "Starter protocol hash mismatch")
    print(f"Verified starter protocol and all three reference files: {recorded_hash}")
    return protocol


def frozen_bundle(data_dir, output_dir, protocol):
    for name, expected in FROZEN_SOURCES.items():
        raw = (ROOT / name).read_bytes()
        frozen_bytes = raw.replace(b'\r\n', b'\n')
        if name == "starter/starter.py":
            frozen_bytes = frozen_bytes.replace(b'\n', b'\r\n')
        require(hashlib.sha256(frozen_bytes).hexdigest() == expected,
                f"Frozen source hash mismatch: {name}")
    args = SimpleNamespace(
        data_dir=data_dir, starter_dir=ROOT / "starter", output_dir=output_dir,
        cutoff=protocol["cutoff"], horizon_days=30, max_users=100, candidate_count=50,
    )
    bundle = load(args)
    _, assets, _, _, _, _, _, _, manifest = bundle
    require(
        len(assets) == 767 and sorted(assets) == protocol["eligible_assets"]
        and len(manifest["users"]) == 100 and manifest["users"] == protocol["users"]
        and manifest["cutoff"] == protocol["cutoff"]
        and manifest["end_exclusive"] == protocol["end_exclusive"]
        and manifest["data_sha256"] == protocol["source_sha256"]
        and manifest["candidate_count"] == protocol["candidate_count"]
        and manifest["repeat_allowed"] and not protocol["exclude_seen"]
        and protocol["cohort_selection"] == "warm" and protocol["k"] == 10,
        "Local data or evaluation settings differ from the saved starter protocol",
    )
    return args, bundle


def asset_inputs(assets, protocol):
    return {
        "cutoff": protocol["cutoff"], "end_exclusive": protocol["end_exclusive"],
        "protocol_hash": PROTOCOL_HASH, "asset_metadata_sha256": protocol["source_sha256"]["asset_information.csv"],
        "model": MODEL, "revision": REVISION, "source_sha256": FROZEN_SOURCES,
        "assets": [
            {"asset_id": isin, "as_of_date": assets[isin]["timestamp"],
             "fields": {field: assets[isin][field] for field in ASSET_FIELDS}}
            for isin in protocol["eligible_assets"]
        ],
    }


def read_rankings(path, methods):
    rankings = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["method"] not in methods:
                continue
            key = row["user_id"], row["method"], int(row["rank"])
            require(key not in rankings, f"Duplicate rank in {path}: {key}")
            rankings[key] = row["ISIN"]
    return rankings


def restore_baselines(bundle, saved_ranks):
    starter, assets, history, _, _, baseline, _, _, manifest = bundle
    rebuilt = {(uid, method, rank): isin
               for uid, methods in baseline.items()
               for method, ranking in methods.items()
               for rank, isin in enumerate(ranking, 1)}
    require(len(saved_ranks) == 4000 and rebuilt.keys() == saved_ranks.keys(),
            "Saved and reconstructed starter ranking coverage differs")
    differences = {}
    for key, fresh in rebuilt.items():
        if fresh != saved_ranks[key]:
            uid, method, rank = key
            require(method == "cbf", f"Unexpected difference in starter {method}")
            differences.setdefault(uid, []).append((rank, fresh, saved_ranks[key]))

    features = {isin: starter.tokens(asset) for isin, asset in assets.items()}
    checked = []
    for uid, rows in differences.items():
        scores = starter.content_scores(history[uid], features)
        for rank, fresh, saved in rows:
            gap = abs(scores[fresh] - scores[saved])
            require(gap < 1e-12, "Reconstructed CBF differs beyond the near-tie tolerance")
            checked.append({"user_id": uid, "rank": rank, "fresh_isin": fresh,
                            "saved_isin": saved, "absolute_score_gap": gap})
    for uid in manifest["users"]:
        baseline[uid]["cbf"] = [saved_ranks[uid, "cbf", rank] for rank in range(1, 11)]
    return checked


class SavedResponses:
    def __init__(self, responses):
        self.records = iter(sorted(responses.items()))
        self.used = []

    def generate(self, prompt, max_new_tokens=160):
        asset_id, row = next(self.records)
        descriptor = {"prompt": prompt, "max_new_tokens": max_new_tokens}
        key = hashlib.sha256(json.dumps(descriptor, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        require(prompt == row["prompt"] and max_new_tokens == 160 and key == row["cache_key"],
                f"Saved prompt or cache key mismatch: {asset_id}")
        self.used.append(asset_id)
        return row["response"]


def replay_views(documents, export):
    replay = SavedResponses(export["responses"])
    qwen, templates, failures, audit = encoder.generate_views(documents, replay, "fallback")
    expected_ids = [i for i, row in audit.items() if row["actual_word_count"] is not None]
    require(replay.used == expected_ids == sorted(export["responses"]), "Saved response coverage mismatch")
    require(audit == export["audit"] and failures == export["failures"], "Colab audit did not reproduce")
    for failure in failures:
        i = failure["asset_id"]
        require(qwen[i] == templates[i] == documents[i], f"Shared fallback mismatch: {i}")
    return qwen, templates, failures


def ranking_comparison(predictions, truth):
    counts = Counter()
    for uid, methods in predictions.items():
        qwen, template = methods["llmrec_encoder"], methods["template_text_cbf"]
        if qwen != template:
            counts["different_ordered_top_10"] += 1
            counts["different_ordered_nonempty_truth"] += bool(truth.get(uid))
            if set(qwen) != set(template):
                counts["different_top_10_membership"] += 1
                counts["different_membership_nonempty_truth"] += bool(truth.get(uid))
            else:
                counts["different_order_only"] += 1
    keys = ("different_ordered_top_10", "different_top_10_membership", "different_order_only",
            "different_ordered_nonempty_truth", "different_membership_nonempty_truth")
    return {key: counts[key] for key in keys}


def mulberry32(seed):
    """Yield the original audit's Mulberry32 stream with 32-bit unsigned arithmetic."""
    mask = 0xFFFFFFFF
    state = seed
    while True:
        state = (state + 0x6D2B79F5) & mask
        mixed = ((state ^ (state >> 15)) * (state | 1)) & mask
        mixed ^= (mixed + (((mixed ^ (mixed >> 7)) * (mixed | 61)) & mask)) & mask
        yield ((mixed ^ (mixed >> 14)) & mask) / 2**32


def percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (position - low) * (sorted_values[high] - sorted_values[low])


def paired_bootstrap(records, users):
    scores = {(row["user_id"], row["method"]): row["ndcg_at_10"] for row in records}
    deltas = [[scores[uid, "llmrec_encoder"] - scores[uid, method] for uid in users]
              for method in CONTROLS]
    size = len(users)
    rng = mulberry32(BOOTSTRAP_SEED)
    means = [[], []]
    for _ in range(BOOTSTRAP_DRAWS):
        indices = [int(next(rng) * size) for _ in users]
        for draws, differences in zip(means, deltas):
            total = 0.0
            for index in indices:
                total += differences[index]
            draws.append(total / size)
    results = {}
    for method, differences, draws in zip(CONTROLS, deltas, means):
        draws.sort()
        results[method] = {
            "mean_qwen_minus_control": sum(differences) / size,
            "percentile_95_ci": [percentile(draws, 0.025), percentile(draws, 0.975)],
            "qwen_higher": sum(delta > 0 for delta in differences),
            "tied": sum(delta == 0 for delta in differences),
            "qwen_lower": sum(delta < 0 for delta in differences),
        }
    return {
        "settings": {"draws": BOOTSTRAP_DRAWS, "users_per_draw": size, "seed": BOOTSTRAP_SEED,
                     "rng": "Mulberry32", "resampling": "paired users with replacement, shared draws",
                     "user_order": "saved starter protocol", "empty_truth_included": True,
                     "quantiles": [0.025, 0.975], "interpolation": "linear type 7"},
        "comparisons": results,
    }


def score(args, bundle, protocol, cli):
    _, assets, history, truth, contexts, _, _, pop, manifest = bundle
    start = perf_counter()
    spec = read_json(cli.asset_inputs)
    expected_spec = asset_inputs(assets, protocol)
    require(spec == expected_spec, "Asset input bundle differs from verified as-of fields or frozen settings")
    export = read_json(cli.generation_export)
    runtime = export["runtime"]
    require(export["protocol_hash"] == PROTOCOL_HASH and export["input_sha256"] == sha256(cli.asset_inputs),
            "Colab export does not match the provided asset bundle")
    require(runtime["model"] == MODEL and runtime["requested_revision"] == runtime["resolved_revision"] == REVISION
            and runtime["dtype"] == "torch.float16" and runtime["device"] == "cuda"
            and "T4" in runtime["gpu"] and runtime["do_sample"] is False and runtime["thinking"] is False,
            "Generation did not use the frozen T4 FP16 configuration")

    reference = read_json(cli.starter_reference / "metrics.json")
    methods = reference["methods"]
    saved_ranks = read_rankings(cli.starter_reference / "predictions.csv", methods)
    changed = restore_baselines(bundle, saved_ranks)
    documents = {i: encoder.source_text(asset) for i, asset in assets.items()}
    qwen, templates, failures = replay_views(documents, export)
    replay_seconds = perf_counter() - start

    start = perf_counter()
    original = encoder.tfidf(documents)
    predictions = encoder.predict(
        contexts, history, original,
        encoder.combine_views(original, encoder.tfidf(qwen)), pop,
        template=encoder.combine_views(original, encoder.tfidf(templates)),
    )
    ranking_seconds = perf_counter() - start
    args.output_dir.mkdir(parents=True)
    save_results(args, bundle, predictions, failures, "encoder_v3", {
        "starter_protocol_hash": PROTOCOL_HASH, "starter_reference_sha256": REFERENCE_HASHES,
        "asset_input_sha256": sha256(cli.asset_inputs), "generation_export_sha256": sha256(cli.generation_export),
        "frozen_source_sha256": FROZEN_SOURCES, "generation_summary": export["summary"],
        "colab_runtime": runtime, "local_verify_and_replay_seconds": replay_seconds,
        "local_tfidf_and_ranking_seconds": ranking_seconds,
    })
    require(read_rankings(args.output_dir / "predictions.csv", methods) == saved_ranks,
            "Scored starter rankings differ from the 4,000 saved positions")
    metrics = read_json(args.output_dir / "metrics.json")
    require(len(metrics) == 7 and all(metrics[method] == row for method, row in methods.items()),
            "Starter metrics did not reproduce")
    require(all(row["users"] == 100 and row["nonempty_truth_users"] == 11 for row in metrics.values()),
            "Scoring cohort differs from the reference")
    checks = {
        "protocol_hash": PROTOCOL_HASH, "saved_baseline_rank_positions_verified": len(saved_ranks),
        "saved_cbf_ordering_used": True, "near_tie_absolute_tolerance": 1e-12,
        "fresh_cbf_differing_position_count": len(changed), "fresh_cbf_differing_positions": changed,
        "qwen_vs_template_rankings": ranking_comparison(predictions, truth),
        "paired_bootstrap": paired_bootstrap(read_json(args.output_dir / "per_user.json"), manifest["users"]),
    }
    write_json(args.output_dir / "replay_checks.json", checks)
    for method, row in metrics.items():
        print(f"{method:25s} {row['ndcg_at_10']:.4f} all / {row['ndcg_at_10_nonempty']:.4f} nonempty")
    print(f"Verified all {len(saved_ranks)} saved ranks. Fresh CBF near-ties: {len(changed)}.")
    print(json.dumps(checks["qwen_vs_template_rankings"], indent=2))
    print(json.dumps(checks["paired_bootstrap"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-reference", "prepare", "score"):
        command = commands.add_parser(name)
        command.add_argument("--starter-reference", type=Path, required=True)
        if name != "verify-reference":
            command.add_argument("--data-dir", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
        if name == "score":
            command.add_argument("--asset-inputs", type=Path, required=True)
            command.add_argument("--generation-export", type=Path, required=True)
    cli = parser.parse_args()
    protocol = verify_reference(cli.starter_reference)
    if cli.command == "verify-reference":
        return

    output_dir = cli.output_dir.resolve()
    require(not output_dir.exists(), f"Use a fresh output directory: {output_dir}")
    marker = output_dir.relative_to(ROOT) / "run.json"
    ignored = subprocess.run(["git", "check-ignore", "-q", "--", str(marker)], cwd=ROOT)
    require(ignored.returncode == 0, f"Output directory must be Git-ignored: {output_dir}")

    args, bundle = frozen_bundle(cli.data_dir, output_dir, protocol)
    if cli.command == "prepare":
        output_dir.mkdir(parents=True)
        path = output_dir / "asset_inputs.json"
        write_json(path, asset_inputs(bundle[1], protocol))
        print(f"{path}  SHA-256 {sha256(path)}")
    else:
        score(args, bundle, protocol, cli)


if __name__ == "__main__":
    main()
