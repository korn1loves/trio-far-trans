# Latest-buy fixed-candidate Ranker — Artem Bondarenko, Team Trio

HSE LLM4Rec Homework 2, Ranker track. This directory adds a separate experiment
without modifying the four supplied classical algorithms or Session 1 artifacts.
Python 3.10+ and the standard library are sufficient for classical evaluation,
preparation, and tests. Optional real inference additionally needs PyTorch and
Transformers. Nothing is installed automatically; inference is an explicit command
and downloads are disabled unless explicitly permitted.

## Research question and status

Does zero-shot LLM reranking of a **fixed latest_buy top-50** set improve binary
nDCG@10 over original order and the predefined frequency-recency control?

- **A:** canonical latest-buy first ten; measured when the CLI completes.
- **B:** deterministic frequency-recency; measured when the CLI completes.
- **C:** Qwen3-1.7B reranking; **NOT RUN**. The real runtime is implemented but has
  not been executed with a real model. Injected/stub-library tests are **TEST ONLY**,
  never Qwen measurements.

The user reports **PERSONAL RANKER CHECK: PASSED** for one real user's Top-10.
That personal verification is distinct from automated tests and baseline parity.

The instructor's reference at revision
`f6850861ea20a51e1fb018ee4e5f840c2ac759e0` uses **CBF** candidates. This experiment
does not use that candidate policy, the stock CBF reranker seam, or CBF fallback.

## Reproduce from the repository root

All generated files belong under the existing ignored `starter/outputs/`.
Disable bytecode writes; temporary fixtures are also directed there.

```bash
# Original suite: expected 58 tests.
TMPDIR="$PWD/starter/outputs" PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s starter -p 'test_starter.py' -v

# New synthetic contracts; no real dataset or model required.
TMPDIR="$PWD/starter/outputs" PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s ranker_artem -p 'test_*.py' -v

# Complete fixed 100-user real-data A/B run, including saved-baseline parity.
PYTHONDONTWRITEBYTECODE=1 python3 -B -m ranker_artem.ranker \
  --data-dir "$HOME/Project/data/FAR-Trans" \
  --baseline-dir starter/outputs/session2_baseline \
  --output-dir starter/outputs/session3_ranker_artem_ab_check

# Safe: prepare all 100 prompts/provenance without loading even the tokenizer.
PYTHONDONTWRITEBYTECODE=1 python3 -B -m ranker_artem.runtime prepare \
  --data-dir "$HOME/Project/data/FAR-Trans" \
  --baseline-dir starter/outputs/session2_baseline \
  --output-dir starter/outputs/session3_ranker_artem_prepared \
  --device cuda --precision bfloat16
```

Choose a fresh output directory for each additional run. Existing directories are
rejected before loading data. The CLI also works as `python3 -B ranker_artem/ranker.py`.

## Frozen data protocol

- Deterministic starter warm cohort: 100 users, SHA256 `42:` selection.
- Cutoff **2021-06-01 exclusive**; future Buy truth window
  `[2021-06-01, 2021-07-01)`; repeats allowed.
- Same 767 eligible assets: latest strictly pre-cutoff metadata and at least one
  positive historical close price.
- Exact-full-row deduplication; negative transaction IDs excluded. Repeated
  transaction IDs with different full rows remain. Sell events do not affect A/B
  Buy counts, recency, or popularity; the original Buy-or-Sell baseline is retained.
- Global popularity is distinct historical buying users over the whole population,
  not only the evaluation cohort.
- The loader requires the saved Session 1 protocol hash:
  `50efd4aeb415ba6f0c66dd0793a7c2033cdc32a42b86b58eebbca3e81eaf3417`.
  It verifies the hash, all four data hashes, ordered users, universe, predictions,
  row-exclusion counts, full-precision per-user scores, and aggregate metrics.
  Missing or incompatible baseline artifacts cause failure, not silent adaptation.

Future truth stays in the evaluation bundle and is never accepted by ranking or
prompt functions. No user is chosen or removed based on future labels. The real
run verifies 11 nonempty and 89 empty-truth users after freezing inputs.

## Canonical candidates and controls

`adapter.build_inputs` calls the unmodified
`starter.recency_rankings(..., k=50)` and selects `latest_buy`. It sorts personally
bought assets by last Buy date descending, then ISIN ascending, and fills from
global popularity descending, then ISIN ascending, skipping duplicates. Every
user must have exactly 50 unique eligible original IDs. Its first ten must equal
the separately generated original latest-buy baseline and saved predictions.

An immutable `RankerInput` contains only cutoff and 50 immutable candidate records.
A, B, and future C share that one canonical input. No CBF candidate pool is
substituted. A returns the canonical first ten.

For B, normalize **within the frozen candidate set**:

```text
F = log1p(personal_buy_count) / log1p(max candidate personal_buy_count)
P = log1p(global_distinct_buyers) / log1p(max candidate global_distinct_buyers)
R = 2 ** (-calendar_days_since_last_buy / 180) for purchased assets; otherwise 0
score = 0.45*F + 0.45*R + 0.10*P
```

A zero denominator gives a zero feature. Ties use canonical position, then ISIN.
The coefficients and half-life were fixed in Session 1, not selected using future
labels. Positive personal counts without historical dates are rejected.

## LLM contract

`build_prompt` emits deterministic instructions plus an allowlisted JSON payload:
cutoff, repeat permission, and all 50 original ISINs with as-of names, personal
historical Buy counts, latest historical Buy dates, and global distinct-buyer
popularity. Missing names become `unknown`. Names are explicitly passive data.
There are no customer IDs, profiles, future labels/events, or external facts.
Post-cutoff dates and metadata are rejected at the input boundary.

`validate_response` requires native JSON containing exactly ten unique candidate
strings. Prose, code fences, invalid types, duplicate IDs, and noncandidate IDs
are rejected. No substring extraction, ID repair, or regeneration is used.
`response_or_fallback` returns the canonical latest-buy first ten and a nonempty
failure reason for invalid output or a supplied recoverable inference error.
It never calls a model itself. `runtime.py` persists the raw response, validation
decision and failure reason for every user; fallback preserves the evaluation slot.

The implemented runtime uses `Qwen/Qwen3-1.7B`, model/tokenizer revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; non-thinking chat template, greedy
decoding, one beam, batch size one, seed 42, inference/evaluation mode, 8192 total
tokens and at most 512 new tokens. It sets evaluation mode and `torch.inference_mode`,
decodes only newly generated tokens, and does not truncate candidates or prompts.
Prompts requiring more than 7680 input tokens cause a recorded per-user fallback.
Preparation cannot measure tokenizer lengths; actual token counts are recorded
only during inference. Greedy decoding and fixed seeds do not guarantee identical
outputs across hardware/backends; deterministic torch algorithms are not forced.

## External NVIDIA GPU execution — commands for after approval only

**No installation, model download, or real inference was performed in Phase 3.**
The following commands are instructions for a future external Linux/NVIDIA runtime.
Copy this repository, matching data CSVs, and all five files from the saved
`starter/outputs/session2_baseline/` to that machine. Matching hashes are mandatory.

Reference dependency pins: **Python 3.12**, **torch 2.11.0** (reference CUDA build:
`2.11.0+cu128`), **transformers 4.57.6**. The wrapper records actual versions of
torch, transformers, tokenizers, huggingface-hub, safetensors, numpy, and Python.
These installation commands and wheel/driver compatibility have not been tested
on an external GPU in this session. `accelerate` is not required.

```bash
# FUTURE EXTERNAL MACHINE ONLY, after explicit installation/download approval.
python3.12 -m venv "$HOME/venvs/trio-qwen"
"$HOME/venvs/trio-qwen/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0'
"$HOME/venvs/trio-qwen/bin/python" -m pip install 'transformers==4.57.6'
nvidia-smi

# From this repository's root on the GPU host:
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$PWD/starter/outputs/hf_cache"
PYTHON="$HOME/venvs/trio-qwen/bin/python"
DATA="/path/to/FAR-Trans"
RUN="starter/outputs/qwen_latest_buy_gpu_01"

# Standard-library preparation only; fresh directory required.
"$PYTHON" -B -m ranker_artem.runtime prepare \
  --data-dir "$DATA" --output-dir "$RUN" \
  --device cuda --precision bfloat16

# Actual inference and explicit FIRST download permission, only after approval.
"$PYTHON" -B -m ranker_artem.runtime infer \
  --data-dir "$DATA" --output-dir "$RUN" --resume \
  --device cuda --precision bfloat16 --allow-download

# Resume compatible cached-model execution, with network disabled:
HF_HUB_OFFLINE=1 "$PYTHON" -B -m ranker_artem.runtime infer \
  --data-dir "$DATA" --output-dir "$RUN" --resume \
  --device cuda --precision bfloat16
```

Use BF16-capable NVIDIA hardware for those exact commands. For an FP16-only device
such as T4, choose `--precision float16` in **both** preparation and inference.
`auto` chooses BF16 when CUDA supports it, otherwise FP16; the resolved choice is
recorded. There is no automatic CPU fallback. Explicit CPU mode uses float32.
GPU availability and memory fit are unmeasured; model-loading failure is an
incomplete run, not a fabricated all-user fallback result. `infer` itself is the
explicit execution action; without `--allow-download`, model/tokenizer loading
uses `local_files_only=True`, always with the immutable revision and
`trust_remote_code=False`.

### Resumption, evidence, and failure rules

- `prepare` writes protocol, canonical candidates, exact prompts, and
  `run_state.json` with **PREPARED / Qwen NOT RUN**. No metric or prediction is
  generated, and neither torch nor transformers is imported. It can be resumed
  directly by `infer --resume` with identical device/precision settings.
- The experiment hash binds the baseline protocol, ordered-candidate hash,
  ordered prompt hashes, pinned model/tokenizer, settings, fixed control B
  parameters, and source-file hashes. The instructor's CBF candidate policy is
  never selected. All 100 users use exactly the A/B canonical candidates.
- `responses/*.json` holds atomic, checksummed per-user records immediately after
  completion. Completed invalid outputs/errors are cached as fallbacks, not retried
  until success. An OS lock prevents concurrent writers and releases on death.
- `--resume` rejects changed code, prompts, candidates, data/protocol, precision,
  backend hardware/dependencies, modified prepared files, or corrupted responses.
  Pending inference checks the actual backend against persisted metadata before
  generating new text. Replaying a complete intact run needs no model loading.
- Recoverable per-user generation errors (including OOM/runtime errors, timeout,
  or context overflow) and invalid output use the canonical latest-buy first ten.
  Unexpected programming/disk errors, interrupts, or model startup errors leave
  the run **INCOMPLETE** with existing response evidence intact. They do not remove
  users or create partial-cohort measured metrics.
- Only complete runs publish `responses.jsonl`, seven-method `predictions.csv`
  and `per_user.csv`, `metrics.json`, and `experiment_manifest.json`. Metrics use
  the unchanged starter scorer/aggregator; all 100 users and both truth means
  remain. The result is explicitly Qwen **plus fallback**, with invalid-output,
  inference-error, and fallback counts.
- Evidence includes input and rendered chat prompts, raw generated text, token
  counts, generation/request seconds, model-call flags, GPU peak allocated memory,
  hardware/precision, requested/resolved revisions, tokenizer/template identity,
  effective generation config, actual dependency versions, and artifact hashes.
  Failed requests retain available measurements; missing token counts are counted
  explicitly. Summed generation time uses original responses, not cached replay;
  `invocations.jsonl` separately records loading and resumption information.
- Injected generators are always **TEST ONLY**, including all written metrics and
  manifests. Their job hashes cannot be resumed as real-model jobs. Stubbed
  Transformers tests check wrapper behavior without loading any real libraries or
  weights; synthetic token/timing values are not model measurements.

## Evaluation and outputs

The classical CLI uses original `starter.ndcg10` and `starter.aggregate` for six rows:
four original methods plus A and B. Truth is intersected with the eligible universe,
not with retrieved candidates; empty truth counts as zero. Both all-user and
nonempty-truth means are reported. All saved baseline values and A are compared
using exact full-precision equality, not rounded display values.

Each completed run writes:

| File | Contents |
|---|---|
| `protocol.json` | Byte-identical saved baseline protocol, full hashes retained |
| `canonical_candidates.jsonl` | Ordered IDs and historical features for each user |
| `predictions.csv` | Six methods × 100 users × ten original ISINs |
| `per_user.csv` | Six methods × 100 user-level binary nDCG scores |
| `control_b_scores.csv` | All 50 candidates' F/R/P/score and zero-based positions |
| `metrics.json` | Full-precision means; MEASURED/NOT RUN status; C has no metric |
| `experiment_manifest.json` | Protocol, source/artifact/code hashes, candidate policy, fixed parameters, verification, whole-computation timing, and C NOT RUN |

The canonical ordered-candidate hash is SHA256 over UTF-8 compact JSON
`[[user_id,[ordered IDs]],...]` in protocol user order, `sort_keys=True` and
`ensure_ascii=False`. It binds both user and candidate order. The separate
experiment hash binds the candidate policy, ordered-candidate hash, baseline
protocol, parameters, and hashes of these four implementation files. The manifest
includes output artifact hashes (excluding itself). Timing covers the complete
classical loading/verification/evaluation, excluding output writing; it is not a
per-method measurement. Console metrics use four decimals; files retain precision.

Local audit artifacts contain user linkage and should stay in ignored outputs.
Only the allowlisted prompt payload is intended for a future model.
