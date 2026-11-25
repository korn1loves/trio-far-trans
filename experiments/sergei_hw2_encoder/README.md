# LLM Selection of Recent-Purchase Weights for Content-Based Ranking on FAR-Trans

Kornilov Sergei Yuryevich · HSE University · LLM4Rec · Team Trio

Homework 2 · Individual Encoder Track · User Representation

## Research question

On the 100 warm users in the deterministic FAR-Trans teaching cohort, does using an LLM to choose the weight of the last 90 days from {0.25, 0.50, 0.75} improve mean nDCG@10 over a deterministic rule that rounds the recent purchase share to the same set, with a fixed CBF scorer, a 1 June 2021 cutoff, and a 30-day purchase window?

## Method

The recent window is `[2021-03-03, 2021-06-01)`. The older window contains retained Buy events before `2021-03-03`. Each window's counts are normalized independently. Asset weights follow `w_ui = alpha * p_recent_ui + (1 - alpha) * p_older_ui`. An asset present in both windows receives both contributions. A sole nonempty window receives weight 1 and bypasses inference. The deterministic selector rounds recent Buy share to 0.25, 0.50 or 0.75 with ties going to 0.50.

The shared starter filters exact duplicate rows and negative transaction IDs. Eligibility uses the latest pre-cutoff metadata and at least one positive pre-cutoff closing price. Both Encoders feed saved `history_weights` to the original CBF scorer, which ranks the full eligible universe with repeat purchases allowed. Evaluation uses binary nDCG@10 on purchases in `[2021-06-01, 2021-07-01)`; empty truth contributes zero.

The frozen [prompt](prompt.txt) is followed by a JSON object with the two historical window summaries. Each contains Buy-event and distinct-asset counts, event-weighted histograms and missing counts of asset category, subcategory, sector, industry and market ID. User IDs, ISINs, dates of individual events, profiles, prices, volumes and truth stay local. The validator demands one JSON object with one key, `alpha`, and one permitted numeric value. Invalid output and recoverable inference errors use the predefined 0.50 mixture. Model-load or fatal device errors stop inference. Checkpoints retain terminal responses and fallback reasons, and resumed runs skip recorded cases.

- `encoder.py` builds histories and weights and validates the frozen output contract.
- `run.py` builds the deterministic control and verifies parity with the classical reference.
- `runtime.py` prepares inputs, checkpoints inference, exports saved responses and controls paired evaluation.
- `qwen_backend.py` loads the pinned model and tokenizer and generates one response at a time.
- `prompt.txt`, `model_config.json` and `freeze.sha256` pin the research contract.
- `test_encoder.py` and `test_runtime.py` cover synthetic fixtures and optional real-run checks.

## Reproduce from a fresh checkout

Use the official [FAR-Trans release](https://researchdata.gla.ac.uk/1658/) and point `data` at the extracted CSV directory. The evaluated setup used Python 3.13.5 on Apple M4 arm64 with 16 GB, MPS and float16. The [dependency lock](requirements.txt) records the installed environment. Run from the repository root.

```sh
data=/absolute/path/to/FAR-Trans
exp=experiments/sergei_hw2_encoder
ref="$exp/outputs/reference-june"
control=deterministic-repro
llm=qwen-repro

python3 -m venv "$exp/.venv"
py="$exp/.venv/bin/python"
"$py" -m pip --isolated install --index-url https://pypi.org/simple -r "$exp/requirements.txt"
export HF_HOME="$PWD/$exp/.venv/hf"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60

"$py" -c 'import torch; assert torch.backends.mps.is_built() and torch.backends.mps.is_available(); assert (torch.ones(2,device="mps",dtype=torch.float16)*2).sum().item()==4'
(cd "$exp" && shasum -c freeze.sha256)
"$py" starter/starter.py run --data-dir "$data" --output-dir "$ref"
"$py" -m experiments.sergei_hw2_encoder.run \
  --data-dir "$data" --reference "$ref" --run-name "$control"
"$py" -m experiments.sergei_hw2_encoder.runtime prepare \
  --data-dir "$data" --reference "$ref" --run-name "$llm"
"$py" -m experiments.sergei_hw2_encoder.runtime status --run-name "$llm"
"$py" -m experiments.sergei_hw2_encoder.runtime infer --run-name "$llm" \
  --device mps --dtype float16 --execute-inference --allow-download
"$py" -m experiments.sergei_hw2_encoder.runtime evaluate --run-name "$llm" \
  --data-dir "$data" --reference "$ref" --phase2 "$exp/outputs/$control"
```

Use fresh output names for reproduction. Keep the same environment, `HF_HOME`, run name and options to resume, omitting `--allow-download` once the cache is complete. Both tokenizer and model use the revision in `model_config.json`. The chat template uses non-thinking mode, greedy decoding, one beam, batch size one, seed 42 and a 32-new-token cap. Preparation includes 35 cases. The 65 single-window users bypass inference and all 100 receive saved responses. `responses.sqlite3` records raw outputs, validation, applied alpha, token counts and timing. Evaluation writes `evaluation/llm_summary.json` and the original starter artifacts locally.

```sh
"$py" -m unittest discover -s starter -p 'test_*.py'
ENCODER_RUN="$exp/outputs/$control" ENCODER_REFERENCE="$ref" \
  ENCODER_PREPARED="$exp/outputs/$llm" \
  "$py" -m unittest experiments.sergei_hw2_encoder.test_encoder \
    experiments.sergei_hw2_encoder.test_runtime
```

Run the prepared-artifact runtime check immediately after preparation if running tests on a reproduction. That check expects a clean inference checkpoint. The committed tests passed 58 starter, 12 Encoder and 11 runtime checks. Keep the local data, model cache, raw outputs, rankings and checkpoints outside Git.

## Measured June results

| Method | nDCG@10, all 100 | nDCG@10, nonempty truth 11 |
|---|---:|---:|
| Popularity | 0.0286 | 0.2599 |
| CBF | 0.0201 | 0.1827 |
| Latest buy | 0.0277 | 0.2520 |
| Latest buy or sell | 0.0271 | 0.2464 |
| Deterministic purchase-share Encoder | 0.0162 | 0.1473 |
| LLM weight selection Encoder | 0.0145 | 0.1320 |

All four classical baselines matched the original June reference at full precision and by ordered predictions. The protocol hash was `50efd4aeb415ba6f0c66dd0793a7c2033cdc32a42b86b58eebbca3e81eaf3417`. Out of 35 real local generations, 6 were valid and 29 used the prescribed equal-mixture fallback. The all-user paired difference versus the deterministic control was −0.001683. The descriptive 95% paired bootstrap interval was [−0.015552, +0.011054]. The result refers to the frozen selector and fallback on this cohort. April tuning and additional fixed-weight ablations were outside the implementation.
