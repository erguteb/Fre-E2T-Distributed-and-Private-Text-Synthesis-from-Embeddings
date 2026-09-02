# Fre-E2T: Distributed and Private Textual Data Synthesis from Embeddings

Code for the PoPETs paper *Distributed and Private Textual Data Synthesis from
Embeddings*. It contains the Fre-E2T mechanism used for the main results
(Table 1) and the script that computes the end-to-end privacy guarantee
(Theorem 4, hyperparameters in Appendix A.1).

Pipeline: 
1. embed each text with GTR-T5-base
2. random Gaussian projection to k dimensions
3. bucketize with a randomly offset grid
4. release heavy buckets via subsampling + thresholding
5. add Gaussian noise to the per-bucket embedding sums
6. invert the noisy centroids back to text with vec2text.

## Contents

| File | Purpose |
| --- | --- |
| `data/<name>/preprocessing.py` | download each dataset and write `texts.pkl` |
| `data/load_data.py` | dataset/embedding loader |
| `1_build_histogram.py` | embed, project, bucketize; save histogram + bucket averages/members |
| `2_generate_data.py` | DP release (subsample, threshold, noise) and vec2text inversion |
| `service_embedding.py`, `service_nn.py`, `service_dp.py` | embedding model, bucketization, DP wrapper |
| `end_to_end_DP_analysis.py` | end-to-end (ε, δ) accounting |
| `pyproject.toml`, `uv.lock` | pinned environment (Python 3.12) |

## Setup

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

A CUDA GPU is required for `1_build_histogram.py` and `2_generate_data.py`
(embedding + vec2text inversion). `end_to_end_DP_analysis.py` runs on CPU and
only needs numpy/scipy/mpmath.

## Data

The datasets are public and are not redistributed here. Each preprocessing
script downloads from Hugging Face and writes `$DATA_DIR/<name>/texts.pkl`
(`DATA_DIR` defaults to `./data`):

```bash
python data/taylor/preprocessing.py       # TaylorAI/user_queries_dataset (English only)
python data/om/preprocessing.py           # wentingzhao/one-million-instructions ("Instructions-2M" in the paper)
python data/lmsys_chat/preprocessing.py   # lmsys/lmsys-chat-1m (gated; request access on HF; English user turns)
python data/yelp/preprocessing.py         # Yelp/yelp_review_full, train split only (650k reviews)
```

## Running the mechanism

Environment variables (defaults in parentheses): `DATA_DIR` (`./data`),
`RESULTS_DIR` (`./results`), `SYNTHETIC_DATA_DIR` (`./results_synthetic_data`).

Main-result setup (Appendix A.1): neighborhood radius r = 0.5, frequency
threshold t = 100 (a text is "frequent" if at least t others lie within
distance r of it in embedding space), projection dimension k = 20, seed 42.
p_s is the client subsampling rate; a bucket is released if its subsampled
count reaches τ = p_s·t. v is the budget factor that sets the heavy-hitter
ε (Lemma 1), and u = 2.4 is the sensitivity ratio: each bucket sum has L2
sensitivity Δ_dist = u·r with high probability. σ is the std of the Gaussian
noise added to each released bucket's embedding sum. Per privacy target:

| target ε | p_s | v | τ = p_s·t | noise std σ |
| --- | --- | --- | --- | --- |
| 4  | 0.3 | 4 | 30 | 2.136 |
| 8  | 0.5 | 4 | 50 | 1.14  |
| 16 | 0.6 | 3 | 60 | 0.516 |
| ∞  | 1 (no subsampling) | – | 100 | 0 |

Step 1 — build the bucket histogram (once per dataset):

```bash
python 1_build_histogram.py --dataset_name taylor --r 0.5 \
    --random_projection_K 20 --random_seed 42 --tau 20
```

`--tau 20` here is only a build-time prefilter on raw bucket counts, kept
below every release threshold; the actual frequency filtering happens in
step 2, on subsampled counts. Since a subsampled count never exceeds the raw
count, a bucket with ≤ 20 members can never pass the thresholds τ ≥ 30 below.
The projected space is clipped to the maximum absolute projected value by
default (`--c` overrides).

Step 2 — DP release + text generation, e.g. for ε = 4 (`--mask_prob` is p_s,
`--noise_std` is σ from the table; note `--tau` takes t = 100 here, not
τ — the release threshold τ = p_s·t is applied internally to the subsampled
counts):

```bash
python 2_generate_data.py --dataset_name taylor --setting dp --r 0.5 \
    --random_projection_K 20 --random_seed 42 --tau 100 \
    --mask_prob 0.3 --noise_std 2.136 --batch_size 2048
```

For the non-DP reference (ε = ∞): `--setting non_dp --tau 100`. Outputs go to
`$SYNTHETIC_DATA_DIR/<run>_synthetic_data.pkl`; each released bucket stores its
noisy centroid and the synthetic text(s) inverted from it, plus bookkeeping
fields used for evaluation (the bucket's true member indices and count). The
bookkeeping fields are ground truth, not part of the DP release — do not
publish the pickle as-is.

## Privacy accounting

The mechanism releases two things about the private texts: which buckets are
heavy, and the noisy embedding sums of those buckets. The overall guarantee
(Theorem 4) is

- ε = ε_fre + ε_agg  and  δ = δ_fre + δ_sens + δ_agg, where
- ε_fre = v·ln(1/(1−p_s)) and δ_fre = exp(−(τ/q)·D(q‖p_s)) with
  q = 1−(1−p_s)^(v+1) — the subsample-and-threshold heavy-hitter release
  (Lemma 1); D(·‖·) is the KL divergence between Bernoulli distributions;
- δ_sens = f(2/u)^k — the failure probability of the sensitivity bound
  Δ_dist = u·r for one bucket sum (Lemma 2); f is the per-coordinate
  collision-probability function from Eq. 7 of the paper, implemented as
  `f()` in the script;
- ε_agg — the Gaussian mechanism on the bucket sums with noise multiplier
  m = σ/Δ_dist, accounted at budget δ_agg = δ − δ_fre − δ_sens using the
  **analytic Gaussian mechanism** (Balle & Wang, ICML 2018). The script inverts
  the analytic bound by bisection; this is exact for the Gaussian mechanism,
  i.e. neither a zCDP nor an RDP relaxation.

The noise scales σ above are the analytic-Gaussian-calibrated ones, which we
use in the centralized-setting simulation (continuous Gaussian noise).
Theorem 3 in the paper states the Gaussian guarantee in CDP form; that
formulation is what provides the formal guarantee under the distributed
implementation, where the noise is discrete Gaussian and composes in CDP.

`end_to_end_DP_analysis.py` prints every component and the total. The three
main configurations (`--HH_thres` is τ, `--HH_eps_factor` is v, `--sens_to_r`
is u, `--noise_multiplier` is m = σ/(u·r); note σ = m·u·r is exactly the
`--noise_std` passed to `2_generate_data.py`):

```bash
python end_to_end_DP_analysis.py --ps 0.3 --HH_eps_factor 4 --HH_thres 30 --K 20 --sens_to_r 2.4 --noise_multiplier 1.78 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.5 --HH_eps_factor 4 --HH_thres 50 --K 20 --sens_to_r 2.4 --noise_multiplier 0.95 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.6 --HH_eps_factor 3 --HH_thres 60 --K 20 --sens_to_r 2.4 --noise_multiplier 0.43 --delta 1e-6
```

Resulting guarantees at overall δ = 10⁻⁶ (the δ split matches Table 11 in the
paper):

| target ε | ε_fre | δ_fre | δ_sens | ε_agg | **final ε** | **final δ** |
| --- | --- | --- | --- | --- | --- | --- |
| 4  | 1.4267 | 2.93×10⁻¹⁰ | 8.95×10⁻¹¹ | 2.5627  | **3.9894**  | **10⁻⁶** |
| 8  | 2.7726 | 3.80×10⁻¹³ | 8.95×10⁻¹¹ | 5.1808  | **7.9534**  | **10⁻⁶** |
| 16 | 2.7489 | 1.76×10⁻¹¹ | 8.95×10⁻¹¹ | 13.2165 | **15.9654** | **10⁻⁶** |

So each run is certified at slightly below its nominal budget. The commented
block at the end of `end_to_end_DP_analysis.py` lists the invocations for the
other projection dimensions k ∈ {10, 15, 25, 30} used in the ablations.

## License

MIT, see `LICENSE`.
