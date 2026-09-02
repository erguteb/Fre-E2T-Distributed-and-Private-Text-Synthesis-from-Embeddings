# Artifact Appendix

Paper title: **Distributed and Private Textual Data Synthesis from Embeddings**

Requested Badge(s):
  - [x] **Available**

## Description

This artifact accompanies the paper *Distributed and Private Textual Data
Synthesis from Embeddings* of submission #46, Proceedings on Privacy Enhancing Technologies
(PoPETs), 2027.

It contains the implementation of the Fre-E2T mechanism used for the paper's
main results: dataset preprocessing for the four evaluated corpora
(TaylorAI, Instructions-2M, LMSYS-Chat-1M, Yelp), the pipeline that embeds
texts, projects them to a low-dimensional space, releases heavy buckets under
subsampling and thresholding, perturbs the per-bucket embedding sums with
Gaussian noise, and inverts the noisy centroids back to text with vec2text.

It also contains the standalone accounting script `end_to_end_DP_analysis.py` that computes
the end-to-end (epsilon, delta)-DP guarantee of Theorem 4 for the hyperparameters in
Appendix A.1 of the paper. 

`README.md` documents the setup, the exact commands for
the main configurations, and the resulting privacy parameters.

### Security/Privacy Issues and Ethical Concerns

The artifact does not disable any security mechanism and contains no attack or
exploit code; it runs standard PyTorch/Hugging Face code. The preprocessing
scripts download public datasets and pretrained models (GTR-T5-base, the
vec2text corrector, GPT-2) from Hugging Face at run time; no dataset is
redistributed within the artifact.

One dataset, LMSYS-Chat-1M, is gated on Hugging Face and consists of real
user–LLM conversations that may contain personal or offensive content. Access
requires accepting the dataset's terms of use, and the downloaded data should
be handled according to those terms. The remaining datasets (TaylorAI user
queries, Instructions-2M, Yelp reviews) are publicly available. The artifact
involves no user study.

## Environment

### Accessibility

The artifact is hosted on GitHub:
https://github.com/<your-account>/fl_embedding/tree/main

The repository is self-contained: `README.md` covers installation (a pinned
Python 3.12 environment via `uv sync` from the included `pyproject.toml` and
`uv.lock`), data preparation, how to run the mechanism, and how to compute the
privacy guarantees. The code is released under the MIT license (`LICENSE`).

## On Reusability

The overall pipeline is dataset-agnostic: any text corpus can be plugged in (we provide scripts for four datasets). 
The embedding model is encapsulated in `service_embedding.py` and can be swapped for any text-to-vector model.
The bucketization and DP release (`service_nn.py`, `service_dp.py`) operate on arbitrary embedding dimensions. 

`end_to_end_DP_analysis.py` is a standalone accountant for the sample-and-threshold histogram plus the Gaussian mechanism, and can
calibrate the parameters under different (epsilon,delta)-DP guarantees. 
