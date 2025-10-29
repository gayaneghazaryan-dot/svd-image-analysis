---
title: "SVDlab: A Reproducible Toolkit for SVD-based Image Compression, Denoising, and PCA with Adaptive Rank Selection"

authors:
  - name: Gayane Ghazaryan
    affiliation: 1
  - name: Artashes Ghazaryan
    affiliation: 2

affiliations:
  - name: Institute of Physics, Yerevan State University, Armenia
    index: 1
  - name: Provectus Research
    index: 2

date: 2025-10-11
bibliography: paper.bib
link-citations: true

tags:
  - Singular Value Decomposition
  - Image Compression
  - Denoising
  - Principal Component Analysis
  - Reproducibility
  - Python
---

# Summary

**SVDlab** is an open-source Python toolkit that demonstrates how the **Singular Value Decomposition (SVD)** can be applied in three classical yet powerful contexts: **image compression**, **image denoising**, and **principal component analysis (PCA)**. It is designed as a compact and reliable framework that connects the mathematical theory of SVD with practical, fully reproducible workflows.  

A key question in these applications is how many singular values to retain. **SVDlab** implements an **adaptive rank-selection rule** that automatically balances reconstruction quality and efficiency by combining a cumulative-energy threshold with elbow detection (*Kneedle*). The rule chooses  
$k^* = \max(k_\tau, k_e)$, providing stable results across datasets and domains [@satopaa2011kneedle; @eckart1936approximation; @jolliffe2002pca].

All figures, tables, and metrics — including PSNR, SSIM, energy retention, and runtime — can be **reproduced from four one-line commands**. The toolkit records metadata to trace every artifact back to its code, parameters, and dependency versions. Benchmarks against eigenvalue decomposition (EVD) and pivoted thin QR demonstrate consistent, transparent performance across noise levels and hardware backends [@gu1996efficient; @wang2004ssim].

![SVD compression comparison at 99.5 % retained energy. The adaptive rank (k*) selected by the Kneedle method achieves higher PSNR and SSIM than the fixed manual rank (k = 50). Reproducible via `code/svd_compression_merged.py`.](figures/astronaut_svd_comparison_995.png)

# Statement of Need

Despite its central role in data analysis and imaging [@golub2013matrix; @andrews1976svd], the choice of rank in SVD-based methods remains inconsistent, hindering reproducibility and fair comparison across studies [@jolliffe2002pca].  

**SVDlab** addresses this by offering a single, coherent toolkit that integrates preprocessing, adaptive rank selection, and reproducible artifact generation for multiple tasks. It aligns with the **FAIR principles** [@wilkinson2016fair] and current best practices in computational research transparency [@stodden2016enhancing].  

The framework is intended for both researchers and educators who value auditable, theory-grounded experiments. It serves equally as a foundation for algorithmic research and for teaching laboratories where students can reproduce all results with minimal setup.

# Novelty and Relation to Prior Work

While many libraries implement SVD or PCA, few unify them across tasks with a consistent design for **reproducibility and adaptive rank control**. **SVDlab** distinguishes itself through:

1. **Cross-task standardization** — one interface for compression, denoising, and PCA with identical preprocessing, metrics, and outputs.  
2. **Adaptive, task-agnostic rank selection** — the hybrid cumulative-energy and elbow rule yields stable $k^*$ across datasets, grounded in low-rank approximation theory [@eckart1936approximation; @satopaa2011kneedle].  
3. **Reproducibility by design** — deterministic generation of all figures and tables from four simple commands [@wilkinson2016fair; @stodden2016enhancing].  
4. **Methodological breadth** — standardized benchmarking of SVD, EVD, and QR factorizations with PSNR and SSIM evaluation [@gu1996efficient; @wang2004ssim].

These elements turn linear algebra concepts into a transparent and hands-on computational toolkit suitable for both research and teaching.

# Features

- **Adaptive rank selection** combining cumulative-energy and Kneedle-based elbow detection.  
- **Four ready-to-use Python scripts** covering image compression, denoising, PCA, and benchmarking.  
- **Deterministic results** ensured by fixed random seeds and pinned dependencies.  
- **Automatic export** of all outputs as PDF, CSV, and LaTeX tables.  
- **Cross-platform support** (Linux, macOS, Windows; Python ≥ 3.10).  
- **Minimal execution effort** — all results from this paper can be reproduced with four simple commands.  
- **Sensible defaults and fallbacks** — e.g., $\tau\approx0.995$ for high-fidelity compression, $\tau\in[0.95,0.99]$ for denoising; if the elbow is inconclusive, the energy rule applies.  
- **PCA compatibility** — cumulative energy $\eta_k$ coincides with explained variance for mean-centered data.

**Adaptive rank selection summary:**
1. Compute singular values $\{\sigma_i\}$ and cumulative energy $\eta_k = \sum_{i \le k}\sigma_i^2 / \sum_{i=1}^{r}\sigma_i^2$.  
2. Threshold rule: $k_\tau = \min\{k : \eta_k \ge \tau\}$.  
3. Elbow rule: detect $k_e$ via Kneedle.  
4. Default: $k^* = \max(k_\tau, k_e)$ (aggressive: $\min(k_\tau, k_e)$). If no elbow is detected, use $k_\tau$ and clip $k$ to $[1, r]$.

![Progressive SVD reconstructions of the “astronaut” image at increasing ranks (k = 5 … 200), showing the trade-off between compression efficiency and visual fidelity. Generated via `code/svd_compression_merged.py`.](figures/astronaut_svd_fixed_rank_reconstructions.png)

# Example Usage

```bash
# 1) Image compression
python3 code/svd_compression_merged.py

# 2) Image denoising
python3 code/svd_denoising.py

# 3) Factorization benchmarks
python3 code/benchmark_and_plots.py

# 4) PCA with adaptive component selection
python3 code/pca_adaptive_combined.py
```
Each script automatically saves results under results/Figures/ and results/Tables/, allowing readers to reproduce every figure and table from a clean environment.

# Limitations and Future Work

The current release focuses on exact (non-randomized) factorizations for moderate-scale problems. Future work will extend SVDlab with randomized and streaming variants [@halko2011randomized], GPU acceleration, and support for color images and video — while maintaining full reproducibility and auditability.

# Acknowledgements

We thank the open-source communities behind **NumPy**, **SciPy**, **scikit-image**, **scikit-learn**, and **Matplotlib**, whose work made this toolkit possible.  
We also appreciate the support of colleagues at **Yerevan State University** and **Provectus** for their helpful feedback and collaboration.  

For citation and reproducibility, please refer to the archived version of the software [@ghazaryan2025svdlab].  
The complete source code and Zenodo release are available at  
[https://doi.org/10.5281/zenodo.17313445](https://doi.org/10.5281/zenodo.17313445).

.

# References