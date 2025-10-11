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

**SVDlab** is an open-source Python toolkit that demonstrates how the **Singular Value Decomposition (SVD)** can be applied in three classical yet powerful contexts: **image compression**, **image denoising**, and **principal component analysis (PCA)**. It is designed for readers who want a small, reliable toolkit that connects the theory of SVD with practical, reproducible workflows.  

The central practical question is *how many* singular values to keep. SVDlab provides an adaptive rank-selection rule that automatically balances reconstruction quality and efficiency. The strategy combines a cumulative-energy threshold with elbow detection (Kneedle), choosing  
$k^* = \max(k_\tau, k_e)$ — a simple default that avoids under-selection and keeps results stable across datasets [@satopaa2011kneedle; @eckart1936approximation; @jolliffe2002pca].

All results can be **reproduced** from four one-line commands. The toolkit generates standardized outputs — figures, tables, and metrics such as PSNR, SSIM, energy retention, and runtime — and records metadata so that every artifact can be traced back to its code, parameters, and versions. Benchmarks against eigenvalue decomposition (EVD) and pivoted thin QR show how adaptive rank selection improves reproducibility and consistency across noise levels and hardware backends [@gu1996efficient; @wang2004ssim].

# Statement of Need

Although SVD is central to modern data analysis and imaging [@golub2013matrix; @andrews1976svd], choosing how many singular values to keep — *rank selection* — remains inconsistent across studies and implementations. This makes results difficult to reproduce and compare, especially in image processing and dimensionality reduction [@jolliffe2002pca].  

At the same time, reproducibility has become a core value in computational research, reflected in the **FAIR** principles [@wilkinson2016fair] and broader calls for transparent scientific software [@stodden2016enhancing]. Yet practical tools that integrate preprocessing, adaptive rank selection, and reproducible artifact generation across multiple tasks are still uncommon.

**SVDlab** addresses this need by providing a single, reproducible framework that:
- Unifies preprocessing, adaptive rank selection, and evaluation across different tasks.  
- Implements a principled adaptive rule that combines cumulative-energy and elbow criteria [@satopaa2011kneedle].  
- Generates publication-ready artifacts (figures, tables, CSV/LaTeX outputs) with complete metadata.  
- Ensures one-command reproducibility through fixed random seeds and version-pinned dependencies.  

The toolkit supports both researchers and educators who seek transparent and auditable SVD experiments that bridge mathematical theory with computational practice. It also offers a flexible foundation for reproducible teaching laboratories and algorithmic research.

# Novelty and Relation to Prior Work

Many existing libraries provide SVD or PCA functions, and some illustrate their use for specific tasks such as compression or denoising. **SVDlab**, however, contributes in several distinctive ways:

1. **Cross-task standardization** — a single interface for compression, denoising, and PCA, ensuring consistent preprocessing, metrics, and outputs. Storage is reported with a simple parameter-count proxy \(mk + nk + k\) for \(U_k,\Sigma_k,V_k\), enabling like-for-like comparisons across methods and ranks.  
2. **Adaptive, task-agnostic rank selection** — a hybrid cumulative-energy plus elbow-based rule that yields stable $k^*$ across datasets and domains [@satopaa2011kneedle], grounded in the classical theory of low-rank approximation [@eckart1936approximation].  
3. **Reproducible artifacts by design** — all figures and tables are generated deterministically from four commands, aligning with current best practices in reproducible computational research [@wilkinson2016fair; @stodden2016enhancing].  
4. **Methodological breadth** — standardized benchmarking of SVD, EVD, and QR factorizations under identical conditions, using widely accepted image-quality metrics (PSNR and SSIM) [@gu1996efficient; @wang2004ssim].  

By combining these elements, **SVDlab** turns theoretical linear algebra concepts into a coherent, hands-on toolkit that is equally suitable for research, teaching, and reproducible experimentation.

# Features

- Adaptive rank selection that merges cumulative-energy and Kneedle-based elbow detection [@satopaa2011kneedle].  
- Four ready-to-use Python scripts covering image compression, denoising, PCA, and benchmarking.  
- Deterministic results guaranteed by fixed random seeds and pinned dependencies.  
- Automatic export of results as PDF, CSV, and LaTeX tables.  
- Cross-platform support (Linux, macOS, Windows; Python ≥ 3.10).  
- Lightweight execution: all figures and tables from this paper can be generated with four simple commands.  
- Sensible defaults and graceful fallbacks — e.g., \(\tau\approx 0.995\) for high-fidelity compression and \(\tau\in[0.95,0.99]\) for denoising; if elbow detection is inconclusive, the energy rule is used.  
- PCA parity — cumulative energy \(\eta_k\) coincides with explained variance for mean-centered data, so the same policy selects the number of components.

**Adaptive rank selection (concise):**
1) Compute singular values \(\{\sigma_i\}\) and cumulative energy \(\eta_k=\sum_{i\le k}\sigma_i^2/\sum_{i}\sigma_i^2\).  
2) Threshold rule: \(k_\tau=\min\{k:\eta_k\ge\tau\}\).  
3) Elbow rule: apply Kneedle to the cumulative curve to get \(k_e\).  
4) Default: \(k^*=\max(k_\tau,k_e)\). (Aggressive option: \(\min(k_\tau,k_e)\).)  
5) Fallbacks: if no elbow is detected, use \(k_\tau\); clip \(k\) to \([1,r]\).

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
Each script automatically saves results under results/Figures/ and results/Tables/, allowing reviewers and readers to reproduce every figure and table from a clean environment. All scripts fix random seeds and pin dependency versions to ensure deterministic runs.

# Limitations and Future Work
The current scope targets exact (non-randomized) factorizations on moderate-scale problems; large-scale randomized and streaming variants are future work. The present release focuses on deterministic CPU-based implementations; performance and rank stability may vary on GPU backends. Future versions will include randomized SVD for large-scale problems [@halko2011randomized], support for color images and video, and optional GPU acceleration — while maintaining full reproducibility and auditability.

# Acknowledgements
We are deeply grateful to the open-source communities behind NumPy, SciPy, scikit-image, scikit-learn, and Matplotlib, whose work made this toolkit possible. I also thank my colleagues at Yerevan State University and Provectus for their continuous support and feedback. The complete source and archived release are available at Zenodo (DOI: 10.5281/zenodo.17313445).


# References