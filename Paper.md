---
title: "SVDlab: A Reproducible Toolkit for SVD-based Image Compression, Denoising, and PCA with Adaptive Rank Selection"
tags:
  - Python
  - Singular Value Decomposition
  - Image Compression
  - Denoising
  - Principal Component Analysis
  - Reproducible Research
authors:
  - name: Gayane Ghazaryan
    affiliation: 1
  - name: Artashes Ghazaryan
    affiliation: 2
affiliations:
  - name: Yerevan State University, Yerevan, Armenia
    index: 1
  - name: Provectus
    index: 2
date: 4 October 2025
bibliography: paper.bib
---

# Summary

**SVDlab** is an open-source Python toolkit that operationalizes the Singular Value Decomposition (SVD) for three canonical tasks—image compression, image denoising, and principal component analysis (PCA)—using a *task-agnostic adaptive rank selection rule*. The adaptive policy combines a cumulative-energy threshold with elbow detection (Kneedle algorithm) to select a stable truncation rank \(k^\star=\max(k_\tau,k_e)\), balancing reconstruction fidelity and over-truncation [@satopaa2011kneedle; @eckart1936approximation; @jolliffe2016pca].

The toolkit standardizes datasets, metrics (PSNR, SSIM, energy retention, runtime), and figure/table generation, ensuring that **all results are reproducible from four one-line commands**. Benchmarks against eigenvalue decomposition (EVD) and pivoted thin QR clarify accuracy–speed trade-offs, showing that adaptive rank selection improves reproducibility and consistency across noise levels and hardware backends [@gu1996efficient; @wang2004ssim].

# Statement of Need

Although SVD is central to modern data analysis and imaging [@golub2013matrix; @andrews1976svd], *rank selection*—the choice of how many singular values to retain—remains inconsistent across studies and implementations. This variability undermines reproducibility and comparability in applications such as image compression, denoising, and PCA [@jolliffe2016pca]. The broader research community increasingly expects computational work to follow **FAIR** and reproducible-research practices [@wilkinson2016fair; @stodden2016enhancing], yet turnkey toolkits that unify preprocessing, rank selection, metrics, and artifact generation across tasks are uncommon.

**SVDlab** addresses this gap by providing a single, reproducible framework that:

- Unifies preprocessing, adaptive rank selection, and evaluation across multiple tasks.  
- Implements a principled adaptive rule combining cumulative-energy and elbow criteria [@satopaa2011kneedle].  
- Generates publication-ready artifacts (figures, tables, CSV/LaTeX outputs) with recorded metadata.  
- Enables one-command reproducibility and deterministic results through fixed seeds and version-pinned dependencies.

The toolkit benefits researchers and educators seeking transparent, auditable SVD experiments that bridge mathematical theory and computational practice. It also supports systematic benchmarking of matrix factorizations (SVD, EVD, QR) and serves as a lightweight foundation for reproducible teaching labs or algorithmic research.

# Novelty & Relation to Prior Work

Many libraries expose SVD/PCA primitives or demonstrate task-specific uses (e.g., compression or denoising), but **SVDlab** contributes:

1. **Cross-task standardization**: a *single* interface for compression, denoising, and PCA with consistent preprocessing, metrics, and outputs, facilitating like-for-like comparisons.  
2. **Adaptive rank selection, task-agnostic**: a combined cumulative-energy + elbow policy that yields stable \(k^\star\) across tasks and datasets [@satopaa2011kneedle], grounded in classical low-rank approximation theory [@eckart1936approximation].  
3. **Reproducible artifacts by default**: all figures/tables regenerate from four commands with fixed seeds and pinned dependencies, aligning with modern reproducibility guidance [@wilkinson2016fair; @stodden2016enhancing].  
4. **Methodological breadth**: side-by-side SVD/EVD/QR comparisons under identical conditions, with rank-revealing QR included as a baseline [@gu1996efficient] and standardized quality metrics (PSNR/SSIM) [@wang2004ssim].

# Features

- **Adaptive rank selection** combining cumulative-energy thresholds with Kneedle-based elbow detection [@satopaa2011kneedle].
- **Four unified scripts** for image compression, denoising, PCA, and benchmarking.
- **Reproducibility by design**: fixed random seeds, pinned dependencies, and CPU determinism.
- **Metrics and outputs**: PSNR, SSIM [@wang2004ssim], energy retention, runtime, with automatic PDF/CSV/LaTeX generation.
- **Cross-platform** (Linux, macOS, Windows; Python ≥ 3.10).
- **Lightweight execution**: each task reproduces all paper figures/tables via a single command.

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

Each script regenerates fixed outputs under `results/Figures/` and `results/Tables/`, enabling reviewers to reproduce all results from a clean environment.

# Limitations & Future Work

The current release emphasizes deterministic CPU backends for reproducibility and uses exact factorizations; performance and rank stability on heterogeneous GPU backends may vary. Future work includes optional randomized SVD for large-scale problems, extended datasets/tasks (e.g., color images and video), and GPU-accelerated paths while preserving auditability.

# Acknowledgements

The authors thank the open-source communities behind NumPy, SciPy, scikit-image, scikit-learn, and Matplotlib. We also acknowledge foundational references in linear algebra, SVD/PCA, and inverse problems that inform this work. Institutional support from Yerevan State University and Provectus is gratefully acknowledged.



