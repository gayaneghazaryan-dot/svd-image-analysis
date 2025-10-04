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
  - name: Provectus, Armenia
    index: 2
date: 2025-10-04
bibliography: paper.bib
---

# Summary

**SVDlab** is an open-source Python toolkit that operationalizes the Singular Value Decomposition (SVD) for three canonical tasks—image compression, image denoising, and principal component analysis (PCA)—using a *task-agnostic adaptive rank selection rule*.  
The adaptive policy combines a cumulative-energy threshold with elbow detection (Kneedle algorithm) to select a stable truncation rank \(k^\star=\max(k_\tau,k_e)\), balancing reconstruction fidelity and over-truncation.  

The toolkit standardizes datasets, metrics (PSNR, SSIM, energy retention, runtime), and figure/table generation, ensuring that **all results are reproducible from four one-line commands**.  
Benchmarks against eigenvalue decomposition (EVD) and pivoted thin QR clarify accuracy–speed trade-offs, showing that adaptive rank selection improves reproducibility and consistency across noise levels and hardware backends.

# Statement of Need

Although SVD is central to modern data analysis, *rank selection*—the choice of how many singular values to retain—remains inconsistent across studies and implementations.  
This variability undermines reproducibility and comparability among applications such as image compression, denoising, and PCA.  

**SVDlab** addresses this gap by providing a single, reproducible framework that:

- Unifies preprocessing, rank selection, and evaluation across tasks.
- Implements a principled adaptive rule combining cumulative-energy and elbow criteria.
- Produces publication-ready artifacts (figures, tables, CSV/LaTeX outputs) with recorded metadata.
- Enables one-command reproducibility and deterministic outputs via fixed seeds and version-pinned dependencies.

The toolkit benefits researchers and educators seeking transparent, auditable SVD experiments that bridge mathematical theory and computational practice.  
It supports direct benchmarking of factorization methods (SVD, EVD, QR) and offers a lightweight foundation for reproducible teaching labs or further algorithmic research.

# Features

- 🧠 **Adaptive rank selection** combining cumulative-energy thresholds with Kneedle-based elbow detection.  
- ⚙️ **Four unified scripts** for image compression, denoising, PCA, and benchmarking.  
- 🧩 **Reproducibility by design**: fixed random seeds, pinned dependencies, and CPU determinism.  
- 📊 **Metrics and outputs**: PSNR, SSIM, energy retention, runtime, with automatic PDF/CSV/LaTeX generation.  
- 🧱 **Cross-platform** (Linux, macOS, Windows; Python ≥ 3.10).  
- 🪶 **Lightweight execution**: each task reproduces all paper figures/tables via a single command.  

# Example Usage

```bash
# 1) Image compression
python code/svd_compression_merged.py

# 2) Image denoising
python code/svd_denoising.py

# 3) Factorization benchmarks
python code/benchmark_and_plots.py

# 4) PCA with adaptive component selection
python code/pca_adaptive_combined.py
```
Each script regenerates fixed outputs under results/Figures/ and results/Tables/, enabling reviewers to reproduce all results from a clean environment.

# Acknowledgements

The authors thank the open-source communities of NumPy, SciPy, scikit-image, scikit-learn, and Matplotlib.  
Institutional support from Yerevan State University and Provectus Armenia is gratefully acknowledged.


