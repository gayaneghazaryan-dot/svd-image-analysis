#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined PCA (via SVD) with clean outputs.

Figures  -> results/Figures (PDF):
  - pca_rank_selection.pdf : cumulative variance with τ and selected k*
  - pca_2d_scatter.pdf     : 2D PCA scatter (PC1 vs PC2) with class legend

Tables   -> results/Tables (CSV):
  - pca_variance_report.csv     : per-component singular values, variance, explained & cumulative ratios
  - pca_explained_variance.csv  : lighter report (explained & cumulative ratios only)
"""
import argparse
import os
import csv
from typing import Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless/CI
import matplotlib.pyplot as plt


# ----------------------------
# Data helpers
# ----------------------------
def make_synthetic(n_per: int = 50, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    C1 = rng.normal([2.0, 0.0, 0.0, 0.0], 0.35, size=(n_per, 4))
    C2 = rng.normal([0.0, 2.0, 0.0, 0.0], 0.35, size=(n_per, 4))
    C3 = rng.normal([0.0, 0.0, 2.0, 0.0], 0.35, size=(n_per, 4))
    X = np.vstack([C1, C2, C3]).astype(float)
    y = np.array([0]*n_per + [1]*n_per + [2]*n_per)
    names = ["class-0", "class-1", "class-2"]
    return X, y, names


def load_data(dataset_mode: str = "auto", n_per: int = 50, seed: int = 42):
    mode = str(dataset_mode).lower().strip()
    if mode == "synthetic":
        return make_synthetic(n_per=n_per, seed=seed)
    if mode == "iris":
        try:
            from sklearn.datasets import load_iris  # type: ignore
        except Exception:
            return make_synthetic(n_per=n_per, seed=seed)
        iris = load_iris()
        return iris.data.astype(float), iris.target, list(iris.target_names)
    # auto: try iris, else synthetic
    try:
        from sklearn.datasets import load_iris  # type: ignore
        iris = load_iris()
        return iris.data.astype(float), iris.target, list(iris.target_names)
    except Exception:
        return make_synthetic(n_per=n_per, seed=seed)


# ----------------------------
# Core PCA via SVD
# ----------------------------
def pca_svd(X: np.ndarray):
    """Mean-center X, compute SVD, return (U, S, Vt) and variance curves."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    variances = S**2
    total = float(np.sum(variances)) if variances.size else 0.0
    if total == 0.0:
        explained = np.zeros_like(variances)
        cumulative = np.zeros_like(variances)
    else:
        explained = variances / total
        cumulative = np.cumsum(explained)
    return U, S, Vt, variances, explained, cumulative


# ----------------------------
# CLI
# ----------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        description="PCA (via SVD) with adaptive rank selection and clean outputs."
    )
    ap.add_argument('--outdir', type=str, default='results',
                    help='Root output directory (creates results/Figures and results/Tables).')
    ap.add_argument('--tau', type=float, default=0.95,
                    help='Target cumulative variance threshold in [0,1].')
    ap.add_argument('--dataset', choices=['auto', 'iris', 'synthetic'], default='auto',
                    help='Dataset to use.')
    ap.add_argument('--n-per', type=int, default=50,
                    help='Samples per synthetic class (used when dataset=synthetic or when falling back).')
    ap.add_argument('--seed', type=int, default=42,
                    help='Random seed for synthetic data.')
    return ap


# ----------------------------
# Main
# ----------------------------
def main():
    args = build_parser().parse_args()

    # --- Create output folders (force lowercase last component, like other tools) ---
    outroot = os.path.abspath(args.outdir)
    parent, base = os.path.dirname(outroot), os.path.basename(outroot)
    outroot = os.path.join(parent, base.lower())
    figs_dir = os.path.join(outroot, 'Figures')
    tabs_dir = os.path.join(outroot, 'Tables')
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(tabs_dir, exist_ok=True)

    # --- Load data ---
    X, y, names = load_data(args.dataset, n_per=args.n_per, seed=args.seed)

    # --- PCA via SVD ---
    U, S, Vt, variances, explained, cumulative = pca_svd(X)
    ks = np.arange(1, len(cumulative) + 1)
    if len(cumulative):
        k_tau_raw = int(np.searchsorted(cumulative, float(args.tau)) + 1)
        k_tau = int(max(1, min(k_tau_raw, len(S))))
    else:
        k_tau = 0

    # --- Figure 1: cumulative variance curve ---
    plt.figure(figsize=(7, 4.5))
    if len(cumulative):
        plt.plot(ks, cumulative, marker='o', linewidth=2)
    if 0.0 <= args.tau <= 1.0 and k_tau >= 1:
        plt.axhline(y=args.tau, linestyle='--')
        plt.axvline(x=k_tau, linestyle='--')
        plt.title(f'Adaptive Rank (PCA via SVD): τ={args.tau:.2f}, k*={k_tau}')
    else:
        plt.title('Adaptive Rank (PCA via SVD)')
    plt.xlabel('Number of Components (k)')
    plt.ylabel('Cumulative Variance Explained')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = os.path.join(figs_dir, 'pca_rank_selection.pdf')
    plt.savefig(curve_path, bbox_inches='tight')
    plt.close()

    # --- CSV (full) ---
    full_csv = os.path.join(tabs_dir, 'pca_variance_report.csv')
    with open(full_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['component_index', 'singular_value', 'variance', 'explained_ratio', 'cumulative_ratio'])
        for i, (s, var, r, c) in enumerate(zip(S, variances, explained, cumulative), start=1):
            w.writerow([i, float(s), float(var), float(r), float(c)])
        w.writerow([])
        w.writerow(['tau', float(args.tau)])
        w.writerow(['k_tau', int(k_tau)])
        w.writerow(['dataset', args.dataset])

    # --- Figure 2: 2D PCA scatter (PC1 vs PC2) ---
    pct1 = 100.0 * explained[0] if len(explained) >= 1 else 0.0
    pct2 = 100.0 * explained[1] if len(explained) >= 2 else 0.0
    if len(S) >= 2:
        X_pca = U[:, :2] @ np.diag(S[:2])
    else:
        X_pca = np.zeros((X.shape[0], 2), dtype=float)

    plt.figure(figsize=(6.5, 5.5))
    uniq = np.unique(y)
    for lbl in uniq:
        mask = (y == lbl)
        label_name = names[int(lbl)] if int(lbl) < len(names) else str(lbl)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=label_name, edgecolors='k', linewidths=0.5)
    plt.xlabel(f'PC1 ({pct1:.1f}% var)')
    plt.ylabel(f'PC2 ({pct2:.1f}% var)')
    if k_tau >= 1:
        plt.title(f'PCA (SVD) — τ={args.tau:.2f}, k*={k_tau}')
    else:
        plt.title('PCA (SVD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_path = os.path.join(figs_dir, 'pca_2d_scatter.pdf')
    plt.savefig(scatter_path, bbox_inches='tight')
    plt.close()

    # --- CSV (light) ---
    light_csv = os.path.join(tabs_dir, 'pca_explained_variance.csv')
    with open(light_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['component_index', 'explained_ratio', 'cumulative_ratio'])
        for i, (r, c) in enumerate(zip(explained, cumulative), start=1):
            w.writerow([i, float(r), float(c)])
        w.writerow([])
        w.writerow(['tau', float(args.tau)])
        w.writerow(['k_tau', int(k_tau)])
        w.writerow(['dataset', args.dataset])

    # --- Console summary ---
    print('Saved:', curve_path)
    print('Saved:', scatter_path)
    print('Saved:', full_csv)
    print('Saved:', light_csv)
    print(f'Dataset: {args.dataset}')
    print(f'τ={args.tau:.2f}, k*={k_tau}, PC1={pct1:.1f}%, PC2={pct2:.1f}%')


if __name__ == '__main__':
    main()
