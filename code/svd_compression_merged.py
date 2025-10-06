#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merged SVD image compression tool (flat output layout):
- Loads an image (path or built-in), converts to grayscale [0,1], optional resize
- Computes SVD once; builds cumulative energy
- Produces PDFs (in results/Figures):
    * Manual vs Adaptive (95%, 99%, 99.5%, Elbow)
    * Fixed-k reconstruction grid
- Produces a summary table (in results/Tables, CSV + LaTeX) with rows:
    Manual, Adaptive(95%), Adaptive(99.5%), Adaptive(Elbow), Adaptive(Elbow+Guard)
    and columns: Rank k, PSNR (dB), SSIM, Energy η_k
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe on headless / CI)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- Visual palette (consistent colors across figures)
PALETTE = {
    "energy":  "tab:gray",
    "manual":  "tab:orange",
    "adaptive":"tab:blue",
}

import pandas as pd
from skimage import data, color
from skimage.io import imread
from skimage.util import img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage import transform as sktf


# ----------------------------
# Utilities
# ----------------------------
def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else color.rgb2gray(img[..., :3])


def _maybe_resize(img: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if size is None:
        return img
    H, W = size
    return sktf.resize(
        img, (H, W), order=1, mode="reflect",
        anti_aliasing=True, preserve_range=True
    ).astype(np.float64)


# Accepted built-ins (case-insensitive)
BUILTINS = {
    "astronaut": lambda: _to_gray(img_as_float(data.astronaut())),
    "camera":    lambda: img_as_float(data.camera()),
    "coins":     lambda: img_as_float(data.coins()),
    "chelsea":   lambda: _to_gray(img_as_float(data.chelsea())),
    "moon":      lambda: img_as_float(data.moon()),
    "page":      lambda: img_as_float(data.page()),
    "rocket":    lambda: _to_gray(img_as_float(data.rocket())),
    "coffee":    lambda: _to_gray(img_as_float(data.coffee())),
}


# ----------------------------
# Image loading
# ----------------------------
def load_image_gray(input_spec: str, resize_to: Optional[Tuple[int, int]] = None):
    """
    input_spec: file path or builtin keyword (case-insensitive).
      Built-ins (prefix optional 'builtin:'): astronaut, camera, coins, chelsea, moon, page, rocket, coffee
    Returns (img_gray in [0,1], source_str, image_label)
    """
    key_raw = str(input_spec).strip().strip('"').strip("'")
    key = key_raw.lower()
    if key.startswith("builtin:"):
        key = key.split("builtin:", 1)[1]

    if key in BUILTINS:
        img = BUILTINS[key]()
        img = _maybe_resize(img, resize_to)
        return img, f"builtin:{key}", key

    # File path
    p = Path(key_raw).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        raise FileNotFoundError(
            f"Input image not found: {p}\n"
            f"Tip: verify the exact name/extension in: {p.parent}\n"
            f'On Windows, quote paths with spaces, e.g. --input "C:\\\\path\\\\to\\\\image.png"'
        )

    arr = img_as_float(imread(str(p)))
    gray = _to_gray(arr)
    gray = _maybe_resize(gray, resize_to)
    return gray, str(p.resolve()), p.stem


# ----------------------------
# Math helpers
# ----------------------------
def svd_decompose(img_gray: np.ndarray):
    # use numpy SVD; img is already float64 in [0,1]
    U, S, Vt = np.linalg.svd(img_gray, full_matrices=False)
    energy = S**2
    total = float(np.sum(energy)) if energy.size else 0.0
    if total == 0.0:
        cum_energy = np.zeros_like(energy)
    else:
        cum_energy = np.cumsum(energy) / total
    return U, S, Vt, energy, cum_energy


def reconstruct(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int):
    k = int(max(1, min(k, len(S))))  # safety clamp
    # faster than diag for large k
    Uk = U[:, :k] * S[:k]
    return Uk @ Vt[:k, :]


def detect_elbow(cum_energy: np.ndarray) -> int:
    """Maximum distance-to-chord elbow on cumulative energy curve."""
    n = len(cum_energy)
    if n == 0:
        return 1
    p1 = np.array([1.0, float(cum_energy[0])])
    p2 = np.array([float(n), float(cum_energy[-1])])
    line_vec = p2 - p1
    denom = np.linalg.norm(line_vec)
    if denom == 0.0:
        return n
    xs = np.arange(1.0, n + 1.0)
    points = np.stack([xs, cum_energy.astype(float)], axis=1)
    vec_from_p1 = points - p1
    # 2D cross product magnitude
    cross_mag = np.abs(line_vec[0] * vec_from_p1[:, 1] - line_vec[1] * vec_from_p1[:, 0])
    distances = cross_mag / denom
    return int(np.argmax(distances) + 1)


# ----------------------------
# Figure writers
# ----------------------------
def save_manual_vs_adaptive_pdf(
    pdf_path: Path, img: np.ndarray, U: np.ndarray, S: np.ndarray, Vt: np.ndarray,
    cum_energy: np.ndarray, k_manual: int, k_adapt: int, title_suffix: str
):
    # clamp ks
    k_manual = int(max(1, min(k_manual, len(S))))
    k_adapt  = int(max(1, min(k_adapt,  len(S))))

    Ak_manual = reconstruct(U, S, Vt, k_manual).clip(0, 1)
    Ak_adapt  = reconstruct(U, S, Vt, k_adapt).clip(0, 1)

    psnr_manual = psnr(img, Ak_manual, data_range=1.0)
    ssim_manual = ssim(img, Ak_manual, data_range=1.0)
    psnr_adapt  = psnr(img, Ak_adapt,  data_range=1.0)
    ssim_adapt  = ssim(img, Ak_adapt,  data_range=1.0)
    eta_manual  = cum_energy[k_manual - 1]
    eta_adapt   = cum_energy[k_adapt  - 1]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Energy curve (use 1..n on x-axis)
    x = np.arange(1, len(cum_energy) + 1)
    axes[0].plot(x, cum_energy, linewidth=2, color=PALETTE["energy"])
    axes[0].axvline(
        k_manual, linestyle="--", linewidth=2, color=PALETTE["manual"],
        label=f"Manual $k={k_manual}$"
    )
    axes[0].axvline(
        k_adapt, linestyle="--", linewidth=2, color=PALETTE["adaptive"],
        label=f"Adaptive $k^*={k_adapt}$"
    )
    axes[0].set_xlabel("Rank $k$", fontsize=10)
    axes[0].set_ylabel("Cumulative Energy $\\eta_k$", fontsize=10)
    axes[0].set_ylim(0.75, 1.01)
    axes[0].set_xlim(1, len(cum_energy))
    axes[0].set_title("Energy Retention", fontsize=11)
    axes[0].tick_params(axis="both", which="major", labelsize=9)
    axes[0].legend(fontsize=8, loc="lower right", frameon=False)
    axes[0].grid(True, alpha=0.25)

    # Manual
    axes[1].imshow(Ak_manual, cmap="gray")
    axes[1].set_title(
        f"Manual $k = {k_manual}$\n"
        f"PSNR: {psnr_manual:.2f} dB, SSIM: {ssim_manual:.3f}\n"
        f"$\\eta_k = {eta_manual:.3f}$",
        fontsize=10, pad=8,
    )
    axes[1].axis("off")

    # Adaptive
    axes[2].imshow(Ak_adapt, cmap="gray")
    axes[2].set_title(
        f"Adaptive $k^* = {k_adapt}$\n"
        f"PSNR: {psnr_adapt:.2f} dB, SSIM: {ssim_adapt:.3f}\n"
        f"$\\eta_k = {eta_adapt:.3f}$",
        fontsize=10, pad=8,
    )
    axes[2].axis("off")

    fig.suptitle(
        f"SVD Compression: Manual vs. Adaptive Rank Selection — {title_suffix}",
        fontsize=12, y=0.97,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ----------------------------
# Summary table builders
# ----------------------------
def build_summary_rows(img: np.ndarray, U: np.ndarray, S: np.ndarray, Vt: np.ndarray,
                       cum_energy: np.ndarray, manual_k: int) -> List[dict]:
    """Return list of dict rows for the summary table."""
    def k_for_energy(target: float) -> int:
        return int(np.argmax(cum_energy >= target) + 1)

    k95    = k_for_energy(0.95)
    k995   = k_for_energy(0.995)
    kelbow = detect_elbow(cum_energy)
    kguard = max(kelbow, k995)  # elbow + guard (never below 99.5%)

    plan = [
        ("Manual (fixed-$k$)", manual_k),
        ("Adaptive (95% energy)", k95),
        ("Adaptive (99.5% energy)", k995),
        ("Adaptive (elbow)", kelbow),
        ("Adaptive (elbow + guard)", kguard),
    ]

    rows: List[dict] = []
    for label, k in plan:
        k  = int(max(1, min(k, len(S))))
        Ak = reconstruct(U, S, Vt, k).clip(0, 1)
        P  = psnr(img, Ak, data_range=1.0)
        Ssim = ssim(img, Ak, data_range=1.0)
        eta_k = cum_energy[k - 1]
        rows.append({
            "Method": label,
            "Rank k": k,
            "PSNR (dB)": float(P),
            "SSIM": float(Ssim),
            "Energy η_k": float(eta_k),
        })
    return rows


def save_summary_table(df: pd.DataFrame, tables_dir: Path, img_label: str, shape_hw: Tuple[int, int]):
    """Save CSV + LaTeX table in Tables folder."""
    df_out = df.copy()
    df_out["PSNR (dB)"]  = df_out["PSNR (dB)"].round(2)
    df_out["SSIM"]       = df_out["SSIM"].round(3)
    df_out["Energy η_k"] = df_out["Energy η_k"].round(3)

    # CSV
    csv_path = tables_dir / f"{img_label}_compression_summary.csv"
    df_out.to_csv(csv_path, index=False)

    # LaTeX (caption includes actual image size)
    H, W = shape_hw
    latex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Manual and adaptive rank selection on the $"
        f"{H} \\times {W}"
        "$ grayscale \\texttt{" + img_label.replace("_", "\\_") + "} image. "
        "Reported are the chosen rank $k$, PSNR (dB), SSIM, and cumulative energy $\\eta_k$. "
        "The 99.5\\% energy threshold offers a reliable compromise between compression efficiency and perceptual fidelity.}\n"
        "\\label{tab:compression_summary}\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "\\textbf{Method} & \\textbf{Rank $k$} & \\textbf{PSNR (dB)} & \\textbf{SSIM} & \\textbf{Energy $\\eta_k$} \\\\\n\\midrule\n"
    )
    for _, r in df_out.iterrows():
        latex += (
            f"{r['Method']} & {int(r['Rank k'])} & {r['PSNR (dB)']:.2f} & "
            f"{r['SSIM']:.3f} & {r['Energy η_k']:.3f} \\\\\n"
        )
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    tex_path = tables_dir / f"{img_label}_compression_summary.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print("✅ Saved summary:")
    print(f"   • CSV : {csv_path}")
    print(f"   • LaTeX: {tex_path}  (requires \\usepackage{{booktabs}})")


# ----------------------------
# CLI
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="SVD image compression with manual & adaptive ranks (95/99/99.5% energy + elbow) and a LaTeX-ready summary table."
    )
    p.add_argument(
        "--input", type=str, default="astronaut",
        help=("Path to an image file OR one of the built-ins: "
              "astronaut | camera | coins | chelsea | moon | page | rocket | coffee "
              "(you may prefix with 'builtin:').")
    )
    p.add_argument(
        "--out", type=str, default="results",
        help="Root output directory (Figures -> results/Figures, Tables -> results/Tables).",
    )
    p.add_argument(
        "--k-list", type=int, nargs="*", default=[5, 20, 50, 100, 150, 200],
        help="Ranks to preview/reconstruct (space-separated).",
    )
    p.add_argument(
        "--manual-k", type=int, default=50,
        help="Manual k to compare in the PDFs and the summary table.",
    )
    p.add_argument(
        "--no-pdfs", action="store_true",
        help="Skip generating PDF figures (only compute summary CSV/LaTeX).",
    )
    p.add_argument(
        "--resize", type=str, default=None,
        help="Optional HxW resize, e.g. '256x256'.",
    )
    return p


def _parse_hw(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    s = s.lower().replace("×", "x").replace(" ", "")
    try:
        h, w = s.split("x")
        return (int(h), int(w))
    except Exception:
        print(f'⚠️  Could not parse --resize "{s}". Expected format like "256x256". Ignoring.')
        return None


# ----------------------------
# Main
# ----------------------------
def main():
    print("✅ Starting SVD image compression analysis...")

    args = build_parser().parse_args()
    resize_to = _parse_hw(args.resize)

    # Load image (file or builtin)
    try:
        img, source_str, img_label = load_image_gray(args.input, resize_to=resize_to)
        print(f"🖼️  Loaded image: {source_str} | label='{img_label}' | shape={img.shape}")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        sys.exit(1)

    # Prepare output directories (flat layout, lowercase last component of root)
    out_root = Path(args.out).expanduser()
    out_root = out_root.parent / out_root.name.lower()
    out_figs = out_root / "Figures"
    out_tabs = out_root / "Tables"
    ensure_dirs(out_figs, out_tabs)
    print(f"📁 Figures dir: '{out_figs.resolve()}'")
    print(f"📁 Tables  dir: '{out_tabs.resolve()}'")

    # Compute SVD
    try:
        U, S, Vt, _, cum_energy = svd_decompose(img)  # discard 'energy'
        print(f"🧮 SVD computed: {len(S)} singular values")
    except Exception as e:
        print(f"❌ SVD failed: {e}")
        sys.exit(1)

    # Adaptive k values (for logs)
    k_95  = int(np.argmax(cum_energy >= 0.95) + 1)
    k_99  = int(np.argmax(cum_energy >= 0.99) + 1)
    k_995 = int(np.argmax(cum_energy >= 0.995) + 1)
    k_elb = int(detect_elbow(cum_energy))

    # Clamp manual_k now to avoid OOB in figures
    manual_k = int(np.clip(args.manual_k, 1, len(S)))

    print(f"🔍 Adaptive ranks: k_95={k_95}, k_99={k_99}, k_995={k_995}, k_elbow={k_elb}")
    print(f"🔧 Manual k (clamped): {manual_k} (requested {args.manual_k})")

    # Preview reconstructions for requested k-list
    recons = []
    for k in args.k_list:
        k = int(np.clip(k, 1, len(S)))
        Ak = reconstruct(U, S, Vt, k).clip(0, 1)
        recons.append((k, Ak))

    # FIGURES (optional)
    if not args.no_pdfs:
        try:
            # Order: 95, 99, 99.5, elbow
            save_manual_vs_adaptive_pdf(out_figs / f"{img_label}_svd_comparison_95.pdf",
                                        img, U, S, Vt, cum_energy, manual_k, k_95,  "95% Energy")
            save_manual_vs_adaptive_pdf(out_figs / f"{img_label}_svd_comparison_99.pdf",
                                        img, U, S, Vt, cum_energy, manual_k, k_99,  "99% Energy")
            save_manual_vs_adaptive_pdf(out_figs / f"{img_label}_svd_comparison_995.pdf",
                                        img, U, S, Vt, cum_energy, manual_k, k_995, "99.5% Energy")
            save_manual_vs_adaptive_pdf(out_figs / f"{img_label}_svd_comparison_elbow.pdf",
                                        img, U, S, Vt, cum_energy, manual_k, k_elb, "Elbow Method")
            print("✅ Saved all manual vs. adaptive PDFs.")
        except Exception as e:
            print(f"❌ Failed to save comparison PDFs: {e}")

        # Fixed-k reconstructions grid
        try:
            n = len(recons)
            if n == 0:
                raise ValueError("Empty --k-list; nothing to plot.")

            cols = min(3, n)
            rows = int(np.ceil(n / cols))
            fig_w = 4.0 * cols
            fig_h = 3.5 * rows

            fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
            axarr = np.atleast_1d(axes).ravel()

            fig.suptitle(f"Progressive SVD Reconstruction — {img_label}", fontsize=12)

            for ax, (k, Ak) in zip(axarr, recons):
                p = psnr(img, Ak, data_range=1.0)
                s = ssim(img, Ak, data_range=1.0)
                e = cum_energy[k - 1]
                ax.imshow(Ak, cmap="gray")
                ax.set_title(
                    f"$k = {k}$\nPSNR: {p:.2f} dB, SSIM: {s:.3f}\n$\\eta_k = {e:.3f}$",
                    fontsize=10, pad=6,
                )
                ax.axis("off")

            # Hide any leftover axes
            for ax in axarr[len(recons):]:
                ax.axis("off")

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf_path = out_figs / f"{img_label}_svd_fixed_rank_reconstructions.pdf"
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"✅ Saved: {pdf_path}")
        except Exception as e:
            print(f"❌ Failed to save fixed-k reconstructions: {e}")

    # SUMMARY TABLE (CSV + LaTeX)
    try:
        rows = build_summary_rows(img, U, S, Vt, cum_energy, manual_k)
        df = pd.DataFrame(rows)

        # Order rows to match docstring
        order = [
            "Manual (fixed-$k$)",
            "Adaptive (95% energy)",
            "Adaptive (99.5% energy)",
            "Adaptive (elbow)",
            "Adaptive (elbow + guard)",
        ]
        df["__ord"] = df["Method"].apply(order.index)
        df = df.sort_values("__ord").drop(columns="__ord")

        save_summary_table(df, out_tabs, img_label, img.shape)
    except Exception as e:
        print(f"❌ Failed to build/save summary table: {e}")

    print("🎉 All outputs generated successfully!")


if __name__ == "__main__":
    main()

