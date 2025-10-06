#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SVD image denoising: manual vs adaptive rank selection (Kneedle on cumulative energy).

Results/
  Figures/
    <img_label>_spectrum.png
    <img_label>_adaptive_panel.png
    <img_label>_denoising.png
  Tables/
    <img_label>_denoising_comparison.csv
    <img_label>_denoising_comparison.xlsx
    <img_label>_denoising_comparison.tex
    <img_label>_results_summary.json
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

# --- Visual palette (kept for consistency if you expand later)
PALETTE = {
    "spectrum":   "tab:purple",
    "cum_raw":    "tab:gray",
    "k95":        "tab:green",
    "knee":       "tab:red",
    "k_star":     "tab:red",   # dashed red line in panel
}

import pandas as pd
from skimage import transform as sktf
from skimage import io, data, color
from skimage.util import img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from scipy.linalg import svd
from scipy.signal import savgol_filter

# Optional Kneedle
try:
    from kneed import KneeLocator  # type: ignore
except ImportError:
    KneeLocator = None


# -------------------------------
# Utilities
# -------------------------------
def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


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


BUILTINS = {
    "astronaut": lambda: _to_gray(img_as_float(data.astronaut())),
    "camera":    lambda: img_as_float(data.camera()),
    "moon":      lambda: img_as_float(data.moon()),
    "coins":     lambda: img_as_float(data.coins()),
    "page":      lambda: img_as_float(data.page()),
    "rocket":    lambda: _to_gray(img_as_float(data.rocket())),
    "coffee":    lambda: _to_gray(img_as_float(data.coffee())),
    "chelsea":   lambda: _to_gray(img_as_float(data.chelsea())),
}


def load_image_gray(path: Optional[str], resize_to: Optional[Tuple[int, int]] = None):
    if path is None:
        img = BUILTINS["astronaut"]()
        return (_maybe_resize(img, resize_to), "builtin:astronaut", "astronaut")

    key_raw = str(path).strip().strip('"').strip("'")
    key = key_raw.lower()
    if key.startswith("builtin:"):
        key = key.split("builtin:", 1)[1]

    if key in BUILTINS:
        img = BUILTINS[key]()
        img = _maybe_resize(img, resize_to)
        return (img, f"builtin:{key}", key)

    p = Path(key_raw).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        raise FileNotFoundError(
            f"Input image not found: {p}\n"
            f"Tip: check the exact filename and extension in: {p.parent}\n"
            f'On Windows, quote paths with spaces, e.g. --input "C:\\\\path\\\\to\\\\image.png"'
        )
    img = img_as_float(io.imread(str(p)))
    img_gray = _to_gray(img)
    img_gray = _maybe_resize(img_gray, resize_to)
    return (img_gray, str(p.resolve()), p.stem)


def add_noise(img_gray: np.ndarray, sigma: float = 0.10, seed: int = 0) -> np.ndarray:
    if sigma > 1.0:
        sigma = sigma / 255.0
    rng = np.random.default_rng(seed)
    return np.clip(img_gray + sigma * rng.normal(size=img_gray.shape), 0, 1)


def compute_energy_stats(S: np.ndarray):
    energy = (S ** 2)
    total = np.sum(energy)
    if total == 0:
        eta = np.zeros_like(energy)
        cum = np.zeros_like(energy)
    else:
        eta = energy / total
        cum = np.cumsum(eta)
    return eta, cum, total


def k_from_energy(cum: np.ndarray, tau: float = 0.95) -> int:
    tau = float(np.clip(tau, 1e-9, 1.0))
    return int(np.searchsorted(cum, tau) + 1)


def knee_on_cumulative(cum: np.ndarray, smooth: bool = True) -> Optional[int]:
    if KneeLocator is None:
        warnings.warn("kneed not installed; skipping elbow detection (using energy threshold only).")
        return None
    x = np.arange(1, len(cum) + 1)
    y = cum.copy()
    if smooth and len(y) >= 21:
        try:
            y = savgol_filter(y, window_length=21, polyorder=3, mode="interp")
        except Exception:
            y = savgol_filter(y, window_length=21, polyorder=3)
    kl = KneeLocator(x, y, curve="concave", direction="increasing", interp_method="polynomial")
    return int(kl.knee) if getattr(kl, "knee", None) is not None else None


def reconstruct(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    k = int(max(1, min(k, len(S))))
    Uk = U[:, :k] * S[:k]
    return (Uk @ Vt[:k, :]).clip(0, 1)


def energy_retained(cum: np.ndarray, k: int) -> float:
    k = int(max(1, min(k, len(cum))))
    return float(cum[k - 1])


def ssim_gray(img_ref: np.ndarray, img_cmp: np.ndarray) -> float:
    try:
        return float(ssim(img_ref, img_cmp, data_range=1.0, channel_axis=None))
    except TypeError:
        return float(ssim(img_ref, img_cmp, data_range=1.0, multichannel=False))


# -------------------------------
# Core experiment
# -------------------------------
def run_experiment(
    input_path: Optional[str],
    sigma: float = 0.10,
    seed: int = 0,
    manual_ks: Tuple[int, ...] = (40, 80),
    user_k: Optional[int] = None,
    tau: Optional[float] = None,
    root_dir: str = "Results",
    resize_to: Optional[Tuple[int, int]] = None,
) -> None:
    # 0) Dirs
    figures_dir = os.path.join(root_dir, "Figures")
    tables_dir = os.path.join(root_dir, "Tables")
    ensure_dirs(root_dir, figures_dir, tables_dir)

    # 1) Image
    img_gray, source_str, img_label = load_image_gray(input_path, resize_to=resize_to)
    print(f"Loaded image: {source_str}  (label='{img_label}')")
    print("Outputs →", os.path.abspath(root_dir))

    # 2) Noise
    noisy = add_noise(img_gray, sigma=sigma, seed=seed)

    # 3) SVD on noisy image
    U, S, Vt = svd(noisy, full_matrices=False)
    _, cum, _ = compute_energy_stats(S)

    # 4) Candidates
    k95  = k_from_energy(cum, tau=0.95)
    knee = knee_on_cumulative(cum, smooth=True)

    # If tau given, override with its threshold
    k_tau = k_from_energy(cum, tau=float(tau)) if tau is not None else None
    if tau is not None:
        k_star = k_tau
    else:
        k_star = knee if knee is not None else k95

    # 5) Recon + metrics
    den_adapt  = reconstruct(U, S, Vt, k_star)
    psnr_adapt = float(psnr(img_gray, den_adapt, data_range=1.0))
    ssim_adapt = ssim_gray(img_gray, den_adapt)
    psnr_noisy = float(psnr(img_gray, noisy, data_range=1.0))
    ssim_noisy = ssim_gray(img_gray, noisy)
    enr_adapt  = energy_retained(cum, k_star)

    # 6) Manual rows
    rows: List[dict] = []
    ks = list(manual_ks)
    if user_k is not None and user_k not in ks:
        ks.append(int(user_k))
    safe_ks = sorted({int(np.clip(int(k), 1, len(S))) for k in ks})
    for k in safe_ks:
        den_k = reconstruct(U, S, Vt, k)
        rows.append({
            "Method": f"Manual $k = {k}$",
            "k": k,
            "PSNR": float(psnr(img_gray, den_k, data_range=1.0)),
            "SSIM": ssim_gray(img_gray, den_k),
            "EnergyRetention": energy_retained(cum, k),
        })
    rows.append({
        "Method": f"Adaptive ($k^* = {k_star}$)",
        "k": k_star,
        "PSNR": psnr_adapt,
        "SSIM": ssim_adapt,
        "EnergyRetention": enr_adapt,
    })

    # -------------------------------
    # Figures
    # -------------------------------
    # A) Spectrum: one curve + two dashed vlines (k95 green, knee red)
    x = np.arange(1, len(S) + 1)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x, S, marker="o", linewidth=2, label="Singular values")
    ax.set_yscale("log")
    ax.set_xlabel(r"Index $i$")
    ax.set_ylabel(r"Singular value $\sigma_i$")
    ax.grid(True, which="both", alpha=0.35)
    if k95 is not None:
        ax.axvline(k95, linestyle="--", linewidth=2, color=PALETTE["k95"],
                   label=f"95% energy (k={k95})")
    if knee is not None:
        ax.axvline(knee, linestyle="--", linewidth=2, color=PALETTE["knee"],
                   label=f"Elbow (cumulative) (k={knee})")
    ax.set_title("Adaptive Rank Selection (log spectrum)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    path_spectrum_only = os.path.join(figures_dir, f"{img_label}_spectrum.png")
    fig.savefig(path_spectrum_only, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # B) 3-panel: cumulative (one blue curve) + noisy + denoised (k*)
    plt.figure(figsize=(18, 5))

    # (1) cumulative energy
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(np.arange(1, len(cum) + 1), cum, linewidth=2, color="tab:blue",
             label="Cumulative Energy")
    ax1.axvline(k_star, linestyle="--", linewidth=2, color=PALETTE["k_star"],
                label=f"$k^* = {k_star}$")
    ax1.set_xlabel("Rank $k$")
    ax1.set_ylabel("Cumulative Energy Ratio")
    ax1.set_title("Singular Value Spectrum")
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9, frameon=True)

    # (2) Noisy
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(noisy, cmap="gray")
    ax2.set_title(f"Noisy\nPSNR: {psnr_noisy:.2f} dB, SSIM: {ssim_noisy:.3f}")
    ax2.axis("off")

    # (3) Denoised (adaptive)
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(den_adapt, cmap="gray")
    ax3.set_title(f"Denoised (k={k_star})\nPSNR: {psnr_adapt:.2f} dB, SSIM: {ssim_adapt:.3f}")
    ax3.axis("off")

    plt.tight_layout()
    path_appendix_combo = os.path.join(figures_dir, f"{img_label}_adaptive_panel.png")
    plt.savefig(path_appendix_combo, dpi=200, bbox_inches="tight")
    plt.close()

    # C) Main: noisy vs denoised (2-panel)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(noisy, cmap="gray")
    plt.title(f"Noisy\nPSNR: {psnr_noisy:.2f} dB, SSIM: {ssim_noisy:.3f}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(den_adapt, cmap="gray")
    plt.title(f"Denoised (k={k_star})\nPSNR: {psnr_adapt:.2f} dB, SSIM: {ssim_adapt:.3f}")
    plt.axis("off")

    plt.tight_layout()
    path_main_panel = os.path.join(figures_dir, f"{img_label}_denoising.png")
    plt.savefig(path_main_panel, dpi=200, bbox_inches="tight")
    plt.close()

    # -------------------------------
    # Tables / JSON
    # -------------------------------
    summary = {
        "image": source_str,
        "image_label": img_label,
        "sigma": float(sigma),
        "seed": int(seed),
        "knee": int(knee) if knee is not None else None,
        "k95": int(k95) if k95 is not None else None,
        "k_star": int(k_star),
        "psnr_noisy": float(psnr_noisy),
        "ssim_noisy": float(ssim_noisy),
        "psnr_adaptive": float(psnr_adapt),
        "ssim_adaptive": float(ssim_adapt),
        "energy_retention_adaptive": float(enr_adapt),
        "manual_results": [
            {
                "k": int(r["k"]),
                "psnr": float(r["PSNR"]),
                "ssim": float(r["SSIM"]),
                "energy_retention": float(r["EnergyRetention"]),
            }
            for r in rows if r["Method"].startswith("Manual")
        ],
        "figure_paths": {
            "spectrum_only": path_spectrum_only,
            "appendix_combined": path_appendix_combo,
            "main_panel": path_main_panel,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    csv_path  = os.path.join(tables_dir, f"{img_label}_denoising_comparison.csv")
    excel_path = os.path.join(tables_dir, f"{img_label}_denoising_comparison.xlsx")
    tex_path  = os.path.join(tables_dir, f"{img_label}_denoising_comparison.tex")
    json_path = os.path.join(tables_dir, f"{img_label}_results_summary.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Method,k,PSNR,SSIM,EnergyRetention\n")
        for r in rows:
            f.write(f"{r['Method']},{r['k']},{r['PSNR']:.6f},{r['SSIM']:.6f},{r['EnergyRetention']:.6f}\n")

    # Excel
    lock_file = os.path.join(tables_dir, f"~${os.path.basename(excel_path)}")
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            print(f"Removed Excel lock file: {lock_file}")
        except Exception as e:
            print(f"Warning: failed to remove lock file: {e}")

    excel_engine = None
    try:
        import openpyxl  # noqa: F401
        excel_engine = "openpyxl"
    except ImportError:
        try:
            import xlsxwriter  # noqa: F401
            excel_engine = "xlsxwriter"
        except ImportError:
            pass

    if excel_engine:
        try:
            with pd.ExcelWriter(excel_path, engine=excel_engine, mode="w") as writer:
                pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="denoising")
        except Exception as e:
            print(f"ERROR: Failed to write Excel file: {e}")
    else:
        print("WARNING: Excel export skipped (no engine). Install 'openpyxl' or 'XlsxWriter'.")

    # LaTeX
    def fmt_pct(x: float) -> str:
        return f"{100.0 * float(x):.1f}\\%"

    manual_sorted = sorted([r for r in rows if r["Method"].startswith("Manual")], key=lambda r: r["k"])
    adaptive_row = [r for r in rows if r["Method"].startswith("Adaptive")][0]

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n\\centering\n")
        caption_img = img_label.replace("_", "\\_")
        f.write("\\caption{Comparison of manual vs. adaptive rank selection for denoising the noisy ")
        f.write(f"\\texttt{{{caption_img}}} image.}}\n")
        f.write("\\label{tab:denoising-comparison}\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("\\textbf{Method} & \\textbf{PSNR (dB)} & \\textbf{SSIM} & \\textbf{Energy Retention} \\\\\n\\midrule\n")
        for r in manual_sorted:
            f.write(f"Manual $k = {r['k']}$ & {r['PSNR']:.2f} & {r['SSIM']:.3f} & {fmt_pct(r['EnergyRetention'])} \\\\\n")
        f.write(
            f"Adaptive ($k^* = {adaptive_row['k']}$) & "
            f"\\textbf{{{adaptive_row['PSNR']:.2f}}} & "
            f"\\textbf{{{adaptive_row['SSIM']:.3f}}} & "
            f"{fmt_pct(adaptive_row['EnergyRetention'])} \\\\\n"
        )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # Console summary
    print("=== SVD Denoising ===")
    print(f"image={summary['image']} (label={img_label})")
    print(f"sigma={sigma:.3f}, seed={seed}")
    print(f"knee={knee}, k95={k95}, k* (adaptive)={k_star}")
    for r in manual_sorted:
        print(f"Manual k={r['k']:>3d} | PSNR={r['PSNR']:.2f} dB | SSIM={r['SSIM']:.3f} | Energy={r['EnergyRetention']:.3f}")
    print(f"Adaptive k*={k_star:>3d} | PSNR={psnr_adapt:.2f} dB | SSIM={ssim_adapt:.3f} | Energy={enr_adapt:.3f}")
    print(f"Saved figure: {path_spectrum_only}")
    print(f"Saved figure: {path_appendix_combo}")
    print(f"Saved figure: {path_main_panel}")
    print(f"Wrote CSV:   {csv_path}")
    print(f"Wrote Excel: {excel_path}")
    print(f"Wrote LaTeX: {tex_path}")
    print(f"Wrote JSON:  {json_path}")


# -------------------------------
# CLI
# -------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="SVD denoising with manual/adaptive rank selection.")
    p.add_argument("--input", type=str, default=None,
                   help="Path to input image OR builtins (astronaut, camera, moon, coins, page, rocket, coffee, chelsea) or 'builtin:<name>'.")
    p.add_argument("--sigma", type=float, default=0.10,
                   help="Gaussian noise std; accepts [0,1] or 0..255 (auto-scaled).")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--k", type=int, default=None, help="Manual rank k to include in comparison.")
    p.add_argument("--manual_ks", type=int, nargs="*", default=[40, 80],
                   help="Extra manual ranks to compare (space-separated).")
    p.add_argument("--tau", type=float, default=None,
                   help="Energy threshold for adaptive k (e.g., 0.95). If omitted, uses Kneedle with fallback to 0.95.")
    p.add_argument("--root", type=str, default="results",
                   help="Root output directory (creates results/Figures and results/Tables).")
    p.add_argument("--resize", type=str, default=None,
                   help="Optional HxW resize, e.g. '256x256'.")
    # Legacy (accepted, ignored)
    p.add_argument("--figdir", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--resdir", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--tabdir", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--out", type=str, default=None, help=argparse.SUPPRESS)
    return p


def parse_args_friendly(parser: argparse.ArgumentParser):
    """Parse command-line arguments safely in both CLI and notebook environments."""
    argv = sys.argv
    # If running inside Jupyter/IPython, ignore args
    if any("ipykernel" in a or a.endswith(".json") for a in argv):
        return parser.parse_args([])

    try:
        return parser.parse_args()
    except SystemExit:
        # Prevent argparse from exiting the notebook on error
        args, _ = parser.parse_known_args()
        return args


def main():
    args = parse_args_friendly(build_parser())

    # --- parse optional resize argument ---
    resize_to = None
    if getattr(args, "resize", None):
        s = args.resize.lower().replace("×", "x").replace(" ", "")
        try:
            h_str, w_str = s.split("x")
            resize_to = (int(h_str), int(w_str))
        except Exception:
            warnings.warn(
                f'Could not parse --resize "{args.resize}". Expected format like "256x256".'
            )

    # --- normalize output root name to lowercase (last path component only) ---
    root_dir = args.root or "results"
    root_dir = os.path.normpath(root_dir)
    parent, base = os.path.dirname(root_dir), os.path.basename(root_dir)
    root_dir = os.path.join(parent, base.lower())

    # --- run experiment ---
    run_experiment(
        input_path=args.input,
        sigma=args.sigma,
        seed=args.seed,
        manual_ks=tuple(args.manual_ks) if args.manual_ks is not None else (40, 80),
        user_k=args.k,
        tau=args.tau,
        root_dir=root_dir,
        resize_to=resize_to,
    )


if __name__ == "__main__":
    main()
