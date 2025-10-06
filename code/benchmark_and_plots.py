#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
benchmark_and_plots_clean.py
Unified runner: benchmarks (SVD/EVD/QR) + plotting:
- Reconstruction panel (Original | Noisy | SVD(k) | QR(k))
- PSNR vs k (coalesced)
- PSNR vs k (offset)
- SSIM vs k (offset)
- Runtime vs k
- Adaptive summary

Output layout (flat):
results/
  Figures/
    <label>_reconstruction_panel_kXXX.pdf
    <label>_psnr_vs_k_main.pdf
    <label>_psnr_vs_k_fixed_offset.pdf
    <label>_ssim_vs_k_fixed_offset.pdf
    <label>_runtime_vs_k.pdf
    <label>_adaptive_summary.pdf
  Tables/
    <label>_fixed_k.csv
    <label>_adaptive.csv
    <label>_runtime.csv
"""

import os, time, csv, argparse
from pathlib import Path
from typing import Optional, Tuple, List

# Reduce timing variance across runs/kernels
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless/CI
import matplotlib.pyplot as plt

from scipy.linalg import svd as scipy_svd, eigh
from scipy.linalg import qr as scipy_qr

from skimage import data, color
from skimage.io import imread
from skimage.util import img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage import transform as sktf

import pandas as pd

# ---- Optional: kneed (elbow detection) ----
_HAVE_KNEED = True
try:
    from kneed import KneeLocator
except Exception:
    _HAVE_KNEED = False


# ---------------- Utilities ----------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def log_result(filename: str, row: dict, headers: List[str]):
    """Append a row to CSV (create with headers if absent)."""
    ensure_dir(os.path.dirname(filename) or ".")
    new_file = not os.path.isfile(filename)
    with open(filename, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if new_file:
            w.writeheader()
        if "method" in row and isinstance(row["method"], str):
            row["method"] = row["method"].strip().upper()
        w.writerow(row)


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


def _builtin_gray(name: str):
    """Return a grayscale float image [0,1] for known builtins."""
    low = name.lower()
    if low in {"astronaut"}:
        return _to_gray(img_as_float(data.astronaut())), "astronaut"
    if low in {"camera", "cameraman"}:
        g = img_as_float(data.camera())
        return (g if g.ndim == 2 else _to_gray(g)), "camera"  # instead of "cameraman"
    if low in {"coins"}:
        g = img_as_float(data.coins())
        return (g if g.ndim == 2 else _to_gray(g)), "coins"
    if low in {"chelsea"}:
        return _to_gray(img_as_float(data.chelsea())), "chelsea"
    if low in {"moon"}:
        g = img_as_float(data.moon()); return (g if g.ndim == 2 else _to_gray(g)), "moon"
    if low in {"page"}:
        g = img_as_float(data.page()); return (g if g.ndim == 2 else _to_gray(g)), "page"
    if low in {"rocket"}:
        return _to_gray(img_as_float(data.rocket())), "rocket"
    if low in {"coffee"}:
        return _to_gray(img_as_float(data.coffee())), "coffee"
    return None, None


def parse_resize_hw(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    s2 = s.lower().replace("×", "x").replace(" ", "")
    try:
        h, w = s2.split("x")
        return (int(h), int(w))
    except Exception:
        print(f'[warn] Could not parse --resize "{s}". Expected like "256x256". Ignoring.')
        return None


def load_image(input_spec: Optional[str], resize_to: Optional[Tuple[int, int]] = None):
    """
    Accepts a builtin name (optionally prefixed 'builtin:') or a file path.
    Returns (img_gray in [0,1], label, source_str).
    """
    if input_spec is None:
        g, label = _builtin_gray("astronaut")
        g = _maybe_resize(g, resize_to)
        return g, label, "builtin:astronaut"

    key_raw = str(input_spec).strip().strip('"').strip("'")
    key = key_raw.lower()
    if key.startswith("builtin:"):
        key = key.split("builtin:", 1)[1]

    # Try builtins
    g, label = _builtin_gray(key)
    if g is not None:
        g = _maybe_resize(g, resize_to)
        return g, label, f"builtin:{label}"

    # Else treat as file path
    p = Path(key_raw).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        raise FileNotFoundError(
            f"Input image not found: {p}\n"
            f"Tip: verify exact filename/extension in: {p.parent}\n"
            f'On Windows, quote paths with spaces, e.g. --input "C:\\\\path\\\\to\\\\image.png"'
        )
    arr = img_as_float(imread(str(p)))
    gray = _to_gray(arr)
    gray = _maybe_resize(gray, resize_to)
    return gray, p.stem, str(p.resolve())


def add_gaussian_noise(img, sigma=0.10, clip=True, seed=None):
    """sigma in [0,1]; if >1 we assume 8-bit units and auto-scale by /255."""
    if sigma > 1.0:
        sigma = float(sigma) / 255.0
    sigma = max(0.0, float(sigma))
    rng = np.random.default_rng(seed)
    noisy = img + sigma * rng.normal(size=img.shape)
    return np.clip(noisy, 0.0, 1.0) if clip else noisy


# ---------------- Metrics ----------------
def energy_percent(S, k):
    k = max(0, min(int(k), len(S)))
    if k <= 0:
        return 0.0
    tot = float(np.sum(S**2)) + 1e-12
    kept = float(np.sum(S[:k]**2))
    return 100.0 * kept / tot


# ---------------- Methods ----------------
def svd_low_rank(A, k):
    U, S, Vt = scipy_svd(A, full_matrices=False, lapack_driver="gesdd")
    k = int(max(1, min(k, len(S))))
    Ak = (U[:, :k] * S[:k]) @ Vt[:k, :]
    return Ak, S


def evd_low_rank(A, k):
    AtA = A.T @ A
    w, V = eigh(AtA)              # ascending
    idx = np.argsort(w)[::-1]
    w = w[idx]; V = V[:, idx]
    k = int(max(1, min(k, V.shape[1])))
    V_k = V[:, :k]
    Ak = (A @ V_k) @ V_k.T
    S  = np.sqrt(np.clip(w[::-1], 0, None))[::-1] if w.size else w  # descending
    return Ak, S


def qr_low_rank(A, k):
    Q, R, piv = scipy_qr(A, mode="economic", pivoting=True)
    k = int(max(1, min(k, min(Q.shape[1], R.shape[0]))))
    Qk, Rk = Q[:, :k], R[:k, :]
    Ak_perm = Qk @ Rk
    P = np.eye(A.shape[1])[:, piv]
    return Ak_perm @ P.T


# ---------------- Adaptive k ----------------
def k_from_energy(S, target=0.95):
    energy = (S**2) / (np.sum(S**2) + 1e-12)
    cum = np.cumsum(energy)
    k_star = int(np.searchsorted(cum, float(target)) + 1)
    return max(1, min(k_star, len(S)))


def k_from_elbow(S, energy_fallback=0.95):
    # Robust elbow: if kneed missing or knee=None, fall back to energy target
    try:
        if not _HAVE_KNEED:
            raise RuntimeError("kneed not available")
        energy = (S**2) / (np.sum(S**2) + 1e-12)
        cum = np.cumsum(energy)
        x = np.arange(1, len(S) + 1)
        kneedle = KneeLocator(x, cum, curve='concave', direction='increasing')
        k_elbow = int(kneedle.knee) if kneedle.knee is not None else None
        if not k_elbow:
            raise RuntimeError("knee not found")
        return max(1, min(k_elbow, len(S)))
    except Exception as e:
        print(f"[warn][elbow] {e}; falling back to energy={energy_fallback}.")
        return k_from_energy(S, target=energy_fallback)


# ---------------- Runners ----------------
def measure_once(reconstruct_fn, ref):
    t0 = time.perf_counter()
    out = reconstruct_fn()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    _psnr = psnr(ref, out, data_range=1.0)
    _ssim = ssim(ref, out, data_range=1.0)
    return out, _psnr, _ssim, dt_ms


def repeat_time(fn, trials=5):
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


# ---------------- Plot helpers ----------------
def savefig_pdf(path_base):
    plt.tight_layout()
    plt.savefig(f"{path_base}.pdf", bbox_inches="tight")
    plt.close()


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "method" in df.columns:
        df["method"] = df["method"].astype(str).str.strip().str.upper()
    for col in ["k","k_star","psnr_db","ssim","energy_percent","time_ms",
                "time_mean_ms","time_std_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.reset_index(drop=True)
    if "_row" not in df.columns:
        df["_row"] = np.arange(len(df))
    return df


def _last_valid_per_group(df, key_cols, value_cols):
    """Keep the last non-NaN row per key (by original order)."""
    if df.empty:
        return df
    d = df.sort_values("_row")
    for c in value_cols:
        d = d[~d[c].isna()]
    if d.empty:
        return d
    d = d.groupby(key_cols, as_index=False).last()
    return d


# ---------- Reconstruction panel (Original | Noisy | SVD(k) | QR(k)) ----------
def plot_reconstruction_panel(img_clean, img_noisy, figs_dir, label, k_show):
    """Make a 4-up panel (2nd image is ALWAYS the noisy one) and save PDF."""
    ensure_dir(figs_dir)

    # Reconstructions at k_show FROM THE NOISY IMAGE
    svd_rec = np.clip(svd_low_rank(img_noisy, k_show)[0], 0, 1)
    qr_rec  = np.clip(qr_low_rank(img_noisy, k_show), 0, 1)

    # Metrics vs. clean reference
    p_svd = psnr(img_clean, svd_rec, data_range=1.0)
    s_svd = ssim(img_clean, svd_rec, data_range=1.0)
    p_qr  = psnr(img_clean, qr_rec,  data_range=1.0)
    s_qr  = ssim(img_clean, qr_rec,  data_range=1.0)

    plt.figure(figsize=(14, 4))
    panels = [
        ("Original",  img_clean),
        ("Noisy",     img_noisy),  # <- explicitly show the noisy image
        (f"SVD (k={int(k_show)})\nPSNR: {p_svd:.2f} dB, SSIM: {s_svd:.3f}", svd_rec),
        (f"QR (k={int(k_show)})\nPSNR: {p_qr:.2f} dB, SSIM: {s_qr:.3f}",  qr_rec),
    ]
    for i, (title, im) in enumerate(panels, 1):
        ax = plt.subplot(1, 4, i)
        ax.imshow(im, cmap="gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    savefig_pdf(os.path.join(figs_dir, f"{label}_reconstruction_panel_k{int(k_show):03d}"))


# ---------- figure 1: PSNR vs k with coincident SVD/EVD collapse ----------
def plot_psnr_vs_k_coalesced(df_fixed, df_adapt, figs_dir, label):
    ensure_dir(figs_dir)
    df_f = _normalize(df_fixed)
    df_f = _last_valid_per_group(df_f, ["method","k"], ["psnr_db"])

    # k ticks (use common set)
    k_list = sorted(df_f["k"].dropna().unique().tolist())

    # psnr_map
    def series_for(method):
        sub = df_f[df_f["method"]==method].set_index("k")
        return [sub.loc[k,"psnr_db"] if k in sub.index else np.nan for k in k_list]

    psnr_map = {m: series_for(m) for m in ["SVD","EVD","QR"] if (df_f["method"]==m).any()}

    # Coalesce SVD/EVD if identical
    coincident_fixed = False
    if "SVD" in psnr_map and "EVD" in psnr_map:
        a = np.array(psnr_map["SVD"], dtype=float)
        b = np.array(psnr_map["EVD"], dtype=float)
        # Looser tolerance to account for small rounding or CSV parsing drift
        coincident_fixed = np.allclose(a, b, atol=1e-2, equal_nan=True)

    # Adaptive points (prefer energy_tau if present)
    df_a = _normalize(df_adapt)
    use = df_a[df_a["strategy"].astype(str).str.lower().str.contains("energy_tau")]
    if use.empty:
        use = df_a.copy()
    adaptive_pts = {}
    for m in ["SVD","EVD","QR"]:
        sub = use[use["method"]==m]
        if not sub.empty:
            r = sub.iloc[-1]
            adaptive_pts[m] = (float(r["k_star"]), float(r["psnr_db"]))

    def adapt_equal(a, b, tol=1e-2):
        return (a in adaptive_pts and b in adaptive_pts and
                abs(adaptive_pts[a][0] - adaptive_pts[b][0]) <= tol and
                abs(adaptive_pts[a][1] - adaptive_pts[b][1]) <= tol)

    plt.figure(figsize=(7.5,5.2))

    # fixed curves
    if coincident_fixed and "SVD" in psnr_map:
        plt.plot(k_list, psnr_map["SVD"], marker="o", label="SVD/EVD (fixed k)")
    else:
        if "SVD" in psnr_map:
            plt.plot(k_list, psnr_map["SVD"], marker="o", label="SVD (fixed k)")
        if "EVD" in psnr_map:
            plt.plot(k_list, psnr_map["EVD"], marker="s", label="EVD (fixed k)")
    if "QR" in psnr_map:
        plt.plot(k_list, psnr_map["QR"], marker="^", label="QR (fixed k)")

    # adaptive markers
    if adapt_equal("SVD","EVD"):
        ka, psa = adaptive_pts["SVD"]
        plt.scatter([ka],[psa], marker="s", s=60, label="SVD/EVD (adaptive)")
    else:
        if "SVD" in adaptive_pts:
            ka, psa = adaptive_pts["SVD"]
            plt.scatter([ka],[psa], marker="s", s=60, label="SVD (adaptive)")
        if "EVD" in adaptive_pts:
            ka, psa = adaptive_pts["EVD"]
            plt.scatter([ka],[psa], marker="s", s=60, label="EVD (adaptive)")
    if "QR" in adaptive_pts:
        ka, psa = adaptive_pts["QR"]
        plt.scatter([ka],[psa], marker="s", s=60, label="QR (adaptive)")

    plt.xlabel("Rank $k$")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs. Rank $k$ for SVD/EVD and QR (Fixed and Adaptive)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig_pdf(os.path.join(figs_dir, f"{label}_psnr_vs_k_main"))


# ---------- offset-based fixed plots ----------
def _plot_fixed_offset(df_fixed, ycol, ylabel, figs_dir, label, stem):
    ensure_dir(figs_dir)
    df = _normalize(df_fixed)
    d = _last_valid_per_group(df, ["method","k"], [ycol])

    # Small x-offsets to separate overlapping curves
    x_offset = {"EVD": -0.4, "QR": 0.0, "SVD": +0.4}
    styles = {
        "EVD": {"linestyle":"--", "marker":"s", "alpha":0.95},
        "QR":  {"linestyle":"-.", "marker":"^", "alpha":0.95},
        "SVD": {"linestyle":"-",  "marker":"o", "alpha":1.0, "linewidth":2.2}
    }
    methods = [m for m in ["EVD","QR","SVD"] if (d["method"]==m).any()]

    plt.figure(figsize=(6.8, 4.8))
    all_k = sorted(d["k"].dropna().unique().tolist())

    for method in methods:
        sub = d[d["method"] == method].sort_values("k")
        x = sub["k"].to_numpy(dtype=float) + x_offset.get(method, 0.0)
        plt.plot(
            x, sub[ycol],
            label=method,
            linestyle=styles[method].get("linestyle","-"),
            marker=styles[method].get("marker","o"),
            linewidth=styles[method].get("linewidth",1.8),
            alpha=styles[method].get("alpha",1.0),
        )

    plt.xticks(all_k, [str(int(v)) for v in all_k])
    plt.xlabel("Rank $k$")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} versus rank $k$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig_pdf(os.path.join(figs_dir, f"{label}_{stem}"))


def plot_psnr_vs_k_fixed(df_fixed, figs_dir, label):
    _plot_fixed_offset(df_fixed, "psnr_db", "PSNR (dB)", figs_dir, label, "psnr_vs_k_fixed_offset")


def plot_ssim_vs_k_fixed(df_fixed, figs_dir, label):
    _plot_fixed_offset(df_fixed, "ssim", "SSIM", figs_dir, label, "ssim_vs_k_fixed_offset")


# ---------- runtime ----------
def plot_runtime_vs_k(df_runtime, figs_dir, label):
    ensure_dir(figs_dir)
    df = _normalize(df_runtime)
    if df.empty:
        return
    df = df.dropna(subset=["k","time_mean_ms"]).sort_values(["method","k"])
    plt.figure(figsize=(6.8, 4.8))
    for method in ["EVD","QR","SVD"]:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        plt.errorbar(
            sub["k"], sub["time_mean_ms"],
            yerr=sub["time_std_ms"],
            marker="o", capsize=3, label=method
        )
    plt.xlabel("Rank $k$")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime versus rank $k$ (mean ± std)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig_pdf(os.path.join(figs_dir, f"{label}_runtime_vs_k"))


# ---------- adaptive summary ----------
def plot_adaptive_summary(df_adapt, figs_dir, label):
    ensure_dir(figs_dir)
    df = _normalize(df_adapt)
    df = df.dropna(subset=["k_star","psnr_db"])
    if df.empty:
        return
    method_order = ["SVD","EVD","QR"]
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df = df.sort_values("_row").groupby(["method","strategy"], as_index=False).last()

    strategies = df["strategy"].astype(str).unique().tolist()
    x = np.arange(len(method_order)); width = 0.8 / max(1, len(strategies))
    fig, axes = plt.subplots(1,2, figsize=(10.5,4.6))

    for ax_idx, metric in enumerate(["k_star","psnr_db"]):
        ax = axes[ax_idx]
        for i, strat in enumerate(strategies):
            vals = []
            for m in method_order:
                sub = df[(df["method"]==m) & (df["strategy"]==strat)]
                vals.append(sub[metric].iloc[-1] if not sub.empty else np.nan)
            ax.bar(x + i*width - 0.5*width*(len(strategies)-1), vals, width, label=strat)
        ax.set_xticks(x, method_order)
        ax.set_ylabel(r"$k^\ast$" if metric=="k_star" else "PSNR (dB)")
        ax.set_title(r"Adaptive rank $k^\ast$" if metric=="k_star" else "Adaptive PSNR (dB)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

    fig.suptitle("Adaptive selection summary")
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, f"{label}_adaptive_summary.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------- Benchmark phase (writes CSVs) ----------------
def run_benchmark(args, label_override=None):
    # Label & flat dirs
    label = (label_override or derive_label(args.input))
    resdir = args.resdir   # results/Tables
    figdir = args.figdir   # results/Figures
    ensure_dir(resdir); ensure_dir(figdir)

    # Output files (flat; filenames prefixed with label)
    fixed_csv    = os.path.join(resdir, f"{label}_fixed_k.csv")
    adaptive_csv = os.path.join(resdir, f"{label}_adaptive.csv")
    runtime_csv  = os.path.join(resdir, f"{label}_runtime.csv")

    # Fresh outputs
    for f in [fixed_csv, adaptive_csv, runtime_csv]:
        if os.path.exists(f): os.remove(f)

    # RNG seed
    rng_seed = args.seed if args.seed is not None else 0
    print(f"[info] Using seed={rng_seed}")

    # Load image
    img, label2, source = load_image(args.input, resize_to=args.resize_to)
    print(f"[info] Loaded: {source}  (label='{label2}', shape={img.shape})")

    # Create a NOISY version for the panel (always), and choose benchmark input
    sigma = args.sigma
    noisy_for_panel = add_gaussian_noise(img, sigma=sigma, seed=rng_seed)
    A = noisy_for_panel if args.use_noisy else img
    ref = img

    # Make the reconstruction panel PDF (2nd image = noisy)
    plot_reconstruction_panel(ref, noisy_for_panel, figdir, label, args.panel_k)

    # Spectrum
    _, S_A, _ = scipy_svd(A, full_matrices=False, lapack_driver="gesdd")

    # ---------- Fixed-k ----------
    for k in args.fixed_k:
        # SVD
        out, p, q, t_ms = measure_once(lambda: svd_low_rank(A, k)[0], ref)
        log_result(fixed_csv,
            {"method":"SVD","k":k,"psnr_db":f"{p:.2f}","ssim":f"{q:.3f}",
             "energy_percent":f"{energy_percent(S_A,k):.1f}","time_ms":f"{t_ms:.0f}"},
            headers=["method","k","psnr_db","ssim","energy_percent","time_ms"])

        # EVD
        out, p, q, t_ms = measure_once(lambda: evd_low_rank(A, k)[0], ref)
        log_result(fixed_csv,
            {"method":"EVD","k":k,"psnr_db":f"{p:.2f}","ssim":f"{q:.3f}",
             "energy_percent":f"{energy_percent(S_A,k):.1f}","time_ms":f"{t_ms:.0f}"},
            headers=["method","k","psnr_db","ssim","energy_percent","time_ms"])

        # QR
        out, p, q, t_ms = measure_once(lambda: qr_low_rank(A, k), ref)
        log_result(fixed_csv,
            {"method":"QR","k":k,"psnr_db":f"{p:.2f}","ssim":f"{q:.3f}",
             "energy_percent":f"{energy_percent(S_A,k):.1f}","time_ms":f"{t_ms:.0f}"},
            headers=["method","k","psnr_db","ssim","energy_percent","time_ms"])

    # ---------- Runtime mean/std ----------
    for method in ["SVD","EVD","QR"]:
        for k in args.fixed_k:
            if method == "SVD": fn = lambda: svd_low_rank(A, k)[0]
            elif method == "EVD": fn = lambda: evd_low_rank(A, k)[0]
            else: fn = lambda: qr_low_rank(A, k)
            mean_ms, std_ms = repeat_time(fn, trials=args.trials)
            out = fn()
            _psnr = psnr(ref, out, data_range=1.0); _ssim = ssim(ref, out, data_range=1.0)
            log_result(runtime_csv,
                {"method":method,"k":k,"time_mean_ms":f"{mean_ms:.0f}","time_std_ms":f"{std_ms:.0f}",
                 "psnr_db":f"{_psnr:.2f}","ssim":f"{_ssim:.3f}"},
                headers=["method","k","time_mean_ms","time_std_ms","psnr_db","ssim"])

    # ---------- Adaptive-k ----------
    k_energy = k_from_energy(S_A, target=args.energy)
    strat_set = {s.lower() for s in args.strategies}
    strategies_to_run = []
    if "energy" in strat_set:
        strategies_to_run.append((f"energy_tau={args.energy:.3f}", "energy_tau={args.energy:.3f}", k_energy))
    if "elbow" in strat_set:
        k_elbow = k_from_elbow(S_A, energy_fallback=args.energy)
        strategies_to_run.append(("elbow", "elbow", k_elbow))

    for (strategy, strategy_key, kstar) in strategies_to_run:
        for method, runner in [
            ("SVD", lambda kk: svd_low_rank(A, kk)[0]),
            ("EVD", lambda kk: evd_low_rank(A, kk)[0]),
            ("QR",  lambda kk: qr_low_rank(A, kk)),
        ]:
            try:
                kk = int(max(1, min(kstar, min(A.shape))))
                out = runner(kk)
                p = psnr(ref, out, data_range=1.0); q = ssim(ref, out, data_range=1.0)
                e_pct = energy_percent(S_A, kk)
                mean_ms, _ = repeat_time(lambda: runner(kk), trials=args.trials)
                row = {"method":method,"strategy":strategy,"strategy_key":strategy_key,"k_star":kk,
                       "psnr_db":f"{p:.2f}","ssim":f"{q:.3f}","energy_percent":f"{e_pct:.1f}","time_ms":f"{mean_ms:.0f}"}
            except Exception as e:
                print(f"[warn] Adaptive {strategy}/{method} failed: {e}")
                row = {"method":method,"strategy":strategy,"strategy_key":strategy_key,"k_star":int(kstar),
                       "psnr_db":"","ssim":"","energy_percent":"","time_ms":""}
            log_result(adaptive_csv, row,
                       headers=["method","strategy","strategy_key","k_star","psnr_db","ssim","energy_percent","time_ms"])

    print(f"[info] CSVs written under: {resdir}")
    return resdir, figdir, label


# ---------------- Plot phase (reads CSVs and saves figures) ----------------
def run_plots(args, resdir, figdir, label):
    fixed_csv    = os.path.join(resdir, f"{label}_fixed_k.csv")
    adaptive_csv = os.path.join(resdir, f"{label}_adaptive.csv")
    runtime_csv  = os.path.join(resdir, f"{label}_runtime.csv")

    def maybe_read(path):
        return pd.read_csv(path, encoding="utf-8-sig") if os.path.isfile(path) else pd.DataFrame()

    df_fixed   = maybe_read(fixed_csv)
    df_adapt   = maybe_read(adaptive_csv)
    df_runtime = maybe_read(runtime_csv)

    if not df_fixed.empty:
        plot_psnr_vs_k_coalesced(df_fixed, df_adapt, figdir, label)
        plot_psnr_vs_k_fixed(df_fixed, figdir, label)
        plot_ssim_vs_k_fixed(df_fixed, figdir, label)

    if not df_runtime.empty:
        plot_runtime_vs_k(df_runtime, figdir, label)

    if not df_adapt.empty:
        plot_adaptive_summary(df_adapt, figdir, label)


# ---------------- Helpers ----------------
def derive_label(input_spec: Optional[str]):
    if input_spec is None:
        return "astronaut"
    key = str(input_spec).strip().strip('"').strip("'")
    low = key.lower()
    # Handle 'builtin:<name>' explicitly
    if low.startswith("builtin:"):
        low = low.split("builtin:", 1)[1]
    # Builtins
    _, label = _builtin_gray(low)
    if label is not None:
        return label
    # File path
    return Path(key).stem


# ---------------- Main CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Unified benchmark + plots for SVD/EVD/QR (fixed & adaptive ranks).")
    # Input & noise
    p.add_argument("--input", default="astronaut",
                   help=("Image spec: FILE PATH or builtin name "
                         "(astronaut | camera/cameraman | coins | chelsea | moon | page | rocket | coffee). "
                         "You may prefix with 'builtin:'."))
    p.add_argument("--resize", type=str, default=None,
                   help="Optional HxW resize, e.g. '256x256'.")
    p.add_argument("--sigma", type=float, default=0.10,
                   help="Gaussian noise sigma (in [0,1]); if >1, interpreted as 8-bit (auto-scaled by /255).")
    p.add_argument("--use_noisy", action="store_true", help="Benchmark on noisy input; metrics vs. clean.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for noise & timings.")

    # Adaptive
    p.add_argument("--energy", type=float, default=0.995, help="Energy threshold tau in [0,1] for adaptive-k.")
    p.add_argument("--strategies", nargs="+", default=["energy","elbow"],
                   help="Adaptive strategies to run: any of {energy, elbow}.")

    # Fixed-k & runtime
    p.add_argument("--fixed_k", type=int, nargs="+", default=[10, 30, 50, 100],
                   help="Fixed ranks to evaluate.")
    p.add_argument("--trials", type=int, default=5, help="Trials for runtime mean/std.")

    # NEW: reconstruction panel rank
    p.add_argument("--panel_k", type=int, default=50,
                   help="Rank k for the reconstruction panel figure.")

    # I/O (flat layout under 'results')
    p.add_argument("--resdir", default=os.path.join("results", "Tables"),
                   help="Directory for CSV outputs (default: results/Tables).")
    p.add_argument("--figdir", default=os.path.join("results", "Figures"),
                   help="Directory for figure outputs as PDF (default: results/Figures).")

    # Flow control
    p.add_argument("--plot_only", action="store_true", help="Only generate plots from existing CSVs.")
    p.add_argument("--benchmark_only", action="store_true", help="Only run benchmarks; skip plots.")

    args = p.parse_args()

    if args.plot_only and args.benchmark_only:
        p.error("--plot_only and --benchmark_only are mutually exclusive.")

    # Parse resize now and stash as tuple for convenience
    args.resize_to = parse_resize_hw(args.resize)
    return args


def main():
    args = parse_args()

    label = derive_label(args.input)
    resdir = args.resdir  # results/Tables
    figdir = args.figdir  # results/Figures

    do_bench = not args.plot_only
    do_plot  = not args.benchmark_only

    if do_bench:
        resdir, figdir, label = run_benchmark(args, label_override=label)

    if do_plot:
        ensure_dir(resdir); ensure_dir(figdir)
        run_plots(args, resdir, figdir, label)

    print(f"Done. CSVs in '{resdir}', figures in '{figdir}'.")


if __name__ == "__main__":
    main()
