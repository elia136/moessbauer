import os
import re
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})


# ---------------- CONFIG ----------------
DATA_SUBDIR = "data"
STYLE_FILE = "custom.mplstyle"

PICKS_CSV = os.path.join("results", "picked_peaks.csv")
OUT_SUBDIR = os.path.join("results", "calib_energy_plots")
SUMMARY_CSV = os.path.join("results", "calibration_summary.csv")

# Calibration energies (keV) for true 2-point cases
E_LOW = 6.40
E_HIGH = 14.41

FIG_W_IN = 4.2
FIG_H_IN = 3.0
# --------------------------------------
# For showing the gaussian of the 14.4 keV Mossbauer line

def gaussian(x, A, mu, sigma, C):
    """
    A * exp(-(x - mu)^2 / (2 sigma^2)) + C
    A     : amplitude
    mu    : center
    sigma : standard deviation
    C     : constant background
    """
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + C

def fixed_peak_window(x, y, half_width=80):
    """
    Select a fixed-width window around the maximum of the peak.

    half_width: number of channels to include on each side
    """
    n = len(y)
    peak_idx = np.argmax(y)

    left = max(0, peak_idx - half_width)
    right = min(n - 1, peak_idx + half_width)

    return left, right


def auto_peak_window(x, y, frac_height=0.2, min_width=20):
    """
    Find a window [x_left, x_right] around the largest peak by thresholding.

    frac_height : keep all points where y is above
                  baseline + frac_height * (y_max - baseline)
    min_width   : minimal number of channels in the window
    """
    n = len(y)
    # rough background estimate: median of full spectrum
    baseline = np.median(y)

    # index of global maximum (assumes a positive peak)
    idx_max = np.argmax(y)
    y_max = y[idx_max]

    # threshold for keeping points near the peak
    thr = baseline + frac_height * (y_max - baseline)

    # move left from peak until we drop below threshold
    left = idx_max
    while left > 0 and y[left] > thr:
        left -= 1

    # move right from peak until we drop below threshold
    right = idx_max
    while right < n - 1 and y[right] > thr:
        right += 1

    # ensure at least min_width points
    if right - left < min_width:
        half_extra = (min_width - (right - left)) // 2
        left = max(0, left - half_extra)
        right = min(n - 1, right + half_extra)

    return left, right


# --- Main fitting function --------------------------------------------------

def fit_gaussian_to_file(folder_path, file_name, out_dir, frac_height=0.2, min_width=20):

    """
    Read a spectrum, automatically zoom into the main peak,
    fit a Gaussian, and plot the zoomed region with the fit.

    Returns (popt, pcov) from curve_fit.
    """
    file_path = os.path.join(folder_path, file_name)
    x, y = load_asc_1col(file_path)

    # 1) choose window around main peak
    left, right = fixed_peak_window(x, y, half_width=100)


    x_fit = x[left:right+1]
    y_fit = y[left:right+1]

    # 2) initial guesses for Gaussian parameters
    A0 = y_fit.max() - np.median(y_fit)
    mu0 = x_fit[np.argmax(y_fit)]
    sigma0 = (x_fit[-1] - x_fit[0]) / 6.0  # rough width guess
    C0 = np.median(y_fit)

    p0 = [A0, mu0, sigma0, C0]

    # 3) fit
    popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0)

    A, mu, sigma, C = popt
    fwhm = 2.354820045 * abs(sigma)  # FWHM of a Gaussian

    # 4) plot zoomed region in x, full y-range
    x_dense = np.linspace(x_fit[0], x_fit[-1], 5 * len(x_fit))
    y_dense = gaussian(x_dense, *popt)

    # ---- Gaussian uncertainty band (1σ) ----
    # Compute Jacobian numerically for error propagation
    def gaussian_jacobian(x, A, mu, sigma, C):
        dA = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        dmu = A * dA * ((x - mu) / sigma**2)
        dsigma = A * dA * ((x - mu)**2 / sigma**3)
        dC = np.ones_like(x)
        return np.vstack([dA, dmu, dsigma, dC]).T

    J = gaussian_jacobian(x_dense, *popt)
    y_var = np.einsum("ij,jk,ik->i", J, pcov, J)
    y_err = np.sqrt(np.maximum(y_var, 0))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(x_fit, y_fit, s=10, label="Data")
    ax.plot(x_dense, y_dense, color="black", linewidth=2, label="Gaussian fit")

    # --- Error band ---
    ax.fill_between(
        x_dense,
        y_dense - 1.96*y_err,
        y_dense + 1.96*y_err,
        color="gray",
        alpha=0.3,
        label= r"95\% CI"
    )

    ax.set_xlim(x_fit[0], x_fit[-1])
    ax.set_ylim(min(y), max(y) + 100)

    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts")

    # Title stays at the top
    fig.suptitle(f"Gaussian fit of 14.4 keV peak", y=0.98)

    # Legend OUTSIDE axes, under the title (figure coords)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),  # under suptitle
        ncol=3,
        frameon=False
    )

    ax.grid(True)

    # Leave top room for title + legend
    fig.tight_layout(rect=[0, 0, 1, 0.995])

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(file_name)[0]
    save_path = os.path.join(out_dir, f"14.4keV_gaussfit.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved Gaussian-fit plot to: {save_path}")

    return popt, pcov

# --------------------------------------

def find_git_root(start_path=None):
    if start_path is None:
        start_path = os.path.abspath(os.path.dirname(__file__))
    cur = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("Could not find .git directory. Run inside your git repo.")
        cur = parent

def load_asc_1col(filepath):
    data = np.loadtxt(filepath).astype(float)
    if data.ndim == 2:
        data = data[:, 0]
    ch = np.arange(len(data), dtype=float)
    return ch, data

def voltage_from_filename(filepath):
    return os.path.basename(filepath).split("_")[0]

def voltage_sort_key(filepath):
    v = voltage_from_filename(filepath)
    m = re.search(r"(\d+)", v)
    return int(m.group(1)) if m else v

def compute_calibration_2pt(mu_low, mu_high):
    """Return (a,b) for E = a*ch + b, or (None,None) if invalid."""
    mu_low = float(mu_low)
    mu_high = float(mu_high)
    if not np.isfinite(mu_low) or not np.isfinite(mu_high):
        return None, None
    if mu_high <= mu_low:
        return None, None
    a = (E_HIGH - E_LOW) / (mu_high - mu_low)
    b = E_LOW - a * mu_low
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None, None
    return float(a), float(b)

def load_picks_csv(picks_csv_path):
    """
    Returns dict keyed by basename(file) -> dict with:
      label, mu_low, mu_high
    CSV must contain columns: file,label,mu_low_ch,mu_high_ch (as created earlier).
    """
    picks = {}
    with open(picks_csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        required = {"file", "label", "mu_low_ch", "mu_high_ch"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"{picks_csv_path} missing columns: {sorted(missing)}")

        for row in r:
            base = row["file"].strip()
            mu_low_s = row["mu_low_ch"].strip()
            mu_high_s = row["mu_high_ch"].strip()
            picks[base] = {
                "label": row["label"].strip(),
                "mu_low": float(mu_low_s) if mu_low_s != "" else None,
                "mu_high": float(mu_high_s) if mu_high_s != "" else None,
            }
    return picks


def save_plot_calibrated(outpath, energy_x, y, title):
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    ax.plot(energy_x, y)
    ax.set_title(title)
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel("Counts")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def save_plot_single_peak_label_only(outpath, ch, y, mu_peak, title):
    """
    Plot vs channel, but label the x-axis as energy [keV] and show ONLY one tick:
    the picked peak position labeled '14.4'.
    """
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    ax.plot(ch, y)
    ax.set_title(title)
    ax.set_ylabel("Counts")

    # Keep the axis label, but show only one tick
    ax.set_xlabel("Energy [keV]")

    ax.set_xticks([mu_peak])
    ax.set_xticklabels(["14.4"])

    # Optional: make the single tick look clean
    ax.tick_params(axis="x", which="both", bottom=True, top=False)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def save_plot_channel(outpath, ch, y, title):
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    ax.plot(ch, y)
    ax.set_title(title)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def main():
    repo_root = find_git_root()
    plt.style.use("./custom.mplstyle")

    data_dir = os.path.join(repo_root, DATA_SUBDIR)
    out_dir = os.path.join(repo_root, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(repo_root, "results"), exist_ok=True)

    picks_path = os.path.join(repo_root, PICKS_CSV)
    if not os.path.exists(picks_path):
        raise RuntimeError(f"Could not find picks CSV at: {picks_path}")

    picks = load_picks_csv(picks_path)

    files = sorted(
        glob.glob(os.path.join(data_dir, "*calib_energy.asc")) +
        glob.glob(os.path.join(data_dir, "*calib_energy.ASC")),
        key=voltage_sort_key
    )
    if not files:
        raise RuntimeError(f"No calibration files found in {data_dir}")

    summary_rows = []

    for fp in files:
        base = os.path.basename(fp)
        label = voltage_from_filename(fp)
        ch, y = load_asc_1col(fp)

        row = picks.get(base, None)
        mu_low = row["mu_low"] if row else None
        mu_high = row["mu_high"] if row else None

        # Order-proof if both exist
        if mu_low is not None and mu_high is not None and mu_high < mu_low:
            mu_low, mu_high = mu_high, mu_low

        outpath = os.path.join(out_dir, f"{label}.pdf")

        # Case 1: two peaks -> true 2-point calibration
        if mu_low is not None and mu_high is not None:
            a, b = compute_calibration_2pt(mu_low, mu_high)
            if a is None:
                # If picks were nonsense, fall back to channel plot so you see the issue
                save_plot_channel(outpath, ch, y, title=f"{label}")
                summary_rows.append({
                    "file": base, "label": label,
                    "mu_low_ch": f"{mu_low:.3f}", "mu_high_ch": f"{mu_high:.3f}",
                    "mode": "2pt", "status": "FAIL_2PT"
                })
            else:
                energy = a * ch + b
                save_plot_calibrated(outpath, energy, y, title=f"{label}")
                summary_rows.append({
                    "file": base, "label": label,
                    "mu_low_ch": f"{mu_low:.3f}", "mu_high_ch": f"{mu_high:.3f}",
                    "mode": "2pt", "status": "OK_2PT"
                })
            continue

        # Case 2: exactly one peak -> NO calibration, label only 14.4 keV at that peak
        mu_single = None
        if mu_high is not None:
            mu_single = mu_high
        elif mu_low is not None:
            mu_single = mu_low

        if mu_single is not None:
            save_plot_single_peak_label_only(
                outpath, ch, y, mu_peak=float(mu_single),
                title=f"{label}"
            )
            summary_rows.append({
                "file": base, "label": label,
                "mu_low_ch": "" if mu_low is None else f"{mu_low:.3f}",
                "mu_high_ch": "" if mu_high is None else f"{mu_high:.3f}",
                "mode": "label_only", "status": "OK_14.4_LABEL_ONLY"
            })
            continue

        # Case 3: no pick -> plain channel plot
        save_plot_channel(outpath, ch, y, title=f"{label}")
        summary_rows.append({
            "file": base, "label": label,
            "mu_low_ch": "", "mu_high_ch": "",
            "mode": "none", "status": "NO_PICK"
        })

    # Write summary
    summary_path = os.path.join(repo_root, SUMMARY_CSV)
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["file", "label", "mu_low_ch", "mu_high_ch", "mode", "status"]
        )
        w.writeheader()
        w.writerows(summary_rows)

    fit_gaussian_to_file(data_dir, "14_4_keV_line.asc", out_dir)

    print(f"Saved plots to: {out_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
