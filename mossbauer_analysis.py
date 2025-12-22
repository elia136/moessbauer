import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import find_peaks

plt.style.use("./custom.mplstyle")


class Spectrum:
    """
    Class for storing and processing raw Mössbauer spectra.
    """

    def __init__(self, counts):
        counts = np.asarray(counts, dtype=float)
        # ensure dimensions
        if counts.ndim != 1:
            raise ValueError("counts must be 1D")
        if len(counts) % 2:
            counts = counts[:-1]
        self.raw_counts = counts
        self.folded_counts = None
        self.folded_sigma = None
        self.folded_idx = None

    @classmethod
    def from_folded(cls, counts, sigma, idx, half_len):
        """Create Spectrum instance from folded data."""
        obj = cls(np.zeros(2 * int(half_len), dtype=float))
        obj.folded_counts = np.asarray(counts, dtype=float)
        obj.folded_sigma = np.asarray(sigma, dtype=float)
        obj.folded_idx = np.asarray(idx, dtype=int)
        obj.folded_half_len = int(half_len)
        obj.split_index = int(half_len)
        obj.align_shift = 0
        return obj

    # ---- Main folding method ----
    def fold(self, clip_start=0):
        """Fold the raw spectrum by finding the optimal split and alignment."""
        counts = self.raw_counts.copy()
        if clip_start > 0:
            counts[:clip_start] = np.nan
        n = len(counts)
        # split and fold in the middle, no shift
        self.split_index = n // 2
        self.align_shift = 0
        first = counts[:self.split_index]
        second = counts[self.split_index:][::-1]
        # make halves equal length
        half_len = min(len(first), len(second))
        first = first[:half_len]
        second = second[:half_len]
        # sum halves and ignore nans
        folded_full = np.nan_to_num(first, nan=0.0) + np.nan_to_num(second, nan=0.0)
        # poisson errors sig ~ sqrt(counts)
        sigma_full = np.sqrt(np.clip(folded_full, 1.0, np.inf))
        # mask bins where either half was nan
        valid = (
            np.isfinite(folded_full)
            & np.isfinite(sigma_full)
            & (np.isfinite(first) & np.isfinite(second))
        )
        # store masked folded spectrum
        self.folded_half_len = half_len
        self.folded_idx = np.arange(half_len)[valid]
        self.folded_counts = folded_full[valid]
        self.folded_sigma = sigma_full[valid]

        return self.folded_counts, self.folded_sigma

# ---- Velocity axis conversions ----
def velocity_axis(spectrum, v_max):
    """Generate velocity axis for folded spectrum, accounting for masked nan bins."""
    if spectrum.folded_idx is None:
        raise RuntimeError("Call fold() first")
    half_len = getattr(spectrum, "folded_half_len", None)
    center = half_len / 2.0
    scale = float(v_max) / (half_len / 2.0)
    return scale * (spectrum.folded_idx - center)

def velocity_to_bin_offset(velocity, half_len, v_max):
    """Convert velocity (mm/s) to bin offset from center."""
    center = half_len / 2.0
    scale = v_max / center
    return velocity / scale

def bin_offset_to_velocity(bin_offset, half_len, v_max):
    """Convert bin offset from center to velocity (mm/s)."""
    scale = v_max / (half_len / 2.0)
    return bin_offset * scale

# ---- Lorentzian model and fitting ----
def lorentzian(velocity, center, fwhm):
    """Lorentzian function normalized to peak height 1 at center."""
    g = 0.5 * fwhm
    return g * g / ((velocity - center) ** 2 + g * g)

class LorentzianModel:
    """
    Multi-peak Lorentzian absorption model with linear baseline.
    Parameter layout:
    [baseline, slope, fwhm, amp1, t1, amp2, d2, amp3, d3, ...]
    where:
        baseline:   constant offset
        slope:      linear slope
        fwhm:       full width at half maximum (same for all peaks)
        ampk:       amplitude of peak k (positive value)
        t1:         center of first peak
        dk:         log gap between peak k and peak k-1 (for k >= 2)
    """
    def __init__(self, n_peaks):
        self.n_peaks = int(n_peaks)

    def centers(self, params):
        """Compute peak centers from parameters."""
        tail = params[3:]
        centers = [tail[1]]
        idx = 2
        for _ in range(2, self.n_peaks + 1):
            centers.append(centers[-1] + np.exp(tail[idx + 1]))
            idx += 2
        return np.array(centers)

    def amplitudes(self, params):
        """Extract peak amplitudes from parameters."""
        tail = params[3:]
        amps = [tail[0]]
        idx = 2
        for _ in range(2, self.n_peaks + 1):
            amps.append(tail[idx])
            idx += 2
        return np.array(amps)

    def __call__(self, velocity, params):
        """Evaluate the model at given velocity points."""
        baseline, slope, fwhm = params[:3]
        model = baseline + slope * velocity
        centers = self.centers(params)
        amplitudes = self.amplitudes(params)
        for amp, cen in zip(amplitudes, centers):
            model -= amp * lorentzian(velocity, cen, fwhm)

        return model

def make_p0(spectrum, n_peaks, v_max, smooth_window=9, depth=0.05, fwhm0=0.25):
    """Generate initial parameter guesses for fitting."""
    counts = spectrum.folded_counts
    n = len(counts)
    velocity = velocity_axis(spectrum, v_max)
    # Smooth the spectrum to find dips
    smooth_window = max(3, int(smooth_window) | 1)
    smoothed = np.convolve(counts, np.ones(smooth_window) / smooth_window, mode="same")
    # Estimate baseline as median of smoothed spectrum
    baseline = np.median(smoothed)
    # Dip signal for peak finding
    dip_signal = np.maximum(baseline - smoothed, 0.0)
    min_separation = max(2, n // max(4 * n_peaks, 1))
    # Find peaks in dip signal
    indices, _ = find_peaks(dip_signal, distance=min_separation)
    # if not enough dips found, raise error
    if len(indices) < n_peaks:
        raise RuntimeError("Not enough absorption dips found for initial guess")
    # select strongest peaks and sort
    strongest = indices[np.argsort(dip_signal[indices])[-n_peaks:]]
    center_guesses = np.sort(velocity[strongest])
    # Construct initial parameter array
    amplitude = depth * baseline
    params = [baseline, 0.0, fwhm0, amplitude, center_guesses[0]]
    names = ["baseline", "slope", "fwhm", "amp1", "t1"]
    if n_peaks > 1:
        gaps = np.diff(center_guesses)
        gaps = np.maximum(gaps, 1e-3)
        for i, gap in enumerate(gaps, start=2):
            params += [amplitude, np.log(gap)]
            names += [f"amp{i}", f"d{i}"]

    return np.array(params), names

def make_bounds(n_peaks, v_max):
    """Generate parameter bounds for fitting."""
    lower = [-np.inf, -np.inf, 0.01, 0.0, -v_max]
    upper = [np.inf, np.inf, 2.0, np.inf, v_max]
    for _ in range(2, n_peaks + 1):
        lower += [0.0, -20.0]
        upper += [np.inf, 20.0]
    return np.array(lower), np.array(upper)

def fit_spectrum(spectrum, n_peaks, v_max, v_max_err=None, *, loss="wls"):
    """Fit the folded Mössbauer spectrum using weighted least squares or Poisson deviance."""
    velocity = velocity_axis(spectrum, v_max)
    model = LorentzianModel(n_peaks)
    # initial parameters and bounds
    p0, names = make_p0(spectrum, n_peaks, v_max)
    bounds = make_bounds(n_peaks, v_max)
    # data and errors
    y = spectrum.folded_counts
    dy = spectrum.folded_sigma
    # ---- Residuals definitions ----
    def wls_residuals(params):
        """Weighted least squares residuals."""
        mu = model(velocity, params)
        return (mu - y) / dy
    def poisson_residuals(params):
        """Poisson deviance residuals."""
        mu = model(velocity, params)
        # avoid log(0) issues
        mu = np.clip(mu, 1e-12, np.inf)
        y_clip = np.clip(y, 0.0, np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.where(y_clip > 0, y_clip * np.log(y_clip / mu), 0.0)
        deviance = 2.0 * (mu - y_clip + log_term)
        deviance = np.clip(deviance, 0.0, np.inf)
        return np.sign(mu - y_clip) * np.sqrt(deviance)
    # select residuals function
    residuals = wls_residuals if loss == "wls" else poisson_residuals
    # perform least squares fitting
    result = least_squares(residuals, p0, bounds=bounds)
    # extract centers from fitted parameters
    centers = model.centers(result.x)
    output = {
        "velocity": velocity,
        "model": model,
        "params": result.x,
        "param_names": names,
        "centers": centers,
        "fwhm": result.x[2],
        "loss": loss,
    }
    return output

# ---- Monte Carlo bootstrap ----
def monte_carlo(
    raw_counts: np.ndarray,
    n_peaks: int,
    v_max: float,
    *,
    physics: dict | None = None,
    reference_bin_offset_samples: np.ndarray | None = None,
    v_max_err: float | None = None,
    model_systematic: bool = True,
    clip_start: int = 0,
    B: int = 500,
    seed: int = 0,
    loss: str = "poisson",
):
    """
    Monte Carlo resampling using parametric bootstrap to estimate uncertainties in Mössbauer spectral fits.
    Sampling includes Poisson noise, velocity scale systematics, and evaluation of physsics quantities.
    """
    rng = np.random.default_rng(seed)

    # initial fold of the original raw spectrum
    spec0 = Spectrum(raw_counts)
    spec0.fold(clip_start=clip_start)

    # initial fit of the folded spectrum
    fit0 = fit_spectrum(
        spec0,
        n_peaks=n_peaks,
        v_max=v_max,
        v_max_err=None,
        loss=loss  # keep for diagnostics if you like
    )
    # masked velocity axis
    v0 = fit0["velocity"]
    ci_grid = v0.copy()
    mu0 = fit0["model"](v0, fit0["params"])
    mu0 = np.clip(mu0, 0.0, np.inf)

    idx0 = spec0.folded_idx.copy()
    half_len0 = spec0.folded_half_len
    # prepare storage for bootstrap samples
    physics = physics or {}
    physics_samples = {key: [] for key, enabled in physics.items() if enabled}
    centers_s = []
    fwhm_s = []
    rel_intensity_s = []
    params_s = []
    mu_s = []
    v_max_s = []
    mu_star_s = []
    mu_pred_s = []
    # counters for failed folds/fits
    n_fail_fold = 0
    n_fail_fit = 0
    # --- Monte Carlo loop ---
    for _ in range(B):
        # systematic: sample v_max
        v_max_b = v_max
        if model_systematic and v_max_err is not None:
            v_max_b = float(rng.normal(v_max, v_max_err))
        n_b = rng.poisson(mu0).astype(float)
        sigma_b = np.sqrt(np.clip(n_b, 1.0, np.inf))
        spec_b = Spectrum.from_folded(
            counts=n_b,
            sigma=sigma_b,
            idx=idx0,
            half_len=half_len0,
        )
        v_b = velocity_axis(spec_b, v_max_b)
        # fit folded spectrum
        try:
            fit_b = fit_spectrum(
                spec_b,
                n_peaks=n_peaks,
                v_max=v_max_b,
                v_max_err=None,
                loss=loss
            )
        except Exception:
            n_fail_fit += 1
            continue
        # extract relative intensities
        amps_b = fit_b["model"].amplitudes(fit_b["params"])
        amps_b = np.asarray(amps_b, dtype=float)
        rel_intensity_b = amps_b / np.sum(amps_b)
        rel_intensity_s.append(rel_intensity_b)
        # compute half length for reference velocity calculation
        half_len_b = spec_b.folded_half_len
        ref_off_b = None
        # if requested, calculate bin offset relative to half length and v_max
        if reference_bin_offset_samples is not None:
            ref_off_b = float(
                reference_bin_offset_samples[
                    rng.integers(0, len(reference_bin_offset_samples))
                ]
            )
        # compute physics quantities
        for key, enabled in (physics or {}).items():
            if not enabled:
                continue
            if key == "reference_bin_offset":
                # compute reference bin offset
                value = physics_reference_bin_offset(
                    fit_b["centers"], half_len_b, v_max_b
                )
            elif key == "isomer_shift":
                v_cent = np.mean(fit_b["centers"])
                # convert to bin offset relative to reference 
                off_cent = velocity_to_bin_offset(v_cent, half_len_b, v_max_b)
                # compute isomer shift relative to reference bin offset (done in bins because v_max may vary)
                delta_off = off_cent - ref_off_b
                # convert back to velocity
                value = bin_offset_to_velocity(delta_off, half_len_b, v_max_b)
            else:
                # other physics quantities
                value = PHYSICS_FUNCTIONS[key](fit_b["centers"], fit_b["fwhm"])
            physics_samples[key].append(value)

        # evaluate model on ci_grid
        mu_b = fit_b["model"](v_b, fit_b["params"])
        mu_b = np.clip(mu_b, 0.0, np.inf)
        # interpolate to ci_grid
        mu_star_b = np.interp(ci_grid, v_b, mu_b)
        mu_star_b = np.clip(mu_star_b, 0.0, np.inf)

        centers_s.append(fit_b["centers"])
        fwhm_s.append(fit_b["fwhm"])
        params_s.append(fit_b["params"])
        mu_s.append(mu_b)
        mu_pred_s.append(rng.poisson(mu_star_b).astype(float))
        mu_star_s.append(mu_star_b)
        v_max_s.append(v_max_b)
    # convert lists to arrays
    centers_s = np.asarray(centers_s)
    fwhm_s = np.asarray(fwhm_s)
    params_s = np.asarray(params_s)
    rel_intensity_s = np.asarray(rel_intensity_s)
    mu_s = np.asarray(mu_s)
    v_max_s = np.asarray(v_max_s)
    mu_pred_s = np.asarray(mu_pred_s)
    mu_star_s = np.asarray(mu_star_s)
    for key in physics_samples:
        physics_samples[key] = np.asarray(physics_samples[key])

    # prepare output dictionary
    out = {
        "fit0": fit0,
        "spec0": spec0,
        "mu0": mu0,
        "mu_samples": mu_s,
        "mu_conf_samples": mu_star_s,
        "mu_pred_samples": mu_pred_s,
        "centers_samples": centers_s,
        "ci_grid": ci_grid,
        "fwhm_samples": fwhm_s,
        "params_samples": params_s,
        "rel_intensity_samples": rel_intensity_s,
        "v_max_samples": v_max_s,
        "physics_samples": physics_samples,
        "n_fail_fit": n_fail_fit,
    }

    return out


def envelope_band(mu_samples, keep=0.68):
    """Envelope band around the median curve from mu_samples."""
    # mu_samples shape: (B, N)
    med = np.median(mu_samples, axis=0)
    # L2 distance to median curve as a simple ranking
    d = np.sum((mu_samples - med) ** 2, axis=1)
    k = max(1, int(np.floor(keep * len(d))))
    idx = np.argsort(d)[:k]
    lo = np.min(mu_samples[idx], axis=0)
    hi = np.max(mu_samples[idx], axis=0)
    return lo, hi


def summarize_samples(samples: np.ndarray, level: float = 0.68, axis: int = 0):
    """Percentile CI summary for bootstrap samples."""
    alpha = (1.0 - level) / 2.0
    lo = 100.0 * alpha
    hi = 100.0 * (1.0 - alpha)
    med = np.percentile(samples, 50.0, axis=axis)
    qlo = np.percentile(samples, lo, axis=axis)
    qhi = np.percentile(samples, hi, axis=axis)
    return med, qlo, qhi

def plot_results(
    spectrum,
    fit,
    physics,
    *,
    title="Title",
    save_path="results/fit_plot.pdf",
    ci_lo=None,
    ci_hi=None,
    pi_hi=None,
    pi_lo=None,
):
    """Plot Mössbauer spectrum with fit, confidence intervals, predicted intervals, and residuals."""
    # masked velocity axis
    v = fit["velocity"]
    # data and errors (Poisson)
    y = spectrum.folded_counts
    dy = spectrum.folded_sigma
    # fitted curve
    y_fit = fit["model"](v, fit["params"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [1.85, 1], "hspace": 0.1}
    )
    title_artist = ax1.set_title(title, pad=20)
    # --- data + fit ---
    ax1.errorbar(v, y, yerr=dy, fmt=".", ms=2, zorder=4)
    ax1.plot(v, y_fit, lw=1.3, label="Fit", zorder=5)
    for c in fit["centers"]:
        ax1.axvline(c, color="k", lw=0.8, ls=":", alpha=0.8)
    # --- optional CI ribbon (mean curve) ---
    if ci_lo is not None and ci_hi is not None:
        ax1.fill_between(v, ci_lo, ci_hi, color="#4D4D4D", alpha=0.4, label="68% CI", zorder=3)
    # --- optional PI ribbon (predicted counts) ---
    if pi_lo is not None and pi_hi is not None:
        ax1.fill_between(
            v, pi_lo, pi_hi, color="#4D4D4D", alpha=0.2, label="95% PI", zorder=2
        )
    ax1.set_ylabel("Counts")
    ax1.grid(True, zorder=0)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # normalized residuals even if fit used Poisson deviance)
    ax2.axhline(0, color="gray", lw=1)
    ax2.plot(v, (y - y_fit) / dy, ".", ms=3)
    ax2.axhline(+1, color="k", lw=0.8, ls=":", alpha=0.8)
    ax2.axhline(-1, color="k", lw=0.8, ls=":", alpha=0.8)
    ax2.set_xlabel("Velocity (mm/s)")
    ax2.set_ylabel(r"Residuals ($\sigma$)")
    ax2.grid(True)

    # legend placed under title
    fig.canvas.draw()
    title_box = title_artist.get_window_extent(fig.canvas.renderer)  # type: ignore
    title_y = title_box.y1 / fig.bbox.ymax
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, title_y - 0.03), ncol=3)
    plt.savefig(save_path, backend="pgf")
    plt.close()

# ---- physics calculations ----
def physics_reference_bin_offset(centers, half_len, v_max):
    """Compute reference bin offset from peak centers."""
    v_ref = np.mean(centers)
    return velocity_to_bin_offset(v_ref, half_len, v_max)

def physics_reference_velocity(centers):
    """Compute reference velocity from peak centers."""
    return np.mean(centers)

def physics_hyperfine_field(centers):
    """Compute hyperfine field from peak centers."""
    dv = centers[-1] - centers[0]
    # doppler energy splitting
    dE = E_GAMMA_J * (dv * 1e-3) / C
    B_hf = dE / (2.0 * (MU_G - MU_E))
    return B_hf

def physics_excited_magnetic_moment(centers):
    """Compute excited state magnetic moment from peak centers."""
    dg = centers[5] - centers[0]
    de = centers[4] - centers[1]
    I_g, I_e = 0.5, 1.5
    mu_e = MU_G * (I_e / I_g) * (np.abs(de) / np.abs(dg))
    return mu_e

def physics_lifetime(fwhm):
    """Compute excited state lifetime from FWHM."""
    GAMMA = E_GAMMA_J * (fwhm * 1e-3) / C
    tau = HBAR / GAMMA
    return tau

def physics_quadrupole_splitting(centers):
    """Compute quadrupole splitting from peak centers."""
    return centers[1] - centers[0]

def physics_isomer_shift(centers, reference_velocity=0.0):
    """Compute isomer shift from peak centers."""
    return np.mean(centers) - reference_velocity

# ---- constants and physics presentations ----
C = 299792458  # m/s
E_GAMMA = 14.4e3  # eV
E_GAMMA_J = E_GAMMA * 1.60218e-19  # J
HBAR = 1.0545718e-34  # J.s
MU_N = 5.0507837e-27  # J/T
MU_G = 0.0906 * MU_N  # J/T
MU_E = -0.154 * MU_N  # J/T

# physics calculation functions mapping
PHYSICS_FUNCTIONS = {
    "reference_bin_offset": lambda centers, fwhm: np.nan,
    "hyperfine_field": lambda centers, fwhm: physics_hyperfine_field(centers),
    "excited_magnetic_moment": lambda centers, fwhm: physics_excited_magnetic_moment(
        centers
    ),
    "lifetime": lambda centers, fwhm: physics_lifetime(fwhm),
    "quadrupole": lambda centers, fwhm: physics_quadrupole_splitting(centers),
    "isomer_shift": lambda centers, fwhm: physics_isomer_shift(centers),
}
# presentation metadata for physics quantities
PHYSICS_PRESENTATIONS = {
    "reference_bin_offset": {
        "name": "Reference Bin Offset",
        "unit": "bins",
        "scale": lambda x: x,
        "latex": r"\Delta n_{\mathrm{ref}}",
    },
    "hyperfine_field": {
        "name": "Hyperfine Field",
        "unit": "T",
        "scale": lambda x: x,
        "latex": r"B_{\mathrm{HF}}",
    },
    "excited_magnetic_moment": {
        "name": "Excited State Magnetic Moment",
        "unit": r"\mu_N",
        "scale": lambda x: x / MU_N,
        "latex": r"\mu_e",
    },
    "lifetime": {
        "name": "Excited State Lifetime",
        "unit": "ns",
        "scale": lambda x: x * 1e9,
        "latex": r"\tau",
    },
    "quadrupole": {
        "name": "Quadrupole Splitting",
        "unit": "mm/s",
        "scale": lambda x: x,
        "latex": r"\Delta Q",
    },
    "isomer_shift": {
        "name": "Isomer Shift",
        "unit": "mm/s",
        "scale": lambda x: x,
        "latex": r"\delta",
    },
}

def main():
    measurements = [
        {
            "absorber": "iron",
            "file": "data/Fe_1950V_2d.asc",
            "n_peaks": 6,
            "v_max": 6.0,
            "v_max_err": 0.3,
            "physics": {
                "reference_bin_offset": True,
                "hyperfine_field": True,
                "excited_magnetic_moment": True,
            },
        },
        {
            "absorber": "magnetized_iron",
            "file": "data/Fe_1950V_9mms_1d_magn.asc",
            "n_peaks": 6,
            "v_max": 9.0,
            "v_max_err": 0.45,
        },
        {
            "absorber": "steel",
            "file": "data/stainless_steel_1950V_4mms_3d.asc",
            "n_peaks": 1,
            "v_max": 4.0,
            "v_max_err": 0.4,
            "physics": {"lifetime": True},
        },
        {
            "absorber": "ferrocyanide",
            "file": "data/potassium_ferrocyanide_1950V_9mms_2d.asc",
            "n_peaks": 1,
            "v_max": 9.0,
            "v_max_err": 0.45,
            "physics": {"isomer_shift": True},
        },
        {
            "absorber": "sulphate",
            "file": "data/ferrous_sulphate_1950V_9mms_2d.asc",
            "n_peaks": 2,
            "v_max": 9.0,
            "v_max_err": 0.45,
            "physics": {"quadrupole": True},
        },
        {
            "absorber": "dust",
            "file": "data/space_dust_1950V_9mms_2d.asc",
            "n_peaks": 1,
            "v_max": 9.0,
            "v_max_err": 0.45,
            "physics": {"isomer_shift": True},
        },
    ]
    ref_bin_samples = None
    for meas in measurements:
        print(f"Absorber: {meas['absorber']}")
        raw_counts = np.loadtxt(meas["file"])
        # perform Monte Carlo bootstrap
        boot = monte_carlo(
            raw_counts,
            n_peaks=meas["n_peaks"],
            v_max=meas["v_max"],
            v_max_err=meas["v_max_err"],
            physics=meas.get("physics", {}),
            reference_bin_offset_samples=ref_bin_samples,
            clip_start=2,
            B=1000,
            seed=1,
            loss="poisson",
        )
        if "reference_bin_offset" in boot["physics_samples"]:
            ref_bin_samples = boot["physics_samples"]["reference_bin_offset"]

        # The fit for plotting should be the original fit returned by bootstrap
        fit0 = boot["fit0"]
        spec0 = boot["spec0"]

        centers_med, centers_lo, centers_hi = summarize_samples(
            boot["centers_samples"], level=0.68
        )
        fwhm_med, fwhm_lo, fwhm_hi = summarize_samples(boot["fwhm_samples"], level=0.68)
        rel_intensity_med, rel_intensity_lo, rel_intensity_hi = summarize_samples(
            boot["rel_intensity_samples"], level=0.68
        )

        for key, samples in boot["physics_samples"].items():
            meta = PHYSICS_PRESENTATIONS.get(key, None)
            if meta is None:
                continue
            values = meta["scale"](samples)
            med, qlo, qhi = summarize_samples(values, level=0.68)
            if key == "reference_bin_offset":
                print(
                    f"  Reference Velocity = {bin_offset_to_velocity(med, spec0.folded_half_len, meas["v_max"]):.6g} mm/s"
                )
            print(
                f"  {meta['name']} = {med:.6g} 68% CI [{qlo:.6g}, {qhi:.6g}] {meta['unit']} (deltas: +{(med - qlo):.6g}/-{(qhi - med):.6g})"
            )
        print("------ fit parameters ------")
        print(
            f"  FWHM = {fwhm_med:.6g} mm/s 68% CI [{fwhm_lo:.6g}, {fwhm_hi:.6g}] (deltas: +{(fwhm_med - fwhm_lo):.6g}/-{(fwhm_hi - fwhm_med):.6g})"
        )
        for i, c_med in enumerate(centers_med):
            c_lo = centers_lo[i]
            c_hi = centers_hi[i]
            print(
                f"  Center {i+1} = {c_med:.6g} mm/s 68% CI [{c_lo:.6g}, {c_hi:.6g}] (deltas: +{(c_med - c_lo):.6g}/-{(c_hi - c_med):.6g})"
            )
        for i, ri_med in enumerate(rel_intensity_med):
            ri_lo = rel_intensity_lo[i]
            ri_hi = rel_intensity_hi[i]
            print(
                f"  Rel. Intensity {i+1} = {ri_med:.6g} 68% CI [{ri_lo:.6g}, {ri_hi:.6g}] (deltas: +{(ri_med - ri_lo):.6g}/-{(ri_hi - ri_med):.6g})"
            )
        print(
            f"  (Failed fits: {boot['n_fail_fit']})"
        )
        print("----------------------------")

        v_star = boot["ci_grid"]
        _, ci_lo, ci_hi = summarize_samples(boot["mu_conf_samples"], level=0.68)
        _, pi_lo, pi_hi = summarize_samples(boot["mu_pred_samples"], level=0.95)
        plot_results(
            spec0,
            fit0,
            physics=boot["physics_samples"],
            title=f"{meas['absorber'].capitalize()}: Lorentzian Fit and Monte Carlo CI",
            save_path=f"results/fit_plot_{meas['absorber']}_bootstrapCI.pdf",
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            pi_lo=pi_lo,
            pi_hi=pi_hi,
        )

if __name__ == "__main__":
    main()