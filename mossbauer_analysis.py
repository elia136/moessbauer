import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import find_peaks

plt.style.use("./custom.mplstyle")

class Spectrum:
    def __init__(self, counts):
        counts = np.asarray(counts, dtype=float)
        if counts.ndim != 1:
            raise ValueError("counts must be 1D")
        if len(counts) % 2:
            counts = counts[:-1]

        self.raw_counts = counts
        self.folded_counts = None
        self.folded_sigma = None
        self.folded_idx = None
        self.split_index = None
        self.align_shift = None

    @staticmethod
    def _moving_average(signal, window):
        window = max(5, int(window) | 1)
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode="same")

    @staticmethod
    def _highpass(signal, window):
        return signal - Spectrum._moving_average(signal, window)

    @staticmethod
    def _shift_with_nan(signal, shift):
        shifted = np.full_like(signal, np.nan)
        if shift < 0:
            shifted[:shift] = signal[-shift:]
        elif shift > 0:
            shifted[shift:] = signal[:-shift]
        else:
            shifted[:] = signal
        return shifted

    @staticmethod
    def _best_shift_by_correlation(reference, moving, max_shift):
        reference = reference - np.mean(reference)
        reference_norm = np.linalg.norm(reference) + 1e-12

        best_shift = 0
        best_score = -np.inf

        for shift in range(-max_shift, max_shift + 1):
            shifted = Spectrum._shift_with_nan(moving, shift)
            valid = np.isfinite(shifted) & np.isfinite(reference)
            if np.count_nonzero(valid) < 20:
                continue

            m = shifted[valid] - np.mean(shifted[valid])
            score = np.dot(reference[valid], m) / (
                reference_norm * (np.linalg.norm(m) + 1e-12)
            )

            if score > best_score:
                best_score = score
                best_shift = shift

        return best_shift

    def fold(self,
             trim=10,
             detrend_window=101,
             max_shift=200,
             split_search=10,
             clip_start=0,
             freeze_split=None,
             freeze_shift=None
    ):
        counts = self.raw_counts.copy()
        if clip_start > 0:
            counts[:clip_start] = np.nan
        n = len(counts)
        nominal_split = n // 2

        if freeze_split is not None and freeze_shift is not None:
            best_split = freeze_split
            best_shift = freeze_shift
            self.split_index = best_split
            self.align_shift = best_shift
        else:
            best_score = -np.inf
            best_split = nominal_split
            best_shift = 0

            for split in range(nominal_split - split_search, nominal_split + split_search + 1):
                if split <= trim or split >= n - trim:
                    continue

                first = counts[:split]
                second = counts[split:][::-1]
                half_len = min(len(first), len(second))

                first = first[:half_len]
                second = second[:half_len]

                core_first = self._highpass(first[trim:-trim], detrend_window)
                core_second = self._highpass(second[trim:-trim], detrend_window)

                shift = self._best_shift_by_correlation(core_first, core_second, max_shift)
                shifted = self._shift_with_nan(core_second, shift)
                valid = np.isfinite(shifted) & np.isfinite(core_first)
                if np.count_nonzero(valid) < 20:
                    continue
                score = np.corrcoef(core_first[valid], shifted[valid])[0, 1]
                if score > best_score:
                    best_score = score
                    best_split = split
                    best_shift = shift

        self.split_index = best_split
        self.align_shift = best_shift

        first = counts[:best_split]
        second = counts[best_split:][::-1]
        half_len = min(len(first), len(second))
        first = first[:half_len]
        second = second[:half_len]

        second = self._shift_with_nan(second, best_shift)

        # sum halves and ignore nans
        folded_full = np.nan_to_num(first, nan=0.0) + np.nan_to_num(second, nan=0.0)
        # poisson errors sig ~ sqrt(counts)
        sigma_full = np.sqrt(np.clip(folded_full, 1.0, np.inf))

        valid = (
            np.isfinite(folded_full) &
            np.isfinite(sigma_full) &
            (np.isfinite(first) & np.isfinite(second))
        )
        self.folded_half_len = half_len
        self.folded_idx = np.arange(half_len)[valid]
        self.folded_counts = folded_full[valid]
        self.folded_sigma = sigma_full[valid]

        return self.folded_counts, self.folded_sigma

    def rebin(self, factor):
        if self.folded_counts is None:
            raise RuntimeError("Call fold() first")

        factor = int(factor)
        if factor <= 1:
            return self.folded_counts, self.folded_sigma

        n = (len(self.folded_counts) // factor) * factor
        counts = self.folded_counts[:n].reshape(-1, factor)
        sigma = self.folded_sigma[:n].reshape(-1, factor)   # type: ignore

        self.folded_counts = counts.sum(axis=1)
        self.folded_sigma = np.sqrt((sigma**2).sum(axis=1))
        return self.folded_counts, self.folded_sigma


def velocity_axis_mask(spectrum, v_max):
    """
    Generate velocity axis for folded spectrum, accounting for masked nan bins.
    """
    if spectrum.folded_idx is None:
        raise RuntimeError("Call fold() first")
    half_len = getattr(spectrum, "folded_half_len", None)

    center = half_len / 2.0
    scale = float(v_max) / (half_len / 2.0)
    return scale * (spectrum.folded_idx - center)

def velocity_axis_full(spectrum, v_max):
    """
    Generate velocity axis for folded spectrum, including nan bins.
    """
    if spectrum.folded_idx is None:
        raise RuntimeError("Call fold() first")
    half_len = getattr(spectrum, "folded_half_len", None)

    center = half_len / 2.0
    scale = float(v_max) / (half_len / 2.0)
    folded_idx_full = np.arange(half_len)
    return scale * (folded_idx_full - center)



def lorentzian(velocity, center, fwhm):
    g = 0.5 * fwhm
    return g * g / ((velocity - center) ** 2 + g * g)


class LorentzianModel:
    def __init__(self, n_peaks):
        self.n_peaks = int(n_peaks)

    def centers(self, params):
        tail = params[3:]
        centers = [tail[1]]
        idx = 2
        for _ in range(2, self.n_peaks + 1):
            centers.append(centers[-1] + np.exp(tail[idx + 1]))
            idx += 2
        return np.array(centers)

    def amplitudes(self, params):
        tail = params[3:]
        amps = [tail[0]]
        idx = 2
        for _ in range(2, self.n_peaks + 1):
            amps.append(tail[idx])
            idx += 2
        return np.array(amps)

    def __call__(self, velocity, params):
        baseline, slope, fwhm = params[:3]
        model = baseline + slope * velocity

        centers = self.centers(params)
        amplitudes = self.amplitudes(params)

        for amp, cen in zip(amplitudes, centers):
            model -= amp * lorentzian(velocity, cen, fwhm)

        return model

class SextetModel(LorentzianModel):
    def amplitudes(self, params):
        """
        parameter layout: [baseline, slope, fwhm, amp, eta, t1, d2, d3, d4, d5, d6]
        """
        tail = params[3:]
        amp = tail[0]
        eta = tail[1]
        base = np.array([3, 2, 1, 1, 2, 3], dtype=float)
        mod = np.array([np.exp(-eta), 1.0, np.exp(+eta), np.exp(+eta), 1.0, np.exp(-eta)], dtype=float)
        ratios = base * mod
        ratios /= ratios.mean()
        return amp * ratios
    
    def centers(self, params):
        tail = params[3:]
        t1 = tail[2]
        # log spacing of deltas
        dlogs = tail[3:]
        centers = [t1]
        for dj in dlogs:
            centers.append(centers[-1] + np.exp(dj))
        return np.array(centers)

def make_p0(spectrum, n_peaks, v_max, smooth_window=9, depth=0.05, fwhm0=0.25):
    counts = spectrum.folded_counts
    n = len(counts)
    velocity = velocity_axis_mask(spectrum, v_max)

    smooth_window = max(3, int(smooth_window) | 1)
    smoothed = np.convolve(counts, np.ones(smooth_window) / smooth_window, mode="same")

    baseline = np.median(smoothed)

    dip_signal = np.maximum(baseline - smoothed, 0.0)

    min_separation = max(2, n // max(4 * n_peaks, 1))
    indices, _ = find_peaks(dip_signal, distance=min_separation)

    if len(indices) < n_peaks:
        raise RuntimeError("Not enough absorption dips found for initial guess")

    strongest = indices[np.argsort(dip_signal[indices])[-n_peaks:]]
    center_guesses = np.sort(velocity[strongest])

    if n_peaks == 6:
        amp0 = depth * baseline
        eta0 = 0.0
        params = [baseline, 0.0, fwhm0, amp0, eta0, center_guesses[0]]
        names = ["baseline", "slope", "fwhm", "amp", "eta", "t1"]
        gaps = np.diff(center_guesses)
        gaps = np.maximum(gaps, 1e-3)
        for i, gap in enumerate(gaps, start=2):
            params += [np.log(gap)]
            names += [f"d{i}"]
        return np.array(params), names

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
    if n_peaks == 6:
        lower = [-np.inf, -np.inf, 0.01, 0.0, -2.0, -v_max]
        upper = [ np.inf,  np.inf, 2.0,  np.inf,  2.0,  v_max]
        for _ in range(5):
            lower += [-20.0]
            upper += [ 20.0]
        return np.array(lower), np.array(upper)
    lower = [-np.inf, -np.inf, 0.01, 0.0, -v_max]
    upper = [ np.inf,  np.inf, 2.0,  np.inf,  v_max]
    for _ in range(2, n_peaks + 1):
        lower += [0.0, -20.0]
        upper += [np.inf, 20.0]
    return np.array(lower), np.array(upper)


def covariance_from_jacobian(jacobian, cost, n_data, n_params, rcond=1e-12):
    dof = max(1, n_data - n_params)
    scale = 2 * cost / dof

    U, s, Vt = np.linalg.svd(jacobian, full_matrices=False)
    cutoff = rcond * s[0]
    inv_s2 = np.array([1 / si**2 if si > cutoff else 0.0 for si in s])

    return scale * (Vt.T * inv_s2) @ Vt


def fit_spectrum(
        spectrum,
        n_peaks,
        v_max, 
        v_max_err=None,
        *,
        loss="wls",
        compute_cov=True
    ):
    velocity = velocity_axis_mask(spectrum, v_max)
    model = SextetModel(n_peaks) if n_peaks == 6 else LorentzianModel(n_peaks)

    p0, names = make_p0(spectrum, n_peaks, v_max)
    bounds = make_bounds(n_peaks, v_max)

    y = spectrum.folded_counts
    dy = spectrum.folded_sigma

    def wls_residuals(params):
        mu = model(velocity, params)
        return (mu - y) / dy
    
    def poisson_residuals(params):
        mu = model(velocity, params)
        mu = np.clip(mu, 1e-12, np.inf)    
        y_clip = np.clip(y, 0.0, np.inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_term = np.where(y_clip > 0, y_clip * np.log(y_clip / mu), 0.0)
        deviance = 2.0 * (mu - y_clip + log_term)
        deviance = np.clip(deviance, 0.0, np.inf)
        return np.sign(mu - y_clip) * np.sqrt(deviance) 
    
    residuals = wls_residuals if loss == "wls" else poisson_residuals

    result = least_squares(residuals, p0, bounds=bounds)
    centers = model.centers(result.x)
    output = {
        "velocity": velocity,
        "model": model,
        "params": result.x,
        "param_names": names,
        "centers" : centers,
        "fwhm": result.x[2],
        "loss": loss,
    }

    if not compute_cov:
        output["covariance"] = None
        output["centers_err"] = None
        output["fwhm_err"] = None
        return output
    
    # covariance from jacobian
    cov = covariance_from_jacobian(
        result.jac,
        result.cost,
        n_data=len(y),
        n_params=len(result.x)
    )

    # center covariance via analytic Jacobian
    n_params = len(result.x)
    J = np.zeros((n_peaks, n_params))

    if n_peaks == 6:
        # Sextet layout: [b, m, fwhm, amp, eta, t1, d2, d3, d4, d5, d6]
        t1_idx = 5
        J[:, t1_idx] = 1.0

        # d2..d6 are at indices 6..10
        for j in range(2, 7):   # j = 2..6
            d_idx = 4 + j       # j=2->6, j=3->7, ..., j=6->10
            step = np.exp(result.x[d_idx])
            J[j-1:, d_idx] = step
    else:
        # Parameter layout:
        # [baseline, slope, fwhm, amp1, t1, amp2, d2, amp3, d3, ...]
        t1_idx = 4
        J[:, t1_idx] = 1.0  # all centers shift with t1

        # d(center_k)/d(dj) = exp(dj) for k >= j
        # d2 is at index 6, d3 at 8, ...
        for j in range(2, n_peaks + 1):
            d_idx = 2 * j + 2   # j=2 -> 6, j=3 -> 8, ...
            step = np.exp(result.x[d_idx])
            J[j-1:, d_idx] = step

    cov_centers = J @ cov @ J.T
    center_err = np.sqrt(np.clip(np.diag(cov_centers), 0.0, np.inf))
    output["covariance"] = cov

    if v_max_err is not None:
        rel = v_max_err / v_max
        output["centers_err"] = np.sqrt(center_err**2 + (np.abs(centers) * rel)**2)
        output["fwhm_err"] = np.sqrt((np.sqrt(cov[2,2]))**2 + (abs(result.x[2]) * rel)**2)
    else:
        output["centers_err"] = center_err
        output["fwhm_err"] = np.sqrt(cov[2,2])

    return output

def _estimate_half_ratio(raw_counts: np.ndarray, clip_start: int = 0) -> float:
    """
    Estimate r = total_first_half / (total_first_half + total_second_half)
    from the REAL raw scan. Applies clip_start by ignoring those bins.
    """
    x = np.asarray(raw_counts, dtype=float).copy()
    if clip_start > 0:
        x[:clip_start] = np.nan

    n = len(x)
    n2 = n // 2
    first = x[:n2]
    second = x[n2:]

    s1 = np.nansum(first)
    s2 = np.nansum(second)
    denom = s1 + s2
    if denom <= 0:
        return 0.5

    r = float(s1 / denom)
    return float(np.clip(r, 0.05, 0.95))

def _construct_raw_spectrum(mu_fold: np.ndarray, r: float) -> np.ndarray:
    """
    Construct a mean RAW spectrum of length 2*L from a mean folded spectrum mu_fold (length L),
    consistent with sum folding.

    We assume the folded expectation splits into two independent Poisson halves:
        E[N1_k] = r * mu_fold_k
        E[N2_k] = (1-r) * mu_fold_k

    Raw layout expected by your folding code:
        first half (forward):  mu_first = r * mu_fold
        second half (reverse): mu_second[::-1] = (1-r) * mu_fold   -> so raw second half is reversed
    """
    mu_fold = np.asarray(mu_fold, dtype=float)
    mu_fold = np.clip(mu_fold, 0.0, np.inf)

    mu_first = r * mu_fold
    mu_second = (1.0 - r) * mu_fold

    # raw = [first_half, second_half_reversed]
    return np.concatenate([mu_first, mu_second[::-1]])


def parametric_bootstrap(
    raw_counts: np.ndarray,
    n_peaks: int,
    v_max: float,
    *,
    v_max_err: float | None = None,
    clip_start: int = 0,
    fold_kwargs: dict | None = None,
    B: int = 500,
    seed: int = 0,
    ci_grid: np.ndarray | None = None,
    loss: str = "poisson",  # "poisson" recommended for raw counts
):
    """
    parametric bootstrap:

    0) Fit real data once: raw -> fold -> fit
    1) Build mean folded model mu_fold0 on the real folded velocity axis
    2) Estimate half-ratio r from the real raw scan
    3) Build mean raw model mu_raw0 (length 2*folded_half_len) consistent with sum folding
    4) For each replicate:
         a) sample v_max (optional systematic)
         b) draw synthetic raw counts ~ Poisson(mu_raw0)
         c) fold using SAME fold() (captures split/shift variability + AND-mask)
         d) fit using Poisson deviance (or WLS)
         e) store centers, fwhm, and model curve on ci_grid for CI band
    """
    rng = np.random.default_rng(seed)
    fold_kwargs = fold_kwargs or {}

    # ---- Fit the real data once ----
    spec0 = Spectrum(raw_counts)
    spec0.fold(clip_start=clip_start, **fold_kwargs)

    freeze_split = spec0.split_index
    freeze_shift = spec0.align_shift

    fit0 = fit_spectrum(
        spec0,
        n_peaks=n_peaks,
        v_max=v_max,
        v_max_err=None,
        loss=loss,
        compute_cov=True,   # keep for diagnostics if you like
    )

    v0 = fit0["velocity"]  # nominal folded velocity axis (masked)
    half_len0 = spec0.folded_half_len
    v_full = velocity_axis_full(spec0, v_max)

    mu_fold_full = fit0["model"](v_full, fit0["params"])
    mu_fold_full = np.clip(mu_fold_full, 0.0, np.inf)

    # ---- Half-ratio split from real raw scan ----
    r = _estimate_half_ratio(raw_counts, clip_start=clip_start)

    # ---- Mean raw model consistent with folding ----
    mu_raw0 = _construct_raw_spectrum(mu_fold_full, r=r)

    # Synthetic raw length will be exactly what the folding uses (2*folded_half_len)
    # This is the region that matters under your AND-mask logic.
    # Also apply clip_start to the synthetic raw by letting fold() do it (same as real data).

    # ---- CI plotting grid ----
    # CI band is computed on this fixed x-axis so all replicates are comparable.
    if ci_grid is None:
        ci_grid = v0.copy()

    centers_s = []
    fwhm_s = []
    params_s = []
    mu_ci_s = []
    v_max_s = []

    n_fail_fold = 0
    n_fail_fit = 0

    for _ in range(B):
        # a) systematic: sample v_max
        v_max_b = v_max
        if v_max_err is not None and v_max_err > 0:
            v_max_b = float(rng.normal(v_max, v_max_err))

        # b) statistical: Poisson synthetic raw counts
        raw_b = rng.poisson(mu_raw0).astype(float)

        # c) fold (includes split/shift instability)
        spec_b = Spectrum(raw_b)
        try:
            spec_b.fold(
                clip_start=clip_start,
                **fold_kwargs,
                freeze_split=freeze_split,
                freeze_shift=freeze_shift
            )
        except Exception:
            n_fail_fold += 1
            continue

        if spec_b.folded_counts is None or len(spec_b.folded_counts) < max(30, 6 * n_peaks):
            n_fail_fold += 1
            continue

        # d) fit (fast: skip covariance)
        try:
            fit_b = fit_spectrum(
                spec_b,
                n_peaks=n_peaks,
                v_max=v_max_b,
                v_max_err=None,
                loss=loss,
                compute_cov=False,
            )
        except Exception:
            n_fail_fit += 1
            continue

        centers_s.append(fit_b["centers"])
        fwhm_s.append(fit_b["fwhm"])
        params_s.append(fit_b["params"])
        mu_ci_s.append(fit_b["model"](ci_grid, fit_b["params"]))
        v_max_s.append(v_max_b)

    centers_s = np.asarray(centers_s)
    fwhm_s = np.asarray(fwhm_s)
    params_s = np.asarray(params_s)
    mu_ci_s = np.asarray(mu_ci_s)
    v_max_s = np.asarray(v_max_s)

    return {
        "fit0": fit0,
        "spec0": spec0,
        "half_ratio_r": r,
        "mu_fold_full": mu_fold_full,
        "mu_raw0": mu_raw0,
        "ci_grid": ci_grid,
        "centers_samples": centers_s,
        "fwhm_samples": fwhm_s,
        "params_samples": params_s,
        "mu_ci_samples": mu_ci_s,
        "v_max_samples": v_max_s,
        "n_fail_fold": n_fail_fold,
        "n_fail_fit": n_fail_fit,
    }
def envelope_band(mu_samples, keep=0.68):
    # mu_samples shape: (B, N)
    med = np.median(mu_samples, axis=0)
    # L2 distance to median curve as a simple ranking
    d = np.sum((mu_samples - med)**2, axis=1)
    k = max(1, int(np.floor(keep * len(d))))
    idx = np.argsort(d)[:k]
    lo = np.min(mu_samples[idx], axis=0)
    hi = np.max(mu_samples[idx], axis=0)
    return lo, hi

def summarize_bootstrap(samples: np.ndarray, level: float = 0.68, axis: int = 0):
    """
    Percentile CI summary for bootstrap samples.
    level=0.68 gives 16-84%, level=0.95 gives 2.5-97.5%.
    """
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
    *,
    title="Title",
    save_path="results/fit_plot.pdf",
    ci_grid=None,
    ci_lo=None,
    ci_hi=None,
    ci_label="68% CI (bootstrap)",
    ci_alpha=0.35,
):
    v = fit["velocity"]
    y = spectrum.folded_counts
    dy = spectrum.folded_sigma
    y_fit = fit["model"](v, fit["params"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={"height_ratios": [1.85, 1], "hspace": 0.1}
    )
    title_artist = ax1.set_title(title, pad=20)

    # --- data + fit ---
    ax1.errorbar(v, y, yerr=dy, fmt=".", ms=2, label="Data")
    ax1.plot(v, y_fit, lw=1.3, label="Fit", zorder=10)

    # --- optional CI ribbon (mean curve) ---
    if ci_grid is not None and ci_lo is not None and ci_hi is not None:
        ax1.fill_between(ci_grid, ci_lo, ci_hi, alpha=ci_alpha, label=ci_label, zorder=2)

    ax1.set_ylabel("Counts")
    ax1.grid(True)

    # --- residuals (use intuitive normalized residuals even if fit used Poisson deviance) ---
    ax2.axhline(0, color="gray", lw=1)
    ax2.plot(v, (y - y_fit) / dy, ".", ms=3)
    ax2.set_xlabel("Velocity (mm/s)")
    ax2.set_ylabel(r"Residuals ($\sigma$)")
    ax2.grid(True)

    # --- legend placed under title, like you already do ---
    fig.canvas.draw()
    title_box = title_artist.get_window_extent(fig.canvas.renderer)  # type: ignore
    title_y = title_box.y1 / fig.bbox.ymax
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, title_y - 0.03), ncol=3)

    plt.savefig(save_path, backend="pgf")
    plt.close()


if __name__ == "__main__":
    measurements = [
        {
            "absorber": "iron",
            "file": "data/Fe_1950V_2d.asc",
            "n_peaks": 6,
            "v_max": 6.0,
            "v_max_err": 0.3
        },
        {
            "absorber": "steel",
            "file": "data/stainless_1950V_9mms_2d.asc",
            "n_peaks": 1,
            "v_max": 9.0,
            "v_max_err": 0.45
        },
        {
            "absorber": "ferrocyanide",
            "file": "data/potassium_ferrocyanide_1950V_9mms_2d.asc",
            "n_peaks": 1,
            "v_max": 9.0,
            "v_max_err": 0.45
        },
        {
            "absorber": "sulphate",
            "file": "data/ferrous_sulphate_1950V_9mms_2d.asc",
            "n_peaks": 2,
            "v_max": 9.0,
            "v_max_err": 0.45
        },
    ]
    for meas in measurements:
        print(f"Processing {meas['absorber']}...")
        raw_counts = np.loadtxt(meas["file"])
        boot = parametric_bootstrap(
            raw_counts,
            n_peaks=meas["n_peaks"],
            v_max=meas["v_max"],
            v_max_err=meas["v_max_err"],
            clip_start=5,
            fold_kwargs={"trim": 10, "detrend_window": 101, "max_shift": 200, "split_search": 10},
            B=200,
            seed=1,
            loss="poisson",
        )
        # The fit for plotting should be the original fit returned by bootstrap
        fit0 = boot["fit0"]
        spec0 = boot["spec0"]

        # CI band for the mean curve
        v_star = boot["ci_grid"]
        mu_med, mu_lo, mu_hi = summarize_bootstrap(boot["mu_ci_samples"], level=0.68)
        ci_lo, ci_hi = envelope_band(boot["mu_ci_samples"], keep=0.68)

        plot_results(
            spec0,
            fit0,
            title=f"Lorentzian Fit with 68% bootstrap CI ({meas['absorber'].capitalize()})",
            save_path=f"results/fit_plot_{meas['absorber']}_bootstrapCI.pdf",
            ci_grid=v_star,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            ci_label="68% CI (bootstrap)"
        )
