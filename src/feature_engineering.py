"""Signal processing and window-level feature extraction for each modality.

Feature prefixes (one prefix = one feature group used in the ablation experiments):
    hr_   heart rate from the device pulse stream            -> HR-only BASELINE
    hrv_  heart-rate variability from R-peaks detected in the raw ECG
    ecg_  ECG signal-quality / motion-artifact indicators (raw 250 Hz ECG)
    temp_ skin temperature                                    (add-on; not in every session)
    resp_ chest-impedance respiration                         (add-on; not in every session)
    acc_  accelerometer                                       (descriptive only; 5 sessions)
    val_  validation-only columns (never used as model features)
"""
import numpy as np
import pandas as pd
from scipy import signal, stats

from .config import ECG_FS, RESP_FS, WINDOW_SECONDS

NAN = np.nan


def _slope_per_min(t_ms: np.ndarray, y: np.ndarray) -> float:
    """Least-squares trend of y over time, in units per minute."""
    if len(y) < 3 or np.ptp(t_ms) == 0:
        return NAN
    return float(np.polyfit((t_ms - t_ms[0]) / 60000.0, y, 1)[0])


def _band_power(f, pxx, lo, hi):
    m = (f >= lo) & (f < hi)
    return np.trapezoid(pxx[m], f[m]) if m.sum() > 1 else 0.0


# =========================================================================== HR (baseline)
HR_KEYS = ["hr_mean", "hr_median", "hr_min", "hr_max", "hr_std", "hr_range",
           "hr_p10", "hr_p90", "hr_slope"]


def hr_features(t_ms: np.ndarray, hr: np.ndarray) -> dict:
    """Summary statistics of the device-reported heart rate inside one window."""
    keep = (hr >= 30) & (hr <= 220)            # physiologically plausible device HR
    t_ms, hr = t_ms[keep], hr[keep].astype(float)
    if len(hr) < 10:
        return dict.fromkeys(HR_KEYS, NAN)
    p10, p90 = np.percentile(hr, [10, 90])
    return dict(hr_mean=hr.mean(), hr_median=np.median(hr), hr_min=hr.min(), hr_max=hr.max(),
                hr_std=hr.std(ddof=1), hr_range=np.ptp(hr), hr_p10=p10, hr_p90=p90,
                hr_slope=_slope_per_min(t_ms, hr))


# =========================================================================== R-peak detection
_ECG_BP = signal.butter(3, [5, 20], btype="bandpass", fs=ECG_FS, output="sos")


def detect_session_beats(t_ms: np.ndarray, ecg: np.ndarray, fs: float = ECG_FS) -> pd.DataFrame:
    """Detect R-peaks over a whole session (simplified Pan-Tompkins).

    band-pass 5-20 Hz -> derivative -> square -> 150 ms moving integration -> peaks above
    0.3 x the local (10 s block) 98th percentile, refractory 250 ms -> refine to max |filtered|.
    The ECG is split into contiguous segments at gaps > 1 s (contact loss); RR intervals are
    only formed between beats of the same segment.
    Returns DataFrame[timestamp, rr_ms, r_amp] with one row per beat that has a preceding beat.
    """
    out = []
    breaks = np.where(np.diff(t_ms) > 1000)[0] + 1
    for seg in np.split(np.arange(len(t_ms)), breaks):
        if len(seg) < 10 * fs:
            continue
        x = ecg[seg].astype(float)
        filtered = signal.sosfiltfilt(_ECG_BP, x - x.mean())
        win = int(0.15 * fs)
        energy = np.convolve(np.diff(filtered, prepend=filtered[0]) ** 2, np.ones(win) / win, "same")
        block = int(10 * fs)                       # local adaptive threshold per 10 s block
        n_blocks = int(np.ceil(len(energy) / block))
        thr_blocks = np.array([np.percentile(energy[i * block:(i + 1) * block], 98) for i in range(n_blocks)])
        threshold = 0.3 * np.repeat(thr_blocks, block)[:len(energy)]
        cand, _ = signal.find_peaks(energy, height=threshold, distance=int(0.25 * fs))
        half = int(0.075 * fs)
        peaks = np.unique([max(0, c - half) + np.argmax(np.abs(filtered[max(0, c - half): c + half]))
                           for c in cand])
        if len(peaks) < 2:
            continue
        beat_t = t_ms[seg][peaks]
        out.append(pd.DataFrame({"timestamp": beat_t[1:], "rr_ms": np.diff(beat_t).astype(float),
                                 "r_amp": np.abs(filtered[peaks[1:]])}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["timestamp", "rr_ms", "r_amp"])


# =========================================================================== HRV
HRV_KEYS = ["hrv_mean_rr", "hrv_median_rr", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_cv_rr"]


def clean_rr_series(rr_ms: np.ndarray) -> np.ndarray:
    """Session-level RR artifact detection. Returns boolean mask of valid beats.

    1. Physiological range 300-2000 ms (HR 30-200 bpm).
    2. Local-median filter: a beat deviating > 20 % from the median of its 5-beat
       neighbourhood is treated as an artifact/ectopic beat.
    Applied once per session (unsupervised, label-free) so the rolling median has context.
    """
    rr = rr_ms.astype(float)
    valid = (rr >= 300) & (rr <= 2000)
    local_median = pd.Series(np.where(valid, rr, np.nan)).rolling(5, center=True, min_periods=3).median()
    valid &= np.abs(rr - local_median.to_numpy()) <= 0.2 * local_median.to_numpy()
    return np.asarray(valid, dtype=bool)


def hrv_features(rr_ms: np.ndarray, valid: np.ndarray, prefix: str = "hrv_") -> dict:
    """Time-domain HRV for one window. RMSSD/pNN50 use only successive pairs where BOTH
    beats are valid, so a removed artifact does not create a fake large difference.
    Requires >= 20 valid beats and >= 15 valid successive pairs."""
    keys = [prefix + k[4:] for k in HRV_KEYS]
    out = dict.fromkeys(keys, NAN)
    rr = rr_ms[valid].astype(float)
    pair_ok = valid[1:] & valid[:-1]
    diffs = np.diff(rr_ms.astype(float))[pair_ok]
    if len(rr) < 20 or len(diffs) < 15:
        return out
    vals = [rr.mean(), np.median(rr), rr.std(ddof=1), np.sqrt(np.mean(diffs ** 2)),
            np.mean(np.abs(diffs) > 50) * 100, rr.std(ddof=1) / rr.mean()]
    return dict(zip(keys, vals))


# =========================================================================== ECG quality / motion
ECG_KEYS = ["ecg_kurtosis", "ecg_hf_noise_ratio", "ecg_lf_wander_ratio", "ecg_ramp_cv",
            "ecg_beat_artifact_frac"]


def ecg_quality_features(ecg: np.ndarray, fs: float = ECG_FS) -> dict:
    """Spectral signal-quality / motion-artifact indicators of the raw ECG in one window.

    hf_noise_ratio : power 30-45 + 55-100 Hz (muscle/EMG & motion noise) / power 1-30 Hz (QRS).
                     The 45-55 Hz mains band is excluded so the electrical environment
                     (indoor vs outdoor) cannot act as a hidden label.
    lf_wander_ratio: power 0-1 Hz (baseline wander from movement & breathing) / power 1-30 Hz.
    kurtosis       : clean ECG is 'spiky' (high kurtosis); noise lowers it.
    """
    out = dict.fromkeys(ECG_KEYS[:3], NAN)
    if len(ecg) < 0.9 * WINDOW_SECONDS * fs:     # require >= 90 % coverage
        return out
    x = ecg.astype(float) - np.mean(ecg)
    f, pxx = signal.welch(x, fs=fs, nperseg=1024)
    qrs = _band_power(f, pxx, 1, 30)
    out["ecg_hf_noise_ratio"] = (_band_power(f, pxx, 30, 45) + _band_power(f, pxx, 55, 100)) / qrs
    out["ecg_lf_wander_ratio"] = _band_power(f, pxx, 0, 1) / qrs
    out["ecg_kurtosis"] = stats.kurtosis(signal.sosfiltfilt(_ECG_BP, x))
    return out


# =========================================================================== Respiration (add-on)
RESP_KEYS = ["resp_rate", "resp_bbi_cv", "resp_amp_std", "resp_spectral_purity"]
# 0.1-1.0 Hz = 6-60 breaths/min (device reports up to 43 br/min during squash)
_RESP_BP = signal.butter(2, [0.1, 1.0], btype="bandpass", fs=RESP_FS, output="sos")


def resp_features(resp: np.ndarray, fs: float = RESP_FS) -> dict:
    """Breathing rate (peak counting — validated best against the device rate), breath-to-breath
    variability, amplitude and spectral purity (regularity / quality) for one window."""
    out = dict.fromkeys(RESP_KEYS, NAN)
    if len(resp) < 0.9 * WINDOW_SECONDS * fs:
        return out
    filtered = signal.sosfiltfilt(_RESP_BP, resp.astype(float) - np.mean(resp))
    f, pxx = signal.welch(filtered, fs=fs, nperseg=len(filtered))
    band = (f >= 0.1) & (f <= 1.0)
    f_dom = f[band][np.argmax(pxx[band])]
    out["resp_spectral_purity"] = (_band_power(f, pxx, f_dom - 0.05, f_dom + 0.05)
                                   / _band_power(f, pxx, 0.1, 1.0))
    out["resp_amp_std"] = filtered.std(ddof=1)
    peaks, _ = signal.find_peaks(filtered, distance=int(1.0 * fs), prominence=0.3 * filtered.std())
    if len(peaks) >= 4:
        bbi = np.diff(peaks) / fs                       # breath-to-breath intervals (s)
        out["resp_rate"] = 60 / bbi.mean()
        out["resp_bbi_cv"] = bbi.std(ddof=1) / bbi.mean()
    return out


# =========================================================================== Temperature (add-on)
TEMP_KEYS = ["temp_mean", "temp_min", "temp_max", "temp_std", "temp_delta", "temp_slope"]


def temp_features(t_ms: np.ndarray, temp: np.ndarray) -> dict:
    if len(temp) < 3:
        return dict.fromkeys(TEMP_KEYS, NAN)
    temp = temp.astype(float)
    return dict(temp_mean=temp.mean(), temp_min=temp.min(), temp_max=temp.max(),
                temp_std=temp.std(ddof=1), temp_delta=temp[-1] - temp[0],
                temp_slope=_slope_per_min(t_ms, temp))


# =========================================================================== Motion (descriptive)
ACC_KEYS = ["acc_mag_mean", "acc_mag_std", "acc_enmo", "acc_jerk", "acc_mag_p95"]


def acc_features(t_ms: np.ndarray, xyz: np.ndarray) -> dict:
    """Accelerometer features; units verified to be g in Phase 1 (median |a| ~ 1)."""
    if len(xyz) < 0.5 * WINDOW_SECONDS * 25:
        return dict.fromkeys(ACC_KEYS, NAN)
    mag = np.linalg.norm(xyz.astype(float), axis=1)
    dt_s = np.diff(t_ms) / 1000
    dt_s = np.where(dt_s > 0, dt_s, np.nan)
    return dict(acc_mag_mean=mag.mean(), acc_mag_std=mag.std(ddof=1),
                acc_enmo=np.mean(np.maximum(mag - 1.0, 0)),          # Euclidean norm minus one
                acc_jerk=np.nanmean(np.abs(np.diff(mag)) / dt_s),
                acc_mag_p95=np.percentile(mag, 95))


# Model feature groups. hrv_mean_rr / hrv_median_rr are computed (descriptive + validation)
# but NOT used as HRV model features: they equal 60000/HR (Spearman rho = -0.997 with hr_mean
# on the development set), so including them would let an "HR+HRV" model gain simply from a
# second copy of heart rate, not from heart-rate *variability*.
HRV_MODEL_KEYS = ["hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_cv_rr"]

FEATURE_GROUPS = {
    "HR": HR_KEYS,
    "HRV": HRV_MODEL_KEYS,
    "ECG": ECG_KEYS,
    "TEMP": TEMP_KEYS,
    "RESP": RESP_KEYS,
    "ACC": ACC_KEYS,
}
