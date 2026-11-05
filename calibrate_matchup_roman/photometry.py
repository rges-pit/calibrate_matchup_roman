"""
Zero-point estimation: automatic choice of the linear magnitude window and a
robust zero point within it.
"""

import numpy as np


def robust_stats(x):
    """
    Median and normalized-MAD sigma of finite values.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    med, sig : float
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    med = float(np.median(x))
    sig = float(1.4826 * np.median(np.abs(x - med)))

    return med, sig


def binned_medians(mag, dmag, bin_width=0.25, min_n=20):
    """
    Median of dmag in magnitude bins.

    Parameters
    ----------
    mag, dmag : numpy arrays
    bin_width : float, optional
    min_n : int, optional
        Bins with fewer stars are dropped.

    Returns
    -------
    lo, hi, med, n : numpy arrays
        Bin edges, medians, and counts for bins with >= min_n stars.
    """
    edges = np.arange(np.floor(mag.min()), np.ceil(mag.max()) + bin_width, bin_width)
    idx = np.digitize(mag, edges) - 1

    lo, hi, med, n = [], [], [], []
    for k in range(len(edges) - 1):
        sel = idx == k
        if sel.sum() >= min_n:
            lo.append(edges[k])
            hi.append(edges[k + 1])
            med.append(np.median(dmag[sel]))
            n.append(sel.sum())

    return np.array(lo), np.array(hi), np.array(med), np.array(n)


def find_zp_window(mag, dmag, bin_width=0.25, min_n=20, tol=0.03):
    """
    Find the magnitude window where dmag is flat (the linear regime).

    Searches all runs of consecutive magnitude bins and keeps the run with the
    most stars whose bin medians all lie within +/- tol of the run's median.
    This excludes the bright end (saturation) and faint end (crowding) where
    pipeline PSF photometry departs from Jay's.

    Parameters
    ----------
    mag : numpy array
        Jay instrumental magnitudes of matched stars.
    dmag : numpy array
        MAST AB mag - Jay instrumental mag for the same stars.
    bin_width : float, optional
        Bin width (mag).
    min_n : int, optional
        Minimum stars per bin.
    tol : float, optional
        Allowed deviation of bin medians from the window median (mag).

    Returns
    -------
    window : tuple of float
        (mag_lo, mag_hi) in Jay instrumental magnitudes.
    """
    lo, hi, med, n = binned_medians(mag, dmag, bin_width, min_n)
    if len(med) == 0:
        raise ValueError('Too few matched stars to determine a zero-point window.')

    # Exhaustive search over runs of consecutive bins.
    best, best_n = (0, 0), -1
    for i in range(len(med)):
        for j in range(i, len(med)):
            # Runs must be contiguous in magnitude (no dropped bins inside).
            if j > i and not np.allclose(lo[i + 1:j + 1], hi[i:j]):
                break
            run = med[i:j + 1]
            if np.max(np.abs(run - np.median(run))) <= tol and n[i:j + 1].sum() > best_n:
                best, best_n = (i, j), n[i:j + 1].sum()

    window = (float(lo[best[0]]), float(hi[best[1]]))

    return window


def compute_zeropoint(mag, dmag, window=None, clip=3.0, n_iter=5, **window_kwargs):
    """
    Robust zero point (MAST AB - Jay instrumental) within a magnitude window.

    Parameters
    ----------
    mag, dmag : numpy arrays
        Jay instrumental magnitudes and (MAST AB - Jay) for matched stars.
    window : tuple of float or None, optional
        (mag_lo, mag_hi). If None, determined with `find_zp_window`.
    clip : float, optional
        Sigma-clipping threshold.
    n_iter : int, optional
        Clipping iterations.
    **window_kwargs
        Passed to `find_zp_window`.

    Returns
    -------
    zp : dict
        'zp' (median), 'zp_err' (error on the median), 'zp_sigma' (robust
        per-star scatter), 'n' (stars used), 'window', 'window_auto'.
    """
    good = np.isfinite(mag) & np.isfinite(dmag)
    mag, dmag = mag[good], dmag[good]

    window_auto = window is None
    if window_auto:
        window = find_zp_window(mag, dmag, **window_kwargs)

    # Stars in the window, then iterative sigma clipping about the median.
    d = dmag[(mag >= window[0]) & (mag < window[1])]
    keep = np.ones(d.size, dtype=bool)
    for _ in range(n_iter):
        med, sig = robust_stats(d[keep])
        new_keep = np.abs(d - med) < clip * sig
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    med, sig = robust_stats(d[keep])

    # Error on the median ~ 1.253 sigma / sqrt(N).
    n = int(keep.sum())
    zp = {'zp': med, 'zp_err': float(1.2533 * sig / np.sqrt(max(n, 1))), 'zp_sigma': sig,
          'n': n, 'window': (float(window[0]), float(window[1])), 'window_auto': window_auto}

    return zp
