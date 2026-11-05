"""
Diagnostic plots for a CalibrationResult.

Every function takes a `CalibrationResult` (from `calibrate` or
`CalibrationResult.read`), returns the matplotlib Figure, and optionally saves
it with ``path=``.

    plot_zeropoint           MAST AB - Jay mbar vs mbar, ZP window, histogram
    plot_catalog_zeropoints  per-catalog ZP and bulk astrometric offsets
    plot_footprints          Jay field + MAST catalog footprints (colored by ZP)
    plot_astrometry_map      Jay - MAST offset vectors on the sky (per star, binned)
    plot_residuals           Jay - MAST (ΔRA, ΔDec, Δmag) vs RA, Dec, magnitude
    plot_gaia_fit            residuals of the pixel -> sky fit to Gaia
    plot_mag_errors          msig_abmag vs mbar_abmag and magnitude distributions
    plot_errors              position and magnitude error vs mbar_abmag
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from .io import matchup_quality, parse_s_region
from .photometry import binned_medians


def short_name(catalog):
    """
    Compact label for a MAST catalog filename.

    Parameters
    ----------
    catalog : str

    Returns
    -------
    label : str
        e.g. 'v02002001002018 267m27x30y49' for sky cells, the obs_id otherwise.
    """
    name = str(catalog).replace('_cat.parquet', '')
    parts = name.split('_')
    if len(parts) >= 4 and parts[1] == 'p':
        label = f'{parts[2]} {parts[3]}'
    else:
        label = name

    return label


def finish(fig, path):
    """
    Optionally save a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str or Path or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if path is not None:
        fig.savefig(Path(path), dpi=130, bbox_inches='tight')

    return fig


def running_stats(x, y, bins, min_n=15):
    """
    Running median and 16th/84th percentiles of y in bins of x.

    Parameters
    ----------
    x, y : numpy arrays
    bins : numpy array
        Bin edges.
    min_n : int, optional

    Returns
    -------
    centers, med, lo, hi : numpy arrays
    """
    centers = 0.5 * (bins[1:] + bins[:-1])
    med, lo, hi = (np.full(len(centers), np.nan) for _ in range(3))

    idx = np.digitize(x, bins) - 1
    for k in range(len(centers)):
        yk = y[(idx == k) & np.isfinite(y)]
        if yk.size >= min_n:
            lo[k], med[k], hi[k] = np.percentile(yk, [16, 50, 84])

    return centers, med, lo, hi


def plot_zeropoint(res, path=None, mag_range=1.5):
    """
    (MAST PSF AB - Jay mbar) vs mbar with the ZP window, plus a histogram.

    Parameters
    ----------
    res : CalibrationResult
    path : str or Path or None, optional
    mag_range : float, optional
        Half-range of the y axis around the ZP (mag).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    m = res.meta
    d = res.pair_arrays()
    jm, dm = d['mbar'], d['mast_mag'] - d['mbar']
    zp, lo_w, hi_w = m['ZP'], m['ZPLO'], m['ZPHI']

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={'width_ratios': [2.3, 1]})

    # Δ vs mbar: density, bin medians, window, ZP.
    ax = axes[0]
    ax.hexbin(jm, dm, gridsize=120, bins='log', cmap='Greys',
              extent=(np.nanpercentile(jm, 0.2), np.nanpercentile(jm, 99.8),
                      zp - mag_range, zp + mag_range))
    blo, bhi, bmed, _ = binned_medians(jm, dm, 0.25, 20)
    ax.plot(0.5 * (blo + bhi), bmed, 'C3.-', label='bin median')
    ax.axvspan(lo_w, hi_w, color='C0', alpha=0.12,
               label=f"ZP window{' (auto)' if m['ZPAUTO'] else ''}")
    ax.axhline(zp, color='C0', ls='--', label=f"ZP = {zp:.3f} ± {m['ZPERR']:.3f}")
    ax.set_ylim(zp - mag_range, zp + mag_range)
    ax.set_xlabel('Jay mbar (instrumental)')
    ax.set_ylabel('MAST psf AB − Jay mbar')
    ax.set_title(f"{m['FILTER']}: {len(jm):,} pairs (psf_flags = 0), {m['NMASTCAT']} catalogs")
    ax.legend(fontsize=8, loc='upper right')

    # Residuals in the window.
    ax = axes[1]
    inw = (jm >= lo_w) & (jm < hi_w)
    ax.hist(dm[inw] - zp, bins=np.linspace(-0.5, 0.5, 101), color='0.5')
    ax.axvline(0, color='C0', ls='--')
    ax.set_xlabel('(MAST − Jay) − ZP in window (mag)')
    ax.set_ylabel('N')
    ax.set_title(f"σ = {m['ZPSIG']:.3f}, N = {m['ZPN']:,}")

    fig.tight_layout()
    fig = finish(fig, path)

    return fig


def plot_catalog_zeropoints(res, path=None):
    """
    Per-catalog zero points and bulk astrometric offsets.

    Parameters
    ----------
    res : CalibrationResult
    path : str or Path or None, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    c, m = res.catalogs, res.meta
    x = np.arange(len(c))
    labels = [short_name(n) for n in c['catalog']]

    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.55 * len(c) + 3), 7.5), sharex=True)

    # ZP per catalog with error on the median, and the global ZP ± ZPERR.
    ax = axes[0]
    err = 1.2533 * np.asarray(c['zp_cat_sig']) / np.sqrt(np.maximum(c['zp_cat_n'], 1))
    ax.errorbar(x, c['zp_cat'], yerr=err, fmt='o', color='C3', capsize=3)
    ax.axhline(m['ZP'], color='C0', ls='--', label=f"pooled ZP = {m['ZP']:.3f}")
    ax.axhspan(m['ZP'] - m['ZPERR'], m['ZP'] + m['ZPERR'], color='C0', alpha=0.12,
               label=f"± ZPERR ({m['ZPERR']:.3f})")
    for xi, n in zip(x, c['zp_cat_n']):
        ax.annotate(f'{n}', (xi, ax.get_ylim()[0]), fontsize=7, ha='center', va='bottom',
                    color='0.4')
    ax.set_ylabel('ZP (MAST AB − Jay mbar)')
    ax.set_title('Per-catalog zero point (numbers: stars in ZP window)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bulk offsets (MAST − Jay) with residual scatter as error bars.
    ax = axes[1]
    ax.errorbar(x - 0.1, c['dxi_mas'], yerr=c['sig_xi_mas'], fmt='o', capsize=3,
                label='Δξ (ΔRA cos δ)')
    ax.errorbar(x + 0.1, c['deta_mas'], yerr=c['sig_eta_mas'], fmt='s', capsize=3, label='Δη (ΔDec)')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('bulk offset MAST − Jay (mas)\n(error bar = per-star σ)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig = finish(fig, path)

    return fig


def plot_footprints(res, path=None, n_show=20000, seed=0):
    """
    Jay's stars and the MAST catalog footprints, colored by per-catalog ZP offset.

    Parameters
    ----------
    res : CalibrationResult
    path : str or Path or None, optional
    n_show : int, optional
        Number of Jay stars drawn.
    seed : int, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    t, c, m = res.table, res.catalogs, res.meta
    ok, _phot_ok = matchup_quality(t)
    ra, dec = np.asarray(t['ra'])[ok], np.asarray(t['dec'])[ok]
    cosd = np.cos(np.radians(np.nanmedian(dec)))

    fig, ax = plt.subplots(figsize=(8, 7.5))

    # Random subset of Jay's stars.
    rng = np.random.default_rng(seed)
    idx = rng.choice(ra.size, size=min(n_show, ra.size), replace=False)
    ax.plot(ra[idx], dec[idx], ',', color='0.6', alpha=0.6, rasterized=True)

    # Footprints colored by ZP_cat − ZP.
    dz = np.asarray(c['zp_cat']) - m['ZP']
    vmax = max(0.02, np.nanmax(np.abs(dz)))
    cmap = plt.get_cmap('coolwarm')
    labels = {}
    for i, row in enumerate(c):
        if not row['s_region']:
            continue
        poly = parse_s_region(row['s_region'])
        col = cmap(0.5 + 0.5 * dz[i] / vmax)
        ax.add_patch(Polygon(poly, closed=True, fc=col, ec=col, alpha=0.25, lw=1.2))
        ax.add_patch(Polygon(poly, closed=True, fill=False, ec=col, lw=1.2))

        # Group labels of catalogs sharing a footprint (e.g. two visits, same sky cell).
        center = tuple(np.round(poly.mean(axis=0), 4))
        labels.setdefault(center, []).append(str(i))

    # View = Jay's field plus a margin; labels are clamped inside it.
    ra_lo, ra_hi = np.nanmin(ra) - 0.01, np.nanmax(ra) + 0.01
    dec_lo, dec_hi = np.nanmin(dec) - 0.01, np.nanmax(dec) + 0.01
    pad_ra, pad_dec = 0.08 * (ra_hi - ra_lo), 0.04 * (dec_hi - dec_lo)
    for (cra, cdec), idx in labels.items():
        lx = np.clip(cra, ra_lo + pad_ra, ra_hi - pad_ra)
        ly = np.clip(cdec, dec_lo + pad_dec, dec_hi - pad_dec)
        ax.text(lx, ly, ','.join(idx), fontsize=8, ha='center', va='center')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    fig.colorbar(sm, ax=ax, label='ZP(catalog) − ZP (mag)', shrink=0.8)

    ax.set_xlim(ra_hi, ra_lo)
    ax.set_ylim(dec_lo, dec_hi)
    ax.set_aspect(1. / cosd)
    ax.ticklabel_format(useOffset=False)
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.set_title(f"{m['FILTER']}: Jay stars (grey) and {len(c)} MAST catalogs (numbers = "
                 f"row in catalog table)", fontsize=10)
    fig = finish(fig, path)

    return fig


def binned_offsets(ra, dec, dra, ddec, cell_arcsec=15., min_n=15):
    """
    Median offsets in square sky cells.

    Parameters
    ----------
    ra, dec : numpy arrays (deg)
    dra, ddec : numpy arrays (mas)
    cell_arcsec : float, optional
    min_n : int, optional

    Returns
    -------
    bra, bdec, bdx, bdy : numpy arrays
    """
    cosd = np.cos(np.radians(np.median(dec)))
    cell = cell_arcsec / 3600.

    # Integer cell indices in (RA cos δ, Dec).
    ira = np.floor((ra - ra.min()) * cosd / cell).astype(int)
    idec = np.floor((dec - dec.min()) / cell).astype(int)
    keys = ira * 100000 + idec

    out = [[], [], [], []]
    for key in np.unique(keys):
        sel = keys == key
        if sel.sum() < min_n:
            continue
        for lst, val in zip(out, (np.median(ra[sel]), np.median(dec[sel]),
                                  np.median(dra[sel]), np.median(ddec[sel]))):
            lst.append(val)
    bra, bdec, bdx, bdy = (np.array(v) for v in out)

    return bra, bdec, bdx, bdy


def plot_astrometry_map(res, path=None, cell_arcsec=15., n_show=3000, seed=42):
    """
    Jay − MAST offset vectors on the sky (each catalog's bulk offset removed).

    Parameters
    ----------
    res : CalibrationResult
    path : str or Path or None, optional
    cell_arcsec : float, optional
        Cell size for the binned panel.
    n_show : int, optional
        Stars drawn in the per-star panel.
    seed : int, optional

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    Arrows point from the MAST position to Jay's. Per-catalog bulk offsets are
    shown in `plot_catalog_zeropoints`.
    """
    d = res.pair_arrays()
    ra, dec, dra, ddec = d['ra'], d['dec'], d['dra_mas'], d['ddec_mas']
    cosd = np.cos(np.radians(np.median(dec)))

    # 20 mas of offset -> 5 arcsec on the plot.
    scale = 20. / (5. / 3600.)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Per-star vectors for a random subset.
    ax = axes[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(ra.size, size=min(n_show, ra.size), replace=False)
    amp = np.hypot(dra[idx], ddec[idx])
    q = ax.quiver(ra[idx], dec[idx], dra[idx] / cosd, ddec[idx], np.clip(amp, 0, 50),
                  angles='xy', scale_units='xy', scale=scale, width=0.0018, cmap='viridis')
    fig.colorbar(q, ax=ax, label='|Jay − MAST| (mas)', shrink=0.75, pad=0.02)
    ax.quiverkey(q, 0.08, -0.12, 20, '20 mas', labelpos='E', coordinates='axes')
    ax.set_title(f'Per star: random {idx.size:,} of {ra.size:,} pairs', fontsize=10)

    # Binned medians, 4x longer arrows.
    ax = axes[1]
    bra, bdec, bdx, bdy = binned_offsets(ra, dec, dra, ddec, cell_arcsec)
    q2 = ax.quiver(bra, bdec, bdx / cosd, bdy, np.hypot(bdx, bdy), angles='xy',
                   scale_units='xy', scale=scale / 4., width=0.004, cmap='plasma')
    fig.colorbar(q2, ax=ax, label='|median offset| (mas)', shrink=0.75, pad=0.02)
    ax.quiverkey(q2, 0.08, -0.12, 5, '5 mas', labelpos='E', coordinates='axes')
    ax.set_title(f'Median per {cell_arcsec:.0f}″ cell (≥15 pairs), arrows 4× longer',
                 fontsize=10)

    for ax in axes:
        ax.set_xlim(ra.max() + 0.003, ra.min() - 0.003)
        ax.set_ylim(dec.min() - 0.003, dec.max() + 0.003)
        ax.set_aspect(1. / cosd)
        ax.ticklabel_format(useOffset=False)
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('Dec (deg)')

    fig.suptitle(f"{res.filter}: Jay − MAST astrometric residuals (per-catalog bulk offset "
                 f"removed; arrows point MAST → Jay)", y=1.0)
    fig = finish(fig, path)

    return fig


def plot_residuals(res, path=None, pos_lim=100., mag_lim=1.0):
    """
    3x3 grid: Jay − MAST (ΔRA cos δ, ΔDec, Δmag) vs RA, Dec, and mbar_abmag.

    Parameters
    ----------
    res : CalibrationResult
    path : str or Path or None, optional
    pos_lim : float, optional
        y-range half-width for positions (mas).
    mag_lim : float, optional
        y-range half-width for Δmag (mag).

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    Positions have each catalog's bulk offset removed; Δmag = mbar_abmag − MAST psf AB.
    """
    d = res.pair_arrays()
    xs = [(d['ra'], 'RA (deg)', True), (d['dec'], 'Dec (deg)', False),
          (d['mbar_abmag'], f'{res.filter} mbar_abmag (AB)', False)]
    ys = [(d['dra_mas'], 'Jay − MAST ΔRA cos δ (mas)', (-pos_lim, pos_lim)),
          (d['ddec_mas'], 'Jay − MAST ΔDec (mas)', (-pos_lim, pos_lim)),
          (d['dmag'], 'Jay − MAST Δmag', (-mag_lim, mag_lim))]

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharey='row')

    for r, (y, ylab, ylim) in enumerate(ys):
        for c, (x, xlab, flip) in enumerate(xs):
            ax = axes[r, c]
            ax.plot(x, y, ',', color='k', alpha=0.3, rasterized=True)

            # Running median and 16–84% band.
            bins = np.linspace(np.nanpercentile(x, 0.2), np.nanpercentile(x, 99.8), 31)
            cen, med, lo, hi = running_stats(x, y, bins)
            ax.plot(cen, med, 'C3-', lw=2)
            ax.plot(cen, lo, 'C3--', lw=1)
            ax.plot(cen, hi, 'C3--', lw=1)
            ax.axhline(0, color='C0', lw=0.8)

            ax.set_ylim(*ylim)
            ax.set_xlim(bins[0], bins[-1])
            if flip:
                ax.invert_xaxis()
            if c < 2:
                ax.ticklabel_format(axis='x', useOffset=False)
            if r == 2:
                ax.set_xlabel(xlab)
            if c == 0:
                ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)

    fig.suptitle(f'{res.filter}: Jay − MAST residuals for {d["ra"].size:,} pairs '
                 f'(psf_flags = 0; red = running median and 16–84%)', y=0.995)
    fig.tight_layout()
    fig = finish(fig, path)

    return fig


def plot_gaia_fit(res, path=None):
    """
    Residuals of the pixel -> sky fit to the Gaia link stars.

    Parameters
    ----------
    res : CalibrationResult
    path : str or Path or None, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    g, m = res.gaia, res.meta
    k = np.asarray(g['used'], dtype=bool)
    rx, ry = np.asarray(g['resid_xi_mas']), np.asarray(g['resid_eta_mas'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Residual vectors in the pixel frame.
    ax = axes[0]
    q = ax.quiver(g['XC'][k], g['YC'][k], rx[k], ry[k], angles='xy', scale_units='xy',
                  scale=0.05, width=0.003)
    ax.quiverkey(q, 0.75, 1.04, 20, '20 mas', labelpos='E')
    ax.plot(g['XC'][~k], g['YC'][~k], 'rx', label='clipped')
    ax.set_aspect('equal')
    ax.set_xlabel('x1 (pix)')
    ax.set_ylabel('y1 (pix)')
    ax.legend(fontsize=8, loc='lower right')

    # Residuals vs Gaia G.
    ax = axes[1]
    ax.plot(g['GMAG'][k], rx[k], '.', label='Δξ')
    ax.plot(g['GMAG'][k], ry[k], '.', label='Δη')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Gaia G')
    ax.set_ylabel('Gaia − fit (mas)')
    ax.legend(fontsize=8)

    # Histograms.
    ax = axes[2]
    bins = np.linspace(-60, 60, 41)
    ax.hist(rx[k], bins=bins, histtype='step', lw=1.5, label='Δξ')
    ax.hist(ry[k], bins=bins, histtype='step', lw=1.5, label='Δη')
    ax.set_xlabel('Gaia − fit (mas)')
    ax.legend(fontsize=8)

    fig.suptitle(f"Pixel → sky fit (order {m['ASTORDER']}): {m['ASTNUSE']}/{m['ASTNREF']} "
                 f"Gaia stars, rms {m['ASTRMSX']:.1f} / {m['ASTRMSY']:.1f} mas, "
                 f"scale {m['PIXSCALE']:.3f} mas/pix")
    fig.tight_layout()
    fig = finish(fig, path)

    return fig


def plot_mag_errors(res, path=None):
    """
    msig_abmag vs mbar_abmag, and mbar_abmag distributions (all vs MAST-matched).

    Parameters
    ----------
    res : CalibrationResult
    path : str or Path or None, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    t, m = res.table, res.meta
    _pos_ok, ok = matchup_quality(t)
    mag = np.asarray(t['mbar_abmag'], dtype=float)[ok]
    err = np.asarray(t['msig_abmag'], dtype=float)[ok]
    matched = np.asarray(t['mast_matched'], dtype=bool)[ok]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Error vs magnitude (log y), with the ZP-error floor.
    ax = axes[0]
    ax.hexbin(mag, err, gridsize=120, bins='log', yscale='log', cmap='Greys')
    ax.axhline(m['ZPERR'], color='C0', ls='--', label=f"ZPERR = {m['ZPERR']:.4f}")
    ax.set_xlabel(f"{m['FILTER']} mbar_abmag (AB)")
    ax.set_ylabel('msig_abmag')
    ax.set_title(f"ERRMODE = {m['ERRMODE']}")
    ax.legend(fontsize=8)

    # Magnitude distributions.
    ax = axes[1]
    bins = np.arange(np.floor(np.nanmin(mag)), np.ceil(np.nanmax(mag)) + 0.25, 0.25)
    ax.hist(mag, bins=bins, histtype='step', lw=1.5, label=f'with photometry ({mag.size:,})')
    ax.hist(mag[matched], bins=bins, histtype='step', lw=1.5,
            label=f'MAST-matched ({matched.sum():,})')
    ax.axvspan(m['ZPLO'] + m['ZP'], m['ZPHI'] + m['ZP'], color='C0', alpha=0.12,
               label='ZP window')
    ax.set_yscale('log')
    ax.set_xlabel(f"{m['FILTER']} mbar_abmag (AB)")
    ax.set_ylabel('N per 0.25 mag')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig = finish(fig, path)

    return fig


def plot_errors(res, path=None):
    """
    Position and magnitude errors versus calibrated magnitude.

    Parameters
    ----------
    res : CalibrationResult
    path : str or Path or None, optional

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    The position error is the uncertainty of the mean position,
    ``hypot(xsig_mas, ysig_mas) / sqrt(Ng)``. ``xsig_mas`` and
    ``ysig_mas`` are the rms of the exposures. The magnitude error is
    ``msig_abmag``.
    """
    t, m = res.table, res.meta
    _pos_ok, ok = matchup_quality(t)
    mag = np.asarray(t['mbar_abmag'], dtype=float)[ok]
    mag_err = np.asarray(t['msig_abmag'], dtype=float)[ok]
    xsig = np.asarray(t['xsig_mas'], dtype=float)[ok]
    ysig = np.asarray(t['ysig_mas'], dtype=float)[ok]
    ng = np.asarray(t['Ng'], dtype=float)[ok]

    # xsig_mas, ysig_mas are the rms of the exposures. The catalog
    # position is their mean, so divide by sqrt(Ng).
    pos = np.hypot(xsig, ysig) / np.sqrt(np.maximum(ng, 1.0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.arange(np.floor(np.nanmin(mag)), np.ceil(np.nanmax(mag)) + 0.5, 0.5)

    # Position error. The line is the median in 0.5 mag bins.
    ax = axes[0]
    ax.hexbin(mag, pos, gridsize=120, bins='log', yscale='log', cmap='Greys')
    cx, med, _lo, _hi = running_stats(mag, pos, bins)
    finite = np.isfinite(med)
    ax.plot(cx[finite], med[finite], color='C0', lw=1.5, label='median')
    ax.set_xlabel(f"{m['FILTER']} mbar_abmag (AB)")
    ax.set_ylabel('position error (mas)')
    ax.set_title(r'hypot(xsig, ysig) / $\sqrt{N_g}$')
    ax.legend(fontsize=8)

    # Magnitude error, with the zero-point contribution marked.
    ax = axes[1]
    ax.hexbin(mag, mag_err, gridsize=120, bins='log', yscale='log', cmap='Greys')
    cx, med, _lo, _hi = running_stats(mag, mag_err, bins)
    finite = np.isfinite(med)
    ax.plot(cx[finite], med[finite], color='C0', lw=1.5, label='median')
    ax.axhline(m['ZPERR'], color='C1', ls='--', label=f"ZPERR = {m['ZPERR']:.4f}")
    ax.set_xlabel(f"{m['FILTER']} mbar_abmag (AB)")
    ax.set_ylabel('msig_abmag')
    ax.set_title(f"ERRMODE = {m['ERRMODE']}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig = finish(fig, path)

    return fig


ALL_PLOTS = [('zeropoint', plot_zeropoint), ('catalog_zeropoints', plot_catalog_zeropoints),
             ('footprints', plot_footprints), ('astrometry_map', plot_astrometry_map),
             ('residuals', plot_residuals), ('gaia_fit', plot_gaia_fit),
             ('mag_errors', plot_mag_errors), ('errors', plot_errors)]


def save_all(res, out_dir, prefix=None):
    """
    Save every diagnostic plot as PNG.

    Parameters
    ----------
    res : CalibrationResult
    out_dir : str or Path
    prefix : str or None, optional
        Filename prefix (default: the filter name).

    Returns
    -------
    paths : list of Path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or res.filter

    paths = []
    for name, func in ALL_PLOTS:
        p = out_dir / f'{prefix}_{name}.png'
        plt.close(func(res, path=p))
        paths.append(p)

    return paths
