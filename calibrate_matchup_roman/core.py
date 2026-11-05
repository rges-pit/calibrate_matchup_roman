"""
End-to-end calibration of a MATCHUP star list.

1. Pixel -> RA/Dec from the Gaia link stars (polynomial fit, default affine).
2. Find/download MAST Roman source catalogs in the same filter (optional).
3. Match Jay's stars to each MAST catalog after removing its bulk offset.
4. Zero point = robust median of (MAST PSF AB - Jay mbar) over the linear
   magnitude window (found automatically unless given).
5. Output = all MATCHUP columns plus xbar_asec, ybar_asec, mbar_abmag,
   xsig_mas, ysig_mas, msig_abmag, ra, dec, and mast_matched, wrapped in
   a `CalibrationResult`.
"""

import datetime
import re
from pathlib import Path

import numpy as np
from astropy.table import Table

from . import astrometry as ast
from . import io
from . import photometry as phot
from .result import CalibrationResult


def filter_from_filename(path):
    """
    Infer the filter from a MATCHUP filename like 'MATCHUP.XYMEEE.00.F213'.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    filt : str or None
    """
    m = re.search(r'(F\d{3})', Path(path).name.upper())
    filt = m.group(1) if m else None

    return filt


def catalog_level(path):
    """
    Level of a MAST catalog from its filename (sky-cell '_p_v...' -> 3, else 2).

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    level : int
    """
    level = 3 if re.search(r'_p_v\d+_', Path(path).name) else 2

    return level


def match_catalogs(paths, xi_j, eta_j, ra0, dec0, radius, min_matches=50, verbose=True):
    """
    Match Jay's stars to each MAST catalog.

    Parameters
    ----------
    paths : list of Path
        MAST catalogs.
    xi_j, eta_j : numpy arrays
        Jay star positions (arcsec, tangent plane about ra0, dec0). NaN = unusable.
    ra0, dec0 : float
        Tangent point (deg).
    radius : float
        Final match radius (arcsec).
    min_matches : int, optional
        Catalogs with fewer matches are skipped.
    verbose : bool, optional

    Returns
    -------
    matches : list of dict
        Per catalog: summary values plus per-pair arrays 'ij' (Jay row indices),
        'mast_ra', 'mast_dec', 'mag' (MAST psf AB), 'good' (psf_flags = 0 and
        finite mag), 'dxi', 'deta' (MAST - Jay after the bulk shift, mas).
    """
    use = np.where(np.isfinite(xi_j))[0]

    matches = []
    for p in paths:
        cat, sreg = io.read_mast_catalog(p)
        xi_m, eta_m = ast.radec_to_tan(cat['ra'], cat['dec'], ra0, dec0)

        # Match after removing the bulk offset.
        ij, im, off, n_coarse = ast.match_with_offset(xi_j[use], eta_j[use], xi_m, eta_m,
                                                      radius)
        if len(ij) < min_matches:
            if verbose:
                print(f'  skip {p.name}: {len(ij)} matches')
            continue

        # Residuals after the shift (mas).
        dxi = (xi_m[im] - xi_j[use][ij] - off[0]) * 1000.
        deta = (eta_m[im] - eta_j[use][ij] - off[1]) * 1000.

        mag = cat['psf_abmag'].to_numpy()[im]
        good = np.isfinite(mag) & (cat['psf_flags'].to_numpy()[im] == 0)
        s_region = ('POLYGON ICRS ' + ' '.join(f'{v:.7f}' for v in sreg.ravel())
                    if sreg is not None else '')

        matches.append({'catalog': p.name, 'level': catalog_level(p), 'n_src': int(len(cat)),
                        'n_coarse': n_coarse, 'n_match': int(len(ij)),
                        'dxi_mas': off[0] * 1000., 'deta_mas': off[1] * 1000.,
                        'sig_xi_mas': phot.robust_stats(dxi)[1],
                        'sig_eta_mas': phot.robust_stats(deta)[1], 's_region': s_region,
                        'ij': use[ij], 'mast_ra': cat['ra'].to_numpy()[im],
                        'mast_dec': cat['dec'].to_numpy()[im], 'mag': mag, 'good': good,
                        'dxi': dxi, 'deta': deta})
        if verbose:
            print(f'  {p.name}: {len(ij):,} matches, offset MAST-Jay = '
                  f'({off[0] * 1000:+.1f}, {off[1] * 1000:+.1f}) mas')

    return matches


def resolve_catalogs(filt, ra, dec, pos_ok, ra0, dec0, catalogs=None, data_dir='mast_catalogs',
                     download=True, proposal_id=None, level='auto', verbose=True):
    """
    Decide which MAST catalogs to use: given, downloaded, or found on disk.

    Parameters
    ----------
    filt : str
    ra, dec : numpy arrays
        Jay star positions (deg); NaN where unusable.
    pos_ok : numpy array of bool
    ra0, dec0 : float
        Tangent point (deg).
    catalogs, data_dir, download, proposal_id, level, verbose
        See `calibrate`.

    Returns
    -------
    paths : list of Path
    """
    data_dir = Path(data_dir)

    if catalogs is not None:
        paths = [Path(c) for c in catalogs]
    elif download:
        from . import mast

        # Cone covering Jay's field.
        xi, eta = ast.radec_to_tan(ra[pos_ok], dec[pos_ok], ra0, dec0)
        cra, cdec = ast.tan_to_radec(np.median(xi), np.median(eta), ra0, dec0)
        rad = float(np.max(np.hypot(xi - np.median(xi), eta - np.median(eta)))) / 3600. * 1.05
        products = mast.find_catalogs(float(cra), float(cdec), rad, filt, proposal_id, level)
        if verbose:
            print(f'MAST: {len(products)} {filt} catalogs within {rad * 60:.1f} arcmin')
        paths = mast.download_catalogs(products, data_dir, verbose=verbose)
    else:
        # Same level rule as the MAST query: 'auto' = level 3 if any, else level 2.
        found = sorted(data_dir.glob(f'*_{filt.lower()}_cat.parquet'))
        by_level = {lv: [p for p in found if catalog_level(p) == lv] for lv in (2, 3)}
        if level == 'auto':
            paths = by_level[3] or by_level[2]
        elif str(level) in ('2', '3'):
            paths = by_level[int(level)]
        else:
            paths = found

    if not paths:
        raise RuntimeError(f'No MAST {filt} catalogs available.')

    return paths


def calibrate(matchup, gaia_link, filt=None, catalogs=None, data_dir='mast_catalogs',
              proposal_id=None, level='auto', download=True, match_radius=0.1, zp_window=None,
              fit_order=1, err_mode='mean', min_matches=50, zp_tol=0.03, verbose=True):
    """
    Calibrate a MATCHUP star list: RA/Dec from Gaia, AB zero point from MAST.

    Parameters
    ----------
    matchup : str or Path
        MATCHUP.XYMEEE.* file (Jay's x1 frame).
    gaia_link : str or Path
        Gaia link file (*.xymrd) with Gaia stars in the same x1 frame.
    filt : str or None, optional
        Filter (e.g. 'F213'); inferred from the MATCHUP filename if None.
    catalogs : list of str/Path or None, optional
        MAST catalogs to use. If None, catalogs are found on MAST (if `download`)
        or globbed from `data_dir` ('*_<filt>_cat.parquet', filtered by `level`).
    data_dir : str or Path, optional
        Where MAST catalogs are downloaded / looked for.
    proposal_id : str or None, optional
        Restrict the MAST query to one proposal.
    level : {'auto', '2', '3', 'both'}, optional
        MAST catalog level (see `mast.find_catalogs`).
    download : bool, optional
        Query MAST and download missing catalogs (needs MAST_API_TOKEN).
    match_radius : float, optional
        Final match radius in arcsec (after removing the bulk offset).
    zp_window : tuple of float or None, optional
        (mbar_lo, mbar_hi) window for the zero point; automatic if None.
    fit_order : int, optional
        Polynomial order for the pixel -> sky fit.
    err_mode : {'mean', 'rms'}, optional
        'mean': per-star error = msig / sqrt(Ng) (msig taken as the per-
        measurement rms); 'rms': error = msig. ZPERR is added in quadrature.
    min_matches : int, optional
        Skip MAST catalogs with fewer matches.
    zp_tol : float, optional
        Flatness tolerance (mag) for the automatic window.
    verbose : bool, optional

    Returns
    -------
    result : CalibrationResult
        .table (calibrated star list), .catalogs (per-MAST-catalog summary),
        .pairs (matched pairs), .gaia (Gaia fit residuals); see CalibrationResult.
    """
    from . import __version__

    matchup, gaia_link = Path(matchup), Path(gaia_link)
    filt = (filt or filter_from_filename(matchup) or '').upper()
    if not filt:
        raise ValueError('Could not infer the filter; pass filt=.')

    # --- 1. Read inputs and quality masks.
    tab = io.read_matchup(matchup)
    pos_ok, phot_ok = io.matchup_quality(tab)
    if verbose:
        print(f'{matchup.name}: {len(tab):,} rows; {pos_ok.sum():,} with positions, '
              f'{phot_ok.sum():,} with photometry ({filt})')

    # --- 2. Pixel -> sky from Gaia; tangent point at the Gaia median position.
    gaia = io.read_gaia_link(gaia_link)
    ra0, dec0 = float(np.median(gaia['RA'])), float(np.median(gaia['DEC']))
    fit = ast.fit_pixel_to_sky(gaia['XC'], gaia['YC'], gaia['RA'], gaia['DEC'], ra0, dec0,
                               order=fit_order)
    if verbose:
        print(f"Gaia fit: {fit['n_used']}/{fit['n_ref']} stars, rms {fit['rms_x_mas']:.1f}/"
              f"{fit['rms_y_mas']:.1f} mas, scale {fit['scale_mas']:.3f} mas/pix")

    ra = np.full(len(tab), np.nan)
    dec = np.full(len(tab), np.nan)
    ra[pos_ok], dec[pos_ok] = ast.apply_pixel_to_sky(fit, tab['xbar'][pos_ok],
                                                     tab['ybar'][pos_ok])

    # --- 3. MAST catalogs.
    paths = resolve_catalogs(filt, ra, dec, pos_ok, ra0, dec0, catalogs, data_dir, download,
                             proposal_id, level, verbose)

    # --- 4. Match (only stars with positions and photometry).
    xi_j = np.full(len(tab), np.nan)
    eta_j = np.full(len(tab), np.nan)
    xi_j[phot_ok], eta_j[phot_ok] = ast.radec_to_tan(ra[phot_ok], dec[phot_ok], ra0, dec0)
    matches = match_catalogs(paths, xi_j, eta_j, ra0, dec0, match_radius, min_matches, verbose)
    if not matches:
        raise RuntimeError('No MAST catalog had enough matches.')

    # --- 5. Zero point from pooled pairs (psf_flags = 0).
    mbar = np.asarray(tab['mbar'], dtype=float)
    jm = np.concatenate([mbar[m['ij'][m['good']]] for m in matches])
    dm = np.concatenate([m['mag'][m['good']] - mbar[m['ij'][m['good']]] for m in matches])
    zp = phot.compute_zeropoint(jm, dm, window=zp_window, tol=zp_tol)

    # Per-catalog zero points in the same window.
    for m in matches:
        g = m['good']
        pz = phot.compute_zeropoint(mbar[m['ij'][g]], m['mag'][g] - mbar[m['ij'][g]],
                                    window=zp['window'])
        m['zp_cat'], m['zp_cat_n'], m['zp_cat_sig'] = pz['zp'], pz['n'], pz['zp_sigma']

    # Systematic term: N-weighted rms of per-catalog ZPs about the global ZP.
    zc = np.array([m['zp_cat'] for m in matches])
    wc = np.array([m['zp_cat_n'] for m in matches], dtype=float)
    ok_c = np.isfinite(zc) & (wc > 0)
    zp['zp_stat'] = zp['zp_err']
    zp['zp_sys'] = (float(np.sqrt(np.sum(wc[ok_c] * (zc[ok_c] - zp['zp']) ** 2) / wc[ok_c].sum()))
                    if ok_c.sum() >= 2 else 0.)
    zp['zp_err'] = float(np.hypot(zp['zp_stat'], zp['zp_sys']))
    if verbose:
        print(f"ZP = {zp['zp']:.4f} +/- {zp['zp_err']:.4f} (stat {zp['zp_stat']:.4f}, "
              f"catalog-to-catalog {zp['zp_sys']:.4f}); sigma {zp['zp_sigma']:.3f}, "
              f"N = {zp['n']:,}, mbar window {zp['window'][0]:.2f} to {zp['window'][1]:.2f}"
              f"{' (auto)' if zp['window_auto'] else ''}")

    # --- 6. Calibrated positions, magnitudes, and errors.
    # Arcsec offsets use one fixed plate scale and Sgr A* as the origin.
    # They do not use the scale or rotation of the Gaia polynomial.
    # xbar/ybar and xsig/ysig stay in pixels in the MATCHUP columns.
    x_sgra, y_sgra = ast.pixel_of_radec(fit, ast.SGRA_RA, ast.SGRA_DEC)
    scale_mas = ast.PLATE_SCALE_MAS
    xbar = np.asarray(tab['xbar'], dtype=float)
    ybar = np.asarray(tab['ybar'], dtype=float)
    xsig = np.asarray(tab['xsig'], dtype=float)
    ysig = np.asarray(tab['ysig'], dtype=float)
    msig = np.asarray(tab['msig'], dtype=float)
    ng = np.asarray(tab['Ng'], dtype=float)
    xbar_asec = np.where(pos_ok, (xbar - x_sgra) * scale_mas / 1000.0, np.nan)
    ybar_asec = np.where(pos_ok, (ybar - y_sgra) * scale_mas / 1000.0, np.nan)
    xsig_mas = np.where(pos_ok, xsig * scale_mas, np.nan)
    ysig_mas = np.where(pos_ok, ysig * scale_mas, np.nan)
    mbar_abmag = np.where(phot_ok, mbar + zp['zp'], np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        stat = msig / np.sqrt(ng) if err_mode == 'mean' else msig
    # Uncertainty of mbar_abmag. msig is the per-exposure rms.
    msig_abmag = np.where(phot_ok, np.sqrt(stat ** 2 + zp['zp_err'] ** 2), np.nan)

    matched = np.zeros(len(tab), dtype=bool)
    for m in matches:
        matched[m['ij']] = True

    # --- 7. Star list: MATCHUP columns + calibrated columns + metadata.
    out = tab.copy()
    out['xbar_asec'] = xbar_asec
    out['xbar_asec'].unit = 'arcsec'
    out['xbar_asec'].description = 'x offset from Sgr A* at 110 mas/pix'
    out['ybar_asec'] = ybar_asec
    out['ybar_asec'].unit = 'arcsec'
    out['ybar_asec'].description = 'y offset from Sgr A* at 110 mas/pix'
    out['mbar_abmag'] = mbar_abmag
    out['mbar_abmag'].unit = 'mag'
    out['mbar_abmag'].description = f'{filt} AB magnitude = mbar + ZP'
    out['xsig_mas'] = xsig_mas
    out['xsig_mas'].unit = 'mas'
    out['xsig_mas'].description = 'xsig * 110 mas/pix (rms of the exposures)'
    out['ysig_mas'] = ysig_mas
    out['ysig_mas'].unit = 'mas'
    out['ysig_mas'].description = 'ysig * 110 mas/pix (rms of the exposures)'
    out['msig_abmag'] = msig_abmag
    out['msig_abmag'].unit = 'mag'
    out['msig_abmag'].description = 'AB mag error; see ERRMODE'
    out['ra'] = ra
    out['ra'].unit = 'deg'
    out['ra'].description = 'ICRS RA (Gaia DR3 frame, at EPOCH)'
    out['dec'] = dec
    out['dec'].unit = 'deg'
    out['dec'].description = 'ICRS Dec (Gaia DR3 frame, at EPOCH)'
    out['mast_matched'] = matched

    out.meta.update({
        'FILTER': filt, 'MAGSYS': 'AB', 'ZP': zp['zp'], 'ZPERR': zp['zp_err'],
        'ZPSTAT': zp['zp_stat'], 'ZPSYS': zp['zp_sys'], 'ZPSIG': zp['zp_sigma'],
        'ZPN': zp['n'], 'ZPLO': zp['window'][0], 'ZPHI': zp['window'][1],
        'ZPAUTO': zp['window_auto'], 'ZPTOL': zp_tol, 'NMASTCAT': len(matches),
        'NMATCH': int(sum(m['n_match'] for m in matches)), 'MATCHRAD': match_radius,
        'ASTORDER': fit['order'], 'ASTNREF': fit['n_ref'], 'ASTNUSE': fit['n_used'],
        'ASTRMSX': fit['rms_x_mas'], 'ASTRMSY': fit['rms_y_mas'], 'PIXSCALE': fit['scale_mas'],
        'PLTSCALE': scale_mas, 'SGRAX': x_sgra, 'SGRAY': y_sgra,
        'SGRARA': ast.SGRA_RA, 'SGRADEC': ast.SGRA_DEC,
        'TANRA': ra0, 'TANDEC': dec0, 'EPOCH': float(np.median(gaia['RDATE'])),
        'ERRMODE': err_mode, 'MATCHUP': matchup.name, 'GAIALINK': gaia_link.name,
        'CALSOFT': f'calibrate_matchup_roman {__version__}',
        'CALDATE': datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds')})

    # --- 8. Per-catalog summary, matched pairs, and Gaia fit residuals.
    keys = ['catalog', 'level', 'n_src', 'n_match', 'dxi_mas', 'deta_mas', 'sig_xi_mas',
            'sig_eta_mas', 'zp_cat', 'zp_cat_sig', 'zp_cat_n', 's_region']
    cats = Table(rows=[[m[k] for k in keys] for m in matches], names=keys)

    pairs = Table()
    pairs['row'] = np.concatenate([m['ij'] for m in matches])
    pairs['cat_idx'] = np.concatenate([np.full(len(m['ij']), i) for i, m in enumerate(matches)])
    pairs['mast_ra'] = np.concatenate([m['mast_ra'] for m in matches])
    pairs['mast_dec'] = np.concatenate([m['mast_dec'] for m in matches])
    pairs['mast_mag'] = np.concatenate([m['mag'] for m in matches])
    pairs['mast_good'] = np.concatenate([m['good'] for m in matches])
    pairs['dxi_mas'] = np.concatenate([m['dxi'] for m in matches])
    pairs['deta_mas'] = np.concatenate([m['deta'] for m in matches])

    xi_g, eta_g = ast.radec_to_tan(gaia['RA'], gaia['DEC'], ra0, dec0)
    A = ast.design_matrix(gaia['XC'], gaia['YC'], fit['order'])
    gt = Table()
    gt['XC'] = gaia['XC'].to_numpy()
    gt['YC'] = gaia['YC'].to_numpy()
    gt['RA'] = gaia['RA'].to_numpy()
    gt['DEC'] = gaia['DEC'].to_numpy()
    gt['GMAG'] = gaia['GMAG'].to_numpy()
    gt['resid_xi_mas'] = (xi_g - A @ fit['cx']) * 1000.
    gt['resid_eta_mas'] = (eta_g - A @ fit['cy']) * 1000.
    gt['used'] = fit['keep']

    result = CalibrationResult(table=out, catalogs=cats, pairs=pairs, gaia=gt)

    return result
