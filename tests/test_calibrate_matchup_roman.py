"""
Tests for calibrate_matchup_roman on synthetic data with a known answer.

Run from the repository root:  python -m pytest tests -q
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from astropy.table import Table

import calibrate_matchup_roman as cmr
from calibrate_matchup_roman import astrometry as ast
from calibrate_matchup_roman import photometry as phot
from calibrate_matchup_roman import plots
from calibrate_matchup_roman.cli import main as cli_main
from calibrate_matchup_roman.io import read_matchup, matchup_quality

RA0, DEC0 = 266.417, -29.008
TRUE_ZP = 26.60
OFFSET_MAS = (15., -20.)


def true_pixel_to_tan(x, y, scale=0.1077, rot_deg=30.):
    """
    Known affine pixel -> tangent-plane mapping used to build the test data.

    Parameters
    ----------
    x, y : numpy arrays
        Pixel positions.
    scale : float, optional
        arcsec / pixel.
    rot_deg : float, optional
        Rotation (deg).

    Returns
    -------
    xi, eta : numpy arrays (arcsec)
    """
    t = np.radians(rot_deg)
    dx, dy = x - 2048., y - 2048.

    # Parity-flipped rotation, like Jay's frame (det < 0).
    xi = scale * (-np.cos(t) * dx + np.sin(t) * dy)
    eta = scale * (np.sin(t) * dx + np.cos(t) * dy)

    return xi, eta


def mast_minus_jay(mbar):
    """
    Synthetic (MAST AB - Jay mbar): flat between -9 and -6.5, curved outside.

    Parameters
    ----------
    mbar : numpy array

    Returns
    -------
    d : numpy array
    """
    d = np.full_like(mbar, TRUE_ZP)

    # Saturation (MAST too faint) and crowding (MAST too bright).
    d += np.where(mbar < -9., 0.5 * (-9. - mbar), 0.)
    d -= np.where(mbar > -6.5, 0.4 * (mbar + 6.5), 0.)

    return d


def write_mast_catalog(path, ra, dec, mab, rng, box):
    """
    Write a synthetic MAST *_cat.parquet with an s_region footprint.

    Parameters
    ----------
    path : Path
    ra, dec, mab : numpy arrays
        Positions (deg) and PSF AB magnitudes.
    rng : numpy.random.Generator
    box : tuple of float
        (ra_lo, ra_hi, dec_lo, dec_hi) footprint.

    Returns
    -------
    path : Path
    """
    flux = 10 ** (-0.4 * (mab - 31.4))
    tab = pa.table({'ra': ra, 'dec': dec, 'psf_flux': flux.astype('f4'),
                    'psf_flux_err': (0.01 * flux).astype('f4'),
                    'psf_flags': np.zeros(ra.size, dtype='i4')})
    a0, a1, d0, d1 = box
    sreg = f'POLYGON ICRS {a0} {d0} {a1} {d0} {a1} {d1} {a0} {d1}'
    tab = tab.replace_schema_metadata({b'roman.meta.wcsinfo.s_region': sreg.encode()})
    pq.write_table(tab, path)

    return path


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    """
    Write a synthetic MATCHUP file, Gaia link file, and two MAST catalogs.

    Returns
    -------
    paths : dict
        File paths plus truth arrays.
    """
    rng = np.random.default_rng(1)
    d = tmp_path_factory.mktemp('syn')
    n = 4000

    # Jay's stars.
    x = rng.uniform(50, 4046, n)
    y = rng.uniform(50, 4046, n)
    mbar = rng.uniform(-10.5, -5.0, n)
    msig = rng.uniform(0.01, 0.05, n)
    ng = np.full(n, 25)

    # Two placeholder rows and one unmeasured row.
    x[:2] = y[:2] = -2500.
    msig[2], ng[2] = 9.0, 0

    # Truth on the sky.
    xi, eta = true_pixel_to_tan(x, y)
    ra, dec = ast.tan_to_radec(xi, eta, RA0, DEC0)

    # MATCHUP file in Jay's format.
    names = ['xbar', 'ybar', 'mbar', 'xsig', 'ysig', 'msig', 'qbar', 'Nf', 'Ng', 'Nm', 'Nmin',
             'Nstar', 'pki', 'pkj', 'pkp', 'pkn', 'pku']
    with open(d / 'MATCHUP.XYMEEE.02.F146', 'w') as f:
        f.write('#....\n#    ' + '  '.join(names) + '\n#    (01)\n#....\n')
        for k in range(n):
            f.write(f'{x[k]:11.4f} {y[k]:11.4f} {mbar[k]:11.4f} 0.0100 0.0100 {msig[k]:11.4f} '
                    f'9.9990 28 {ng[k]} 28 25 N{k + 1:06d} 1 1 34 25 25\n')
        f.write('# INP0000001 "dummy"\n')

    # Gaia link file: 300 stars with exact positions (31 columns).
    with open(d / 'link.xymrd', 'w') as f:
        f.write('# header\n')
        for k in range(3, 303):
            cols = [x[k], y[k], -10.0, 0, 0, 17.0, 0, 0, 0, 0, 2026.75, ra[k], dec[k], ra[k],
                    dec[k], 0.1, 0.1, 17.0, 0.001, 18., 0.01, 16., 0.01, 0, 0.1, 0, 0.1, 0.5,
                    0.1, 4057481821012592640 + k, 1.0]
            f.write(' '.join(str(c) for c in cols) + '\n')

    # Two MAST catalogs (east/west halves): true positions + bulk offset + 5 mas noise.
    good = (x > -2000) & (ng > 0)
    mxi = xi[good] + OFFSET_MAS[0] / 1000. + rng.normal(0, 0.005, good.sum())
    meta = eta[good] + OFFSET_MAS[1] / 1000. + rng.normal(0, 0.005, good.sum())
    mra, mdec = ast.tan_to_radec(mxi, meta, RA0, DEC0)
    mab = mbar[good] + mast_minus_jay(mbar[good]) + rng.normal(0, 0.02, good.sum())
    cats = []
    for k, half in enumerate([mra < RA0, mra >= RA0]):
        box = (mra[half].min(), mra[half].max(), mdec.min(), mdec.max())
        cats.append(write_mast_catalog(
            d / f'r01054_p_v0200200100201{k}_267m27x30y49_f146_cat.parquet',
            mra[half], mdec[half], mab[half], rng, box))

    paths = {'matchup': d / 'MATCHUP.XYMEEE.02.F146', 'gaia': d / 'link.xymrd', 'cats': cats,
             'ra': ra, 'dec': dec, 'dir': d}

    return paths


@pytest.fixture(scope='module')
def result(synthetic):
    """Calibration of the synthetic data (no MAST access)."""
    res = cmr.calibrate(synthetic['matchup'], synthetic['gaia'], catalogs=synthetic['cats'],
                        download=False, verbose=False)

    return res


def test_tan_roundtrip():
    """Gnomonic projection and its inverse agree to < 1 micro-arcsec."""
    xi = np.array([-300., 0., 120.5])
    eta = np.array([250., 0., -80.25])
    ra, dec = ast.tan_to_radec(xi, eta, RA0, DEC0)
    xi2, eta2 = ast.radec_to_tan(ra, dec, RA0, DEC0)
    assert np.allclose(xi, xi2, atol=1e-6) and np.allclose(eta, eta2, atol=1e-6)


def test_mutual_match_is_one_to_one():
    """Two list-1 stars near one list-2 star: only the closer one matches."""
    i1, i2 = ast.mutual_match(np.array([0., 0.05]), np.array([0., 0.]), np.array([0.01]),
                              np.array([0.]), radius=0.1)
    assert list(i1) == [0] and list(i2) == [0]


def test_zp_window_excludes_curved_ends():
    """Automatic window lies inside the flat regime and recovers the ZP."""
    rng = np.random.default_rng(2)
    mbar = rng.uniform(-11, -5, 20000)
    dm = mast_minus_jay(mbar) + rng.normal(0, 0.03, mbar.size)
    zp = phot.compute_zeropoint(mbar, dm)
    assert zp['window'][0] >= -9.25 and zp['window'][1] <= -6.25
    assert abs(zp['zp'] - TRUE_ZP) < 0.003


def test_read_matchup_quality(synthetic):
    """All columns read; placeholder and unmeasured rows flagged."""
    tab = read_matchup(synthetic['matchup'])
    pos_ok, phot_ok = matchup_quality(tab)
    assert len(tab.colnames) == 17 and tab['Nstar'][0] == 'N000001'
    assert not pos_ok[:2].any() and pos_ok[2] and not phot_ok[2]


def test_calibration_values(synthetic, result):
    """Recovers positions (Gaia frame), bulk offsets, and ZP."""
    out = result.table

    # Positions on the Gaia frame (not shifted onto MAST), to < 0.1 mas.
    ok, phot = matchup_quality(out)
    xi, eta = ast.radec_to_tan(out['ra'][ok], out['dec'][ok], RA0, DEC0)
    t_xi, t_eta = ast.radec_to_tan(synthetic['ra'][ok], synthetic['dec'][ok], RA0, DEC0)
    assert np.max(np.hypot(xi - t_xi, eta - t_eta)) < 1e-4

    # Both catalogs used; bulk offsets and ZP recovered.
    assert len(result.catalogs) == 2
    assert np.allclose(result.catalogs['dxi_mas'], OFFSET_MAS[0], atol=1.0)
    assert np.allclose(result.catalogs['deta_mas'], OFFSET_MAS[1], atol=1.0)
    assert abs(result.zp - TRUE_ZP) < 0.005 and result.zp_err < 0.005

    # mbar_abmag = mbar + ZP where photometry is valid; NaN elsewhere.
    # Arcsec offsets are from Sgr A* at the fixed plate scale, not PIXSCALE.
    scale = ast.PLATE_SCALE_MAS
    assert result.meta['PLTSCALE'] == scale
    assert np.allclose(out['mbar_abmag'][phot], out['mbar'][phot] + result.zp)
    assert np.all(np.isnan(out['mbar_abmag'][~phot]))
    assert np.all(out['msig_abmag'][phot] > 0)
    assert np.allclose(out['xbar_asec'][ok],
                       (out['xbar'][ok] - result.meta['SGRAX']) * scale / 1000.)
    assert np.allclose(out['ybar_asec'][ok],
                       (out['ybar'][ok] - result.meta['SGRAY']) * scale / 1000.)
    assert np.allclose(out['ysig_mas'][ok], out['ysig'][ok] * scale)
    assert np.all(np.isnan(out['xbar_asec'][~ok])) and 'mag_cal' not in out.colnames


def test_fits_roundtrip(result, tmp_path):
    """write() -> read() preserves tables and calibration keywords."""
    path = result.write(tmp_path / 'cal.fits')
    back = cmr.read(path)

    for col in ['xbar', 'pku', 'Nstar', 'xbar_asec', 'ybar_asec', 'mbar_abmag',
                'xsig_mas', 'ysig_mas', 'msig_abmag', 'ra', 'dec', 'mast_matched']:
        assert col in back.table.colnames
    assert len(back.pairs) == len(result.pairs) and len(back.gaia) == len(result.gaia)
    assert abs(back.zp - result.zp) < 1e-12 and back.filter == 'F146'
    assert fits.getheader(path, 'STARLIST').comments['ZP'] == 'mbar_abmag = mbar + ZP'

    # Plain Table.read on the star list also works.
    assert len(Table.read(path, hdu='STARLIST')) == len(result.table)


def test_plots_and_report(result, tmp_path):
    """Every plot renders; the report is written with embedded figures."""
    paths = plots.save_all(result, tmp_path / 'figs')
    assert len(paths) == len(plots.ALL_PLOTS) and all(p.stat().st_size > 0 for p in paths)

    html = result.report(tmp_path / 'report.html')
    text = html.read_text()
    assert 'data:image/png;base64' in text and 'Zero point' in text


def test_cli(synthetic, tmp_path):
    """calibrate -> report -> plots through the command-line interface."""
    out = tmp_path / 'cli.fits'
    cli_main(['calibrate', str(synthetic['matchup']), '--gaia', str(synthetic['gaia']),
              '--catalogs', *[str(c) for c in synthetic['cats']], '--out', str(out),
              '--report'])
    assert out.exists() and out.with_suffix('.html').exists()

    cli_main(['report', str(out), '--out', str(tmp_path / 'r2.html')])
    cli_main(['plots', str(out), '--out-dir', str(tmp_path / 'figs')])
    assert (tmp_path / 'r2.html').exists() and len(list((tmp_path / 'figs').glob('*.png'))) == 8
