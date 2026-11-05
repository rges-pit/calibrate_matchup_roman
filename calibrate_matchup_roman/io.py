"""
Readers for Jay's MATCHUP / Gaia-link files and MAST Roman source catalogs.
"""

import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.table import Table

# Gaia link file columns (one row per Gaia star found in Jay's data).
GAIA_COLS = ['XC', 'YC', 'MAG_JAY', 'XFRAME', 'YFRAME', 'GMAG_', 'eXFRAME', 'eYFRAME',
             'XFRAMExPI', 'YFRAMExPI', 'RDATE', 'RA', 'DEC', 'RA2016', 'DEC2016', 'eRA',
             'eDEC', 'GMAG', 'eGMAG', 'BMAG', 'eBMAG', 'RMAG', 'eRMAG', 'muRA', 'emuRA',
             'muDEC', 'emuDEC', 'PI', 'ePI', 'GAIA_ID', 'RUWE']

# MATCHUP columns, used if the header line can't be parsed.
MATCHUP_COLS = ['xbar', 'ybar', 'mbar', 'xsig', 'ysig', 'msig', 'qbar', 'Nf', 'Ng', 'Nm',
                'Nmin', 'Nstar', 'pki', 'pkj', 'pkp', 'pkn', 'pku']

# Placeholder position for stars with no measurements.
PLACEHOLDER_XY = -2500.


def matchup_column_names(path):
    """
    Parse column names from a MATCHUP header ('#    xbar  ybar ...').

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    names : list of str
    """
    names = MATCHUP_COLS
    with open(path) as f:
        for line in f:
            if not line.startswith('#'):
                break
            toks = line.lstrip('#').split()
            if toks[:2] == ['xbar', 'ybar']:
                names = toks
                break

    return names


def read_matchup(path):
    """
    Read a MATCHUP.XYMEEE file into an astropy Table (all columns, all rows).

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    tab : astropy.table.Table
        All MATCHUP columns, with 'Nstar' as a string column.
    """
    names = matchup_column_names(path)

    # Whitespace-separated; trailing '#' footer lines are comments.
    df = pd.read_csv(path, comment='#', sep=r'\s+', header=None, names=names,
                     dtype={'Nstar': str})
    tab = Table.from_pandas(df)

    return tab


def matchup_quality(tab):
    """
    Masks for usable positions and photometry in a MATCHUP table.

    Parameters
    ----------
    tab : astropy.table.Table
        Output of `read_matchup`.

    Returns
    -------
    pos_ok : numpy array of bool
        Position is not the (-2500, -2500) placeholder.
    phot_ok : numpy array of bool
        At least one good measurement (Ng > 0) and 0 < msig < 9.

    Notes
    -----
    Rows with msig = 9 are unmeasured; rows with msig = 0 have mbar ~ 0
    (no flux) in the files inspected so far.
    """
    x = np.asarray(tab['xbar'], dtype=float)
    y = np.asarray(tab['ybar'], dtype=float)
    pos_ok = (x > PLACEHOLDER_XY + 1) & (y > PLACEHOLDER_XY + 1)

    ng = np.asarray(tab['Ng'], dtype=float)
    msig = np.asarray(tab['msig'], dtype=float)
    phot_ok = pos_ok & (ng > 0) & (msig > 0) & (msig < 9)

    return pos_ok, phot_ok


def read_gaia_link(path):
    """
    Read Jay's Gaia link file (*.xymrd).

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    gaia : pandas.DataFrame
        One row per unique Gaia star; positions XC, YC are in Jay's x1 frame,
        RA, DEC at epoch RDATE.
    """
    gaia = pd.read_csv(path, comment='#', sep=r'\s+', header=None, names=GAIA_COLS)
    gaia = gaia.drop_duplicates('GAIA_ID').reset_index(drop=True)

    return gaia


def parse_s_region(s_region):
    """
    Parse an STC-S 'POLYGON ICRS ra1 dec1 ...' string.

    Parameters
    ----------
    s_region : str

    Returns
    -------
    radec : numpy array, shape = [N_vertices, 2]
        Vertex (RA, Dec) in degrees.
    """
    nums = [float(t) for t in s_region.split() if re.match(r'^-?\d+(\.\d+)?$', t)]
    radec = np.array(nums).reshape(-1, 2)

    return radec


def read_mast_catalog(path):
    """
    Read a MAST Roman source catalog (*_cat.parquet) and its footprint.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    cat : pandas.DataFrame
        ra, dec, psf_flux, psf_flux_err, psf_flags, plus psf_abmag and
        psf_abmag_err (from nJy).
    s_region : numpy array or None
        Footprint vertices (RA, Dec) in degrees, if present in the metadata.
    """
    cols = ['ra', 'dec', 'psf_flux', 'psf_flux_err', 'psf_flags']
    pf = pq.ParquetFile(path)
    cat = pf.read(columns=cols).to_pandas()

    # nJy -> AB mag: m = -2.5 log10(f / 3631 Jy) = -2.5 log10(f_nJy) + 31.4.
    flux = cat['psf_flux'].to_numpy(dtype=float)
    ferr = cat['psf_flux_err'].to_numpy(dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        cat['psf_abmag'] = -2.5 * np.log10(flux) + 31.4
        cat['psf_abmag_err'] = 2.5 / np.log(10.) * ferr / flux

    # Footprint from the catalog metadata (both L2 and L3 catalogs carry it).
    md = pf.metadata.metadata or {}
    sreg = md.get(b'roman.meta.wcsinfo.s_region')
    s_region = parse_s_region(sreg.decode()) if sreg else None

    return cat, s_region
