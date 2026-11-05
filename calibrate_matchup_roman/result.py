"""
CalibrationResult: container for a calibrated MATCHUP star list, with FITS
I/O, convenience accessors, plotting shortcuts, and report generation.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from .io import matchup_quality

# FITS header comments for the calibration keywords in the STARLIST HDU.
HEADER_COMMENTS = {
    'FILTER': 'Roman WFI filter',
    'MAGSYS': 'Magnitude system of mbar_abmag',
    'ZP': 'mbar_abmag = mbar + ZP',
    'ZPERR': 'ZP error = ZPSTAT (+) ZPSYS',
    'ZPSTAT': 'Statistical: 1.253 sigma / sqrt(N)',
    'ZPSYS': 'N-weighted rms of per-catalog ZPs',
    'ZPSIG': 'Robust per-star scatter in ZP window',
    'ZPN': 'Stars used for ZP (after clipping)',
    'ZPLO': 'ZP window: min Jay mbar',
    'ZPHI': 'ZP window: max Jay mbar',
    'ZPAUTO': 'ZP window found automatically',
    'ZPTOL': 'Flatness tolerance for auto window (mag)',
    'NMASTCAT': 'MAST catalogs with enough matches',
    'NMATCH': 'Matched pairs (all catalogs)',
    'MATCHRAD': 'Match radius after bulk shift (arcsec)',
    'ASTORDER': 'Polynomial order, pixel -> sky fit',
    'ASTNREF': 'Gaia stars in link file',
    'ASTNUSE': 'Gaia stars used after clipping',
    'ASTRMSX': 'Gaia fit rms in xi (mas)',
    'ASTRMSY': 'Gaia fit rms in eta (mas)',
    'PIXSCALE': 'Fitted pixel scale (mas/pix)',
    'PLTSCALE': 'Fixed plate scale for xbar_asec (mas/pix)',
    'SGRAX': 'Sgr A* x pixel in the MATCHUP frame',
    'SGRAY': 'Sgr A* y pixel in the MATCHUP frame',
    'SGRARA': 'Sgr A* RA (deg)',
    'SGRADEC': 'Sgr A* Dec (deg)',
    'TANRA': 'Tangent point RA (deg)',
    'TANDEC': 'Tangent point Dec (deg)',
    'EPOCH': 'Epoch of Gaia positions (yr)',
    'ERRMODE': 'msig_abmag: msig/sqrt(Ng) or msig, (+) ZPERR',
    'MATCHUP': 'Input MATCHUP file',
    'GAIALINK': 'Input Gaia link file',
    'CALSOFT': 'Software',
    'CALDATE': 'Creation time (UTC)',
}

HDU_NAMES = ('STARLIST', 'MASTCATS', 'PAIRS', 'GAIAFIT')


@dataclass
class CalibrationResult:
    """
    A calibrated MATCHUP star list and everything needed to check it.

    Attributes
    ----------
    table : astropy.table.Table
        Calibrated star list: all MATCHUP columns plus xbar_asec, ybar_asec,
        mbar_abmag, xsig_mas, ysig_mas, msig_abmag, ra, dec, mast_matched.
        Calibration keywords in
        ``table.meta`` (ZP, ZPERR, FILTER, ...).
    catalogs : astropy.table.Table
        One row per MAST catalog used: matches, bulk offset (MAST - Jay),
        residual scatter, per-catalog zero point, footprint (s_region).
    pairs : astropy.table.Table
        Matched pairs: 'row' (index into `table`), 'cat_idx' (index into
        `catalogs`), MAST ra/dec/mag, 'mast_good' (psf_flags = 0), and
        astrometric residuals 'dxi_mas', 'deta_mas' (MAST - Jay, bulk removed).
    gaia : astropy.table.Table
        Gaia link stars with residuals of the pixel -> sky fit.
    """

    table: Table
    catalogs: Table
    pairs: Table
    gaia: Table

    # --- Accessors ---------------------------------------------------------

    @property
    def meta(self):
        """Calibration keywords (dict-like)."""
        return self.table.meta

    @property
    def filter(self):
        """Filter name."""
        return self.meta['FILTER']

    @property
    def zp(self):
        """Zero point: mbar_abmag = mbar + zp."""
        return float(self.meta['ZP'])

    @property
    def zp_err(self):
        """Zero-point error (statistical (+) catalog-to-catalog)."""
        return float(self.meta['ZPERR'])

    def pair_arrays(self, good_only=True):
        """
        Per-pair arrays joined with the star list.

        Parameters
        ----------
        good_only : bool, optional
            Keep only pairs with psf_flags = 0 and finite MAST magnitude.

        Returns
        -------
        d : dict of numpy arrays
            'ra', 'dec' (Jay, deg), 'mbar', 'mbar_abmag', 'mast_mag', 'dmag'
            (mbar_abmag - mast_mag), 'dra_mas', 'ddec_mas' (Jay - MAST, bulk
            removed), 'cat_idx', 'row'.
        """
        p = self.pairs
        sel = np.asarray(p['mast_good'], dtype=bool) if good_only else np.ones(len(p), bool)
        row = np.asarray(p['row'])[sel]
        t = self.table

        d = {'row': row, 'cat_idx': np.asarray(p['cat_idx'])[sel],
             'ra': np.asarray(t['ra'], dtype=float)[row],
             'dec': np.asarray(t['dec'], dtype=float)[row],
             'mbar': np.asarray(t['mbar'], dtype=float)[row],
             'mbar_abmag': np.asarray(t['mbar_abmag'], dtype=float)[row],
             'mast_mag': np.asarray(p['mast_mag'], dtype=float)[sel],
             'dra_mas': -np.asarray(p['dxi_mas'], dtype=float)[sel],
             'ddec_mas': -np.asarray(p['deta_mas'], dtype=float)[sel]}
        d['dmag'] = d['mbar_abmag'] - d['mast_mag']

        return d

    def summary(self):
        """
        One-paragraph text summary of the calibration.

        Returns
        -------
        text : str
        """
        m = self.meta
        _pos_ok, phot_ok = matchup_quality(self.table)
        text = (f"{m['FILTER']} ({m['MATCHUP']}): {len(self.table):,} stars, "
                f"{int(np.sum(phot_ok)):,} with photometry, "
                f"{int(np.sum(self.table['mast_matched'])):,} matched to {m['NMASTCAT']} MAST "
                f"catalogs.\nZP = {m['ZP']:.4f} +/- {m['ZPERR']:.4f} (stat {m['ZPSTAT']:.4f}, "
                f"catalog-to-catalog {m['ZPSYS']:.4f}); per-star sigma {m['ZPSIG']:.3f} from "
                f"{m['ZPN']:,} stars with {m['ZPLO']:.2f} <= mbar < {m['ZPHI']:.2f}.\n"
                f"Astrometry: Gaia fit order {m['ASTORDER']}, {m['ASTNUSE']}/{m['ASTNREF']} stars, "
                f"rms {m['ASTRMSX']:.1f}/{m['ASTRMSY']:.1f} mas, scale {m['PIXSCALE']:.3f} "
                f"mas/pix, epoch {m['EPOCH']}.")

        return text

    # --- I/O ---------------------------------------------------------------

    def write(self, path, overwrite=False):
        """
        Write to a multi-extension FITS file.

        HDUs: STARLIST (calibrated star list + calibration keywords),
        MASTCATS, PAIRS, GAIAFIT.

        Parameters
        ----------
        path : str or Path
        overwrite : bool, optional

        Returns
        -------
        path : Path
        """
        path = Path(path)
        hdus = [fits.PrimaryHDU()]

        for name, tab in zip(HDU_NAMES, (self.table, self.catalogs, self.pairs, self.gaia)):
            hdu = fits.table_to_hdu(tab)
            hdu.name = name

            # Header comments on the calibration keywords.
            if name == 'STARLIST':
                for key, comment in HEADER_COMMENTS.items():
                    if key in hdu.header:
                        hdu.header.comments[key] = comment
            hdus.append(hdu)

        fits.HDUList(hdus).writeto(path, overwrite=overwrite)

        return path

    @classmethod
    def read(cls, path):
        """
        Read a FITS file written by `write`.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        result : CalibrationResult
        """
        # mask_invalid=False keeps NaN as NaN (not masked) for plain numpy use.
        tabs = [Table.read(path, hdu=name, mask_invalid=False) for name in HDU_NAMES]
        result = cls(*tabs)

        return result

    # --- Plots and report ----------------------------------------------------

    def plot_zeropoint(self, **kwargs):
        """Shortcut for `plots.plot_zeropoint(self, ...)`."""
        from . import plots
        return plots.plot_zeropoint(self, **kwargs)

    def plot_catalog_zeropoints(self, **kwargs):
        """Shortcut for `plots.plot_catalog_zeropoints(self, ...)`."""
        from . import plots
        return plots.plot_catalog_zeropoints(self, **kwargs)

    def plot_footprints(self, **kwargs):
        """Shortcut for `plots.plot_footprints(self, ...)`."""
        from . import plots
        return plots.plot_footprints(self, **kwargs)

    def plot_astrometry_map(self, **kwargs):
        """Shortcut for `plots.plot_astrometry_map(self, ...)`."""
        from . import plots
        return plots.plot_astrometry_map(self, **kwargs)

    def plot_residuals(self, **kwargs):
        """Shortcut for `plots.plot_residuals(self, ...)`."""
        from . import plots
        return plots.plot_residuals(self, **kwargs)

    def plot_gaia_fit(self, **kwargs):
        """Shortcut for `plots.plot_gaia_fit(self, ...)`."""
        from . import plots
        return plots.plot_gaia_fit(self, **kwargs)

    def plot_mag_errors(self, **kwargs):
        """Shortcut for `plots.plot_mag_errors(self, ...)`."""
        from . import plots
        return plots.plot_mag_errors(self, **kwargs)

    def plot_errors(self, **kwargs):
        """Shortcut for `plots.plot_errors(self, ...)`."""
        from . import plots
        return plots.plot_errors(self, **kwargs)

    def report(self, path, title=None, figures_dir=None):
        """
        Write a self-contained HTML calibration report.

        Parameters
        ----------
        path : str or Path
            Output HTML file.
        title : str or None, optional
        figures_dir : str or Path or None, optional
            Also save the figures as PNGs here.

        Returns
        -------
        path : Path
        """
        from .report import make_report
        return make_report(self, path, title=title, figures_dir=figures_dir)
