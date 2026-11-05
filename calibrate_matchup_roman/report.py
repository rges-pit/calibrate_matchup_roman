"""
Self-contained HTML calibration report for a CalibrationResult.
"""

import base64
import html
import io as _io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from . import plots
from .io import matchup_quality

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial,
       sans-serif; max-width: 1150px; margin: 0 auto; padding: 0 24px 60px; color: #1d1d1f;
       line-height: 1.5; }
h1 { margin-top: 22px; } h2 { border-bottom: 1px solid #eee; padding-bottom: 4px;
     margin-top: 34px; }
.meta { color: #666; font-size: 0.9em; }
.ok { background: #f1f8f2; border-left: 4px solid #2ca02c; padding: 8px 14px; margin: 8px 0; }
.warn { background: #fff7ed; border-left: 4px solid #e66101; padding: 8px 14px; margin: 8px 0; }
table { border-collapse: collapse; font-size: 0.86em; margin: 10px 0 18px; }
th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f5f5f7; } tr:nth-child(even) td { background: #fafafa; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
pre { background: #f6f8fa; border: 1px solid #e1e4e8; padding: 10px 12px; overflow-x: auto;
      font-size: 0.82em; }
details { margin: 10px 0; } summary { cursor: pointer; color: #0b5cad; }
figure { margin: 16px 0; } figure img { max-width: 100%; border: 1px solid #eee; }
figcaption { color: #555; font-size: 0.9em; }
"""

# Descriptions of the columns added to the MATCHUP list.
NEW_COLUMNS = [
    ('xbar_asec, ybar_asec', 'arcsec',
     'Offset from Sgr A* along xbar, ybar at the fixed 110 mas/pix scale. '
     'NaN if x = y = -2500'),
    ('mbar_abmag', 'mag', 'AB magnitude = mbar + ZP (NaN if Ng = 0 or msig is 0 or 9)'),
    ('xsig_mas, ysig_mas', 'mas',
     'xsig, ysig times 110 mas/pix (rms of the exposures, not / sqrt(Ng))'),
    ('msig_abmag', 'mag', 'sqrt((msig/sqrt(Ng))^2 + ZPERR^2) for ERRMODE=mean; msig for rms'),
    ('ra, dec', 'deg', 'ICRS position on the Gaia DR3 frame at EPOCH (NaN if x = y = -2500)'),
    ('mast_matched', '', 'Matched to at least one MAST catalog'),
]


def esc(x):
    """
    HTML-escape a value.

    Parameters
    ----------
    x : object

    Returns
    -------
    s : str
    """
    s = html.escape(str(x))

    return s


def fmt(v, nd=4):
    """
    Format a table cell.

    Parameters
    ----------
    v : object
    nd : int, optional
        Decimals for floats.

    Returns
    -------
    s : str
    """
    if isinstance(v, (bool, np.bool_)):
        s = 'yes' if v else 'no'
    elif isinstance(v, (int, np.integer)):
        s = f'{v:,}'
    elif isinstance(v, (float, np.floating)):
        s = '' if not np.isfinite(v) else f'{v:,.{nd}f}'
    else:
        s = str(v)

    return s


def table_html(header, rows, num_cols=(), nd=4):
    """
    HTML table.

    Parameters
    ----------
    header : list of str
    rows : list of list
    num_cols : iterable of int, optional
        Right-aligned columns.
    nd : int, optional
        Decimals for floats.

    Returns
    -------
    s : str
    """
    num_cols = set(num_cols)
    out = ['<table><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in header) + '</tr>']
    for row in rows:
        cells = ''.join(f'<td{" class=num" if i in num_cols else ""}>{esc(fmt(v, nd))}</td>'
                        for i, v in enumerate(row))
        out.append(f'<tr>{cells}</tr>')
    out.append('</table>')
    s = '\n'.join(out)

    return s


def figure_html(fig, caption, save_path=None):
    """
    Embed a matplotlib figure as a base64 PNG (and optionally save it).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    caption : str
    save_path : Path or None, optional

    Returns
    -------
    s : str
    """
    buf = _io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    if save_path is not None:
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)

    b64 = base64.b64encode(buf.getvalue()).decode()
    s = (f'<figure><img src="data:image/png;base64,{b64}" alt="{esc(caption)}">'
         f'<figcaption>{esc(caption)}</figcaption></figure>')

    return s


def quality_checks(res):
    """
    Automatic checks on a calibration.

    Parameters
    ----------
    res : CalibrationResult

    Returns
    -------
    checks : list of (bool, str)
        (passed, message) pairs.
    """
    m, c, t = res.meta, res.catalogs, res.table
    checks = []

    # Gaia fit.
    rms = max(m['ASTRMSX'], m['ASTRMSY'])
    checks.append((rms < 25., f"Gaia pixel → sky fit rms {m['ASTRMSX']:.1f} / "
                               f"{m['ASTRMSY']:.1f} mas ({m['ASTNUSE']}/{m['ASTNREF']} stars)"))

    # Zero point: systematic vs statistical.
    sys_ok = m['ZPSYS'] <= max(0.02, 3. * m['ZPSTAT'])
    checks.append((sys_ok, f"Catalog-to-catalog ZP scatter {m['ZPSYS']:.4f} mag vs statistical "
                           f"{m['ZPSTAT']:.4f} mag"
                           + ('' if sys_ok else ' — ZPERR is dominated by differences between '
                                                'MAST catalogs')))

    # Width of the zero-point window.
    width = m['ZPHI'] - m['ZPLO']
    checks.append((width >= 1.0, f"ZP window {m['ZPLO']:.2f} to {m['ZPHI']:.2f} "
                                 f"({width:.2f} mag wide, {m['ZPN']:,} stars)"))

    # Spread of bulk offsets between catalogs.
    if len(c) > 1:
        spread = max(np.ptp(c['dxi_mas']), np.ptp(c['deta_mas']))
        checks.append((spread < 20., f"Bulk offsets MAST − Jay differ between catalogs by up to "
                                     f"{spread:.0f} mas"))

    # Matched fraction of stars with photometry.
    _pos_ok, phot_ok = matchup_quality(t)
    frac = np.sum(t['mast_matched']) / max(np.sum(phot_ok), 1)
    checks.append((frac > 0.1, f"{frac:.0%} of stars with photometry are matched to MAST"))

    return checks


def make_report(res, path, title=None, figures_dir=None):
    """
    Write a self-contained HTML report for a calibration.

    Parameters
    ----------
    res : CalibrationResult
    path : str or Path
        Output HTML file.
    title : str or None, optional
    figures_dir : str or Path or None, optional
        Also save each figure as a PNG in this directory.

    Returns
    -------
    path : Path
    """
    m, c = res.meta, res.catalogs
    path = Path(path)
    title = title or f"{m['FILTER']} calibration: {m['MATCHUP']}"
    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

    def fig(name, func, caption):
        """Render one plot into the report."""
        save = figures_dir / f"{m['FILTER']}_{name}.png" if figures_dir else None
        return figure_html(func(res), caption, save)

    # Summary numbers.
    t = res.table
    pos_ok, phot_ok = matchup_quality(t)
    summ = table_html(['Quantity', 'Value'], [
        ['Filter / magnitude system', f"{m['FILTER']} / {m['MAGSYS']}"],
        ['Zero point (mbar_abmag = mbar + ZP)', f"{m['ZP']:.4f} ± {m['ZPERR']:.4f}"],
        ['  statistical / catalog-to-catalog', f"{m['ZPSTAT']:.4f} / {m['ZPSYS']:.4f}"],
        ['  per-star scatter in window', f"{m['ZPSIG']:.3f} mag ({m['ZPN']:,} stars)"],
        ['  window (Jay mbar)', f"{m['ZPLO']:.2f} to {m['ZPHI']:.2f}"
                                f"{' (automatic, tol ' + str(m['ZPTOL']) + ' mag)' if m['ZPAUTO'] else ''}"],
        ['Stars: total / with positions / with photometry',
         f"{len(t):,} / {int(np.sum(pos_ok)):,} / {int(np.sum(phot_ok)):,}"],
        ['MAST catalogs used / matched pairs', f"{m['NMASTCAT']} / {m['NMATCH']:,} "
                                               f"(radius {m['MATCHRAD']}″ after bulk shift)"],
        ['Astrometry', f"Gaia fit order {m['ASTORDER']}, rms {m['ASTRMSX']:.1f} / "
                       f"{m['ASTRMSY']:.1f} mas, {m['PIXSCALE']:.3f} mas/pix, epoch {m['EPOCH']}"],
        ['Per-star error mode', m['ERRMODE']],
    ])

    checks = ''.join(f'<div class="{"ok" if ok else "warn"}">{"✓" if ok else "!"} {esc(msg)}</div>'
                     for ok, msg in quality_checks(res))

    cats = table_html(['#', 'Catalog', 'Level', 'N src', 'N match', 'MAST−Jay Δξ (mas)',
                       'Δη (mas)', 'σξ (mas)', 'ση (mas)', 'ZP', 'N (ZP)'],
                      [[i, r['catalog'], int(r['level']), int(r['n_src']), int(r['n_match']),
                        f"{r['dxi_mas']:+.1f}", f"{r['deta_mas']:+.1f}", f"{r['sig_xi_mas']:.1f}",
                        f"{r['sig_eta_mas']:.1f}", f"{r['zp_cat']:.4f}", int(r['zp_cat_n'])]
                       for i, r in enumerate(c)], (0, 2, 3, 4, 5, 6, 7, 8, 9, 10))

    hdr = table_html(['Keyword', 'Value'], [[k, v] for k, v in m.items()])
    cols = table_html(['Column', 'Unit', 'Description'], NEW_COLUMNS)

    body = f"""
<p class="meta">{esc(m['CALSOFT'])} · created {esc(m['CALDATE'])} · inputs
<code>{esc(m['MATCHUP'])}</code>, <code>{esc(m['GAIALINK'])}</code></p>
<h2>Summary</h2>{summ}
<h2>Checks</h2>{checks}
<h2>Zero point</h2>
{fig('zeropoint', plots.plot_zeropoint, 'MAST psf AB − Jay mbar vs mbar; shaded = ZP window; '
     'right: residuals in the window.')}
{fig('catalog_zeropoints', plots.plot_catalog_zeropoints, 'Per-catalog ZP (top) and bulk '
     'astrometric offsets MAST − Jay (bottom).')}
<h2>MAST catalogs</h2>
{fig('footprints', plots.plot_footprints, 'Footprints of the MAST catalogs used, colored by '
     'per-catalog ZP offset; grey = Jay stars.')}
{cats}
<h2>Astrometry</h2>
{fig('gaia_fit', plots.plot_gaia_fit, 'Residuals of the pixel → sky fit to the Gaia link stars.')}
{fig('astrometry_map', plots.plot_astrometry_map, "Jay − MAST offsets on the sky after removing "
     "each catalog's bulk offset.")}
{fig('residuals', plots.plot_residuals, 'Jay − MAST residuals in ΔRA cos δ, ΔDec, Δmag vs RA, '
     'Dec, magnitude.')}
<h2>Catalog errors</h2>
{fig('errors', plots.plot_errors, 'Uncertainty of the mean position (left) and '
     'msig_abmag (right) versus mbar_abmag. The position error is '
     'hypot(xsig, ysig) / sqrt(Ng).')}
<h2>Photometric errors</h2>
{fig('mag_errors', plots.plot_mag_errors, 'msig_abmag vs mbar_abmag (left) and magnitude '
     'distributions (right).')}
<h2>Output columns</h2>
<p>All MATCHUP columns are kept unchanged; added columns:</p>{cols}
<details><summary>All calibration keywords (STARLIST header)</summary>{hdr}</details>
"""
    doc = (f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{esc(title)}</title>'
           f'<style>{CSS}</style></head><body><h1>{esc(title)}</h1>{body}</body></html>')
    path.write_text(doc)

    return path
