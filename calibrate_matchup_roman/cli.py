"""
Command-line interface.

    calibrate-matchup-roman calibrate MATCHUP.XYMEEE.00.F213 --gaia link_with_gaia.xymrd \\
        --proposal 1054 --data-dir mast_catalogs --out F213.cal.fits --report
    calibrate-matchup-roman report F213.cal.fits [--out F213.cal.html] [--figures-dir figs]
    calibrate-matchup-roman plots F213.cal.fits --out-dir figs

(`python -m calibrate_matchup_roman ...` works the same without installing.)
"""

import argparse
from pathlib import Path


def add_calibrate_args(p):
    """
    Arguments for the 'calibrate' subcommand.

    Parameters
    ----------
    p : argparse.ArgumentParser

    Returns
    -------
    None
    """
    p.add_argument('matchup', help='MATCHUP.XYMEEE.* file')
    p.add_argument('--gaia', required=True, help='Gaia link file (*.xymrd) in the same frame')
    p.add_argument('--out', default=None, help='output FITS (default: <matchup>.cal.fits)')
    p.add_argument('--filter', default=None, help='filter, e.g. F213 (default: from filename)')
    p.add_argument('--catalogs', nargs='+', default=None,
                   help='use these MAST *_cat.parquet files (skips the MAST query)')
    p.add_argument('--data-dir', default='mast_catalogs', help='MAST catalog download dir')
    p.add_argument('--proposal', default=None, help='MAST proposal ID, e.g. 1054')
    p.add_argument('--level', default='auto', choices=['auto', '2', '3', 'both'],
                   help='MAST catalog level (auto: level 3 if available, else 2)')
    p.add_argument('--no-download', action='store_true',
                   help='do not query MAST; use catalogs already in --data-dir')
    p.add_argument('--radius', type=float, default=0.1,
                   help='match radius in arcsec after bulk shift (default 0.1)')
    p.add_argument('--zp-window', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                   help='Jay mbar window for the ZP (default: automatic)')
    p.add_argument('--zp-tol', type=float, default=0.03,
                   help='flatness tolerance for the automatic window (mag, default 0.03)')
    p.add_argument('--order', type=int, default=1, help='pixel->sky polynomial order')
    p.add_argument('--err-mode', default='mean', choices=['mean', 'rms'],
                   help='mean: msig/sqrt(Ng); rms: msig (ZP error added in quadrature)')
    p.add_argument('--min-matches', type=int, default=50,
                   help='skip MAST catalogs with fewer matches (default 50)')
    p.add_argument('--report', action='store_true',
                   help='also write an HTML report next to the FITS file')
    p.add_argument('--overwrite', action='store_true')

    return None


def run_calibrate(args):
    """
    'calibrate' subcommand.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    None
    """
    from .core import calibrate

    res = calibrate(args.matchup, args.gaia, filt=args.filter, catalogs=args.catalogs,
                    data_dir=args.data_dir, proposal_id=args.proposal, level=args.level,
                    download=not args.no_download, match_radius=args.radius,
                    zp_window=tuple(args.zp_window) if args.zp_window else None,
                    fit_order=args.order, err_mode=args.err_mode, min_matches=args.min_matches,
                    zp_tol=args.zp_tol)

    out = Path(args.out or f'{args.matchup}.cal.fits')
    res.write(out, overwrite=args.overwrite)
    print(f'Wrote {out} ({len(res.table):,} rows, {len(res.table.colnames)} columns)')

    if args.report:
        html = res.report(out.with_suffix('.html'))
        print(f'Wrote {html}')

    return None


def run_report(args):
    """
    'report' subcommand: HTML report from a calibrated FITS file.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    None
    """
    from .result import CalibrationResult

    res = CalibrationResult.read(args.fits)
    out = Path(args.out or Path(args.fits).with_suffix('.html'))
    res.report(out, title=args.title, figures_dir=args.figures_dir)
    print(f'Wrote {out}')

    return None


def run_plots(args):
    """
    'plots' subcommand: save all diagnostic PNGs from a calibrated FITS file.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    None
    """
    from . import plots
    from .result import CalibrationResult

    res = CalibrationResult.read(args.fits)
    for p in plots.save_all(res, args.out_dir):
        print(f'Wrote {p}')

    return None


def main(argv=None):
    """
    Entry point for `calibrate-matchup-roman` / `python -m calibrate_matchup_roman`.

    Parameters
    ----------
    argv : list of str or None, optional

    Returns
    -------
    None
    """
    import matplotlib
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser(
        prog='calibrate-matchup-roman',
        description='Calibrate Jay Anderson MATCHUP star lists: RA/Dec from a Gaia link file '
                    'and an AB zero point from MAST Roman source catalogs (MAST access needs '
                    'MAST_API_TOKEN).')
    sub = parser.add_subparsers(dest='command', required=True)

    p_cal = sub.add_parser('calibrate', help='calibrate a MATCHUP file -> FITS (+ report)')
    add_calibrate_args(p_cal)
    p_cal.set_defaults(func=run_calibrate)

    p_rep = sub.add_parser('report', help='HTML report from a calibrated FITS file')
    p_rep.add_argument('fits')
    p_rep.add_argument('--out', default=None, help='output HTML (default: <fits>.html)')
    p_rep.add_argument('--title', default=None)
    p_rep.add_argument('--figures-dir', default=None, help='also save figures as PNGs here')
    p_rep.set_defaults(func=run_report)

    p_plt = sub.add_parser('plots', help='save all diagnostic plots from a calibrated FITS file')
    p_plt.add_argument('fits')
    p_plt.add_argument('--out-dir', default='.', help='output directory for PNGs')
    p_plt.set_defaults(func=run_plots)

    args = parser.parse_args(argv)
    args.func(args)

    return None


if __name__ == '__main__':
    main()
