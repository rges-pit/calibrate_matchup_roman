# calibrate_matchup_roman

Calibrate Jay Anderson's MATCHUP star lists from Roman WFI data:

- **Astrometry:** RA/Dec from a Gaia link file (Gaia DR3 stars measured in the same
  pixel frame).
- **Photometry:** an AB zero point measured against MAST Roman pipeline source
  catalogs (`*_cat.parquet`) in the same filter.

The output is a FITS table with **every MATCHUP column** plus `xbar_asec`, `ybar_asec`,
`mbar_abmag`, `xsig_mas`, `ysig_mas`, `msig_abmag`, `ra`, `dec`, and `mast_matched`.
Diagnostic plots and a self-contained HTML report can be made from
that file at any time.

## Installation

```bash
git clone https://github.com/rges-pit/calibrate_matchup_roman.git
cd calibrate_matchup_roman
pip install -e .
```

That adds the `calibrate-matchup-roman` command. Without installing, run
`python -m calibrate_matchup_roman` from the repository root.

Querying and downloading from MAST needs a token:

```bash
export MAST_API_TOKEN=<your token>
```

## Command line

There are three subcommands: `calibrate`, `report` and `plots`.

```bash
# Query MAST, download catalogs, calibrate, and write FITS + HTML report.
calibrate-matchup-roman calibrate MATCHUP.XYMEEE.00.F213 \
    --gaia link_with_gaia.xymrd --proposal 1054 \
    --data-dir mast_catalogs --out F213.cal.fits --report

# Offline: use catalogs already in --data-dir (or list them with --catalogs).
calibrate-matchup-roman calibrate MATCHUP.XYMEEE.02.F146 \
    --gaia link_with_gaia.xymrd --data-dir mast_catalogs --no-download \
    --out F146.cal.fits --report

# Re-make the report (optionally saving every figure as PNG) from a FITS file.
calibrate-matchup-roman report F213.cal.fits --figures-dir figures

# Only the diagnostic PNGs.
calibrate-matchup-roman plots F213.cal.fits --out-dir figures
```

Useful `calibrate` options (see `calibrate-matchup-roman calibrate --help`):

| Option | Default | Meaning |
|---|---|---|
| `--filter` | from filename | Filter, e.g. `F213` |
| `--level` | `auto` | MAST catalog level: `auto` (level 3 sky cells if any exist, else level 2), `2`, `3`, `both` |
| `--catalogs f1 f2 ...` | | Use these catalogs; skips the MAST query |
| `--radius` | `0.1` | Match radius in arcsec, after removing each catalog's bulk offset |
| `--zp-window LO HI` | automatic | Jay `mbar` range used for the zero point |
| `--zp-tol` | `0.03` | Flatness tolerance (mag) for the automatic window |
| `--order` | `1` | Polynomial order of the pixel → sky fit |
| `--err-mode` | `mean` | `mean`: `msig/sqrt(Ng)`; `rms`: `msig` (ZP error added in quadrature either way) |
| `--min-matches` | `50` | Skip MAST catalogs with fewer matches |

## Python API

```python
import calibrate_matchup_roman as cmr

# Run the calibration (download=False + data_dir, or catalogs=[...], to stay offline).
res = cmr.calibrate('MATCHUP.XYMEEE.00.F213', 'link_with_gaia.xymrd',
                    proposal_id='1054', data_dir='mast_catalogs')

print(res.summary())
print(res.zp, res.zp_err, res.filter)

# Calibrated star list: MATCHUP columns plus the calibrated columns below.
stars = res.table
good = np.isfinite(stars['mbar_abmag'])

# Per-MAST-catalog results, matched pairs, and Gaia fit residuals.
res.catalogs      # one row per catalog: matches, bulk offset, residual scatter, per-catalog ZP
res.pairs         # matched pairs (row index into res.table, MAST ra/dec/mag, residuals)
res.gaia          # Gaia link stars + residuals of the pixel -> sky fit

# Save, reload, report.
res.write('F213.cal.fits', overwrite=True)
res = cmr.read('F213.cal.fits')          # all plots and the report work from the file
res.report('F213.cal.html', figures_dir='figures')
```

The star list alone can be read with plain astropy:

```python
from astropy.table import Table
stars = Table.read('F213.cal.fits', hdu='STARLIST')
zp = stars.meta['ZP']
```

### Plotting utilities

Every plot takes a `CalibrationResult`, returns a matplotlib `Figure`, and accepts
`path=` to save it. Each is also a method on the result (`res.plot_zeropoint()`, ...).

```python
from calibrate_matchup_roman import plots

plots.plot_zeropoint(res)            # MAST AB - Jay mbar vs mbar, ZP window, residual histogram
plots.plot_catalog_zeropoints(res)   # per-catalog ZP and bulk astrometric offsets
plots.plot_footprints(res)           # Jay field + MAST footprints colored by per-catalog ZP
plots.plot_astrometry_map(res)       # Jay - MAST offset vectors on the sky (per star, binned)
plots.plot_residuals(res)            # Jay - MAST ΔRA, ΔDec, Δmag vs RA, Dec, magnitude
plots.plot_gaia_fit(res)             # residuals of the pixel -> sky fit to Gaia
plots.plot_mag_errors(res)           # msig_abmag vs mbar_abmag; magnitude distributions
plots.save_all(res, 'figures')       # all of the above as PNGs
```

### Report

`res.report(path)`, or `calibrate-matchup-roman report file.fits`, writes one HTML file
with embedded figures. It contains:

- a summary table;
- automatic **checks**: Gaia fit rms, catalog-to-catalog zero-point scatter versus the
  statistical error, width of the zero-point window, spread of bulk offsets between
  catalogs, and the fraction of stars matched;
- the per-catalog table;
- all diagnostic plots;
- the output column descriptions and all calibration keywords.

## Output FITS file

| HDU | Contents |
|---|---|
| `STARLIST` | All MATCHUP columns, plus `xbar_asec`, `ybar_asec` [arcsec], `mbar_abmag` [AB mag], `xsig_mas`, `ysig_mas` [mas], `msig_abmag` [AB mag], `ra`, `dec` [deg], and `mast_matched`. Calibration keywords are in the header (below). |
| `MASTCATS` | One row per MAST catalog used: `n_match`, bulk offset `dxi_mas`/`deta_mas` (MAST − Jay), residual scatter, per-catalog `zp_cat`, footprint `s_region` |
| `PAIRS` | Matched pairs: `row` (into STARLIST), `cat_idx` (into MASTCATS), MAST `ra`/`dec`/`mag`, `mast_good`, residuals `dxi_mas`/`deta_mas` |
| `GAIAFIT` | Gaia link stars with residuals of the pixel → sky fit and a `used` flag |

`STARLIST` header keywords:

- **Zero point:**
  - `ZP`: the zero point itself, `mbar_abmag = mbar + ZP`.
  - `ZPERR` = `ZPSTAT` ⊕ `ZPSYS`, where `ZPSTAT` is the statistical error and `ZPSYS`
    is the star-weighted rms of the per-catalog zero points.
  - `ZPSIG`: per-star scatter in the window. `ZPN`: number of stars used.
  - `ZPLO`/`ZPHI`: the `mbar` window. `ZPAUTO`: whether it was chosen automatically.
- **Matching:** `NMASTCAT` (catalogs used), `NMATCH` (matched pairs), `MATCHRAD`
  (match radius).
- **Astrometric fit:** `ASTORDER`, `ASTNREF`/`ASTNUSE`, `ASTRMSX`/`ASTRMSY`,
  `PIXSCALE`, `TANRA`/`TANDEC`, `EPOCH`.
- **Provenance:** `ERRMODE`, `MATCHUP`, `GAIALINK`, `CALSOFT`, `CALDATE`.

## Method and assumptions

1. **Positions.** A polynomial (affine by default) is fitted from Jay's x1 pixels to
   gnomonic (ξ, η) using the Gaia link stars, with 4σ clipping. RA/Dec are on the
   Gaia DR3 frame at the link-file epoch; they are not shifted onto MAST.
2. **Matching.** For each MAST catalog, a coarse 1″ mutual nearest-neighbour match
   measures the bulk offset. Then a one-to-one match is made within `--radius`.
3. **Zero point.** MAST PSF AB = −2.5 log₁₀(`psf_flux`/nJy) + 31.4, using only
   sources with `psf_flags = 0`.
   - The window is the run of consecutive 0.25-mag bins holding the most stars whose
     bin medians stay within ±`zp_tol` of each other. This excludes saturated bright
     stars and crowded faint stars, where pipeline photometry departs from Jay's.
   - ZP is a 3σ-clipped median of the pairs in that window, pooled over all catalogs.
4. **Errors.**
   - `mbar_abmag = mbar + ZP`. `msig_abmag = sqrt((msig/sqrt(Ng))^2 + ZPERR^2)`.
     This treats `msig` as the per-measurement rms; to confirm with Jay, and
     `--err-mode rms` uses `msig` directly.
   - `xbar_asec` and `ybar_asec` are offsets from Sgr A* along `xbar` and `ybar`,
     times a fixed 110 mas/pix. They do not use the fitted Gaia scale or rotation.
     `xsig_mas` and `ysig_mas` are the exposure rms times that same scale, not
     divided by `sqrt(Ng)`.
   - Rows with x = y = −2500 have no position (`ra`, `dec`, `xbar_asec`, `ybar_asec` are NaN).
   - Rows with `Ng = 0`, or `msig` equal to 0 or 9, have no valid photometry
     (`mbar_abmag` and `msig_abmag` are NaN).

## Tests

```bash
pip install -e ".[test]"
python -m pytest tests -q
```

The tests use synthetic data with a known rotation, offset, zero point and
bright/faint curvature. They check the recovered values, the FITS round trip, every
plot, the report, and the CLI.
