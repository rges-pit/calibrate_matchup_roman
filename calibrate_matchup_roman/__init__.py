"""
calibrate_matchup_roman
-----------------------
Put Jay Anderson's MATCHUP star lists (Roman WFI) on RA/Dec, via a Gaia link
file, and on the AB magnitude scale, via a zero point measured against MAST
Roman pipeline source catalogs.

Python API
----------
>>> import calibrate_matchup_roman as cmr
>>> res = cmr.calibrate('MATCHUP.XYMEEE.00.F213', 'link_with_gaia.xymrd',
...                     proposal_id='1054', data_dir='mast_catalogs')
>>> res.write('F213.cal.fits')
>>> res.report('F213.cal.html')
>>> res = cmr.read('F213.cal.fits')          # reload later; plots/report still work

Command line
------------
    calibrate-matchup-roman calibrate MATCHUP.XYMEEE.00.F213 --gaia link.xymrd ...
    calibrate-matchup-roman report F213.cal.fits
    python -m calibrate_matchup_roman --help

See README.md in this directory for details.
"""

__version__ = '0.2.0'

from .core import calibrate
from .result import CalibrationResult
from . import plots
from .report import make_report

read = CalibrationResult.read

__all__ = ['calibrate', 'read', 'CalibrationResult', 'make_report', 'plots', '__version__']
