"""
Find and download MAST Roman source catalogs (*_cat.parquet).

Level-3 (sky-cell) catalogs are listed in the MAST product list. Level-2
(per-exposure, per-detector) catalogs are not listed there, but are served by
the Roman retrieve_product endpoint as '<obs_id>_cat.parquet'.

Authentication uses the MAST_API_TOKEN environment variable.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

ROMAN_RETRIEVE_URL = 'https://mast.stsci.edu/search/roman/api/v0.1/retrieve_product'


def get_token():
    """
    Return the MAST API token from the environment.

    Returns
    -------
    token : str

    Raises
    ------
    RuntimeError
        If MAST_API_TOKEN is not set.
    """
    token = os.environ.get('MAST_API_TOKEN')
    if not token:
        raise RuntimeError('MAST_API_TOKEN is not set; export it to query/download from MAST.')

    return token


def find_catalogs(ra, dec, radius_deg, filt, proposal_id=None, level='auto'):
    """
    List MAST Roman source catalogs for one filter in a cone.

    Parameters
    ----------
    ra, dec : float
        Cone center (deg).
    radius_deg : float
        Cone radius (deg).
    filt : str
        Roman filter, e.g. 'F146'.
    proposal_id : str or None, optional
        Restrict to one proposal.
    level : {'auto', '2', '3', 'both'}, optional
        Catalog level. 'auto' uses level 3 if any exist, else level 2.

    Returns
    -------
    products : list of dict
        Keys: 'obs_id', 'filename', 'size' (bytes or None), 'level'.
    """
    from astroquery.mast import Observations
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    Observations.login(token=get_token())

    # Cone search for this filter (and proposal).
    criteria = dict(obs_collection='Roman', filters=filt.upper())
    if proposal_id is not None:
        criteria['proposal_id'] = str(proposal_id)
    obs = Observations.query_criteria(coordinates=SkyCoord(ra, dec, unit='deg'),
                                      radius=radius_deg * u.deg, **criteria)
    if len(obs) == 0:
        return []

    lev = np.array([int(v) for v in obs['calib_level']])

    # Level 3: catalogs are in the product list.
    l3 = []
    if np.any(lev == 3):
        prod = Observations.get_product_list(obs[lev == 3])
        for p in prod:
            fn = str(p['productFilename'])
            if fn.endswith('_cat.parquet'):
                l3.append({'obs_id': str(p['obs_id']), 'filename': fn,
                           'size': int(p['size']), 'level': 3})

    # Level 2: one '<obs_id>_cat.parquet' per exposure/detector (size unknown).
    l2 = [{'obs_id': str(o), 'filename': f'{o}_cat.parquet', 'size': None, 'level': 2}
          for o in obs['obs_id'][lev == 2]]

    if level == 'auto':
        products = l3 if l3 else l2
    elif str(level) == '3':
        products = l3
    elif str(level) == '2':
        products = l2
    else:
        products = l3 + l2

    return products


def download_one(obs_id, filename, out_dir, token, expected_size=None, timeout=300):
    """
    Stream one product from the Roman retrieve_product endpoint.

    Parameters
    ----------
    obs_id : str
        Observation ID (URL path prefix).
    filename : str
        Product filename.
    out_dir : Path
        Output directory.
    token : str
        MAST API token.
    expected_size : int or None, optional
        Expected size in bytes; if None, the server Content-Length is used.
    timeout : float, optional
        Request timeout (s).

    Returns
    -------
    status : str
        'OK', 'EXISTS', or an error message.

    Notes
    -----
    Writes to '<filename>.part' and renames only after the size check passes.
    """
    out_path = out_dir / filename
    if out_path.exists() and (expected_size is None or out_path.stat().st_size == expected_size):
        return 'EXISTS'

    url = f'{ROMAN_RETRIEVE_URL}?product_name={obs_id}%2F{filename}'
    part = out_dir / (filename + '.part')

    try:
        with requests.get(url, headers={'Authorization': f'token {token}'}, stream=True,
                          timeout=timeout) as resp:
            resp.raise_for_status()
            if expected_size is None:
                expected_size = int(resp.headers['Content-Length'])
            with open(part, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8 << 20):
                    f.write(chunk)
    except Exception as err:
        part.unlink(missing_ok=True)
        return f'{type(err).__name__}: {err}'

    # Size check before promoting the file.
    if part.stat().st_size != expected_size:
        part.unlink(missing_ok=True)
        return f'size mismatch (expected {expected_size})'
    part.rename(out_path)

    return 'OK'


def download_catalogs(products, out_dir, n_workers=4, verbose=True):
    """
    Download catalogs, skipping files already present.

    Parameters
    ----------
    products : list of dict
        Output of `find_catalogs`.
    out_dir : str or Path
        Output directory.
    n_workers : int, optional
        Parallel downloads.
    verbose : bool, optional
        Print a one-line summary and any failures.

    Returns
    -------
    paths : list of Path
        Local paths of catalogs available after the download.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    token = get_token()

    # Parallel downloads; collect status per file.
    status = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(download_one, p['obs_id'], p['filename'], out_dir, token,
                            p['size']): p['filename'] for p in products}
        for fut in as_completed(futs):
            status[futs[fut]] = fut.result()

    paths = [out_dir / f for f, s in status.items() if s in ('OK', 'EXISTS')]
    if verbose:
        n_new = sum(s == 'OK' for s in status.values())
        n_old = sum(s == 'EXISTS' for s in status.values())
        print(f'MAST catalogs: {n_new} downloaded, {n_old} already present, '
              f'{len(status) - n_new - n_old} failed.')
        for f, s in sorted(status.items()):
            if s not in ('OK', 'EXISTS'):
                print(f'  FAILED {f}: {s}')

    return sorted(paths)
