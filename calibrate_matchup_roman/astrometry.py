"""
Pixel -> sky mapping from Gaia link stars, tangent-plane helpers, and matching.
"""

import numpy as np
from scipy.spatial import cKDTree

# Fixed WFI plate scale, the value in Jay's XY() link header. The Gaia
# polynomial has its own fitted scale; these arcsec columns do not use it.
PLATE_SCALE_MAS = 110.0

# Sgr A*, ICRS. Same position as the MAST search cone.
SGRA_RA = 266.4168
SGRA_DEC = -29.0078


def radec_to_tan(ra, dec, ra0, dec0):
    """
    Gnomonic (tangent-plane) projection.

    Parameters
    ----------
    ra, dec : array_like
        Coordinates in degrees.
    ra0, dec0 : float
        Tangent point in degrees.

    Returns
    -------
    xi, eta : numpy arrays
        East and north offsets in arcsec.
    """
    ra = np.radians(np.asarray(ra, dtype=float))
    dec = np.radians(np.asarray(dec, dtype=float))
    a0, d0 = np.radians(ra0), np.radians(dec0)

    # Standard gnomonic projection.
    cosc = np.sin(d0) * np.sin(dec) + np.cos(d0) * np.cos(dec) * np.cos(ra - a0)
    xi = np.cos(dec) * np.sin(ra - a0) / cosc
    eta = (np.cos(d0) * np.sin(dec) - np.sin(d0) * np.cos(dec) * np.cos(ra - a0)) / cosc

    # Radians -> arcsec.
    xi = np.degrees(xi) * 3600.
    eta = np.degrees(eta) * 3600.

    return xi, eta


def tan_to_radec(xi, eta, ra0, dec0):
    """
    Inverse gnomonic projection.

    Parameters
    ----------
    xi, eta : array_like
        East and north offsets in arcsec.
    ra0, dec0 : float
        Tangent point in degrees.

    Returns
    -------
    ra, dec : numpy arrays
        Coordinates in degrees.
    """
    x = np.radians(np.asarray(xi, dtype=float) / 3600.)
    y = np.radians(np.asarray(eta, dtype=float) / 3600.)
    a0, d0 = np.radians(ra0), np.radians(dec0)

    # Standard inverse gnomonic projection.
    rho = np.hypot(x, y)
    c = np.arctan(rho)
    with np.errstate(invalid='ignore', divide='ignore'):
        dec = np.arcsin(np.cos(c) * np.sin(d0) +
                        np.where(rho > 0, y * np.sin(c) * np.cos(d0) / rho, 0.))
    ra = a0 + np.arctan2(x * np.sin(c), rho * np.cos(d0) * np.cos(c) - y * np.sin(d0) * np.sin(c))

    ra = np.degrees(ra) % 360.
    dec = np.degrees(dec)

    return ra, dec


def design_matrix(x, y, order=1, center=2048., half=2048.):
    """
    Polynomial design matrix in scaled pixel coordinates.

    Parameters
    ----------
    x, y : array_like
        Pixel positions.
    order : int, optional
        Polynomial order (1 = affine).
    center, half : float, optional
        Scaling: (x - center) / half, for conditioning.

    Returns
    -------
    A : numpy array, shape = [N, N_terms]
    """
    xs = (np.asarray(x, dtype=float) - center) / half
    ys = (np.asarray(y, dtype=float) - center) / half

    terms = [np.ones_like(xs)]
    for n in range(1, order + 1):
        for i in range(n + 1):
            terms.append(xs ** (n - i) * ys ** i)
    A = np.column_stack(terms)

    return A


def fit_pixel_to_sky(x, y, ra, dec, ra0, dec0, order=1, clip=4.0, n_iter=10):
    """
    Fit pixel (x, y) -> tangent-plane (xi, eta) with iterative sigma clipping.

    Parameters
    ----------
    x, y : array_like
        Pixel positions of reference stars (e.g. Gaia stars in Jay's x1 frame).
    ra, dec : array_like
        Reference coordinates (deg) at the epoch of the data.
    ra0, dec0 : float
        Tangent point (deg).
    order : int, optional
        Polynomial order.
    clip : float, optional
        Clip stars with residual radius > clip * sigma. The default (4) only
        removes clear outliers; Gaia errors have heavy tails at faint G.
    n_iter : int, optional
        Maximum clipping iterations.

    Returns
    -------
    fit : dict
        'cx', 'cy' (coefficients), 'order', 'ra0', 'dec0', 'keep' (mask),
        'rms_x_mas', 'rms_y_mas', 'n_ref', 'n_used', 'scale_mas' (mas/pix).
    """
    xi, eta = radec_to_tan(ra, dec, ra0, dec0)
    A = design_matrix(x, y, order)
    keep = np.ones(len(xi), dtype=bool)

    for _ in range(n_iter):
        # Least squares on the kept stars.
        cx, *_ = np.linalg.lstsq(A[keep], xi[keep], rcond=None)
        cy, *_ = np.linalg.lstsq(A[keep], eta[keep], rcond=None)
        rx, ry = xi - A @ cx, eta - A @ cy

        # Robust per-axis sigma from the residual radius (2D Gaussian).
        rr = np.hypot(rx, ry)
        sig = np.median(rr[keep]) / np.sqrt(2. * np.log(2.))
        new_keep = rr < clip * sig
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    # Pixel scale from the linear terms (coefficients are per `half` pixels).
    J = np.array([[cx[1], cx[2]], [cy[1], cy[2]]]) / 2048.
    scale_mas = float(np.sqrt(abs(np.linalg.det(J))) * 1000.)

    fit = {'cx': cx, 'cy': cy, 'order': int(order), 'ra0': float(ra0), 'dec0': float(dec0),
           'keep': keep, 'n_ref': int(len(xi)), 'n_used': int(keep.sum()),
           'rms_x_mas': float(np.std(rx[keep]) * 1000.),
           'rms_y_mas': float(np.std(ry[keep]) * 1000.), 'scale_mas': scale_mas}

    return fit


def apply_pixel_to_sky(fit, x, y):
    """
    Map pixel positions to RA/Dec with a `fit_pixel_to_sky` result.

    Parameters
    ----------
    fit : dict
    x, y : array_like

    Returns
    -------
    ra, dec : numpy arrays (deg)
    """
    A = design_matrix(x, y, fit['order'])
    ra, dec = tan_to_radec(A @ fit['cx'], A @ fit['cy'], fit['ra0'], fit['dec0'])

    return ra, dec


def pixel_of_radec(fit, ra, dec, center=2048., half=2048.):
    """
    Pixel position of one sky coordinate in an order-1 pixel-to-sky fit.

    Parameters
    ----------
    fit : dict
        Result of `fit_pixel_to_sky`. ``order`` must be 1.
    ra, dec : float
        Sky position (deg).
    center, half : float, optional
        Same scaling as `design_matrix`.

    Returns
    -------
    x, y : float
        Pixel position in the same frame as the fit.
    """
    if int(fit['order']) != 1:
        raise ValueError('pixel_of_radec inverts an order-1 fit only')

    xi, eta = radec_to_tan(ra, dec, fit['ra0'], fit['dec0'])
    cx = np.asarray(fit['cx'], dtype=float)
    cy = np.asarray(fit['cy'], dtype=float)

    # xi = c0 + c1 xs + c2 ys, xs = (x - center) / half.
    A = np.array([[cx[1], cx[2]], [cy[1], cy[2]]])
    xs, ys = np.linalg.solve(A, [float(np.asarray(xi)) - cx[0],
                                  float(np.asarray(eta)) - cy[0]])
    x = float(xs) * half + center
    y = float(ys) * half + center

    return x, y


def mutual_match(xi1, eta1, xi2, eta2, radius):
    """
    One-to-one (mutual nearest neighbour) match within a radius.

    Parameters
    ----------
    xi1, eta1, xi2, eta2 : numpy arrays
        Positions (arcsec) of list 1 and list 2.
    radius : float
        Match radius (arcsec).

    Returns
    -------
    i1, i2 : numpy arrays of int
        Indices of matched pairs into list 1 and list 2.
    """
    p1 = np.column_stack([xi1, eta1])
    p2 = np.column_stack([xi2, eta2])

    # Nearest neighbour in each direction.
    d12, j12 = cKDTree(p2).query(p1, distance_upper_bound=radius)
    _, j21 = cKDTree(p1).query(p2, distance_upper_bound=radius)

    # Keep pairs that are each other's nearest neighbour.
    i1 = np.where(np.isfinite(d12))[0]
    i2 = j12[i1]
    mutual = j21[i2] == i1
    i1, i2 = i1[mutual], i2[mutual]

    return i1, i2


def match_with_offset(xi_j, eta_j, xi_m, eta_m, radius, coarse_radius=1.0):
    """
    Match two lists after removing their bulk offset.

    Parameters
    ----------
    xi_j, eta_j : numpy arrays
        List 1 (e.g. Jay) positions in arcsec.
    xi_m, eta_m : numpy arrays
        List 2 (e.g. MAST) positions in arcsec.
    radius : float
        Final match radius (arcsec).
    coarse_radius : float, optional
        Radius for the coarse match used to measure the bulk offset.

    Returns
    -------
    ij, im : numpy arrays of int
        Matched indices into list 1 and list 2.
    offset : tuple of float
        Bulk offset (list 2 - list 1) in arcsec, (dxi, deta).
    n_coarse : int
        Pairs in the coarse match.
    """
    # Coarse match -> median offset.
    ij, im = mutual_match(xi_j, eta_j, xi_m, eta_m, coarse_radius)
    if len(ij) == 0:
        return ij, im, (np.nan, np.nan), 0
    offset = (float(np.median(xi_m[im] - xi_j[ij])), float(np.median(eta_m[im] - eta_j[ij])))
    n_coarse = int(len(ij))

    # Final match after shifting list 1 onto list 2.
    ij, im = mutual_match(xi_j + offset[0], eta_j + offset[1], xi_m, eta_m, radius)

    return ij, im, offset, n_coarse
