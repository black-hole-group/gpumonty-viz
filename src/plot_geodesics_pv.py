#!/usr/bin/env python
"""
PyVista helper functions for interactive 3D visualization of GRMHD density
and geodesic trajectories.

Imports pure-NumPy data-loading functions from data_utils.py; only the
rendering layer is replaced with PyVista (VTK backend).

Jupyter interactive backend requires trame:
    pip install pyvista[jupyter]   # installs trame + trame-vuetify + trame-vtk
"""

import numpy as np
import pyvista as pv

from data_utils import bl_to_cartesian


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def rho_to_pyvista_grid(rho_cart, bbox, r_max):
    """
    Wrap a Cartesian density cube in a pyvista.ImageData object.

    Parameters
    ----------
    rho_cart : ndarray, shape (N, N, N)
        Raw density values (code units) on a uniform Cartesian grid.
    bbox : list of [min, max] pairs
        Physical extent along each axis, e.g. [[-r_max, r_max], ...].
    r_max : float
        Half-width of the box in r_g.

    Returns
    -------
    grid : pyvista.ImageData
        Contains point data field ``"log_density"`` = log10(max(rho, 1e-30)).
    """
    N = rho_cart.shape[0]
    spacing = 2.0 * r_max / N

    grid = pv.ImageData()
    grid.dimensions = (N, N, N)
    grid.origin = (-r_max, -r_max, -r_max)
    grid.spacing = (spacing, spacing, spacing)

    # VTK ImageData is x-varies-fastest (Fortran order); NumPy meshgrid with
    # indexing="ij" is C-order, so ravel with order="F" to match.
    log_rho = np.log10(np.maximum(rho_cart, 1e-30))
    grid.point_data["log_density"] = log_rho.ravel(order="F")

    return grid


# ---------------------------------------------------------------------------
# Geodesic polylines
# ---------------------------------------------------------------------------

def geodesics_to_polydata(r_all, th_all, ph_all, nsteps, idx, r_max, follow_idx=None):
    """
    Convert geodesic trajectories to PyVista PolyData polylines.

    Each geodesic is stored as a single connected polyline (not paired segments).
    Points outside ``r_max`` are dropped; geodesics with fewer than 2 valid points
    are skipped.

    Parameters
    ----------
    r_all, th_all, ph_all : ndarray, shape (nph, max_steps)
        Boyer-Lindquist coordinates for all geodesics.
    nsteps : ndarray, shape (nph,)
        Number of valid steps per geodesic.
    idx : array-like
        Indices of geodesics to include (subset of [0, nph)).
    r_max : float
        Clipping radius in r_g.
    follow_idx : int or None
        If given, this geodesic is returned as the gold PolyData (highlighted).

    Returns
    -------
    cyan_lines : pyvista.PolyData or None
    gold_lines : pyvista.PolyData or None
    """
    def _build_polydata(indices):
        """Build a single PolyData with one polyline per geodesic in *indices*."""
        all_points = []
        all_lines  = []
        pt_offset  = 0

        for i in indices:
            ns = nsteps[i]
            if ns < 2:
                continue

            r  = r_all[i, :ns]
            th = th_all[i, :ns]
            ph = ph_all[i, :ns]

            inside = r <= r_max
            # Extract contiguous inside-runs as separate polylines
            pts_xyz = np.stack(bl_to_cartesian(r, th, ph), axis=1)  # (ns, 3)

            # Build one polyline from the inside points (keep connectivity)
            valid_pts = pts_xyz[inside]
            n_valid = len(valid_pts)
            if n_valid < 2:
                continue

            all_points.append(valid_pts)
            # VTK polyline cell: [n_pts, idx0, idx1, ..., idx_{n-1}]
            line_cell = [n_valid] + list(range(pt_offset, pt_offset + n_valid))
            all_lines.extend(line_cell)
            pt_offset += n_valid

        if not all_points:
            return None

        points = np.vstack(all_points)
        poly = pv.PolyData()
        poly.points = points
        poly.lines  = np.array(all_lines, dtype=np.int_)
        return poly

    idx = np.asarray(idx)

    if follow_idx is not None:
        cyan_idx = idx[idx != follow_idx]
        gold_idx = np.array([follow_idx])
    else:
        cyan_idx = idx
        gold_idx = np.array([], dtype=int)

    cyan_lines = _build_polydata(cyan_idx) if len(cyan_idx) > 0 else None
    gold_lines = _build_polydata(gold_idx) if len(gold_idx) > 0 else None

    return cyan_lines, gold_lines


def geodesics_to_polydata_at_step(r_all, th_all, ph_all, nsteps, idx,
                                   r_max, step, follow_idx=None,
                                   captured_idx=None):
    """
    Like geodesics_to_polydata but truncates each geodesic at min(step, nsteps[i]).

    Parameters
    ----------
    r_all, th_all, ph_all : ndarray, shape (nph, max_steps)
    nsteps : ndarray, shape (nph,)
    idx : array-like
    r_max : float
    step : int
        Current reveal step; geodesics are clipped to this many points.
    follow_idx : int or None
    captured_idx : array-like or None
        If provided, these indices are rendered as a separate (crimson) group.

    Returns
    -------
    cyan_lines : pyvista.PolyData or None
    gold_lines : pyvista.PolyData or None
    captured_lines : pyvista.PolyData or None
    """
    def _build_polydata_at_step(indices):
        all_points = []
        all_lines  = []
        pt_offset  = 0

        for i in indices:
            ns = min(step, int(nsteps[i]))
            if ns < 2:
                continue

            r  = r_all[i, :ns]
            th = th_all[i, :ns]
            ph = ph_all[i, :ns]

            inside = r <= r_max
            pts_xyz = np.stack(bl_to_cartesian(r, th, ph), axis=1)

            valid_pts = pts_xyz[inside]
            n_valid = len(valid_pts)
            if n_valid < 2:
                continue

            all_points.append(valid_pts)
            line_cell = [n_valid] + list(range(pt_offset, pt_offset + n_valid))
            all_lines.extend(line_cell)
            pt_offset += n_valid

        if not all_points:
            return None

        points = np.vstack(all_points)
        poly = pv.PolyData()
        poly.points = points
        poly.lines  = np.array(all_lines, dtype=np.int_)
        return poly

    idx = np.asarray(idx)

    captured_set = set(captured_idx.tolist()) if captured_idx is not None and len(captured_idx) > 0 else set()

    if follow_idx is not None:
        remaining = idx[idx != follow_idx]
        gold_idx = np.array([follow_idx])
    else:
        remaining = idx
        gold_idx = np.array([], dtype=int)

    if captured_set:
        cyan_idx = remaining[~np.isin(remaining, list(captured_set))]
        cap_idx  = remaining[np.isin(remaining, list(captured_set))]
    else:
        cyan_idx = remaining
        cap_idx  = np.array([], dtype=int)

    cyan_lines     = _build_polydata_at_step(cyan_idx) if len(cyan_idx) > 0 else None
    gold_lines     = _build_polydata_at_step(gold_idx) if len(gold_idx) > 0 else None
    captured_lines = _build_polydata_at_step(cap_idx)  if len(cap_idx)  > 0 else None

    return cyan_lines, gold_lines, captured_lines


# ---------------------------------------------------------------------------
# Black hole sphere
# ---------------------------------------------------------------------------

def make_bh_sphere(a):
    """
    Return a PyVista sphere for the black hole event horizon.

    Parameters
    ----------
    a : float
        Dimensionless spin parameter (|a| <= 1).

    Returns
    -------
    pyvista.PolyData sphere at the origin with radius r_h = 1 + sqrt(1 - a^2).
    """
    r_h = 1.0 + np.sqrt(max(1.0 - a**2, 0.0))
    return pv.Sphere(radius=r_h, center=(0.0, 0.0, 0.0), theta_resolution=64, phi_resolution=64)


# ---------------------------------------------------------------------------
# Plotter builder
# ---------------------------------------------------------------------------

def build_pv_plotter(grid, cyan_lines, gold_lines, r_max, grid_params,
                     cam_position=None, density_render=True, opacity_multiplier=2,
                     show_scalar_bar=False):
    """
    Assemble a PyVista Plotter with volume rendering and geodesic overlays.

    Parameters
    ----------
    grid : pyvista.ImageData
        From :func:`rho_to_pyvista_grid`.
    cyan_lines, gold_lines : pyvista.PolyData or None
        From :func:`geodesics_to_polydata`.
    r_max : float
        Half-width of the scene in r_g (used for axis labels).
    grid_params : dict
        Must contain key ``"a"`` (black hole spin).
    cam_position : array-like of length 3, or None
        Camera position in r_g. Defaults to (2*r_max, 0, 0).
    density_render : bool
        If True, add inferno volume rendering of log-density.

    Returns
    -------
    plotter : pyvista.Plotter
    """
    if cam_position is None:
        cam_position = (2.0 * r_max, 0.0, 0.0)

    plotter = pv.Plotter(window_size=(1024, 1024))
    plotter.set_background("black")

    # --- Volume rendering ---
    if density_render:
        log_rho = grid.point_data["log_density"]
        floor = 1e-30
        # Bounds: ignore the masked floor (log10(1e-30) = -30)
        valid = log_rho[log_rho > np.log10(10 * floor)]
        log_min = float(valid.min()) if valid.size > 0 else -10.0
        log_max = float(log_rho.max())
        print(f"  log10 density range for TF: [{log_min:.2f}, {log_max:.2f}]")

        plotter.add_volume(
            grid,
            scalars="log_density",
            cmap="inferno",
            opacity="linear",
            opacity_unit_distance=grid.length / (np.mean(grid.dimensions) - 1) * opacity_multiplier,
            clim=[log_min, log_max],
            shade=False,
            show_scalar_bar=show_scalar_bar,
        )

    # --- Geodesic tubes ---
    CYAN = (0.0, 1.0, 0.88)   # fluorescent cyan
    GOLD = (1.0, 0.85, 0.0)

    if cyan_lines is not None:
        tubes = cyan_lines.tube(radius=0.035)
        plotter.add_mesh(tubes, color=CYAN, opacity=0.7, smooth_shading=True)

    if gold_lines is not None:
        tubes = gold_lines.tube(radius=0.07)
        plotter.add_mesh(tubes, color=GOLD, opacity=0.95, smooth_shading=True)

    # --- Black hole sphere ---
    a = grid_params.get("a", 0.0)
    bh = make_bh_sphere(a)
    plotter.add_mesh(bh, color="#111118", opacity=1.0, smooth_shading=True)

    # --- Camera ---
    plotter.camera.position    = tuple(cam_position)
    plotter.camera.focal_point = (0.0, 0.0, 0.0)
    plotter.camera.up          = (0.0, 0.0, 1.0)

    return plotter
