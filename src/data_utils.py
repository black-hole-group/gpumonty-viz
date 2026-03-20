"""
Pure-NumPy/scipy/h5py data utilities shared across all visualization scripts.

No yt dependency — safe to import in PyVista-only scripts and Jupyter notebooks.
"""

import glob
import os
import shutil
import subprocess

import h5py
import numpy as np
from scipy.ndimage import map_coordinates


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def bl_to_cartesian(r, theta, phi):
    """Convert Boyer-Lindquist spherical coords to Cartesian."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def mks_to_bl(X1, X2, X3, hslope):
    """Convert MKS coordinates to Boyer-Lindquist (r, theta, phi)."""
    r = np.exp(X1)
    theta = np.pi * X2 + 0.5 * (1.0 - hslope) * np.sin(2.0 * np.pi * X2)
    phi = X3
    return r, theta, phi


# ---------------------------------------------------------------------------
# GRMHD data loading
# ---------------------------------------------------------------------------

def load_grmhd_density(dump_path, r_max):
    """
    Read iharm3d HDF5 dump and return density array with grid parameters.

    Returns
    -------
    rho : ndarray, shape (n1, n2, n3)
    grid_params : dict with keys startx1, dx1, startx2, dx2, startx3, dx3,
                  hslope, a, n1, n2, n3
    """
    with h5py.File(dump_path, "r") as f:
        geom = f["header/geom"]
        mks  = geom["mks"]

        n1 = int(f["header/n1"][()])
        n2 = int(f["header/n2"][()])
        n3 = int(f["header/n3"][()])

        startx1 = float(geom["startx1"][()])
        startx2 = float(geom["startx2"][()])
        startx3 = float(geom["startx3"][()])
        dx1 = float(geom["dx1"][()])
        dx2 = float(geom["dx2"][()])
        dx3 = float(geom["dx3"][()])

        hslope = float(mks["hslope"][()])
        a      = float(mks["a"][()])

        # prims layout: [n1, n2, n3, nprim]; index 0 = RHO
        rho = f["prims"][:, :, :, 0].astype(np.float64)

    grid_params = dict(
        n1=n1, n2=n2, n3=n3,
        startx1=startx1, dx1=dx1,
        startx2=startx2, dx2=dx2,
        startx3=startx3, dx3=dx3,
        hslope=hslope, a=a,
    )
    return rho, grid_params


# ---------------------------------------------------------------------------
# Cartesian interpolation
# ---------------------------------------------------------------------------

def interpolate_to_cartesian(rho, grid_params, r_max, N_cart):
    """
    Interpolate GRMHD density from MKS grid onto a uniform Cartesian cube.

    Returns
    -------
    rho_cart : ndarray, shape (N_cart, N_cart, N_cart)
    bbox : list of [min, max] pairs for each axis
    """
    gp = grid_params
    hslope = gp["hslope"]

    r_in = np.exp(gp["startx1"] + 0.5 * gp["dx1"])
    a = gp["a"]
    r_h = 1.0 + np.sqrt(max(0.0, 1.0 - a**2))
    r_mask_inner = max(r_in, r_h)

    xyz = np.linspace(-r_max, r_max, N_cart)
    Xc, Yc, Zc = np.meshgrid(xyz, xyz, xyz, indexing="ij")

    r_c     = np.sqrt(Xc**2 + Yc**2 + Zc**2)
    theta_c = np.arccos(np.clip(Zc / np.where(r_c > 0, r_c, 1e-30), -1.0, 1.0))
    phi_c   = np.arctan2(Yc, Xc) % (2.0 * np.pi)

    X1_c  = np.log(np.where(r_c > 0, r_c, 1e-30))
    i_frac = (X1_c - gp["startx1"]) / gp["dx1"] - 0.5

    X2_table    = np.linspace(0.0, 1.0, 10000)
    theta_table = np.pi * X2_table + 0.5 * (1.0 - hslope) * np.sin(2.0 * np.pi * X2_table)
    X2_c  = np.interp(theta_c, theta_table, X2_table)
    j_frac = (X2_c - gp["startx2"]) / gp["dx2"] - 0.5

    X3_c  = phi_c
    k_frac = (X3_c - gp["startx3"]) / gp["dx3"] - 0.5

    coords = np.array([i_frac.ravel(), j_frac.ravel(), k_frac.ravel()])
    rho_cart = map_coordinates(rho, coords, order=1, mode="nearest").reshape(N_cart, N_cart, N_cart)

    floor = 1e-30
    mask = (r_c < r_mask_inner) | (r_c > r_max)
    rho_cart[mask] = floor
    rho_cart = np.maximum(rho_cart, floor)

    bbox = [[-r_max, r_max], [-r_max, r_max], [-r_max, r_max]]
    return rho_cart, bbox


# ---------------------------------------------------------------------------
# Geodesic loading
# ---------------------------------------------------------------------------

def load_geodesics(h5path, n_plot):
    """Load geodesic trajectories from GPUmonty HDF5 output."""
    with h5py.File(h5path, "r") as f:
        nph     = int(f["nph"][()])
        r_all   = f["r"][:]
        th_all  = f["theta"][:]
        ph_all  = f["phi"][:]
        nsteps  = f["nsteps"][:].astype(int)

    n_plot = min(n_plot, nph)
    idx = np.random.choice(nph, size=n_plot, replace=False)
    return r_all, th_all, ph_all, nsteps, idx


# ---------------------------------------------------------------------------
# Geodesic classification
# ---------------------------------------------------------------------------

def classify_geodesics(r_all, nsteps, idx, r_horizon):
    """
    Classify geodesic indices into escaped vs captured.

    A photon is captured if its final radius is < 3 * r_horizon (catches both
    horizon-crossers and near-trapped orbits). Everything else is escaped.

    Parameters
    ----------
    r_all : ndarray, shape (nph, max_steps)
    nsteps : ndarray, shape (nph,)
    idx : array-like
        Indices of geodesics to classify.
    r_horizon : float
        Event horizon radius r_h = 1 + sqrt(1 - a^2).

    Returns
    -------
    escaped_idx : ndarray
    captured_idx : ndarray
    """
    idx = np.asarray(idx)
    captured_mask = np.array([
        r_all[i, int(nsteps[i]) - 1] < 3.0 * r_horizon
        for i in idx
    ])
    return idx[~captured_mask], idx[captured_mask]


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------

def clean_frame_dir(path):
    """Remove all PNG files in *path* and ensure the directory exists."""
    os.makedirs(path, exist_ok=True)
    for f in glob.glob(os.path.join(path, "*.png")):
        os.remove(f)


def assemble_video(frame_dir, output_video, fps):
    """Assemble PNG frames into an H.264 MP4 via ffmpeg."""
    if shutil.which("ffmpeg") is None:
        print("Warning: ffmpeg not found. Skipping video assembly.")
        print(f"  Frames are in: {frame_dir}")
        return

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        output_video,
    ]
    print(f"Assembling video: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}")
    else:
        print(f"Saved video: {output_video}")
