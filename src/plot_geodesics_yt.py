#!/usr/bin/env python
"""
Volume-render GRMHD density with geodesic trajectories overlaid using yt.

Usage:
    python plot_geodesics_yt.py [dump] [options]

Arguments:
    dump                GRMHD HDF5 dump file (default: data/dump_SANE.h5)
    --geodesics PATH    Geodesic trajectory HDF5 file (default: output/geodesics.h5)
    --n N               Number of geodesics to plot (default: 50)
    --r-max R           Max radius for visualization clipping in r_g (default: 50.0)
    --resolution N      Cartesian grid resolution per axis (default: 256)
    --no-geodesics      Render density only, skip geodesic lines
    --output PATH       Output image path (default: geodesics_yt.png)
    --cam-position X Y Z
                        Camera position in r_g (default: along +x at 2*r_max)

What it does:
    Reads iharm3d GRMHD density on its native MKS grid, interpolates onto a
    uniform Cartesian grid via scipy trilinear interpolation, loads into yt as
    a uniform Cartesian dataset, and volume-renders log10(density). Geodesic
    lines are overlaid as LineSource objects in fluorescent cyan. Output is a
    static PNG image.
"""

import argparse
import sys
import numpy as np
import h5py
from scipy.ndimage import map_coordinates
import yt
from yt.visualization.volume_rendering.api import LineSource


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
    log_rho : ndarray, shape (N_cart, N_cart, N_cart)
        log10(rho) with a tiny floor for masked/zero regions.
    bbox : list of [min, max] pairs for each axis
    """
    gp = grid_params
    hslope = gp["hslope"]

    # Smallest physical radius in the grid
    r_in = np.exp(gp["startx1"] + 0.5 * gp["dx1"])

    # Build Cartesian sampling grid
    xyz = np.linspace(-r_max, r_max, N_cart)
    Xc, Yc, Zc = np.meshgrid(xyz, xyz, xyz, indexing="ij")

    # Cartesian → spherical
    r_c   = np.sqrt(Xc**2 + Yc**2 + Zc**2)
    theta_c = np.arccos(np.clip(Zc / np.where(r_c > 0, r_c, 1e-30), -1.0, 1.0))
    phi_c   = np.arctan2(Yc, Xc) % (2.0 * np.pi)

    # Spherical → MKS fractional indices
    X1_c = np.log(np.where(r_c > 0, r_c, 1e-30))
    i_frac = (X1_c - gp["startx1"]) / gp["dx1"] - 0.5

    # Invert theta → X2 via lookup table (theta monotonic in X2 on [0,1])
    X2_table    = np.linspace(0.0, 1.0, 10000)
    theta_table = np.pi * X2_table + 0.5 * (1.0 - hslope) * np.sin(2.0 * np.pi * X2_table)
    X2_c  = np.interp(theta_c, theta_table, X2_table)
    j_frac = (X2_c - gp["startx2"]) / gp["dx2"] - 0.5

    X3_c  = phi_c
    k_frac = (X3_c - gp["startx3"]) / gp["dx3"] - 0.5

    # Trilinear interpolation
    coords = np.array([i_frac.ravel(), j_frac.ravel(), k_frac.ravel()])
    rho_cart = map_coordinates(rho, coords, order=1, mode="nearest").reshape(N_cart, N_cart, N_cart)

    # Mask exterior/interior points
    floor = 1e-30
    mask = (r_c < r_in) | (r_c > r_max)
    rho_cart[mask] = floor
    rho_cart = np.maximum(rho_cart, floor)

    # Apply a small floor to avoid log(0), but keep raw density for yt
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


def geodesics_to_line_segments(r_all, th_all, ph_all, nsteps, idx, r_max):
    """
    Convert geodesic data to yt LineSource segment arrays.

    Returns
    -------
    segments : ndarray, shape (N_segments, 2, 3)
    colors   : ndarray, shape (N_segments, 4)  RGBA fluorescent cyan
    """
    seg_list = []
    for i in idx:
        ns = nsteps[i]
        if ns < 2:
            continue
        r  = r_all[i, :ns]
        th = th_all[i, :ns]
        ph = ph_all[i, :ns]
        x, y, z = bl_to_cartesian(r, th, ph)
        pts = np.stack([x, y, z], axis=1)  # (ns, 3)

        # Clip at r_max: keep only steps inside the volume
        inside = r <= r_max
        # Walk along trajectory; emit a segment whenever both endpoints are inside
        for s in range(len(pts) - 1):
            if inside[s] and inside[s + 1]:
                seg_list.append([pts[s], pts[s + 1]])

    if not seg_list:
        return None, None

    segments = np.array(seg_list, dtype=np.float64)   # (N, 2, 3)
    colors   = np.tile([0.0, 1.0, 0.88, 0.6], (len(seg_list), 1))
    return segments, colors


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def build_scene(ds, segments, colors, cam_position, r_max):
    """Volume-render density (log-scaled by yt) and optionally overlay geodesic lines."""
    from yt.visualization.volume_rendering.transfer_functions import ColorTransferFunction

    ad = ds.all_data()
    rho_all = np.array(ad[("stream", "density")])
    rho_max = float(rho_all.max())
    # Use the 1st percentile of non-floor values for the lower bound so the TF
    # focuses on actual density structure rather than the masked-region floor.
    floor = 1e-30
    valid = rho_all[rho_all > 10 * floor]
    rho_min_valid = float(valid.min()) if valid.size > 0 else 1e-10
    log_min = np.log10(rho_min_valid)
    log_max = np.log10(max(rho_max, 1e-30))
    print(f"  density range: [{rho_min_valid:.3e}, {rho_max:.3e}]  "
          f"log10 range: [{log_min:.2f}, {log_max:.2f}]")

    # Layered TF: N gaussians across log-density range reveal turbulent structure
    tf = ColorTransferFunction((log_min, log_max))
    tf.add_layers(N=30, colormap="inferno", alpha=[0.3]*30, w=0.1)

    # create_scene sets up camera, resolution, and a default TF; render in log space
    sc = yt.create_scene(ds, ("stream", "density"))
    vol = sc[0]
    vol.set_log(True)
    vol.transfer_function = tf

    # Overlay geodesic lines
    if segments is not None:
        lines = LineSource(segments, colors)
        sc.add_source(lines)

    # Camera: set position, focus, and north_vector before switch_orientation so
    # the "up" direction is always z (the disk rotation axis), preventing tilt.
    cam = sc.camera
    cam.set_width(ds.arr(2.0 * r_max, "code_length"))
    cam.position = ds.arr(cam_position, "code_length")
    cam.focus    = ds.arr([0.0, 0.0, 0.0], "code_length")
    cam.north_vector = np.array([0.0, 0.0, 1.0])
    cam.switch_orientation()
    cam.set_resolution((800, 800))

    return sc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dump", nargs="?", default="data/dump_SANE.h5",
                        help="GRMHD HDF5 dump file (default: data/dump_SANE.h5)")
    parser.add_argument("--geodesics", default="output/geodesics.h5",
                        help="Geodesic trajectory file (default: output/geodesics.h5)")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of geodesics to plot (default: 50)")
    parser.add_argument("--r-max", type=float, default=50.0,
                        help="Max radius for visualization clipping in r_g (default: 50.0)")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Cartesian grid resolution per axis (default: 256)")
    parser.add_argument("--no-geodesics", action="store_true",
                        help="Render density only, skip geodesic lines")
    parser.add_argument("--output", default="geodesics_yt.png",
                        help="Output image path (default: geodesics_yt.png)")
    parser.add_argument("--cam-position", nargs=3, type=float,
                        metavar=("X", "Y", "Z"), default=None,
                        help="Camera position in r_g (default: (2*r_max, 0, 0))")
    args = parser.parse_args()

    r_max = args.r_max
    cam_position = args.cam_position if args.cam_position else [2.0 * r_max, 0.0, 0.0]

    # --- Load and interpolate density ---
    print(f"Loading GRMHD density from {args.dump} ...")
    rho, grid_params = load_grmhd_density(args.dump, r_max)

    print(f"Interpolating to {args.resolution}^3 Cartesian grid ...")
    rho_cart, bbox = interpolate_to_cartesian(rho, grid_params, r_max, args.resolution)

    # --- Build yt dataset ---
    data = {"density": (rho_cart, "g/cm**3")}   # units are nominal; we use code-unit density
    bbox_arr = np.array(bbox)
    ds = yt.load_uniform_grid(
        data,
        rho_cart.shape,
        length_unit="code_length",
        bbox=bbox_arr,
        periodicity=(False, False, False),
    )

    # --- Load geodesics ---
    segments = colors = None
    if not args.no_geodesics:
        print(f"Loading geodesics from {args.geodesics} ...")
        try:
            r_all, th_all, ph_all, nsteps, idx = load_geodesics(args.geodesics, args.n)
            segments, colors = geodesics_to_line_segments(
                r_all, th_all, ph_all, nsteps, idx, r_max
            )
            if segments is not None:
                print(f"  {len(segments)} line segments from {len(idx)} geodesics")
            else:
                print("  No valid segments found; rendering density only.")
        except FileNotFoundError:
            print(f"Warning: geodesic file not found ({args.geodesics}), skipping.")

    # --- Render ---
    print("Building yt scene and rendering ...")
    sc = build_scene(ds, segments, colors, cam_position, r_max)
    sc.save(args.output, sigma_clip=4.0)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
