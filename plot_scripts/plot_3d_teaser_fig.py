"""Simulate coded depth maps of the face scan for each coding scheme.

Ground-truth depth comes from the peak of each pixel's raw histogram: the bin
with the most photon counts gives the time of flight, and depth = c * t / 2.
Those depths then replace the depth sweep in test.py: every pixel is simulated
at its scan depth, Poisson noise is added, and the result is decoded with ZNCC.

The figure has two rows: Open3D Poisson-mesh renders of each depth map on top (rendered
straight into the figure instead of saved as PNG screenshots, unlike
3d_recon_single.py), and each scheme's gates and illumination pulse below.
"""
import os
import sys
sys.path.append('..')
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.ndimage import median_filter

from utils.tof_utils import (
    get_code,
    simulate_counts,
    simulate_counts_shared_illum,
    decode_simulation_depths,
    calculate_tof_domain_params, scale_photon_count,
)

# =============================================================================
# Scan parameters
# =============================================================================
c = 3e8          # speed of light (m/s)
t_res = 8e-12    # histogram bin width (s)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SCAN_DIR = os.path.join(DATA_DIR, "20181105_face", "opt_flux")
HIST_PATH = os.path.join(SCAN_DIR, "raw-hist-img_r-240-c-150_tres-8ps_tlen-100000ps.npy")
GT_DEPTH_CACHE = os.path.join(SCAN_DIR, "gt-depth-img_r-240-c-150.npy")

DEPTH_VMIN, DEPTH_VMAX = 6.1, 6.3   # colour range for the depth maps (m)

# =============================================================================
# 3-D render parameters
# =============================================================================
FOV_WIDTH_DEG = 2.0      # field of view across the image width (degrees)
RENDER_W, RENDER_H = 800, 1000   # render size in pixels
VIEW_AZIM_DEG = 30.0     # default view: rotation around the vertical axis
VIEW_ELEV_DEG = 10.0     # default view: rotation above the horizontal
VIEW_ZOOM     = 0.7
SHOW_RENDER_WINDOWS = False   # set True if the hidden-window renders come out blank
CROP_PAD_PX     = 10     # white margin kept around the mesh when cropping the renders
CROP_MIN_PIXELS = 5      # rows/cols with fewer mesh pixels than this are cropped as specks

# --- Poisson mesh reconstruction ---
MEDIAN_FILTER_SIZE = 5   # median filter on each depth map before meshing (1 = off)
POISSON_DEPTH  = 8       # octree depth (7=smoother/faster, 9=more detail/slower)
SMOOTH_ITERS   = 5       # Laplacian smoothing iterations
NORMAL_RADIUS_PX = 5     # neighbourhood for normal estimation, in pixel spacings
DENSITY_TRIM_QUANTILE = 0.05   # fraction of lowest-density vertices to trim

# Set PICK_VIEW = True to open the ground-truth mesh in a viewer first. Press S
# there to save the camera view to CAMERA_PARAMS_FILE, then close the window.
# Every render uses the saved view whenever that file exists.
PICK_VIEW = False
CAMERA_PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "camera_params_teaser.json")
# =============================================================================
# Plot Parameters
# =============================================================================

FONT_SIZE = 22        # paper-sized text
GATE_LW   = 2.5
PULSE_LW  = 4
GATE_FILLS = ("0.75", "0.9")   # alternating gate shading (gray levels)
GATE_START_MARKER, GATE_START_COLOR, GATE_START_MS = "^", "tab:red", 9   # sliding-gate start ticks
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE - 2,
    'legend.fontsize': FONT_SIZE - 2,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'xtick.major.size': 6,
})

COL_W, GATE_H = 5, 3.2   # inches per column / height of the gate row

# =============================================================================
# Simulation parameters (same meaning as in test.py)
# =============================================================================
N_TBINS      = 10000
TRIALS       = 1        # one noisy capture per pixel
PHOTON_COUNT = 400_000
SBR          = 100.0
SPLIT_ACQUISITION = True
REP_RATE = 10e6         # 100 ns period, matches the scan
REP_TAU  = float(1 / REP_RATE)

K_0 = 8
PULSE_WIDTH = int((N_TBINS // (K_0)) / (2 * np.sqrt(np.log(2))))

print(PULSE_WIDTH)

RUNS = [
    # {'type': 'ham',        'k': 4, 'photon_count': PHOTON_COUNT, 'pulse_width': PULSE_WIDTH},
    # {'type': 'coarse',     'k': 4, 'photon_count': PHOTON_COUNT},
    # {'type': 'trapcoarse', 'k': 4, 'photon_count': PHOTON_COUNT},
    {'type': 'coarsepw', 'k': 8, 'photon_count': PHOTON_COUNT, 'pulse_width': PULSE_WIDTH ,  'simulated': True},
    #{'type': 'coarsepw', 'k': 12, 'photon_count': PHOTON_COUNT, 'pulse_width': PULSE_WIDTH, 'simulated': True},
    {'type': 'coarsepw', 'k': 16, 'photon_count': PHOTON_COUNT, 'pulse_width': PULSE_WIDTH, 'simulated': True},
    {'type': 'sliding', "pulse_width": PULSE_WIDTH, "shift": 300,  'photon_count': PHOTON_COUNT, "gate_width": N_TBINS // 2,  'simulated': True},
    {'type': 'sliding', "pulse_width": PULSE_WIDTH, "shift": 100,  'photon_count': PHOTON_COUNT, "gate_width": N_TBINS // 2, 'simulated': True},

]

# =============================================================================
# Ground-truth depth from the histogram peak
# =============================================================================
if os.path.isfile(GT_DEPTH_CACHE):
    gt_depth = np.load(GT_DEPTH_CACHE)
else:
    # (rows, cols, time bins); memory-mapped because the file is 3.6 GB
    hist = np.load(HIST_PATH, mmap_mode="r")
    print("histogram cube:", hist.shape, hist.dtype)
    peak_bin = np.argmax(hist, axis=-1)
    gt_depth = (c / 2) * peak_bin * t_res   # metres
    np.save(GT_DEPTH_CACHE, gt_depth)

# =============================================================================
# ToF domain
# =============================================================================
(rep_tau, rep_freq, tbin_res,
 t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(N_TBINS, REP_TAU)

depths = gt_depth.ravel()

# Coded values only depend on a pixel's time bin, so simulate each distinct
# bin once and look the pixels up, instead of simulating every pixel.
pixel_tbins = (depths / tbin_depth_res).astype(int) % N_TBINS
unique_tbins, pixel_to_unique = np.unique(pixel_tbins, return_inverse=True)
unique_depths = (unique_tbins + 0.5) * tbin_depth_res

# =============================================================================
# Run each scheme
# =============================================================================
results = []

for run in RUNS:
    cap_type  = run['type']
    k  = run.get('k', None)
    pulse_width  = run.get('pulse_width', None)
    gate_width = run.get('gate_width', None)
    shift = run.get('shift', None)
    k = int(N_TBINS / shift) if k is None else k

    photon_count_base = run['photon_count']
    photon_count = scale_photon_count(photon_count_base, cap_type, k) if SPLIT_ACQUISITION else photon_count_base // k

    label  = f"{cap_type}_k{k}" + (f"_pw{pulse_width}" if pulse_width is not None else "")

    print(f"--- {label} ---")

    # get_code returns (modfs_or_illum, demodfs_or_coding_matrix_T, coding_matrix)
    waveform_or_illum, demodfs_or_cm, coding_matrix = get_code(cap_type, k, N_TBINS,
                                                               pulse_width=pulse_width,
                                                               shift=shift,
                                                               gate_width=gate_width)

    if cap_type == 'ham':
        _, unique_coded_values = simulate_counts(
            waveform=waveform_or_illum,
            demodfs=demodfs_or_cm,
            depths=unique_depths,
            photon_count=photon_count,
            sbr=SBR,
            tbin_depth_res=tbin_depth_res,
            n_tbins=N_TBINS,
            k=k,
        )
    else:
        _, unique_coded_values = simulate_counts_shared_illum(
            illum=waveform_or_illum,
            coding_matrix=demodfs_or_cm,
            depths=unique_depths,
            photon_count=photon_count,
            sbr=SBR,
            tbin_depth_res=tbin_depth_res,
            n_tbins=N_TBINS,
            k=k,
        )

    coded_values = unique_coded_values[pixel_to_unique]

    decoded_depth, rmse, mae = decode_simulation_depths(
        coding_matrix=coding_matrix,
        coded_values=coded_values,
        depths=depths,
        trials=TRIALS,
        tbin_depth_res=tbin_depth_res,
    )

    print(f"  MAE={mae * 1000:.3f} mm  |  RMSE={rmse * 1000:.3f} mm\n")

    results.append({
        'name':      label,
        'depth_map': decoded_depth[0].reshape(gt_depth.shape),
        'illum':     waveform_or_illum,   # (n_tbins,) or (n_tbins, k) for ham
        'gates':     demodfs_or_cm,       # (n_tbins, k)
        'rmse':      rmse * 1000,
        'mae':       mae  * 1000,
        'k': k,
        'type': cap_type,
    })

# =============================================================================
# 3-D reconstruction
# =============================================================================
def depth_to_mesh(depth_map, centroid=None):
    """Back-project a depth map and reconstruct a Poisson mesh centred on `centroid`.

    z is negated so the mesh is right-handed and Open3D's default camera
    looks at it from the sensor side, unmirrored.
    """
    height, width = depth_map.shape
    fx = width / (2 * np.tan(np.radians(FOV_WIDTH_DEG) / 2))
    cx, cy = width / 2.0, height / 2.0

    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_map
    points = np.stack(((x_grid - cx) * z / fx, -(y_grid - cy) * z / fx, -z), axis=-1).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]

    # centre at origin so open3d zoom/pan works correctly
    if centroid is None:
        centroid = points.mean(axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points - centroid)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # the sensor sits at the original origin, so every normal should face it
    pixel_spacing = np.nanmedian(z) / fx
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_PX * pixel_spacing, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(-centroid)

    # n_threads=1: the multi-threaded solver can die with "Failed to close loop"
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH, n_threads=1)

    # Poisson closes the surface into a blob; trim low-density vertices and
    # anything outside the points' bounding box to get back the open face
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, DENSITY_TRIM_QUANTILE))
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())

    if SMOOTH_ITERS > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=SMOOTH_ITERS, lambda_filter=0.5)
    mesh.remove_non_manifold_edges()

    vertex_depth = -(np.asarray(mesh.vertices)[:, 2] + centroid[2])
    z_norm = np.clip((vertex_depth - DEPTH_VMIN) / (DEPTH_VMAX - DEPTH_VMIN), 0, 1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(plt.get_cmap("gray")(0.3 + 0.4 * z_norm)[:, :3])  # stay in mid-gray range
    mesh.compute_vertex_normals()
    print(f"  mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    return mesh, centroid


def load_camera_params():
    if os.path.isfile(CAMERA_PARAMS_FILE):
        return o3d.io.read_pinhole_camera_parameters(CAMERA_PARAMS_FILE)
    return None


def apply_view(vis, camera_params):
    vis.get_render_option().mesh_show_back_face = True
    ctrl = vis.get_view_control()
    if camera_params is not None:
        ctrl.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
    else:
        azim, elev = np.radians(VIEW_AZIM_DEG), np.radians(VIEW_ELEV_DEG)
        ctrl.set_lookat([0.0, 0.0, 0.0])
        ctrl.set_front([np.sin(azim) * np.cos(elev), np.sin(elev), np.cos(azim) * np.cos(elev)])
        ctrl.set_up([0.0, 1.0, 0.0])
        ctrl.set_zoom(VIEW_ZOOM)


def window_size(camera_params):
    # a saved view only reproduces exactly in a window of the size it was saved at
    if camera_params is not None:
        return camera_params.intrinsic.width, camera_params.intrinsic.height
    return RENDER_W, RENDER_H


def pick_view(geometry):
    """Show `geometry` in an interactive viewer; S saves the camera view."""
    camera_params = load_camera_params()
    width, height = window_size(camera_params)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Pick view", width=width, height=height)
    vis.add_geometry(geometry)
    apply_view(vis, camera_params)

    def save_camera(vis):
        params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(CAMERA_PARAMS_FILE, params)
        print(f"Camera saved to {CAMERA_PARAMS_FILE}")
        return False

    vis.register_key_callback(ord("S"), save_camera)
    print(f"Press S to save the current view -> {CAMERA_PARAMS_FILE}, then close the window")
    vis.run()
    vis.destroy_window()


def render_geometry(geometry, title="Open3D"):
    """Render `geometry` with Open3D and return the image as an (H, W, 3) array."""
    camera_params = load_camera_params()
    width, height = window_size(camera_params)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=width, height=height, visible=SHOW_RENDER_WINDOWS)
    vis.add_geometry(geometry)
    apply_view(vis, camera_params)
    vis.poll_events()
    vis.update_renderer()
    image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    return image


# Background pixels (ground truth outside the depth range) are dropped from
# every mesh; decoded outliers on the face are clipped to the range.
face_mask = (gt_depth >= DEPTH_VMIN) & (gt_depth <= DEPTH_VMAX)

def face_only(depth_map):
    depth_map = median_filter(depth_map, size=MEDIAN_FILTER_SIZE)
    return np.where(face_mask, np.clip(depth_map, DEPTH_VMIN, DEPTH_VMAX), np.nan)

# every mesh is centred on the ground-truth centroid so the views line up
gt_mesh, centroid = depth_to_mesh(face_only(gt_depth))

if PICK_VIEW:
    pick_view(gt_mesh)

for res in results:
    mesh, _ = depth_to_mesh(face_only(res['depth_map']), centroid)
    res['render'] = render_geometry(mesh, title=res['name'])


def content_bbox(images, min_pixels=CROP_MIN_PIXELS, pad=CROP_PAD_PX):
    """Bounding box of the non-background pixels shared by all `images`.

    Rows/columns with fewer than `min_pixels` mesh pixels are ignored so stray
    specks don't stretch the box.
    """
    mask = np.any([np.any(img < 0.99, axis=-1) for img in images], axis=0)
    rows = np.flatnonzero(mask.sum(axis=1) >= min_pixels)
    cols = np.flatnonzero(mask.sum(axis=0) >= min_pixels)
    height, width = mask.shape
    return (max(rows[0] - pad, 0), min(rows[-1] + pad + 1, height),
            max(cols[0] - pad, 0), min(cols[-1] + pad + 1, width))

# crop the white margins around the meshes, using one box for every render so
# the views stay aligned
r0, r1, c0, c1 = content_bbox([res['render'] for res in results])
for res in results:
    res['render'] = res['render'][r0:r1, c0:c1]
render_aspect = (r1 - r0) / (c1 - c0)   # height / width of the cropped renders

# =============================================================================
# Plot
# =============================================================================
render_h = COL_W * render_aspect   # top row sized to the cropped render, no letterboxing
fig, axes = plt.subplots(2, len(results), figsize=(COL_W * len(results), render_h + GATE_H + 1),
                         squeeze=False, gridspec_kw={'height_ratios': [render_h, GATE_H]},
                         constrained_layout=True)

t_ns = t_domain * 1e9

def circular_center(waveform):
    """Centre of mass of a periodic waveform, in time bins (handles wrap-around)."""
    phase = 2 * np.pi * np.arange(N_TBINS) / N_TBINS
    return (np.angle(np.sum(waveform * np.exp(1j * phase))) % (2 * np.pi)) * N_TBINS / (2 * np.pi)

first_sliding_col = next((i for i, r in enumerate(results) if r['type'] == 'sliding'), None)

for col, res in enumerate(results):
    axes[0, col].imshow(res['render'])
    # if res['k'] == K_0:
    #     title = 'Less Gates'
    # else:
    #     title = 'More Gates'
    title = f"Gate Count: {res['k']}"
    axes[0, col].set_title(f"{title}\nMAE {res['mae']:.1f} mm", fontsize=30)

    ax = axes[1, col]
    gates = res['gates']
    tau_ns = rep_tau * 1e9
    pulse_center = circular_center(res['illum'].reshape(N_TBINS, -1).sum(axis=1))
    if res['type'] == 'sliding':
        # the many overlapping gates are unreadable over a full period: zoom
        # into the middle quarter period and stack the gates that open inside
        # it, so the shift between consecutive gates shows as a staircase
        lo, hi = 3 * N_TBINS // 8, 5 * N_TBINS // 8
        rising = np.diff(gates[lo - 1:hi], axis=0) > 0
        opening = np.flatnonzero(rising.any(axis=0))
        opening = opening[np.argsort(rising[:, opening].argmax(axis=0))]
        height = len(opening)
        for row, j in enumerate(opening):
            ax.fill_between(t_ns[lo:hi], row, row + 0.8 * gates[lo:hi, j],
                            color=GATE_FILLS[row % 2], lw=0)
            ax.plot(t_ns[lo:hi], row + 0.8 * gates[lo:hi, j], color="k", lw=GATE_LW)
            # mark where each gate opens, i.e. one shift step of the sweep
            t_start = t_ns[lo + rising[:, j].argmax()]
            ax.plot(t_start, row, marker=GATE_START_MARKER, color=GATE_START_COLOR,
                    ms=GATE_START_MS, zorder=4, clip_on=False)
        # the pulse is emitted at t = 0; shift it to the middle of the window
        illum = np.roll(res['illum'], int(round(N_TBINS / 2 - pulse_center)), axis=0)
        ax.plot(t_ns[lo:hi], height * illum[lo:hi] / illum.max(), color="blue", lw=PULSE_LW)
        ax.set_xlim(3 * tau_ns / 8, 5 * tau_ns / 8)
        ax.set_xticks(tau_ns * np.array([3 / 8, 1 / 2, 5 / 8]),
                      labels=[r"$3\tau/8$", r"$\tau/2$", r"$5\tau/8$"])
    else:
        height = 1
        # shade every gate, alternating two grays, so neighbouring gates read as
        # separate windows rather than one row of vertical lines
        for j in range(gates.shape[1]):
            ax.fill_between(t_ns, 0, gates[:, j], color=GATE_FILLS[j % 2], lw=0)
        ax.plot(t_ns, gates, color="k", lw=GATE_LW)
        # the pulse is emitted at t = 0 and wraps around the period; shift it onto
        # the centre of the gate nearest the middle so it shows as one pulse
        # sitting in the middle of a gate
        gate_centers = np.array([circular_center(gates[:, j]) for j in range(gates.shape[1])])
        target = gate_centers[np.argmin(np.abs(gate_centers - N_TBINS / 2))]
        illum = np.roll(res['illum'], int(round(target - pulse_center)), axis=0)
        ax.plot(t_ns, illum / illum.max(), color="blue", lw=PULSE_LW)
        # one laser period, labelled in units of tau instead of nanoseconds
        ax.set_xlim(0, tau_ns)
        ax.set_xticks(tau_ns * np.array([0, 0.25, 0.5, 0.75, 1]),
                      labels=["0", r"$\tau/4$", r"$\tau/2$", r"$3\tau/4$", r"$\tau$"])

    if col == 0:
        ax.plot([], [], color="blue", lw=PULSE_LW, label="Illum.")
        ax.fill_between([], [], [], facecolor=GATE_FILLS[0], edgecolor="k", lw=GATE_LW / 2,
                        label="Gates")
    # the gate-start key goes on the first sliding panel only
    if res['type'] == 'sliding' and col == first_sliding_col:
        ax.plot([], [], ls="none", marker=GATE_START_MARKER, color=GATE_START_COLOR,
                ms=GATE_START_MS, label="Gate start")
    labels = ax.get_legend_handles_labels()[1]
    if labels == ["Gate start"]:
        # marker-only entry: shrink the handle box (sized for line samples) so the
        # triangle sits next to its text
        ax.legend(frameon=False, loc="upper left", handlelength=0.8, handletextpad=0.4)
    elif labels:
        ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.set_ylim(0, 1.5 * height)   # headroom for the legend
    ax.set_xlabel("Time")
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)

for ax in axes[0]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
plt.subplots_adjust(left=0.05, right=0.05, top=0.05, bottom=0.05)
plt.savefig("figures/3d_teaser_fig.pdf", bbox_inches="tight")
plt.show()
