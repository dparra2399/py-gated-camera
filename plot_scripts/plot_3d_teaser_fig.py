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
CROP_PAD_PX     = -10     # white margin kept around the mesh when cropping the renders
CROP_MIN_PIXELS = 5      # rows/cols with fewer mesh pixels than this are cropped as specks

# --- Poisson mesh reconstruction ---
MEDIAN_FILTER_SIZE = 3   # median filter on each depth map before meshing (1 = off)
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
MAX_GATE_ROWS = 4   # gate rows per panel; more gates show the first few, ⋮, and the last
ROW_AMP = 0.7       # waveform height within its row (rows are spaced 1 apart)
X_MARGIN = 0.1     # baseline drawn past each end of the period, as a fraction of tau

# --- traffic-light labels under the gate count and MAE in each title ---
TRAFFIC_COLORS = ("#2ca02c", "#e6b800", "#d62728")   # green / yellow / red = good / medium / bad
BANDWIDTH_EDGES = (10, 80)   # K < 20 low, 20 <= K < 80 medium, K >= 80 high
BANDWIDTH_LABELS = ("Low Bandwidth", "Medium Bandwidth", "High Bandwidth")
# error labels are relative to the schemes shown: the lowest MAE is "low", the
# highest "high", the rest "medium"; schemes within 2 standard errors of the
# best (or worst) count as tied with it and share its label
ERROR_LABELS = ("Low Error", "Medium Error", "High Error")
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

COL_W, GATE_H = 5, 5     # inches per column / height of the gate row

# =============================================================================
# Simulation parameters (same meaning as in test.py)
# =============================================================================
N_TBINS      = 10000
TRIALS       = 100       # noisy captures averaged into the MAE/RMSE (decoded one at a time)
SEED         = 0        # noise seed, so the MAEs are reproducible run to run
PHOTON_COUNT = 400_000
SBR          = 100.0
SPLIT_ACQUISITION = True
REP_RATE = 10e6         # 100 ns period, matches the scan
REP_TAU  = float(1 / REP_RATE)

K_0 = 16
PULSE_WIDTH = int((N_TBINS // (K_0)) / (2 * np.sqrt(np.log(2))))

print(PULSE_WIDTH)

RUNS = [
    # {'type': 'ham',        'k': 4, 'photon_count': PHOTON_COUNT, 'pulse_width': PULSE_WIDTH},
    # {'type': 'coarse',     'k': 4, 'photon_count': PHOTON_COUNT},
    # {'type': 'trapcoarse', 'k': 4, 'photon_count': PHOTON_COUNT},
    {'type': 'coarsepw', 'k': 16, 'photon_count': PHOTON_COUNT, 'pulse_width': PULSE_WIDTH ,  'simulated': True},
    {'type': 'coarsepw', 'k': 64, 'photon_count': PHOTON_COUNT, 'pulse_width': PULSE_WIDTH, 'simulated': True},
    {'type': 'sliding', "pulse_width": PULSE_WIDTH, "shift": 200,  'photon_count': PHOTON_COUNT, "gate_width": N_TBINS // 3,  'simulated': True},
    {'type': 'sliding', "pulse_width": PULSE_WIDTH, "shift": 50,  'photon_count': PHOTON_COUNT, "gate_width": N_TBINS // 3, 'simulated': True},

    {'type': 'ham',        'k': 4, 'photon_count': PHOTON_COUNT, 'pulse_width': PULSE_WIDTH},

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

    # Decode one noisy capture at a time (all trials at once would need ~3 GB
    # per trial) and pool the errors. Each scheme gets its own generator from
    # the same seed, so its numbers don't change when RUNS is edited.
    rng = np.random.default_rng(SEED)
    abs_errs, sq_errs = [], []
    for trial in range(TRIALS):
        decoded_depth, _, _ = decode_simulation_depths(
            coding_matrix=coding_matrix,
            coded_values=coded_values,
            depths=depths,
            trials=1,
            tbin_depth_res=tbin_depth_res,
            rng=rng,
        )
        if trial == 0:
            depth_map = decoded_depth[0].reshape(gt_depth.shape)   # the 3-D render shows one capture
        abs_errs.append(np.mean(np.abs(decoded_depth[0] - depths)))
        sq_errs.append(np.mean((decoded_depth[0] - depths) ** 2))
    mae, rmse = float(np.mean(abs_errs)), float(np.sqrt(np.mean(sq_errs)))

    print(f"  MAE={mae * 1000:.3f} mm  |  RMSE={rmse * 1000:.3f} mm  "
          f"(over {TRIALS} trials, per-trial MAE std {np.std(abs_errs) * 1000:.3f} mm)\n")

    results.append({
        'name':      label,
        'depth_map': depth_map,
        'illum':     waveform_or_illum,   # (n_tbins,) or (n_tbins, k) for ham
        'gates':     demodfs_or_cm,       # (n_tbins, k)
        'rmse':      rmse * 1000,
        'mae':       mae  * 1000,
        'mae_se':    np.std(abs_errs) / np.sqrt(TRIALS) * 1000,   # standard error of the mean MAE
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

def tau_fraction(n_bins):
    """A duration of `n_bins` time bins as a fraction of the period, e.g. tau/8 or 0.18 tau."""
    ratio = N_TBINS / n_bins
    if abs(ratio - round(ratio)) < 0.01:
        return rf"$\tau/{round(ratio)}$"
    return rf"${n_bins / N_TBINS:.2f}\,\tau$"

def width_arrow(ax, t_start, t_end, y, text, side="right"):
    """Double-headed arrow spanning [t_start, t_end] at height y, labelled on `side`."""
    ax.annotate("", xy=(t_start, y), xytext=(t_end, y),
                arrowprops=dict(arrowstyle="|-|", lw=1.5, color="k", shrinkA=0, shrinkB=0))
    pad = 0.02 * rep_tau * 1e9
    if side == "right":
        ax.text(t_end + pad, y, text, va="center", ha="left", fontsize=FONT_SIZE - 4)
    else:
        ax.text(t_start - pad, y, text, va="center", ha="right", fontsize=FONT_SIZE - 4)

def stacked_title(ax, lines, pad=6):
    """Stack (text, colour, fontsize, weight) lines above `ax`, first line on top."""
    y = pad
    for text, color, size, weight in reversed(lines):
        ax.annotate(text, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, y),
                    textcoords="offset points", ha="center", va="bottom",
                    color=color, fontsize=size, fontweight=weight)
        y += size * 1.25

# traffic-light levels: 0 = low (green), 1 = medium (yellow), 2 = high (red)
best = min(results, key=lambda r: r['mae'])
worst = max(results, key=lambda r: r['mae'])
def tied(a, b):
    return abs(a['mae'] - b['mae']) <= 2 * np.hypot(a['mae_se'], b['mae_se'])
for res in results:
    res['bandwidth_level'] = int(np.searchsorted(BANDWIDTH_EDGES, res['k'], side='right'))
    res['error_level'] = 0 if tied(res, best) else 2 if tied(res, worst) else 1

for col, res in enumerate(results):
    axes[0, col].imshow(res['render'])
    # if res['k'] == K_0:
    #     title = 'Less Gates'
    # else:
    #     title = 'More Gates'
    bw, err = res['bandwidth_level'], res['error_level']
    stacked_title(axes[0, col], [
        (f"Gate Count: {res['k']}", "k", 30, "normal"),
        (BANDWIDTH_LABELS[bw], TRAFFIC_COLORS[bw], 24, "bold"),
        (f"MAE {res['mae']:.1f} mm", "k", 30, "normal"),
        (ERROR_LABELS[err], TRAFFIC_COLORS[err], 24, "bold"),
    ])

    ax = axes[1, col]
    gates = res['gates']
    k = gates.shape[1]
    tau_ns = rep_tau * 1e9

    # the pulse is emitted at t = 0 and wraps around the period; shift it to
    # the middle so it shows as one pulse
    illum_zero = res['illum'].reshape(N_TBINS, -1).sum(axis=1)
    illum = np.roll(illum_zero, int(round(N_TBINS / 2 - circular_center(illum_zero))))
    illum_one = np.roll(illum_zero, 1)

    # timing diagram: the illumination on the top row, then one row per gate.
    # Schemes with more gates than fit show the first few, a ⋮ row and the last
    shown = list(range(k)) if k <= MAX_GATE_ROWS else list(range(MAX_GATE_ROWS - 2)) + [None, k - 1]
    labels = [""] + [r"$\vdots$" if j is None else rf"$G_{{{j + 1}}}$" for j in shown]
    waves  = [illum / illum.max()] + [None if j is None else gates[:, j] for j in shown]
    wave_one  = [illum_one / illum_one.max()]

    n_rows = len(waves)
    row_y = n_rows - 1 - np.arange(n_rows)   # baseline of each row, top row highest

    # extend every waveform past both ends of the period, out to the axis limits.
    # A gate open across t = 0 / tau wraps around the period, so it continues
    # periodically into the margins with no edge drawn at 0 or tau. Anything
    # else is zero outside the period, so a gate that really opens at t = 0 or
    # closes at tau still shows that edge
    x_lo, x_hi = -X_MARGIN * tau_ns, (1 + X_MARGIN) * tau_ns
    n_margin = int(np.ceil(X_MARGIN * N_TBINS))
    ext_idx = np.arange(-n_margin, N_TBINS + n_margin)
    t_pad = ext_idx * (t_ns[1] - t_ns[0])
    in_period = (ext_idx >= 0) & (ext_idx < N_TBINS)
    for i, wave in enumerate(waves):
        if wave is None:
            continue
        wraps = wave[0] > 0.5 and wave[-1] > 0.5
        wave = wave[ext_idx % N_TBINS] if wraps else np.where(in_period, wave[ext_idx % N_TBINS], 0)
        if i == 0:
            # one black baseline shared by both pulses, matching the gate rows;
            # each pulse is coloured only where it rises off the baseline
            ax.plot([x_lo, x_hi], [row_y[i]] * 2, color="k", lw=GATE_LW)
            off_base = lambda w: np.where(w > 1e-3, row_y[i] + ROW_AMP * w, np.nan)
            ax.plot(t_pad, off_base(wave), color="blue", linestyle='--', lw=PULSE_LW,
                    label="Received illum.")
            # unwrap the t = 0 pulse: the second half of the period is really
            # just before t = 0, so plot it on [-tau/2, tau/2) as one pulse
            half = N_TBINS // 2
            t_pulse = t_ns - t_ns[half]
            pulse = np.roll(wave_one[0], half)
            ax.plot(t_pulse, off_base(pulse), color="red", lw=PULSE_LW, label="Emitted illum.")

            # t_0: the delay from the red pulse's peak to the blue pulse's peak
            t_red, t_blue = t_pulse[np.argmax(pulse)], t_pad[np.argmax(wave)]
            y_peak = row_y[i] + ROW_AMP
            ax.annotate("", xy=(t_red, y_peak), xytext=(t_blue, y_peak),
                        arrowprops=dict(arrowstyle="<->", lw=1.5, color="k", shrinkA=0, shrinkB=0))
            ax.text((t_red + t_blue) / 2, y_peak - 0.05, r"$t_0$", ha="center", va="top",
                    fontsize=FONT_SIZE - 4)
            # label each pulse directly above its peak, in its own colour; the
            # pulses are the same in every column, so label them only in the first
            if col == 0:
                ax.text(t_red, y_peak + 0.05, "Emitted", color="red", ha="center", va="bottom",
                        fontsize=FONT_SIZE - 4)
                ax.text(t_blue, y_peak + 0.05, "Received", color="blue", ha="center", va="bottom",
                        fontsize=FONT_SIZE - 4)

        else:
            ax.fill_between(t_pad, row_y[i], row_y[i] + ROW_AMP * wave, color=GATE_FILLS[0], lw=0)
            ax.plot(t_pad, row_y[i] + ROW_AMP * wave, color="k", lw=GATE_LW)

    # mark the pulse FWHM on the whole pulse at tau/2, and the first gate's width
    dt_ns = t_ns[1] - t_ns[0]
    above_half = np.flatnonzero(illum >= illum.max() / 2)

    if col == 0:
        width_arrow(ax, t_ns[above_half[0]], t_ns[above_half[-1]] + dt_ns, row_y[0] + ROW_AMP / 2,
                    "FWHM " + tau_fraction(len(above_half)))
    # Hamiltonian gates aren't single windows, so they have no one width or shift
    if res['type'] != 'ham':
        open_bins = np.flatnonzero(gates[:, shown[0]])
        width_arrow(ax, t_ns[open_bins[0]], t_ns[open_bins[0]] + len(open_bins) * dt_ns,
                    row_y[1] + ROW_AMP / 2, "Gate Width " + tau_fraction(len(open_bins)))
        # shift: on the second gate's row, from t = 0 (where the first gate opens)
        # to where the second gate opens; the label sits right of the second gate,
        # where there is free space
        open_bins_2 = np.flatnonzero(gates[:, shown[1]])
        y_shift = row_y[2] + ROW_AMP / 2
        ax.annotate("", xy=(t_ns[open_bins[0]], y_shift), xytext=(t_ns[open_bins_2[0]], y_shift),
                    arrowprops=dict(arrowstyle="|-|", lw=1.5, color="k", shrinkA=0, shrinkB=0))
        ax.text(t_ns[open_bins_2[-1]] + dt_ns + 0.02 * tau_ns, y_shift,
                "Shift " + tau_fraction(open_bins_2[0] - open_bins[0]), va="center",
                fontsize=FONT_SIZE - 4)

    # dotted reference line at t = 0, drawn on top of every row
    ax.axvline(0, color="k", linestyle=":", lw=1.5, zorder=5)
    # pulse legend above the panel; it's the same in every column, so show it once
    #if col == 0:
    #    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

    # one laser period, labelled in units of tau instead of nanoseconds
    ax.set_xlim(x_lo, x_hi)
    ax.set_xticks(tau_ns * np.array([0, 0.25, 0.5, 0.75, 1]),
                  labels=["0", r"$\tau/4$", r"$\tau/2$", r"$3\tau/4$", r"$\tau$"])
    ax.set_yticks(row_y + ROW_AMP / 2, labels=labels, fontsize=FONT_SIZE - 4)
    ax.tick_params(axis="y", length=0)
    # gap below so the last baseline clears the x-axis, headroom above for the pulse labels
    ax.set_ylim(-0.5, n_rows - 1 + ROW_AMP + 0.5)
    ax.set_xlabel("Time")
    ax.spines[["top", "right", "left"]].set_visible(False)

for ax in axes[0]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
plt.subplots_adjust(left=0.05, right=0.05, top=0.05, bottom=0.05)
plt.savefig("figures/3d_teaser_fig.pdf", bbox_inches="tight")
plt.show()
