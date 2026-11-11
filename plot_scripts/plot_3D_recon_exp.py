import numpy as np
import open3d as o3d
from scipy.ndimage import median_filter

from spad_lib.spad512utils import *
from utils.file_utils import *
from plot_scripts.plot_utils import *
from utils.global_constants import *
from utils.tof_utils import build_coding_matrix_from_correlations, get_simulated_coding_matrix, sliding_gate_width_tbins, decode_depth_map, \
    calculate_tof_domain_params, filter_hot_pixels
from utils.parameter_classes import DecodeConfig

# -----------------------------------------------------------------------------
# CONFIG (capitalized)
# -----------------------------------------------------------------------------
EXP_PATH = os.path.join('horse_results', 'k3_HIGHSNR')
SNR_LABEL     = "highsnr" if "highsnr" in EXP_PATH.lower() else "lowsnr"
FIGURES_DIR   = "figures"   # where the panel PDF is saved
VMAX_ERROR    = 3.0 # meters; None = auto (95th percentile across runs)
PLOT_POINT_CLOUD = True  # True = show point cloud PNG in top row; False = show depth map instead
PLOT_ERROR_MAP = False    # True = error-map row on the bottom; False = no error maps, MAE / RMSE written under the top row
ALIGN_DEPTHS = True       # shift each run's depth map so all medians match the first run
N_TBINS = 3000
NUM_TRIALS = 200
TRIAL_SEED = 2   # seed for the random subset of NUM_TRIALS trials
N_GT_TRIALS = 350  # held-out trials for the ground truth; same count for every code so all are scored alike

BAD_ROWS = [(230, 260), (80, 95)] if 'step_stool' in EXP_PATH else None

#PLotting utils for visualization
PLOT_DEPTH_MAPS = True
VMINS = None
VMAXS = None
MEDIAN_FILTER_SIZE = 1

#Masking or normalizing depth maps
NORMALIZE_DEPTH_MAPS = False
MASK_BACKGROUND_PIXELS = True

#Which correlation functions to use
SIMULATED_CORRELATIONS = False
USE_FULL_CORRELATIONS = False

#Smoothing or shifting the correlation functions
SIGMA_SIZE = None #None if no smoothing
SHIFT_SIZE = 70 #None if no shifting

#Corrections to depth map
CORRECT_DEPTH_DISTORTION = False
CORRECT_MASTER = True

# --- point-cloud rendering (top row of the panel) ---
# Point clouds render off-screen straight into the panel (no windows, no PNGs).
# True = also open an interactive viewer per run first, to re-aim the camera and press S.
INTERACTIVE_VIEWER = True
# The point cloud gets its own post-processing, separate from the error maps
PCD_BAD_ROWS = [(230, 260), (80, 95)] if 'step_stool' in EXP_PATH else None
PCD_COL_CROP = (100, -100)  if 'step_stool' in EXP_PATH else (1, -1) # tighter column crop than the error maps' (20, -20)
PCD_MASTER_OFFSET = 0.009    # m added to the master (left) half when CORRECT_MASTER
PCD_VMIN = None              # shared clip range across runs; None = auto
PCD_VMAX = None
FOV_MAJOR_AXIS_DEG = 40.0    # horizontal FOV in degrees
# Press S in the viewer window to save the current camera view; it reloads on the next run.
CAMERA_PARAMS_FILE = "camera_params.json"
VIEW_ZOOM = 0.8
SHOW_AXES = False
PCD_ZOOM_OUT = 0  # mouse-wheel steps applied to the panel renders; bigger = more zoomed out, 0 = saved view


# -----------------------------------------------------------------------------
# POINT-CLOUD HELPERS
# -----------------------------------------------------------------------------
def pointcloud_depth_map(depth_map, im_width, correct_master, mask_background, median_size, background=None):
    """Post-process a decoded depth map for the point cloud (not for the error maps)."""
    depth_map = depth_map.astype(float)
    # mask before the master offset, otherwise zero-depth pixels become z > 0 and survive
    if mask_background:
        depth_map[depth_map == 0] = np.nan
        if background is not None:
            depth_map[background] = np.nan
    if not correct_master:
        depth_map = depth_map[:, : im_width // 2]
    else:
        depth_map[:, : im_width // 2] += PCD_MASTER_OFFSET
    if mask_background:
        depth_map = depth_map[40:450, PCD_COL_CROP[0]:PCD_COL_CROP[1]]
    depth_map = median_filter(depth_map, size=median_size)
    if PCD_BAD_ROWS:
        for start, end in PCD_BAD_ROWS:
            depth_map[start:end + 1, :] = np.nan
    return depth_map


def depth_to_pointcloud(depth_map, fov_major_axis_deg):
    """Back-project a depth map to a 3-D point cloud."""
    height, width = depth_map.shape
    fx = width / (2 * np.tan(np.radians(fov_major_axis_deg) / 2))
    fy = fx
    cx, cy = width / 2.0, height / 2.0

    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_map.astype(np.float32)
    x3d = (x_grid - cx) * z / fx
    y3d = -(y_grid - cy) * z / fy

    points = np.stack((x3d, y3d, z), axis=-1).reshape(-1, 3)
    valid = ~np.isnan(points).any(axis=1) & (points[:, 2] > 0)
    return points[valid]


def setup_view(vis, geometry):
    """Add the geometry (and optional axes) and apply the saved camera, else the default zoom."""
    vis.add_geometry(geometry)
    if SHOW_AXES:
        bbox = geometry.get_axis_aligned_bounding_box()
        axis_size = np.linalg.norm(np.asarray(bbox.max_bound) - np.asarray(bbox.min_bound)) * 0.05
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size))
    ctrl = vis.get_view_control()
    if CAMERA_PARAMS_FILE and os.path.isfile(CAMERA_PARAMS_FILE):
        params = o3d.io.read_pinhole_camera_parameters(CAMERA_PARAMS_FILE)
        ctrl.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    else:
        ctrl.set_zoom(VIEW_ZOOM)
    return ctrl


def aim_camera(geometry, title):
    """Interactive viewer for re-aiming: press S to save the view to CAMERA_PARAMS_FILE."""
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title)
    ctrl = setup_view(vis, geometry)

    def save_camera(vis):
        o3d.io.write_pinhole_camera_parameters(CAMERA_PARAMS_FILE, ctrl.convert_to_pinhole_camera_parameters())
        print(f"\n[{title}] Camera saved to {CAMERA_PARAMS_FILE}")
        return False

    vis.register_key_callback(ord("S"), save_camera)
    print(f"[{title}] Press S to save the current view, then close the window")
    vis.run()
    vis.destroy_window()


def render_offscreen(geometry):
    """Render with the saved camera in a hidden window and return the image (H, W, 3) in [0, 1]."""
    # match the window to the saved camera's size so the view (and the panel's crop) is unchanged
    width, height = 1920, 1080
    if CAMERA_PARAMS_FILE and os.path.isfile(CAMERA_PARAMS_FILE):
        intrinsic = o3d.io.read_pinhole_camera_parameters(CAMERA_PARAMS_FILE).intrinsic
        width, height = intrinsic.width, intrinsic.height
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    ctrl = setup_view(vis, geometry)
    # only here, not in setup_view, so pressing S in the viewer never saves the zoom-out into the camera file
    ctrl.scale(PCD_ZOOM_OUT)
    vis.poll_events()
    vis.update_renderer()
    image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    return image


def render_point_cloud(depth_map, run_tag):
    points = depth_to_pointcloud(depth_map, FOV_MAJOR_AXIS_DEG)
    print(f"  {run_tag}: point cloud {len(points):,} points")
    # centre at origin so open3d zoom/pan works correctly
    points = points - points.mean(axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    z = points[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    pcd.colors = o3d.utility.Vector3dVector(plt.get_cmap("gray")(0.3 + 0.4 * z_norm)[:, :3])

    if INTERACTIVE_VIEWER:
        aim_camera(pcd, title=f"Point Cloud — {run_tag}")
    return render_offscreen(pcd)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    def apply_decode_defaults(cfg: DecodeConfig) -> DecodeConfig:
        defaults = dict(
            exp_path=EXP_PATH,

            n_tbins=N_TBINS,
            num_trials=NUM_TRIALS,

            vmins=VMINS,
            vmaxs=VMAXS,
            median_filter_size=MEDIAN_FILTER_SIZE,

            correct_master=CORRECT_MASTER,
            plot_depth_maps=PLOT_DEPTH_MAPS,
            mask_background_pixels=MASK_BACKGROUND_PIXELS,
            normalize_depth_maps=NORMALIZE_DEPTH_MAPS,

            simulated_correlations=SIMULATED_CORRELATIONS,
            use_full_correlations=USE_FULL_CORRELATIONS,

            smooth_sigma=SIGMA_SIZE,
            shift=SHIFT_SIZE,

            correct_depth_distortion=CORRECT_DEPTH_DISTORTION,

        )

        # fill only missing (None)
        for k, v in defaults.items():
            if getattr(cfg, k) is None:
                setattr(cfg, k, v)

        return cfg

    parser = build_parser_from_config(DecodeConfig)
    args = parser.parse_args()
    cfg = apply_decode_defaults(DecodeConfig(**vars(args)))

    hot_mask = load_hot_mask(get_data_folder(HOT_MASK_PATH_WINDOWS, HOT_MASK_PATH_MAC))
    correlation_folder = get_data_folder(READ_PATH_CORRELATIONS_DEPTH_MAC, READ_PATH_CORRELATIONS_DEPTH_WINDOWS)
    capture_folder = get_data_folder(READ_PATH_CAPTURE_MAC, READ_PATH_CAPTURE_WINDOWS)
    assert cfg.exp_path is not None, 'Must define exp_num to find folder'
    capture_folder = os.path.join(capture_folder, cfg.exp_path)
    # exp_path may point at a .zip; get_capture_folder expects the base name
    # (it appends .zip itself) and returns the unzipped directory.
    if capture_folder.endswith('.zip'):
        capture_folder = capture_folder[:-len('.zip')]
    capture_folder = get_capture_folder(capture_folder)

    capture_paths = os.listdir(capture_folder)
    capture_paths = filter_capture_files(capture_paths)

    depth_map_dict = {}

    for i, coded_vals_name in enumerate(capture_paths):
        if coded_vals_name.startswith('.'): continue
        coded_vals_path = os.path.join(capture_folder, coded_vals_name)
        capture_file = load_npz(coded_vals_path)
        params = capture_file['cfg'].item()
        coded_vals = capture_file['coded_vals']
        im_width = params['im_width']
        mA = params['current']
        mV = params['high_level_amplitude'] * 1000
        capture_type = params['capture_type']
        k = params['k']
        freq = params['rep_rate']
        freq_mhz = freq * 1e-6
        duty = params['duty']
        rep_tau = params['rep_tau']

        #if capture_type == 'ham' and k == 4: continue

        corr_path = os.path.join(correlation_folder, make_correlation_filename(capture_type, k,
                                                                               freq_mhz, mV, mA, duty))


        gt_coded_vals_path = os.path.join(capture_folder,
                                          make_capture_filename(capture_type, k, freq_mhz, mV, mA, duty,
                                                                None, True))

        correlations_total, sliding_rows = load_correlations(corr_path)

        if cfg.simulated_correlations:
            coding_matrix = get_simulated_coding_matrix(capture_type, cfg.n_tbins, k,
                                                        sliding_gate_width_tbins(params.get('sliding_gate_width'), rep_tau, cfg.n_tbins))
        else:
            coding_matrix = build_coding_matrix_from_correlations(
                correlations_total,
                cfg.use_full_correlations,
                cfg.smooth_sigma,
                cfg.shift,
                cfg.n_tbins,
                sliding=sliding_rows,
            )

        # tmp = np.sum(np.sum(coded_vals, axis=0), axis=-1)
        # tmp[tmp < 1.5e4] = 0
        # plt.imshow(tmp)
        # plt.show()
        # mn = coding_matrix.min()
        # mx = coding_matrix.max()
        # coding_matrix = (coding_matrix - mn) / (mx - mn)

        (rep_tau, rep_freq,tbin_res,
         t_domain,max_depth,tbin_depth_res,)= calculate_tof_domain_params(cfg.n_tbins, params['rep_tau'])

        # -----------------------------------------------------------------
        # decode to depth map
        # -----------------------------------------------------------------
        total_trials = coded_vals.shape[0]
        trials = min(total_trials, cfg.num_trials)
        # random trials, not the first N, so slow drift over the acquisition averages out
        trial_order = np.random.default_rng(TRIAL_SEED).permutation(total_trials)
        # ground truth from held-out trials only, so it shares no photons with the depth map
        assert trials + N_GT_TRIALS <= total_trials, (
            f'{trials} trials + {N_GT_TRIALS} ground-truth trials > {total_trials} in the file')
        coded_vals_trials = np.sum(coded_vals[trial_order[:trials], ...], axis=0)
        coded_vals_gt = np.sum(coded_vals[trial_order[trials:trials + N_GT_TRIALS], ...], axis=0)
        depth_map, zncc = decode_depth_map(
            coded_vals_trials,
            coding_matrix,
            im_width,
            tbin_depth_res,
            args.use_full_correlations,
        )


        depth_map = filter_hot_pixels(depth_map, hot_mask)
        # copy: the error-map crop below is a view that gets edited in place
        # same low-photon-count background mask the horse depth map gets below
        background = np.sum(coded_vals, axis=(0, -1)) < 1.5e4 if "horse" in coded_vals_path else None
        pcd_depth_map = pointcloud_depth_map(depth_map.copy(), im_width, cfg.correct_master,
                                             cfg.mask_background_pixels, cfg.median_filter_size, background)

        coded_vals_filt = np.zeros_like(coded_vals_trials)
        for i in range(coded_vals_trials.shape[-1]):
            coded_vals_filt[:, :, i] = filter_hot_pixels(coded_vals_trials[..., i], hot_mask)


        gt_depth_map, zncc = decode_depth_map(
            coded_vals_gt,
            coding_matrix,
            im_width,
            tbin_depth_res,
            args.use_full_correlations,
        )
        gt_depth_map = filter_hot_pixels(gt_depth_map, hot_mask)

        # optional crop for master
        if cfg.correct_master is False:
            depth_map = depth_map[:, : im_width // 2]
            gt_depth_map = gt_depth_map[:, : im_width // 2]
            coded_vals_trials = coded_vals_trials[:, : im_width // 2]

        if cfg.mask_background_pixels:
            if "horse" in coded_vals_path:
                tmp = np.sum(np.sum(coded_vals, axis=0), axis=-1)
                tmp[tmp < 1.5e4] = 0
                depth_map[tmp < 1e4] = np.nan

            else:
                depth_map = depth_map[40:450, 20:-20]
                gt_depth_map = gt_depth_map[40:450, 20:-20]
            depth_map[depth_map == 0] = np.nan
            gt_depth_map[gt_depth_map == 0] = np.nan
            mask = None

        if BAD_ROWS is not None and type(BAD_ROWS) == list:
            depth_map = depth_map.astype(float)
            for BAD_ROW in BAD_ROWS:
                depth_map[BAD_ROW[0]:BAD_ROW[1] + 1, :] = np.nan

        mae = np.nanmean(np.abs(depth_map - gt_depth_map))
        rmse = np.sqrt(np.nanmean((depth_map - gt_depth_map) ** 2))

        cfg_dict = asdict(cfg)
        cfg_dict.update({'depth_map': depth_map, 'gt_depth_map': gt_depth_map,
                         'pcd_depth_map': pcd_depth_map, 'rmse': rmse, 'mae': mae})
        cfg_dict.update(params)
        depth_map_dict[coded_vals_path] = cfg_dict


        #plt.plot(coding_matrix)
        #plt.show()

    # --- median-align all depth maps to the first run ---
    if ALIGN_DEPTHS:
        ref_median = None
        for d in depth_map_dict.values():
            med = np.nanmedian(d['depth_map'])
            if ref_median is None:
                ref_median = med
            else:
                shift = ref_median - med
                d['depth_map']    = d['depth_map']    + shift
                d['gt_depth_map'] = d['gt_depth_map'] + shift
            print(f"  {d['capture_type']} k={d['k']}: median={med:.3f} m  shift={ref_median - med:.3f} m")


    # --- clip depth maps to VMINS / VMAXS ---
    for d in depth_map_dict.values():
        vmin_clip = VMINS if VMINS is not None else np.nanmin(d['depth_map'])
        vmax_clip = VMAXS if VMAXS is not None else np.nanmax(d['depth_map'])
        d['depth_map']    = np.clip(d['depth_map'],    vmin_clip, vmax_clip)
        d['gt_depth_map'] = np.clip(d['gt_depth_map'], vmin_clip, vmax_clip)

    # --- point clouds: align, clip to one shared range, render off-screen for the panel ---
    if PLOT_POINT_CLOUD:
        pcd_maps = [d['pcd_depth_map'] for d in depth_map_dict.values()]
        if ALIGN_DEPTHS:
            ref_median = np.nanmedian(pcd_maps[0])
            pcd_maps = [m + (ref_median - np.nanmedian(m)) for m in pcd_maps]
        all_vals = np.concatenate([m.ravel() for m in pcd_maps])
        pcd_vmin = PCD_VMIN if PCD_VMIN is not None else float(np.nanmin(all_vals))
        pcd_vmax = PCD_VMAX if PCD_VMAX is not None else float(np.nanmax(all_vals))
        print(f"\n  Point-cloud depth range: vmin={pcd_vmin:.3f}  vmax={pcd_vmax:.3f}")
        for d, m in zip(depth_map_dict.values(), pcd_maps):
            d['pcd_image'] = render_point_cloud(np.clip(m, pcd_vmin, pcd_vmax),
                                                f"{d['capture_type']}_k{d['k']}_{SNR_LABEL}")

    # --- 3-D recon comparison panel ---
    recon_runs = []
    for coded_vals_path, d in depth_map_dict.items():
        recon_runs.append({
            'capture_type': d['capture_type'],
            'k':            d['k'],
            'snr_label':    SNR_LABEL,
            'depth_map':    d['depth_map'],
            'gt_depth_map': d['gt_depth_map'],
            'pcd_image':    d.get('pcd_image'),
        })


    save_panel = os.path.join(FIGURES_DIR, f"3d_recon_panel_{SNR_LABEL}_k{d['k']}.pdf")
    plot_3d_recon_comparison(
        recon_runs,
        snr_label=SNR_LABEL,
        vmax_error=VMAX_ERROR,
        figures_dir=FIGURES_DIR,
        save_path=save_panel,
        plot_point_cloud=PLOT_POINT_CLOUD,
        plot_error_map=PLOT_ERROR_MAP,
    )