import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.ndimage import median_filter

from utils.file_utils import (
    get_data_folder, load_hot_mask, load_correlation_npz,
    make_correlation_filename, make_capture_filename,
    get_capture_folder, capture_parse_run, str2bool,
)
from utils.global_constants import (
    READ_PATH_CORRELATIONS_MAC, READ_PATH_CORRELATIONS_WINDOWS,
    READ_PATH_CAPTURE_MAC, READ_PATH_CAPTURE_WINDOWS,
    HOT_MASK_PATH_MAC, HOT_MASK_PATH_WINDOWS,
    PIXEL_PITCH, FOCAL_LENGTH,
)
from utils.tof_utils import (
    build_coding_matrix_from_correlations, get_simulated_coding_matrix,
    decode_depth_map, calculate_tof_domain_params, filter_hot_pixels,
)
from spad_lib.spad512utils import correct_bistatic_distortion
import argparse

# =============================================================================
# CONFIG
# =============================================================================
EXP_PATH = os.path.join('step_stool_results', 'timeslicing_LOWSNR')
N_TBINS = 3000
NUM_TRIALS = 200

MEDIAN_FILTER_SIZE = 1

# --- mesh reconstruction quality/speed ---
VOXEL_SIZE     = 0.01   # downsampling voxel size (m); larger = fewer points = faster
POISSON_DEPTH  = 6       # octree depth for Poisson (6=fast, 7=medium, 8=slow/detailed)
SMOOTH_ITERS   = 5       # Laplacian smoothing iterations
VMIN = None
VMAX = None
BAD_ROWS = [(230, 260), (70, 95)]   # row range to mask out (inclusive), set to None to disable
BASELINE = (0.20, 0.0, 0.0)  # laser position relative to detector in metres (x, y, z)
ALIGN_DEPTHS = True           # shift all runs to match the first run's median depth

MASK_BACKGROUND_PIXELS = True
CORRECT_MASTER = True

SIMULATED_CORRELATIONS = False
USE_FULL_CORRELATIONS = False

SIGMA_SIZE = None
SHIFT_SIZE = 70#70

FOV_MAJOR_AXIS_DEG = 10.0   # horizontal FOV in degrees

# --- 3-D viewer ---
# Press S in the viewer window to save the current camera view to CAMERA_PARAMS_FILE.
# On the next run it will reload automatically.
CAMERA_PARAMS_FILE = "camera_params.json"   # set to None to disable

VIEW_ZOOM     = 0.8
SHOW_AXES     = False  # show XYZ coordinate frame in viewer
SAVE_FIGURES  = True   # save a PNG per run; set False to disable
SNR_LABEL     = "highsnr"  if "highsnr" in EXP_PATH.lower() else "lowsnr"   # "highsnr" or "lowsnr" — written into the filename

DEFAULT_RUNS = [
    "ham,3,10,500,16,20,1",
    # "coarse,3,10, 420,16,30, 1",
    # "trapcoarse,3,10, 420,16,30, 1",

    "ham,4,10,770,16,15,1",
    # "coarse,4,10, 540,16,23, 1",
    # "trapcoarse,4,10, 540,16,23, 1",

    "timeslicing,8,10,1200,16,12,1",
    "timeslicing,12,10,1200,16,12,1",

]


# =============================================================================
# ARGS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="3-D reconstruction from a single depth-map capture")

    p.add_argument("--run", action="append", type=capture_parse_run,
                   help="capture_type,k,freq_mhz,mV,mA,duty,int_time (repeatable)")
    p.add_argument("--exp_path", type=str, default=EXP_PATH)
    p.add_argument("--n_tbins", type=int, default=N_TBINS)
    p.add_argument("--num_trials", type=int, default=NUM_TRIALS)
    p.add_argument("--vmin", type=float, default=VMIN)
    p.add_argument("--vmax", type=float, default=VMAX)
    p.add_argument("--median_filter_size", type=int, default=MEDIAN_FILTER_SIZE)
    p.add_argument("--correct_master", type=str2bool, default=CORRECT_MASTER)
    p.add_argument("--mask_background_pixels", type=str2bool, default=MASK_BACKGROUND_PIXELS)
    p.add_argument("--simulated_correlations", type=str2bool, default=SIMULATED_CORRELATIONS)
    p.add_argument("--use_full_correlations", type=str2bool, default=USE_FULL_CORRELATIONS)
    p.add_argument("--smooth_sigma", type=float, default=SIGMA_SIZE)
    p.add_argument("--shift", type=int, default=SHIFT_SIZE)
    p.add_argument("--fov_major_axis_deg", type=float, default=FOV_MAJOR_AXIS_DEG)

    args = p.parse_args()
    if args.run is None:
        args.run = [capture_parse_run(r) for r in DEFAULT_RUNS]
    return args


# =============================================================================
# HELPERS
# =============================================================================

def depth_to_pointcloud(depth_map, fov_major_axis_deg):
    """Back-project a depth map to a 3-D point cloud."""
    height, width = depth_map.shape
    fov_rad = np.radians(fov_major_axis_deg)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx
    cx, cy = width / 2.0, height / 2.0

    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_map.astype(np.float32)
    x3d = (x_grid - cx) * z / fx
    y3d = -(y_grid - cy) * z / fy

    points = np.stack((x3d, y3d, z), axis=-1).reshape(-1, 3)
    valid = ~np.isnan(points).any(axis=1) & (points[:, 2] > 0)
    return points[valid]


def build_mesh_from_pcd(pcd, voxel_size=0.005, poisson_depth=6, smooth_iters=5):
    """Clean a point cloud and reconstruct a Poisson mesh."""
    # downsample first — biggest speed win
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"  after voxel downsample: {len(pcd.points):,} points")

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.remove_non_finite_points()

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=20)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)

    # trim low-density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    if smooth_iters > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=smooth_iters, lambda_filter=0.5)
    mesh.remove_non_manifold_edges()

    return mesh, pcd


def colorize_mesh_by_depth(mesh):
    verts = np.asarray(mesh.vertices)
    z = verts[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    colors = plt.get_cmap("gray")(0.3 + 0.4 * z_norm)[:, :3]  # stay in mid-gray range
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh

def _rot_matrix(rot_x_deg, rot_y_deg, rot_z_deg):
    rx, ry, rz = np.radians(rot_x_deg), np.radians(rot_y_deg), np.radians(rot_z_deg)
    Rx = np.array([[1,           0,            0],
                   [0,  np.cos(rx), -np.sin(rx)],
                   [0,  np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [          0, 1,           0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [         0,           0, 1]])
    return Rz @ Ry @ Rx


def show_geometry(geometry, title="Open3D", mesh_back=False,
                  zoom=0.8, show_axes=True,
                  camera_params_file=None, save_path=None):
    """
    Display an Open3D geometry.

    Camera view:
      • If camera_params_file exists on disk, it is loaded automatically.
      • Press S in the viewer to save the current view to camera_params_file.
      • On the next run the saved view is restored exactly.

    save_path : if given, saves a PNG screenshot after the window closes.
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title)
    vis.add_geometry(geometry)
    if show_axes:
        bbox      = geometry.get_axis_aligned_bounding_box()
        axis_size = np.linalg.norm(np.asarray(bbox.max_bound) - np.asarray(bbox.min_bound)) * 0.05
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size))
    if mesh_back:
        vis.get_render_option().mesh_show_back_face = True

    ctrl = vis.get_view_control()

    #restore saved camera if available, otherwise auto-fit to geometry
    if camera_params_file and os.path.isfile(camera_params_file):
        params = o3d.io.read_pinhole_camera_parameters(camera_params_file)
        ctrl.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        print(f"[{title}] Loaded camera from {camera_params_file}")
    else:
        ctrl.set_zoom(zoom)

    # S key: save current camera to file
    def save_camera(vis):
        if camera_params_file:
            params = ctrl.convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters(camera_params_file, params)
            print(f"\n[{title}] Camera saved to {camera_params_file}")
        return False

    vis.register_key_callback(ord("S"), save_camera)
    print(f"[{title}] Press S to save the current view → {camera_params_file}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path, do_render=True)
        print(f"[{title}] Screenshot saved to {save_path}")

    vis.run()
    vis.destroy_window()
# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    args = parse_args()

    hot_mask = load_hot_mask(get_data_folder(HOT_MASK_PATH_WINDOWS, HOT_MASK_PATH_MAC))
    correlation_folder = get_data_folder(READ_PATH_CORRELATIONS_MAC, READ_PATH_CORRELATIONS_WINDOWS)
    capture_folder = get_data_folder(READ_PATH_CAPTURE_MAC, READ_PATH_CAPTURE_WINDOWS)
    if args.exp_path is not None:
        capture_folder = get_capture_folder(os.path.join(capture_folder, args.exp_path))

    # =========================================================================
    # PASS 1 — decode every run and median-align; collect depth maps
    # =========================================================================
    ref_median = None
    decoded_runs = []   # list of dicts: {r, depth_map}

    for r in args.run:
        corr_path = os.path.join(
            correlation_folder,
            make_correlation_filename(r['capture_type'], r['k'], r['freq_mhz'], r['mV'], r['mA'], r['duty']),
        )
        coded_vals_path = os.path.join(
            capture_folder,
            make_capture_filename(r['capture_type'], r['k'], r['freq_mhz'], r['mV'], r['mA'], r['duty'],
                                  r['int_time'], False),
        )

        print(f"\n--- {r['capture_type']} k={r['k']} ---")
        print(f"  corr_path:       {corr_path}")
        print(f"  coded_vals_path: {coded_vals_path}")

        correlations_total = load_correlation_npz(corr_path)['correlations']
        print(f"  correlations shape: {correlations_total.shape}")

        if args.simulated_correlations:
            coding_matrix = get_simulated_coding_matrix(r['capture_type'], args.n_tbins, r['k'])
        else:
            coding_matrix = build_coding_matrix_from_correlations(
                correlations_total,
                args.use_full_correlations,
                args.smooth_sigma,
                args.shift,
                args.n_tbins,
            )
        print(f"  coding_matrix shape: {coding_matrix.shape}")

        capture_file = np.load(coded_vals_path, allow_pickle=True)
        cfg = capture_file['cfg'].item()
        coded_vals = capture_file['coded_vals']
        im_width = cfg['im_width']
        print(f"  coded_vals shape: {coded_vals.shape}  im_width={im_width}  rep_tau={cfg['rep_tau']:.2e}")

        (_, _, _, _, _, tbin_depth_res) = calculate_tof_domain_params(args.n_tbins, cfg['rep_tau'])
        print(f"  tbin_depth_res: {tbin_depth_res:.4f} m/bin")

        total_trials = coded_vals.shape[0]
        trials = min(total_trials, args.num_trials)
        coded_vals_trials = np.sum(coded_vals[:trials], axis=0)
        print(f"  using {trials}/{total_trials} trials")

        depth_map, _ = decode_depth_map(
            coded_vals_trials,
            coding_matrix,
            im_width,
            tbin_depth_res,
            args.use_full_correlations,
        )
        print(f"  depth_map after decode:  shape={depth_map.shape}  min={np.nanmin(depth_map):.3f}  max={np.nanmax(depth_map):.3f}  nan%={100*np.isnan(depth_map).mean():.1f}")
        depth_map = filter_hot_pixels(depth_map, hot_mask)

        if not args.correct_master:
            depth_map = depth_map[:, : im_width // 2]
        else:
            depth_map[:, : im_width // 2] = depth_map[:, : im_width // 2] + 0.009

        if args.mask_background_pixels:
            depth_map = depth_map[40:450, 100:-100]
        print(f"  depth_map after crop:    shape={depth_map.shape}")

        depth_map = median_filter(depth_map, size=args.median_filter_size)

        if BAD_ROWS is not None and type(BAD_ROWS) == list:
            depth_map = depth_map.astype(float)
            for BAD_ROW in BAD_ROWS:
                depth_map[BAD_ROW[0]:BAD_ROW[1] + 1, :] = np.nan

        print(f"  depth_map after bistatic: min={np.nanmin(depth_map):.3f}  max={np.nanmax(depth_map):.3f}  nan%={100*np.isnan(depth_map).mean():.1f}")

        if ALIGN_DEPTHS:
            current_median = np.nanmedian(depth_map)
            if ref_median is None:
                ref_median = current_median
            else:
                depth_map = depth_map + (ref_median - current_median)
            print(f"  median depth: {current_median:.3f} m  (ref={ref_median:.3f} m)")

        decoded_runs.append({'r': r, 'depth_map': depth_map})

    # =========================================================================
    # Compute GLOBAL vmin / vmax across all runs (shared colour scale)
    # =========================================================================
    if args.vmin is not None and args.vmax is not None:
        global_vmin, global_vmax = args.vmin, args.vmax
    else:
        all_vals = np.concatenate([d['depth_map'].ravel() for d in decoded_runs])
        global_vmin = float(np.nanmin(all_vals)) if args.vmin is None else args.vmin
        global_vmax = float(np.nanmax(all_vals)) if args.vmax is None else args.vmax
    print(f"\n  Global depth range: vmin={global_vmin:.3f}  vmax={global_vmax:.3f}")

    # =========================================================================
    # PASS 2 — clip to shared range and visualise
    # =========================================================================
    for d in decoded_runs:
        r = d['r']
        depth_map = np.clip(d['depth_map'], global_vmin, global_vmax)
        print(f"\n--- visualising {r['capture_type']} k={r['k']}  vmin={global_vmin:.3f}  vmax={global_vmax:.3f} ---")

        # --- depth map preview ---
        plt.figure()
        plt.imshow(depth_map, cmap='hot', vmin=global_vmin, vmax=global_vmax)
        plt.title(f"{r['capture_type']} k={r['k']}")
        plt.colorbar(label='Depth (m)')
        plt.tight_layout()
        plt.show()

        # --- 3-D reconstruction ---
        points = depth_to_pointcloud(depth_map, args.fov_major_axis_deg)
        print(f"Point cloud: {len(points):,} points")

        # centre at origin so open3d zoom/pan works correctly
        centroid = points.mean(axis=0)
        points_display = points - centroid

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_display)
        z_vals = points_display[:, 2]
        z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-8)
        pcd.colors = o3d.utility.Vector3dVector(plt.get_cmap("gray")(0.3 + 0.4 * z_norm)[:, :3])

        run_tag = f"{r['capture_type']}_k{r['k']}_{SNR_LABEL}"
        pcd_save = f"figures/recon_{run_tag}_pcd.png" if SAVE_FIGURES else None

        show_geometry(pcd, title=f"Point Cloud — {run_tag}",
                      zoom=VIEW_ZOOM, show_axes=SHOW_AXES,
                      camera_params_file=CAMERA_PARAMS_FILE, save_path=pcd_save)

        if len(points) < 10:
            print("Too few points for mesh reconstruction — skipping.")
            continue

        # mesh, pcd_clean = build_mesh_from_pcd(pcd, voxel_size=VOXEL_SIZE,
        #                                        poisson_depth=POISSON_DEPTH, smooth_iters=SMOOTH_ITERS)

        # if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        #     print("Mesh reconstruction produced no geometry — skipping.")
        #     continue
        #
        # if not mesh.has_vertex_normals():
        #     mesh.compute_vertex_normals()
        # mesh = colorize_mesh_by_depth(mesh)
        #
        # mesh_save = f"figures/recon_{run_tag}_mesh.png" if SAVE_FIGURES else None
        # show_geometry(mesh, title=f"Mesh — {run_tag}", mesh_back=True,
        #               zoom=VIEW_ZOOM, show_axes=SHOW_AXES,
        #               camera_params_file=CAMERA_PARAMS_FILE, save_path=mesh_save)
