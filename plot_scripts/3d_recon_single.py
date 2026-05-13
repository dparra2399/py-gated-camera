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
)
from utils.tof_utils import (
    build_coding_matrix_from_correlations, get_simulated_coding_matrix,
    decode_depth_map, calculate_tof_domain_params, filter_hot_pixels,
)
import argparse

# =============================================================================
# CONFIG
# =============================================================================
EXP_PATH = 'exp_2'
N_TBINS = 1500
NUM_TRIALS = 100

MEDIAN_FILTER_SIZE = 5
VMIN = None
VMAX = None

MASK_BACKGROUND_PIXELS = True
CORRECT_MASTER = False

SIMULATED_CORRELATIONS = False
USE_FULL_CORRELATIONS = False

SIGMA_SIZE = None
SHIFT_SIZE = 150

FOV_MAJOR_AXIS_DEG = 5.0   # horizontal FOV in degrees

DEFAULT_RUNS = [
    "ham,3,10,500,16,20,30",
    "coarse,3,10, 420,16,30, 30",

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
    z = depth_map.astype(np.float32) / 20.0
    x3d = (x_grid - cx) * z / fx
    y3d = (y_grid - cy) * z / fy

    points = np.stack((x3d, y3d, z), axis=-1).reshape(-1, 3)
    valid = ~np.isnan(points).any(axis=1) & (points[:, 2] > 0)
    return points[valid]


def build_mesh_from_pcd(pcd):
    """Clean a point cloud and reconstruct a Poisson mesh."""
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    pcd.remove_duplicated_points()
    pcd.remove_non_finite_points()
    pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.02)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=10, lambda_filter=0.5)
    mesh.remove_non_manifold_edges()

    # trim low-density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh, pcd


def colorize_mesh_by_depth(mesh):
    """Color mesh vertices by their z-value using the 'turbo' colormap."""
    verts = np.asarray(mesh.vertices)
    z = verts[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    colors = plt.get_cmap("turbo")(z_norm)[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


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

        correlations_total = load_correlation_npz(corr_path)['correlations']

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

        capture_file = np.load(coded_vals_path, allow_pickle=True)
        cfg = capture_file['cfg'].item()
        coded_vals = capture_file['coded_vals']
        im_width = cfg['im_width']

        (_, _, _, _, _, tbin_depth_res) = calculate_tof_domain_params(args.n_tbins, cfg['rep_tau'])

        total_trials = coded_vals.shape[0]
        trials = min(total_trials, args.num_trials)
        coded_vals_trials = np.sum(coded_vals[:trials], axis=0)

        depth_map, _ = decode_depth_map(
            coded_vals_trials,
            coding_matrix,
            im_width,
            tbin_depth_res,
            args.use_full_correlations,
        )
        depth_map = filter_hot_pixels(depth_map, hot_mask)

        if not args.correct_master:
            depth_map = depth_map[:, : im_width // 2]

        if args.mask_background_pixels:
            depth_map = depth_map[40:450, :]

        depth_map = median_filter(depth_map, size=args.median_filter_size)

        vmin = args.vmin if args.vmin is not None else np.nanmin(depth_map)
        vmax = args.vmax if args.vmax is not None else np.nanmax(depth_map)
        depth_map = np.clip(depth_map, vmin, vmax)

        # --- preview ---
        plt.figure()
        plt.imshow(depth_map, cmap='hot', vmin=vmin, vmax=vmax)
        plt.title(f"{r['capture_type']} k={r['k']}")
        plt.colorbar(label='Depth (m)')
        plt.tight_layout()
        plt.show()

        # --- 3-D reconstruction ---
        points = depth_to_pointcloud(depth_map, args.fov_major_axis_deg)
        print(f"Point cloud: {len(points):,} points")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")

        if len(points) < 10:
            print("Too few points for mesh reconstruction — skipping.")
            continue

        mesh, pcd_clean = build_mesh_from_pcd(pcd)

        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            print("Mesh reconstruction produced no geometry — skipping.")
            continue

        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        mesh = colorize_mesh_by_depth(mesh)

        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, window_name="Mesh")
