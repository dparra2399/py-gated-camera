import os
import glob
from spad_lib.SPAD512S import SPAD512S
from spad_lib.spad512utils import *
import numpy as np
import time
import matplotlib.pyplot  as plt
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter, median_filter, gaussian_filter1d
from felipe_utils import CodingFunctionsFelipe
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import math
import open3d as o3d
from PIL import Image


exp = 3
k = 3
type = 'coarse'
n_tbins = 640
vmin = None
vmax = None
median_filter_size = 3
correct_master = False
mask_background_pixels = True
use_correlations = True
fov_major_axis_deg = 5



test_file = f'/Volumes/velten/Research_Users/David/Gated_Camera_Project/gated_project_data/exp{exp}/{type}k{k}_exp{exp}.npz'

hot_mask_filename = '/Users/davidparra/PycharmProjects/py-gated-camera/masks/hot_pixels.PNG'
hot_mask = np.array(Image.open(hot_mask_filename))
hot_mask[hot_mask < 5000] = 0
hot_mask[hot_mask > 0] = 1

file = np.load(test_file)

coded_vals = file['coded_vals']
total_time = file["total_time"]
im_width = file["im_width"]
bitDepth = file["bitDepth"]
iterations = file["iterations"]
overlap = file["overlap"]
timeout = file["timeout"]
pileup = file["pileup"]
gate_steps = file["gate_steps"]
gate_step_arbitrary = file["gate_step_arbitrary"]
gate_step_size = file["gate_step_size"]
gate_offset = file["gate_offset"]
gate_direction = file["gate_direction"]
gate_trig = file["gate_trig"]
voltage = file["voltage"]
freq = file["freq"]

(rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, 1 / freq)
mhz = int(freq * 1e-6)

if 'coarse' in test_file:
    gate_width = file["gate_width"]
    num_gates = file["num_gates"]
    if num_gates == 3:
        size = 34
        voltage = 7
    elif num_gates == 4:
        size = 25
        voltage = 7.6
    else:
        size = 12
        voltage = 10

    if use_correlations:
        try:
            correlaions_filepath = f'/Users/davidparra/PycharmProjects/py-gated-camera/correlation_functions/coarsek{k}_{mhz}mhz_{voltage}v_{size}w_correlations.npz'
            file = np.load(correlaions_filepath)
        except FileNotFoundError:
            raise 'What? Your file was not found. Sorry ;/'

        correlations_total = file['correlations']
        coding_matrix = np.transpose(np.mean(np.mean(correlations_total[200:400, 100:256], axis=0), axis=0))
        n_tbins = file['n_tbins']
        (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(
            n_tbins, 1. / float(freq))
        coding_matrix = np.roll(coding_matrix, shift=150, axis=0)
        coding_matrix = gaussian_filter1d(coding_matrix, sigma=20, axis=0)
    else:
        irf = get_voltage_function(mhz, voltage, size, 'pulse', n_tbins)
        coding_matrix = get_coarse_coding_matrix(gate_width * 1e3, num_gates, 0, gate_width * 1e3, rep_tau * 1e12,
                                                 n_tbins, irf)
    # plt.imshow(coding_matrix.transpose(), aspect='auto')
    # plt.show()
elif 'ham' in test_file:
    K = coded_vals.shape[-1]
    if 'pulse' in test_file:
        illum_type = 'pulse'
        size = 12
        voltage = 10

    else:
        illum_type = 'square'
        size = 20

    if use_correlations:
        try:
            correlaions_filepath = f'/Users/davidparra/PycharmProjects/py-gated-camera/correlation_functions/hamk{k}_{mhz}mhz_{voltage}v_{size}w_correlations.npz'
            file = np.load(correlaions_filepath)
        except FileNotFoundError:
            raise 'What? Your file was not found. Sorry ;/'
        correlations_total = file['correlations']
        coding_matrix = np.transpose(np.mean(np.mean(correlations_total[200:400, 100:256], axis=0), axis=0))
        n_tbins = file['n_tbins']
        (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(
            n_tbins, 1. / float(freq))
        coding_matrix = np.roll(coding_matrix, shift=150, axis=0)
        coding_matrix = gaussian_filter1d(coding_matrix, sigma=20, axis=0)
    else:
        coding_matrix = get_hamiltonain_correlations(K, mhz, voltage, size, illum_type, n_tbins=n_tbins)
else:
    exit(0)

norm_coding_matrix = zero_norm_t(coding_matrix)

norm_coded_vals = zero_norm_t(coded_vals)

print(norm_coded_vals.shape)
print(norm_coding_matrix.shape)

zncc = np.matmul(norm_coding_matrix, norm_coded_vals[..., np.newaxis]).squeeze(-1)


depths = np.argmax(zncc, axis=-1)
depth_map = np.reshape(depths, (512, 512)) * tbin_depth_res

if correct_master == False:
    depth_map = depth_map[: ,: im_width // 2]

if mask_background_pixels == True:
    depth_map = depth_map[30:400, :]

depth_map = median_filter(depth_map, size=median_filter_size)

if vmin == None:
    vmin = np.min(depth_map)
if vmax == None:
    vmax = np.max(depth_map)

depth_map[depth_map < vmin] = vmin
depth_map[depth_map > vmax] = vmax

plt.imshow(depth_map, cmap='hot')
plt.show()

(nr, nc) = depth_map.shape[0:2]
# FOV along the major axis (in degrees)
# FOV along the major axis (in degrees)
fov_major_axis_rad = np.radians(fov_major_axis_deg)

height, width = depth_map.shape[:2]

# intrinsics (assume FOV is horizontal / major axis = width)
fx = width / (2 * np.tan(fov_major_axis_rad / 2))
fy = fx  # square pixels

# correct principal point!
cx = width / 2.0
cy = height / 2.0

print(f"Estimated Intrinsics:\nfx = {fx:.2f}, fy = {fy:.2f}, cx = {cx:.2f}, cy = {cy:.2f}")

# grid
x, y = np.meshgrid(np.arange(width), np.arange(height))

# your depth scale
z = depth_map.astype(np.float32) / 20.0

# backproject
x3d = (x - cx) * z / fx
y3d = (y - cy) * z / fy

points = np.stack((x3d, y3d, z), axis=-1).reshape(-1, 3)

valid_points = ~np.isnan(points).any(axis=1) & (points[:, 2] > 0)
points = points[valid_points]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
pcd = pcd.select_by_index(ind)

#pcd = pcd.voxel_down_sample(voxel_size=0.0005)
pcd.remove_duplicated_points()
pcd.remove_non_finite_points()
pcd, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.02)

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

o3d.visualization.draw_geometries([pcd], window_name="Mesh ")

#exit(0)
pcd.orient_normals_consistent_tangent_plane(k=30)

o3d.visualization.draw_geometries([pcd], window_name="Mesh ")

#pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))


mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)

mesh = mesh.filter_smooth_laplacian(number_of_iterations=10, lambda_filter=0.5)

mesh.remove_non_manifold_edges()


mesh_subdivided = mesh.subdivide_midpoint(number_of_iterations=2)

vertices_to_remove = densities < np.quantile(densities, 0.05)
mesh.remove_vertices_by_mask(vertices_to_remove)



if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
    raise ValueError("Mesh has no vertices or faces!")


if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()

# Get the normals
normals = np.asarray(mesh.vertex_normals)

# Normalize the normals to [0, 1] range for color mapping
normals_normalized = (normals + 1) / 2  # This gives the range [0, 1]

# Calculate the intensity for the grey color (based on the average of the normals)
# We can use the Y component of the normal to simulate shading, but here we use the magnitude of the normals
intensity = np.linalg.norm(normals_normalized, axis=1)  # This gives a scalar intensity for each vertex

# Ensure the intensity is within the [0, 1] range
intensity = np.clip(intensity * 0.8, 0, 1)

# Set grey shading: Apply intensity to all RGB channels for each vertex
# grey_shading = np.stack([intensity, intensity, intensity], axis=1)  # Apply the same intensity to R, G, B
#
# # Set the colors to the mesh vertices
# mesh.vertex_colors = o3d.utility.Vector3dVector(grey_shading)

verts = np.asarray(mesh.vertices)
z = verts[:, 2]

# Normalize z for colormap
z_min, z_max = z.min(), z.max()
z_norm = (z - z_min) / (z_max - z_min + 1e-8)

# Apply a colormap (choose 'turbo', 'viridis', 'plasma', etc.)
cmap = plt.get_cmap("turbo")
colors = cmap(z_norm)[:, :3]

mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, window_name="Mesh")


# theta = np.radians(25)
#
# R_y = np.array([[-np.cos(theta), 0, -np.sin(theta)],
#                 [0, 1, 0],
#                 [np.sin(theta), 0, -np.cos(theta)]])
# # Apply the rotation matrix to the mesh vertices
# mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) @ R_y.T)  # Apply rotation
#
# R_z = np.array([[-1, 0, 0],
#                 [0, -1, 0],
#                 [0, 0, 1]])
#
# # Apply the rotation matrix to the mesh vertices
# mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) @ R_z.T)
#
# theta = np.deg2rad(15)  # Convert to radians
#
# # Rotation matrix around X-axis
# R_x = np.array([[1, 0, 0],
#                 [0, np.cos(theta), -np.sin(theta)],
#                 [0, np.sin(theta), np.cos(theta)]])
#
# # Apply the rotation matrix to the mesh vertices
# mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) @ R_x.T)  # Apply rotation

#o3d.visualization.draw_geometries([pcd], window_name="Mesh ")
# vis = o3d.visualization.Visualizer()
#
# vis.create_window(visible=False)  # Set visible=False to avoid popping up a window
# vis.add_geometry(mesh)
#
# # Update and render the scene
# vis.poll_events()
# vis.update_renderer()
#
# # Add the mesh to the scene
# vis.add_geometry(mesh)
#
# # Capture the screenshot and save it to a file
# vis.capture_screen_image(f"image.png")  # Save the image as PNG