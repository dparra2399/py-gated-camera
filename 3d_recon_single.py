import os
import glob
from spad_lib.SPAD512S import SPAD512S
from spad_lib.spad512utils import *
import numpy as np
import time
import matplotlib.pyplot  as plt
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter, median_filter
from felipe_utils import CodingFunctionsFelipe
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import math
import open3d as o3d



correct_master = False
exp = 8
n_tbins = 1_024

#filename = f'/Volumes/velten/Research_Users/David/Gated_Camera_Project/gated_project_data/exp{exp}/coarsek3_exp{exp}.npz'
#filename = f'/Volumes/velten/Research_Users/David/Gated_Camera_Project/gated_project_data/exp{exp}/hamK3_exp{exp}.npz'

filename = f'/Volumes/velten/Research_Users/David/Gated_Camera_Project/gated_project_data/exp{exp}/coarse_gt_exp{exp}.npz'
#filename = f"/mnt/researchdrive/research_users/David/Gated_Camera_Project/gated_project_data/exp{exp}/coarse_gt_exp{exp}.npz"



file = np.load(filename)

coded_vals = file['coded_vals']
irf = file['irf']
total_time = file["total_time"]
num_gates = file["num_gates"]
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

if 'coarse' in filename:
    gate_width = file["gate_width"]
    if num_gates == 3:
        size = 34
    elif num_gates == 4:
        size = 25
    else:
        size = 12
    irf = get_voltage_function(mhz, voltage, size, 'pulse', n_tbins)
    coding_matrix = get_coarse_coding_matrix(gate_width * 1e3, num_gates, 0, gate_width * 1e3, rep_tau * 1e12, n_tbins, irf)
    # plt.imshow(coding_matrix.transpose(), aspect='auto')
    # plt.show()
elif 'ham' in filename:
    K = coded_vals.shape[-1]
    coding_matrix = get_hamiltonain_correlations(K, mhz, voltage, 20, n_tbins)
    #plt.plot(coding_matrix)
    #plt.show()
else:
    exit(0)

norm_coding_matrix = zero_norm_t(coding_matrix)

norm_coded_vals = zero_norm_t(coded_vals)

print(norm_coded_vals.shape)
print(norm_coding_matrix.shape)

zncc = np.matmul(norm_coding_matrix, norm_coded_vals[..., np.newaxis]).squeeze(-1)

if correct_master:
    zncc[:, im_width // 2:, :] = np.roll(zncc[:, im_width //2:, :], shift=870)


depths = np.argmax(zncc, axis=-1)

depth_map = np.reshape(depths, (512, 512)) * tbin_depth_res

depth_map = median_filter(depth_map, size=3)

depth_map = depth_map[:, :im_width // 2]
depth_map[depth_map < 6] = 6
depth_map[depth_map > 6.5] = 6.5

(nr, nc) = depth_map.shape[0:2]
# FOV along the major axis (in degrees)
fov_major_axis_deg = 5
fov_major_axis_rad = np.radians(fov_major_axis_deg)  # Convert to radians

# Calculate focal length
fx = nc / (2 * np.tan(fov_major_axis_rad / 2))
fy = fx  # Assume square pixels if no other info is provided

# Principal point (image center)
cx, cy = nr / 2, nc / 2

print(f"Estimated Intrinsics:\nfx = {fx:.2f}, fy = {fy:.2f}, cx = {cx:.2f}, cy = {cy:.2f}")


height, width = depth_map.shape[0:2]

# Generate mesh grid
x, y = np.meshgrid(np.arange(width), np.arange(height))

# Convert depth to meters if necessary (e.g., divide by 1000 if in mm)
z = depth_map.astype(np.float32) / 20.0

# Reproject to 3D using intrinsics
x3d = (x - cx) * z / fx
y3d = (y - cy) * z / fy
points = np.stack((x3d, y3d, z), axis=-1).reshape(-1, 3)

valid_points = ~np.isnan(points).any(axis=1) & (points[:, 2] > 0)
points = points[valid_points]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
pcd = pcd.select_by_index(ind)

pcd = pcd.voxel_down_sample(voxel_size=0.0005)
pcd.remove_duplicated_points()
pcd.remove_non_finite_points()
pcd, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.02)

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

o3d.visualization.draw_geometries([pcd], window_name="Mesh ")

exit(0)
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
grey_shading = np.stack([intensity, intensity, intensity], axis=1)  # Apply the same intensity to R, G, B

# Set the colors to the mesh vertices
mesh.vertex_colors = o3d.utility.Vector3dVector(grey_shading)

o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, window_name="Mesh ")

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