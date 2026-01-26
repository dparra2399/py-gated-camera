from spad_lib.spad512utils import *
from utils.file_utils import *
from plot_scripts.plot_utils import *

# -----------------------------------------------------------------------------
# CONFIG (capitalized)
# -----------------------------------------------------------------------------
PIXEL_PITCH = 16.38 #in uM
FOCAL_LENGTH = 25 #in mm
EXP = 3
K = 8
DATASET_TYPE = 'coarse'  # was `type`

N_TBINS_DEFAULT = 1500
VMIN = None
VMAX = None
MEDIAN_FILTER_SIZE = 5
CORRECT_MASTER = False
MASK_BACKGROUND_PIXELS = True
USE_CORRELATIONS = True
USE_FULL_CORRELATIONS = False
SIGMA_SIZE = None
SHIFT_SIZE = 150
CORRECT_DEPTH_DISTORTION = False
HAM_TMP_CORRELATIONS = ''

# paths

TEST_FILE = f'/Volumes/velten/Research_Users/David/Gated_Camera_Project/gated_project_data/exp{EXP}/{DATASET_TYPE}k{K}_exp{EXP}.npz'
GT_FILE   = f'/Volumes/velten/Research_Users/David/Gated_Camera_Project/gated_project_data/exp{EXP}/{DATASET_TYPE}k{K}_gt_exp{EXP}.npz'
HOT_MASK_PATH = '/Users/davidparra/PycharmProjects/py-gated-camera/masks/hot_pixels.PNG'

EPSILON = 1e-12

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    hot_mask = load_hot_mask(HOT_MASK_PATH)

    gt_depth_map = None
    depth_map = None
    coded_vals_save = None
    coding_matrix_save = None
    coded_vals = None
    coding_matrix = None
    mask = None

    for path in [TEST_FILE, GT_FILE]:

        try:
            f = np.load(path)
        except FileNotFoundError:
            continue

        coded_vals = f["coded_vals"]
        total_time = f["total_time"]
        im_width = f["im_width"]
        freq = float(f["freq"])
        voltage = f["voltage"]
        try:
            size = f["size"]
        except KeyError:
            size = f['duty']
        split_measurements = f["split_measurements"]

        # TOF params -------------------------------------------------------
        (   rep_tau,
             rep_freq,
             tbin_res,
             t_domain,
             max_depth,
             tbin_depth_res,
         ) = calculate_tof_domain_params(N_TBINS_DEFAULT, 1.0 / freq)
        mhz = int(freq * 1e-6)

        # -----------------------------------------------------------------
        # choose coding matrix depending on file type
        # -----------------------------------------------------------------
        if USE_CORRELATIONS:
            if 'coarse' in path:
                name_tmp = 'coarse'
                tmp = ''
            elif 'ham' in path:
                name_tmp = 'ham'
                tmp = HAM_TMP_CORRELATIONS
            else:
                assert False
            corr_path = (
                f"/Users/davidparra/PycharmProjects/py-gated-camera/correlation_functions/"
                f"{name_tmp}k{coded_vals.shape[-1]}_{mhz}mhz_{voltage}v_{size}w_correlations{tmp}.npz"
            )
            correlations_total, n_tbins_corr = load_correlations_file(corr_path)
            coding_matrix = build_coding_matrix_from_correlations(correlations_total, USE_FULL_CORRELATIONS,
                                                                  SIGMA_SIZE, SHIFT_SIZE, N_TBINS_DEFAULT)
        elif "coarse" in path:
            gate_width = f["gate_width"]

            irf = get_voltage_function(mhz, voltage, size, "pulse", N_TBINS_DEFAULT)
            coding_matrix = get_coarse_coding_matrix(
                gate_width * 1e3,
                coded_vals.shape[-1],
                0,
                gate_width * 1e3,
                rep_tau * 1e12,
                N_TBINS_DEFAULT,
                irf,
            )

        elif "ham" in path:
            if "pulse" in path:
                illum_type = "pulse"
            else:
                illum_type = "square"

            coding_matrix = get_hamiltonain_correlations(
                coded_vals.shape[-1], mhz, voltage, size, illum_type, n_tbins=N_TBINS_DEFAULT
            )
        else:
            assert False, 'Path needs to be "hamiltonian" or "coarse"'

        name = get_scheme_name(path, coded_vals.shape[-1])
        print(
            f"{name}\n\tVoltage: {voltage}\n\tSize: {size}\n\tTotal Time: {total_time}\n\tSplit Measurements: {split_measurements}"
        )

        # -----------------------------------------------------------------
        # decode to depth map
        # -----------------------------------------------------------------
        depth_map, zncc = decode_depth_map(
            coded_vals,
            coding_matrix,
            im_width,
            N_TBINS_DEFAULT,
            tbin_depth_res,
            USE_CORRELATIONS,
            USE_FULL_CORRELATIONS,
        )

        if CORRECT_DEPTH_DISTORTION:
            fx, fy, cx, cy = intrinsics_from_pixel_pitch(im_width, im_width, FOCAL_LENGTH, PIXEL_PITCH)

        # hot pixel fix
        filtered = median_filter(depth_map, size=3, mode="nearest")
        depth_map[hot_mask == 1] = filtered[hot_mask == 1]


        # optional crop for master
        if CORRECT_MASTER is False:
            depth_map = depth_map[:, : im_width // 2]
            coded_vals = coded_vals[:, : im_width // 2]

        if MASK_BACKGROUND_PIXELS:
            depth_map = depth_map[20:450, :]
            mask = None
        # assign to GT or test map
        if path == GT_FILE:
            gt_depth_map = np.copy(depth_map)
            # if MASK_BACKGROUND_PIXELS:
            #     mask = cluster_kmeans(np.copy(gt_depth_map), n_clusters=2)
            #     mask[mask == np.nanmax(mask)] = np.nan
            #     mask[mask == np.nanmin(mask)] = 1

        else:
            depth_map = np.copy(depth_map)
            coded_vals_save = np.copy(coded_vals)
            coding_matrix_save = np.copy(coding_matrix)

    # apply mask to depth map if requested
    # if MASK_BACKGROUND_PIXELS and gt_depth_map is not None:
    #     depth_map *= mask
    if CORRECT_MASTER is False:
        hot_mask = hot_mask[:, : im_width // 2]
    if coded_vals_save is None:
        coded_vals_save = np.copy(coded_vals)
        coding_matrix_save = np.copy(coding_matrix)


    coded_vals_filt = np.zeros_like(coded_vals_save)
    for i in range(coded_vals_save.shape[-1]):
        filt = median_filter(np.copy(coded_vals_save[:, :, i]), size=3, mode='nearest')
        filt[hot_mask == 1] = filt[hot_mask == 1]
        coded_vals_filt[:, :, i] = filt


    # ------------------------------------------------------------------
    # Diagnostics plots (coding vs coded at a few points)
    # ------------------------------------------------------------------
    # fallbacks if fewer than K points are given
    x1, y1 = (180, 40)
    x2, y2 = (180, 160)
    x3, y3 = (180, 330)
    points = [(x1, y1), (x2, y2), (x3, y3)]
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']


    plot_gated_images(
            coded_vals_filt,
            depth_map,
            gt_depth_map,
            vmin=VMIN,
            vmax=VMAX,
            median_filter_size=MEDIAN_FILTER_SIZE,
    )

    plot_sample_points(
            coded_vals_save,
            coding_matrix_save,
            points,
            depth_map,
            tbin_depth_res,
            vmin=VMIN,
            vmax=VMAX,
            median_filter_size=MEDIAN_FILTER_SIZE,
    )

    plot_sample_points_simple(
            coded_vals_save,
            coding_matrix_save,
            points,
            depth_map,
            tbin_depth_res,
            USE_FULL_CORRELATIONS,
            colors,
    )

    print(f'min depth map: {np.min(depth_map)}, max depth map: {np.max(depth_map)}')


