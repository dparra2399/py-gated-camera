import glob

from spad_lib.spad512utils import *
from utils.file_utils import *
from plot_scripts.plot_utils import *
from utils.global_constants import *

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
PIXEL_PITCH = 16.38 #in uM
FOCAL_LENGTH = 25 #in mm
EXP_NUM = 3
K_FILTER = [3]
N_TBINS_DEFAULT = 2000
VMIN = 14
VMAX = 16
MEDIAN_FILTER_SIZE = 3
SIGMA_SIZE = 1 #How much to smooth correlations functions
SHIFT_SIZE = 75 #How much to shift correlations functions
USE_CORRELATIONS = True
USE_FULL_CORRELATIONS = False
CORRECT_MASTER = False
MASK_BACKGROUND_PIXELS = True
INCLUDE_SPLIT_MEASUREMENTS = False
CORRECT_DEPTH_DISTORTION = False
NORMALIZE_DEPTH_MAPS = False
HAM_TMP_CORRELATIONS = ''
SUB_FOLDER = ''

SUB_FOLDER = SUB_FOLDER if SUB_FOLDER=='' else SUB_FOLDER + '/'

HOT_MASK_PATH_MAC = "./masks/hot_pixels.PNG"
DATA_FOLDER_MAC = os.path.join(READ_PATH_CAPTURE_MAC, f"{SUB_FOLDER}exp{EXP_NUM}")
DATA_FOLDER_WINDOWS = os.path.join(READ_PATH_CAPTURE_WINDOWS, f"{SUB_FOLDER}exp{EXP_NUM}")

EPSILON = 1e-12


if __name__ == "__main__":
    folder = get_data_folder(DATA_FOLDER_MAC, DATA_FOLDER_WINDOWS)
    correlation_folder = get_data_folder(READ_PATH_CORRELATIONS_MAC, READ_PATH_CORRELATIONS_WINDOWS)
    hot_mask = load_hot_mask(HOT_MASK_PATH_MAC)

    npz_files = glob.glob(os.path.join(folder, "*.npz"))
    npz_files = filter_npz_files(npz_files, K_FILTER)

    depths_maps_dict = {}
    gt_depth_maps_dict = {}

    assert len(npz_files) > 0, f'No files found in {folder}'

    for path in npz_files:
        f = np.load(path)

        coded_vals = f["coded_vals"]
        total_time = f["total_time"]
        im_width = f["im_width"]
        freq = float(f["freq"])
        voltage = f["voltage"]
        try:
            size = f["size"]
        except KeyError:
            size = f['duty']

        split_measurements = bool(f["split_measurements"])
        if split_measurements != INCLUDE_SPLIT_MEASUREMENTS and 'gt' not in os.path.basename(path):
            continue

        # TOF params -------------------------------------------------------
        (rep_tau,
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
                os.path.join(correlation_folder,
                f"{name_tmp}k{coded_vals.shape[-1]}_{mhz}mhz_{voltage}v_{size}w_correlations{tmp}.npz")
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

        # masking / GT handling
        if "GroundTruth" in name and MASK_BACKGROUND_PIXELS:
            # mask = cluster_kmeans(np.copy(depth_map), n_clusters=2)
            # mask[mask == np.nanmax(mask)] = np.nan
            # mask[mask == np.nanmin(mask)] = 1
            depth_map = depth_map[50:450, :]
            mask = None
        elif MASK_BACKGROUND_PIXELS:
            depth_map = depth_map[50:450, :]
            mask = None
        else:
            mask = None

        if "GroundTruth" in name:
            gt_depth_maps_dict[name] = [np.copy(depth_map), mask]
        else:
            depths_maps_dict[name] = np.copy(depth_map)

    # ---------------------------------------------------------------------
    # plotting
    # ---------------------------------------------------------------------
    if NORMALIZE_DEPTH_MAPS:
        depth_maps_normalized = np.stack(
            [dm - np.mean(dm) for dm in depths_maps_dict.values()],
            axis=-1,
        )
    else:
        depth_maps_normalized = np.stack(
            [dm for dm in depths_maps_dict.values()],
            axis=-1,
        )
    vmin = VMIN if VMIN is not None else np.nanmin(depth_maps_normalized)
    vmax = VMAX if VMAX is not None else np.nanmax(depth_maps_normalized)

    x, y = 20, 170
    width, height = 220, 320

    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(len(depths_maps_dict), 4, height_ratios=[1] * len(depths_maps_dict))

    for i, (name, depth_map) in enumerate(depths_maps_dict.items()):
        gt_info = gt_depth_maps_dict.get(name.split("_")[0] + "_GroundTruth", None)

        if gt_info is not None:
            gt_depth_map, mask = gt_info
        else:
            gt_depth_map, mask = None, None

        if gt_depth_map is not None and mask is not None:
            mask[~np.isnan(mask)] = 1
            depth_map = depth_map * mask
            gt_depth_map = gt_depth_map * mask

        depth_map_plot = depth_map - np.mean(depth_map) if NORMALIZE_DEPTH_MAPS else np.copy(depth_map)

        if CORRECT_MASTER:
            depth_map_plot[:, :im_width // 2] = -depth_map_plot[:, :im_width // 2]
        patch = depth_map_plot[y : y + height, x : x + width]

        # full map ---------------------------------------------------------
        ax = fig.add_subplot(gs[i, 0])
        im = ax.imshow(
            median_filter(depth_map_plot, size=MEDIAN_FILTER_SIZE), vmin=vmin, vmax=vmax
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Depth (meters)")
        ax.set_title(name)
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)

        # close-up ---------------------------------------------------------
        ax2 = fig.add_subplot(gs[i, 1])
        im2 = ax2.imshow(
            median_filter(patch, size=MEDIAN_FILTER_SIZE), vmin=vmin, vmax=vmax
        )
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Depth (meters)")
        ax2.set_title("Close-Up")

        # GT / error -------------------------------------------------------
        ax3 = fig.add_subplot(gs[i, 2])
        if gt_depth_map is not None:
            gt_depth_map_plot = gt_depth_map - np.mean(gt_depth_map) if NORMALIZE_DEPTH_MAPS else np.copy(gt_depth_map)

            error = np.nanmean(np.abs(depth_map - gt_depth_map))
            rmse = np.sqrt(np.nanmean((depth_map - gt_depth_map) ** 2))
            im3 = ax3.imshow(
                median_filter(gt_depth_map_plot, size=MEDIAN_FILTER_SIZE), vmin=vmin, vmax=vmax
            )
            fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="Depth (meters)")
            ax3.set_xlabel(f"MAE: {error * 1000: .3f} mm\nRMSE: {rmse * 1000: .3f} mm")
            ax3.set_title("Ground Truth")
        else:
            ax3.set_axis_off()

        # Error map -------------------------------------------------------
        ax4 = fig.add_subplot(gs[i, 3])
        error_map = np.abs(depth_map - gt_depth_map)
        im4 = ax4.imshow(error_map, vmin=0, vmax=0.5)
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label="Absolute Error (m)")
        ax4.set_title("Error Map")

    fig.subplots_adjust(left=0.05, right=0.98, top=0.97, bottom=0.05, wspace=0.05, hspace=0.05)
    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
    plt.show()


