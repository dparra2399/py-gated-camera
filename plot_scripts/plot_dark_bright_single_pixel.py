import pprint
import shutil
import zipfile

from matplotlib import pyplot as plt

from spad_lib.spad512utils import get_gate_shifts
from utils.file_utils import *
from plot_scripts.plot_utils import plot_single_pixel_dist, plot_single_pixel_corr, plot_single_pixel_depth_pairs, \
    get_string_name, get_single_pixel_title, get_cap_color
from utils.global_constants import *
from utils.global_constants import get_single_pixel_coords, get_total_pixels
from utils.parameter_classes import DecodeConfig
from utils.tof_utils import build_coding_matrix_from_correlations, get_simulated_coding_matrix, sliding_gate_width_tbins, \
    calculate_tof_domain_params, decode_single_pixel_experiment
import numpy as np

# -----------------------------------------------------------------------------
# CONFIG (capitalized)
# ----------------------------------------------------------------------------
# List of rows; each row is a list of groups (one subplot per group);
# each group is a list of exp_paths aggregated into that subplot.
EXP_PATHS = [
    # [['k3_HIGHSNR'], ['k3_LOWSNR']],
    # [['k4_HIGHSNR'], ['k4_LOWSNR']]
    [['k8_12_16_HIGHSNR'], ['k8_12_16_LOWSNR']]
]

N_TBINS = 2000
ERROR_TYPE = "MAE"

#Which correlation functions to use
SIMULATED_CORRELATIONS = False

#Smoothing or shifting the correlation functions
SIGMA_SIZE = 1 #None if no smoothing
SHIFT_SIZE = None #None if no shifting

#Not apart of the defaults
# n-pixel sweep knobs. The upper bound is derived per-capture at runtime,
# because the ROI (and so its pixel count) depends on the capture's im_width.
N_PIXELS_START = 11
N_PIXELS_STEP = 10

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    def apply_decode_defaults(cfg: DecodeConfig) -> DecodeConfig:
        defaults = dict(
            n_tbins=N_TBINS,

            simulated_correlations=SIMULATED_CORRELATIONS,

            smooth_sigma=SIGMA_SIZE,
            shift=SHIFT_SIZE,

        )

        # fill only missing (None)
        for k, v in defaults.items():
            if getattr(cfg, k) is None:
                setattr(cfg, k, v)

        return cfg

    parser = build_parser_from_config(DecodeConfig)
    args = parser.parse_args()
    cfg = apply_decode_defaults(DecodeConfig(**vars(args)))

    correlation_folder = get_data_folder(READ_PATH_CORRELATIONS_SINGLE_PIXEL_MAC,
                                         READ_PATH_CORRELATIONS_SINGLE_PIXEL_WINDOWS)
    base_capture_folder = get_data_folder(READ_PATH_SINGLE_PIXEL_MAC, READ_PATH_SINGLE_PIXEL_WINDOWS)

    exp_paths = EXP_PATHS if cfg.exp_path is None else [[[cfg.exp_path]]]

    all_depths_dicts = []  # all_depths_dicts[row][col] -> list of cfg_dicts

    for exp_path_row in exp_paths:
      row_depths_dicts = []

      for exp_path_group in exp_path_row:
        depths_dict = []

        for exp_path in exp_path_group:
            capture_folder, cleanup_dir = get_capture_folder(
                os.path.join(base_capture_folder, exp_path), return_cleanup=True)

            capture_paths = os.listdir(capture_folder)
            capture_paths = [p for p in capture_paths if os.path.isfile(os.path.join(capture_folder, p))]
            capture_paths = filter_capture_files(capture_paths)


            for i, coded_vals_name in enumerate(capture_paths):
                if coded_vals_name.startswith('.'):
                    continue
                coded_vals_path = os.path.join(capture_folder, coded_vals_name)
                if not os.path.isfile(coded_vals_path):
                    continue
                capture_file = load_npz(coded_vals_path)
                params = capture_file['cfg'].item()
                coded_vals = capture_file['coded_vals']
                im_width = params['im_width']
                # ROI depends on the captured frame width (512 vs cropped)
                coords = get_single_pixel_coords(im_width)
                total_pixels = get_total_pixels(coords)
                n_pixels_range = np.arange(N_PIXELS_START, total_pixels, N_PIXELS_STEP)
                mA = params['current']
                mV = params['high_level_amplitude'] * 1000
                capture_type = params['capture_type']
                k = params['k']
                freq = params['rep_rate']
                freq_mhz = freq * 1e-6
                duty = params['duty']
                rep_tau = params['rep_tau']


                gate_widths, gate_starts = get_gate_shifts(capture_type, freq, k, params.get('sliding_gate_width'))
                total_count = sum(len(sublist) for sublist in gate_widths)
                int_time = params['int_time'] * total_count

                #pprint.pprint(params)

                corr_path = os.path.join(correlation_folder, make_correlation_filename(capture_type, k,
                                                                                       freq_mhz, mV, mA, duty))

                gt_coded_vals_path = os.path.join(capture_folder,
                                                  make_capture_filename(capture_type, k, freq_mhz, mV, mA, duty,
                                                                        None, True))

                correlations_total = load_correlation_npz(corr_path)['correlations']

                if cfg.simulated_correlations:
                    coding_matrix = get_simulated_coding_matrix(capture_type, cfg.n_tbins, k,
                                                        sliding_gate_width_tbins(params.get('sliding_gate_width'), rep_tau, cfg.n_tbins))
                else:
                    coding_matrix = build_coding_matrix_from_correlations(
                        correlations_total,
                        False,
                        cfg.smooth_sigma,
                        cfg.shift,
                        cfg.n_tbins,
                    )

                n_tbins = cfg.n_tbins if cfg.n_tbins is not None else coding_matrix.shape[0]
                (rep_tau, rep_freq, tbin_res,
                 t_domain, max_depth, tbin_depth_res,) = calculate_tof_domain_params(n_tbins, rep_tau)

                mae_list = []
                rmse_list = []
                int_times = []

                seed = 0 if k > 4 else 1

                pixel_order = np.random.default_rng(0).permutation(total_pixels)

                #coded_vals_gt = np.load(gt_coded_vals_path, allow_pickle=True)['coded_vals']

                gt_depths, recon_gt, _ = decode_single_pixel_experiment(
                    capture_type + "s",
                    coded_vals,
                    coding_matrix,
                    tbin_depth_res,
                    coords['y'],
                    coords['x'],
                    n_pixels=total_pixels,
                    pixel_order=pixel_order,
                )

                for i, n in enumerate(n_pixels_range):

                    depths, recon, num_pixels = decode_single_pixel_experiment(
                        capture_type + "s",
                        coded_vals,
                        coding_matrix,
                        tbin_depth_res,
                        coords['y'],
                        coords['x'],
                        n_pixels=n,
                        pixel_order=pixel_order
                    )

                    #if capture_type == 'timeslicing': depths = np.roll(depths, -2, axis=-1)

                    phase_shifts = params['phase_shifts']#[2:-2]
                    #depths = depths[:, 2:-2]
                    #gt_depths = gt_depths[:, 2:-2]

                    mae = np.nanmean(np.abs(depths - gt_depths), axis=0) * 1000
                    rmse = np.sqrt(np.nanmean((depths - gt_depths) ** 2))* 1000

                    # mae[3] = np.nan
                    # mae[7] = np.nan

                    mae = np.nanmean(mae)

                    if mae < 100000:
                        mae_list.append(mae)
                        rmse_list.append(rmse)
                        int_times.append(num_pixels * int_time / 1000)

                cfg_dict = asdict(cfg)
                cfg_dict.update({'depths': depths, 'gt_depths': gt_depths,
                                 'rmse': rmse_list, 'mae': mae_list, 'coding_matrix': coding_matrix,
                                 'tbin_res': tbin_res, 'tbin_depth_res': tbin_depth_res,
                                 'phase_shifts': phase_shifts, 'capture_type': capture_type,
                                 'int_times': int_times, "k": k})
                #cfg_dict.update(params)

                depths_dict.append(cfg_dict)
                capture_file.close()  # release the npz handle before deleting

            # done reading this archive — delete the unzipped copy now so at
            # most one archive is expanded on disk at a time
            if cleanup_dir is not None:
                shutil.rmtree(cleanup_dir, ignore_errors=True)

        row_depths_dicts.append(depths_dict)

      all_depths_dicts.append(row_depths_dicts)


    n_rows = len(exp_paths)
    n_cols = max(len(row) for row in exp_paths)
    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    # per-cell bookkeeping for the shared-axis / shared-title post-pass
    cell_title = [[None] * n_cols for _ in range(n_rows)]
    cell_xmax  = [[None] * n_cols for _ in range(n_rows)]

    for i, row_depths_dicts in enumerate(all_depths_dicts):
      # hide any unused axes in a short row
      for j in range(len(row_depths_dicts), n_cols):
          axs[i][j].set_axis_off()

      for j, depths_dict in enumerate(row_depths_dicts):
        ax = axs[i][j]
        xmax = 0
        for idx, inner_dict in enumerate(depths_dict):
            rmse = inner_dict['rmse']
            mae = inner_dict['mae']
            int_times = inner_dict['int_times']
            capture_type = inner_dict['capture_type']
            k = inner_dict['k']
            capture_type = capture_type + "pw" if k > 4 else capture_type

            if ERROR_TYPE == "MAE":
                error = mae
            elif ERROR_TYPE == "RMSE":
                error = rmse
            else:
                raise ValueError(f"Unknown error type {ERROR_TYPE}")
            ax.plot(
                int_times,
                error,
                marker='o',
                linewidth=2,
                markerfacecolor='none',
                markeredgewidth=2,
                label=get_string_name(capture_type, None, True) + f" (K={k})",
                color=get_cap_color(capture_type, k)
            )
            #ax.set_ylim(0, 200)
            xmax = max(xmax, max(int_times))
        ax.set_xlim(-0.01, xmax + 0.01)
        cell_xmax[i][j]  = round(xmax, 6)
        cell_title[i][j] = get_single_pixel_title(exp_paths[i][j])
        ax.legend(fontsize=14, framealpha=1, facecolor='white', edgecolor='black')
        ax.set_xlabel('Total Integration Time (seconds)', fontsize=16)
        if ERROR_TYPE == "MAE":
            ax.set_ylabel('Mean Abs. Error (mm)', fontsize=16)
        elif ERROR_TYPE == "RMSE":
            ax.set_ylabel('Mean Squared Depth Error (mm)', fontsize=16)
        else:
            raise ValueError(f"Unknown error type {ERROR_TYPE}")
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor('black')

    # ---- per-column post-pass: shared x-axis + shared column titles ----
    for j in range(n_cols):
        rows_present = [i for i in range(n_rows) if cell_xmax[i][j] is not None]
        if not rows_present:
            continue

        # Shared x-axis: only when every row in this column spans the same
        # total integration time. Then link them and drop the redundant
        # x tick labels / xlabel on all but the bottom populated row.
        xmaxes = {cell_xmax[i][j] for i in rows_present}
        if len(xmaxes) == 1 and len(rows_present) > 1:
            base = axs[rows_present[-1]][j]
            for i in rows_present[:-1]:
                axs[i][j].sharex(base)
                axs[i][j].tick_params(labelbottom=False)
                axs[i][j].set_xlabel("")

        # Shared title: if all populated rows in the column have the same
        # title, show it once on the top populated row and clear the rest.
        titles = {cell_title[i][j] for i in rows_present}
        if len(titles) == 1:
            top = rows_present[0]
            axs[top][j].set_title(cell_title[top][j], fontsize=24, fontweight='bold')
            for i in rows_present[1:]:
                axs[i][j].set_title("")
        else:
            for i in rows_present:
                axs[i][j].set_title(cell_title[i][j], fontsize=24, fontweight='bold')

    #plt.rcParams['svg.fonttype'] = 'path'
    #timeslicing = if
    plt.subplots_adjust(wspace=0.2, hspace=0.05)
    plt.savefig(f'figures/single_pixel_k{k}.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    print(len(all_depths_dicts))