import pprint
import zipfile

from matplotlib import pyplot as plt

from spad_lib.spad512utils import get_gate_shifts
from utils.file_utils import *
from plot_scripts.plot_utils import plot_single_pixel_dist, plot_single_pixel_corr, plot_single_pixel_depth_pairs, \
    get_string_name, get_single_pixel_title
from utils.global_constants import *
from utils.tof_utils import build_coding_matrix_from_correlations, get_simulated_coding_matrix, \
    calculate_tof_domain_params, decode_single_pixel_experiment
from utils.parameter_classes import DecodeConfig
import numpy as np

# -----------------------------------------------------------------------------
# CONFIG (capitalized)
# ----------------------------------------------------------------------------
EXP_PATHS = [['k3_HIGHSNR'], ['k3_LOWSNR']]
N_TBINS = 1500

#Which correlation functions to use
SIMULATED_CORRELATIONS = False

#Smoothing or shifting the correlation functions
SIGMA_SIZE = 1 #None if no smoothing
SHIFT_SIZE = None #None if no shifting

TOTAL_PIXELS = ((SINGLE_PIXEL_COORDS['y'][1] - SINGLE_PIXEL_COORDS['y'][0])
                * (SINGLE_PIXEL_COORDS['x'][1] - SINGLE_PIXEL_COORDS['x'][0]))
#Not apart of the defaults
N_PIXELS = np.arange(2, TOTAL_PIXELS//3, 5)

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

    correlation_folder = get_data_folder(READ_PATH_CORRELATIONS_MAC, READ_PATH_CORRELATIONS_WINDOWS)
    base_capture_folder = get_data_folder(READ_PATH_SINGLE_PIXEL_MAC, READ_PATH_SINGLE_PIXEL_WINDOWS)

    exp_paths = EXP_PATHS if cfg.exp_path is None else [[cfg.exp_path]]

    all_depths_dicts = []

    for exp_path_group in exp_paths:
        depths_dict = []

        for exp_path in exp_path_group:
            capture_folder = get_capture_folder(os.path.join(base_capture_folder, exp_path))

            capture_paths = os.listdir(capture_folder)
            capture_paths = [p for p in capture_paths if os.path.isfile(os.path.join(capture_folder, p))]
            capture_paths = filter_capture_files(capture_paths)

            for i, coded_vals_name in enumerate(capture_paths):
                if coded_vals_name.startswith('.'):
                    continue
                coded_vals_path = os.path.join(capture_folder, coded_vals_name)
                if not os.path.isfile(coded_vals_path):
                    continue
                capture_file = np.load(coded_vals_path, allow_pickle=True)
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

                gate_widths, gate_starts = get_gate_shifts(capture_type, freq, k)
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
                    coding_matrix = get_simulated_coding_matrix(capture_type, cfg.n_tbins, k)
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

                pixel_order = np.random.default_rng(0).permutation(TOTAL_PIXELS)

                #coded_vals_gt = np.load(gt_coded_vals_path, allow_pickle=True)['coded_vals']

                gt_depths, recon_gt, _ = decode_single_pixel_experiment(
                    capture_type + "s",
                    coded_vals,
                    coding_matrix,
                    tbin_depth_res,
                    SINGLE_PIXEL_COORDS['y'],
                    SINGLE_PIXEL_COORDS['x'],
                    n_pixels=TOTAL_PIXELS,
                    pixel_order=pixel_order,
                )

                for i, n in enumerate(N_PIXELS):

                    depths, recon, num_pixels = decode_single_pixel_experiment(
                        capture_type,
                        coded_vals,
                        coding_matrix,
                        tbin_depth_res,
                        SINGLE_PIXEL_COORDS['y'],
                        SINGLE_PIXEL_COORDS['x'],
                        n_pixels=n,
                        pixel_order=pixel_order
                    )

                    if capture_type == 'timeslicing': depths = np.roll(depths, -2, axis=-1)

                    phase_shifts = params['phase_shifts']#[2:-2]
                    #depths = depths[:, 2:-2]
                    #gt_depths = gt_depths[:, 2:-2]

                    mae = np.nanmean(np.abs(depths - gt_depths)) * 1000
                    rmse = np.sqrt(np.nanmean((depths - gt_depths) ** 2))* 1000
                    if mae < 100000:
                        mae_list.append(mae)
                        rmse_list.append(rmse)
                        int_times.append(num_pixels * int_time)

                cfg_dict = asdict(cfg)
                cfg_dict.update({'depths': depths, 'gt_depths': gt_depths,
                                 'rmse': rmse_list, 'mae': mae_list, 'coding_matrix': coding_matrix,
                                 'tbin_res': tbin_res, 'tbin_depth_res': tbin_depth_res,
                                 'phase_shifts': phase_shifts, 'capture_type': capture_type,
                                 'int_times': int_times, "k": k})
                #cfg_dict.update(params)

                depths_dict.append(cfg_dict)

        all_depths_dicts.append(depths_dict)


    n_paths = len(exp_paths)
    fig, axs = plt.subplots(1, n_paths, figsize=(8 * n_paths, 8))
    if n_paths == 1:
        axs = [axs]

    for j, (ax, depths_dict) in enumerate(zip(axs, all_depths_dicts)):
        for idx, inner_dict in enumerate(depths_dict):
            rmse = inner_dict['rmse']
            mae = inner_dict['mae']
            int_times = inner_dict['int_times']
            capture_type = inner_dict['capture_type']
            k = inner_dict['k']

            ax.plot(
                int_times,
                mae,
                marker='o',
                markerfacecolor='none',
                markeredgewidth=2,
                label=get_string_name(capture_type, k),
            )
            #ax.set_ylim(0, 200)

        ax.set_title(get_single_pixel_title(exp_paths[j]), fontsize=16)
        ax.legend(fontsize=12, framealpha=1, facecolor='white', edgecolor='black')
        ax.set_xlabel('Integration Time (ms)', fontsize=20)
        ax.set_ylabel('Depth Error (mm)', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True)

    #plt.rcParams['svg.fonttype'] = 'path'
    #plt.savefig('single_pixel_decoding_exposure_plot.pdf', dpi=300)
    plt.show()
    print(len(all_depths_dicts))