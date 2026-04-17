import pprint

from matplotlib import pyplot as plt

from utils.file_utils import *
from plot_scripts.plot_utils import plot_single_pixel_dist, plot_single_pixel_corr, plot_single_pixel_depth_pairs, \
    get_string_name
from utils.global_constants import *
from utils.tof_utils import build_coding_matrix_from_correlations, get_simulated_coding_matrix, \
    calculate_tof_domain_params, decode_single_pixel_experiment
from utils.parameter_classes import DecodeConfig
import numpy as np

# -----------------------------------------------------------------------------
# CONFIG (capitalized)
# ----------------------------------------------------------------------------
EXP_PATH = os.path.join('exp_0')
N_TBINS = 1500

#Which correlation functions to use
SIMULATED_CORRELATIONS = False

#Smoothing or shifting the correlation functions
SIGMA_SIZE = 1 #None if no smoothing
SHIFT_SIZE = None #None if no shifting

#Not apart of the defaults
SKIP_PIXELS = np.arange(1, 30, 2)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    def apply_decode_defaults(cfg: DecodeConfig) -> DecodeConfig:
        defaults = dict(
            exp_path=EXP_PATH,

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
    capture_folder = get_data_folder(READ_PATH_SINGLE_PIXEL_MAC, READ_PATH_SINGLE_PIXEL_WINDOWS)
    assert cfg.exp_path is not None, 'Must define exp_num to find folder'
    capture_folder = os.path.join(capture_folder, cfg.exp_path)



    capture_paths = os.listdir(capture_folder)
    capture_paths = filter_capture_files(capture_paths)

    depths_dict = []

    for i, coded_vals_name in enumerate(capture_paths):
        if coded_vals_name.startswith('.'):
            continue
        coded_vals_path = os.path.join(capture_folder, coded_vals_name)
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

        #pprint.pprint(params)

        corr_path = os.path.join(correlation_folder, make_correlation_filename(capture_type, k,
                                                                               freq_mhz, mV, mA, duty))


        gt_coded_vals_path = os.path.join(capture_folder,
                                          make_capture_filename(capture_type, k, freq_mhz, mV, mA, duty,
                                                                None, True))

        correlations_total = np.load(corr_path, allow_pickle=True)['correlations']

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
        (rep_tau, rep_freq,tbin_res,
         t_domain,max_depth,tbin_depth_res,)= calculate_tof_domain_params(n_tbins, rep_tau)

        try:
            coded_vals_gt = np.load(gt_coded_vals_path, allow_pickle=True)['coded_vals']

            gt_depths, zncc, _ = decode_single_pixel_experiment(
                coded_vals_gt,
               # coded_vals,
                coding_matrix,
                tbin_depth_res,
            [190, 210],
            [135, 155],
                1
            )
            #gt_depths = gt_depths[0, ...]

        except FileNotFoundError:
            print('GT Depth Map not found, We need this please ;)')
            exit(0)

        mae_list = []
        rmse_list = []
        int_times = []

        #gt_depths = gt_depths[2:-2]

        for i, skip_pixels in enumerate(SKIP_PIXELS):
            # try:
            #     coded_vals_gt = np.load(gt_coded_vals_path, allow_pickle=True)['coded_vals']
            #
            #     gt_depths, zncc, _ = decode_single_pixel_experiment(
            #         # coded_vals_gt,
            #         coded_vals,
            #         coding_matrix,
            #         tbin_depth_res,
            #         [195, 205],
            #         [145, 155],
            #         1
            #     )
            #     #gt_depths = gt_depths[0, ...]
            #
            # except FileNotFoundError:
            #     print('GT Depth Map not found, We need this please ;)')
            #     exit(0)

            depths, zncc, num_pixels = decode_single_pixel_experiment(
                coded_vals,
                coding_matrix,
                tbin_depth_res,
                [190, 210],
                [135, 155],
                skip_pixels
            )

            phase_shifts = params['phase_shifts']#[2:-2]
            #depths = depths[:, 2:-2]

            mae = np.nanmean(np.abs(depths - gt_depths)) * 1000
            rmse = np.sqrt(np.nanmean((depths - gt_depths) ** 2))* 1000
            if mae < 1000:
                mae_list.append(mae)
                rmse_list.append(rmse)
                int_times.append(num_pixels * params['int_time'])

        cfg_dict = asdict(cfg)
        cfg_dict.update({'depths': depths, 'gt_depths': gt_depths,
                         'rmse': rmse_list, 'mae': mae_list, 'coding_matrix': coding_matrix,
                         'tbin_res': tbin_res, 'tbin_depth_res': tbin_depth_res,
                         'phase_shifts' : phase_shifts, 'capture_type': capture_type,
                         'int_times': int_times})
            #cfg_dict.update(params)

        depths_dict.append(cfg_dict)


    fig, axs = plt.subplots(1, 1, figsize=(8, 8))

    for idx, inner_dict in enumerate(depths_dict):
        rmse = inner_dict['rmse']
        mae = inner_dict['mae']
        int_times = np.log10(inner_dict['int_times'])
        capture_type = inner_dict['capture_type']

        axs.plot(
            int_times,
            mae,
            marker='o',
            markerfacecolor='none',
            markeredgewidth=2,
            label=get_string_name(capture_type),
        )
    axs.legend(fontsize=30, framealpha=1, facecolor='white', edgecolor='black')
    axs.set_xlabel('Integration Time (ms)', fontsize=20)
    axs.set_ylabel('Depth Error (mm)', fontsize=20)
    axs.tick_params(axis='both', labelsize=18)
    plt.grid(True)
    #plt.rcParams['svg.fonttype'] = 'path'
    plt.savefig('single_pixel_decoding_exposure_plot.pdf', dpi=300)
    plt.show()
    print(len(depths_dict))