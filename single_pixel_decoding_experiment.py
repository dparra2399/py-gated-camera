import pprint

from matplotlib import pyplot as plt

from utils.file_utils import *
from plot_scripts.plot_utils import plot_single_pixel_dist, plot_single_pixel_corr, plot_single_pixel_depth_pairs
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

#PLotting utils for visualization
PLOT_SINGLE_PIXEL = True

#Which correlation functions to use
SIMULATED_CORRELATIONS = False

#Smoothing or shifting the correlation functions
SIGMA_SIZE = 1 #None if no smoothing
SHIFT_SIZE = None #None if no shifting

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    def apply_decode_defaults(cfg: DecodeConfig) -> DecodeConfig:
        defaults = dict(
            exp_path=EXP_PATH,

            n_tbins=N_TBINS,

            plot_single_pixel=PLOT_SINGLE_PIXEL,

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

    depths_dict = {}

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
        #plt.imshow(np.sum(coded_vals[0, 0, :, :, :], axis=-1))
        #plt.show()

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




        depths, zncc, _ = decode_single_pixel_experiment(
            coded_vals,
            coding_matrix,
            tbin_depth_res,
            [190, 210],
            [135, 155],
            10
        )

        try:
            coded_vals_gt = np.load(gt_coded_vals_path, allow_pickle=True)['coded_vals']

            gt_depths, zncc, _ = decode_single_pixel_experiment(
                #coded_vals_gt,
                coded_vals,
                coding_matrix,
                tbin_depth_res,
            [190, 210],
            [135, 155],
                1
            )
            gt_depths = gt_depths[0, ...]
        except FileNotFoundError:
            print('GT Depth Map not found, Using Captured Depth Map Instead')
            gt_depths = np.copy(depths)
            coded_vals_gt = None

        #plt.imshow(np.sum(np.sum(coded_vals_gt, axis=0), axis=-1))
        #plt.show()

        print(f'capture type: {capture_type}')
        print(f'depth: {gt_depths[0]:.3f}')
        #print(f'counts: {coded_vals[0, :]}')
        counts = np.sum(coded_vals[0, 0, 280:300, 160:170, :])
        print(f'total counts: {counts:.3f}')
        if coded_vals_gt is not None:
            #print(f'ground truth counts: {coded_vals_gt[0, :]}')
            counts_t = np.sum(coded_vals_gt[0, 280:300, 160:170, :])
            print(f'ground truth total counts: {counts_t:.3f}')
        print('----------------------------------')

        phase_shifts = params['phase_shifts']#[2:-2]
        depths = depths#[:, 2:-2]
        gt_depths = gt_depths#[2:-2]

        mae = np.nanmean(np.abs(depths - gt_depths))
        rmse = np.sqrt(np.nanmean((depths - gt_depths) ** 2))
        cfg_dict = asdict(cfg)
        cfg_dict.update({'depths': depths, 'gt_depths': gt_depths,
                         'rmse': rmse, 'mae': mae, 'coding_matrix': coding_matrix,
                         'tbin_res': tbin_res, 'tbin_depth_res': tbin_depth_res,
                         'phase_shifts' : phase_shifts, 'capture_type': capture_type})
        #cfg_dict.update(params)
        depths_dict[coded_vals_path] = cfg_dict

    if cfg.plot_single_pixel:
        plot_single_pixel_dist(depths_dict)

        plot_single_pixel_depth_pairs(depths_dict)



        plot_single_pixel_corr(depths_dict)