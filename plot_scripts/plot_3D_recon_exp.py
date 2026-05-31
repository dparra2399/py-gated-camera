import numpy as np

from spad_lib.spad512utils import *
from utils.file_utils import *
from plot_scripts.plot_utils import *
from utils.global_constants import *
from utils.tof_utils import build_coding_matrix_from_correlations, get_simulated_coding_matrix, decode_depth_map, \
    calculate_tof_domain_params, filter_hot_pixels
from utils.parameter_classes import DecodeConfig

# -----------------------------------------------------------------------------
# CONFIG (capitalized)
# -----------------------------------------------------------------------------
EXP_PATH = os.path.join('horse_results', 'k4_HIGHSNR_1')
SNR_LABEL     = "highsnr" if "highsnr" in EXP_PATH.lower() else "lowsnr"
FIGURES_DIR   = "figures"   # where 3d_recon_single.py saved its PNGs
VMAX_ERROR    = 0.1  # metres; None = auto (95th percentile across runs)
PLOT_POINT_CLOUD = False  # True = show point cloud PNG in top row; False = show depth map instead
ALIGN_DEPTHS = True       # shift each run's depth map so all medians match the first run
N_TBINS = 3000
NUM_TRIALS = 70

BAD_ROWS = None #[(230, 260), (80, 95)]

#PLotting utils for visualization
PLOT_DEPTH_MAPS = True
VMINS = None
VMAXS = None
MEDIAN_FILTER_SIZE = 1

#Masking or normalizing depth maps
NORMALIZE_DEPTH_MAPS = False
MASK_BACKGROUND_PIXELS = True

#Which correlation functions to use
SIMULATED_CORRELATIONS = False
USE_FULL_CORRELATIONS = False

#Smoothing or shifting the correlation functions
SIGMA_SIZE = None #None if no smoothing
SHIFT_SIZE = 70 #None if no shifting

#Corrections to depth map
CORRECT_DEPTH_DISTORTION = False
CORRECT_MASTER = True


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    def apply_decode_defaults(cfg: DecodeConfig) -> DecodeConfig:
        defaults = dict(
            exp_path=EXP_PATH,

            n_tbins=N_TBINS,
            num_trials=NUM_TRIALS,

            vmins=VMINS,
            vmaxs=VMAXS,
            median_filter_size=MEDIAN_FILTER_SIZE,

            correct_master=CORRECT_MASTER,
            plot_depth_maps=PLOT_DEPTH_MAPS,
            mask_background_pixels=MASK_BACKGROUND_PIXELS,
            normalize_depth_maps=NORMALIZE_DEPTH_MAPS,

            simulated_correlations=SIMULATED_CORRELATIONS,
            use_full_correlations=USE_FULL_CORRELATIONS,

            smooth_sigma=SIGMA_SIZE,
            shift=SHIFT_SIZE,

            correct_depth_distortion=CORRECT_DEPTH_DISTORTION,

        )

        # fill only missing (None)
        for k, v in defaults.items():
            if getattr(cfg, k) is None:
                setattr(cfg, k, v)

        return cfg

    parser = build_parser_from_config(DecodeConfig)
    args = parser.parse_args()
    cfg = apply_decode_defaults(DecodeConfig(**vars(args)))

    hot_mask = load_hot_mask(get_data_folder(HOT_MASK_PATH_WINDOWS, HOT_MASK_PATH_MAC))
    correlation_folder = get_data_folder(READ_PATH_CORRELATIONS_MAC, READ_PATH_CORRELATIONS_WINDOWS)
    capture_folder = get_data_folder(READ_PATH_CAPTURE_MAC, READ_PATH_CAPTURE_WINDOWS)
    assert cfg.exp_path is not None, 'Must define exp_num to find folder'
    capture_folder = os.path.join(capture_folder, cfg.exp_path)

    capture_paths = os.listdir(capture_folder)
    capture_paths = filter_capture_files(capture_paths)

    depth_map_dict = {}

    for i, coded_vals_name in enumerate(capture_paths):
        if coded_vals_name.startswith('.'): continue
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
                cfg.use_full_correlations,
                cfg.smooth_sigma,
                cfg.shift,
                cfg.n_tbins,
            )

        # mn = coding_matrix.min()
        # mx = coding_matrix.max()
        # coding_matrix = (coding_matrix - mn) / (mx - mn)

        (rep_tau, rep_freq,tbin_res,
         t_domain,max_depth,tbin_depth_res,)= calculate_tof_domain_params(cfg.n_tbins, params['rep_tau'])

        # -----------------------------------------------------------------
        # decode to depth map
        # -----------------------------------------------------------------
        total_trials = coded_vals.shape[0]
        trials = min(total_trials, cfg.num_trials)
        coded_vals_trials = np.sum(coded_vals[:trials, ...], axis=0)
        coded_vals_total = np.sum(coded_vals, axis=0)
        depth_map, zncc = decode_depth_map(
            coded_vals_trials,
            coding_matrix,
            im_width,
            tbin_depth_res,
            args.use_full_correlations,
        )


        depth_map = filter_hot_pixels(depth_map, hot_mask)

        coded_vals_filt = np.zeros_like(coded_vals_trials)
        for i in range(coded_vals_trials.shape[-1]):
            coded_vals_filt[:, :, i] = filter_hot_pixels(coded_vals_trials[..., i], hot_mask)


        gt_depth_map, zncc = decode_depth_map(
            coded_vals_total,
            coding_matrix,
            im_width,
            tbin_depth_res,
            args.use_full_correlations,
        )
        gt_depth_map = filter_hot_pixels(gt_depth_map, hot_mask)

        # optional crop for master
        if cfg.correct_master is False:
            depth_map = depth_map[:, : im_width // 2]
            gt_depth_map = gt_depth_map[:, : im_width // 2]
            coded_vals_trials = coded_vals_trials[:, : im_width // 2]

        if cfg.mask_background_pixels:
            if "horse" in coded_vals_path:
                tmp = np.sum(coded_vals_total, axis=-1)
                depth_map[tmp < 1e4] = np.nan

            else:
                depth_map = depth_map[40:450, 20:-20]
                gt_depth_map = gt_depth_map[40:450, 20:-20]
            depth_map[depth_map == 0] = np.nan
            gt_depth_map[gt_depth_map == 0] = np.nan
            mask = None

        if BAD_ROWS is not None and type(BAD_ROWS) == list:
            depth_map = depth_map.astype(float)
            for BAD_ROW in BAD_ROWS:
                depth_map[BAD_ROW[0]:BAD_ROW[1] + 1, :] = np.nan

        mae = np.nanmean(np.abs(depth_map - gt_depth_map))
        rmse = np.sqrt(np.nanmean((depth_map - gt_depth_map) ** 2))

        cfg_dict = asdict(cfg)
        cfg_dict.update({'depth_map': depth_map, 'gt_depth_map': gt_depth_map,
                         'rmse': rmse, 'mae': mae})
        cfg_dict.update(params)
        depth_map_dict[coded_vals_path] = cfg_dict


        #plt.plot(coding_matrix)
        #plt.show()

    # --- median-align all depth maps to the first run ---
    if ALIGN_DEPTHS:
        ref_median = None
        for d in depth_map_dict.values():
            med = np.nanmedian(d['depth_map'])
            if ref_median is None:
                ref_median = med
            else:
                shift = ref_median - med
                d['depth_map']    = d['depth_map']    + shift
                d['gt_depth_map'] = d['gt_depth_map'] + shift
            print(f"  {d['capture_type']} k={d['k']}: median={med:.3f} m  shift={ref_median - med:.3f} m")


    # --- clip depth maps to VMINS / VMAXS ---
    for d in depth_map_dict.values():
        vmin_clip = VMINS if VMINS is not None else np.nanmin(d['depth_map'])
        vmax_clip = VMAXS if VMAXS is not None else np.nanmax(d['depth_map'])
        d['depth_map']    = np.clip(d['depth_map'],    vmin_clip, vmax_clip)
        d['gt_depth_map'] = np.clip(d['gt_depth_map'], vmin_clip, vmax_clip)

    # --- 3-D recon comparison panel ---
    recon_runs = []
    for coded_vals_path, d in depth_map_dict.items():
        recon_runs.append({
            'capture_type': d['capture_type'],
            'k':            d['k'],
            'snr_label':    SNR_LABEL,
            'depth_map':    d['depth_map'],
            'gt_depth_map': d['gt_depth_map'],
        })


    save_panel = os.path.join(FIGURES_DIR, f"3d_recon_panel_{SNR_LABEL}_k{d['k']}.pdf")
    plot_3d_recon_comparison(
        recon_runs,
        snr_label=SNR_LABEL,
        vmax_error=VMAX_ERROR,
        figures_dir=FIGURES_DIR,
        save_path=save_panel,
        plot_point_cloud=PLOT_POINT_CLOUD,
    )