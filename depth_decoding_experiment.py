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
EXP_PATH = 'exp_0' #Only use if inside folder otherwise none
N_TBINS = 1500
VMIN = None
VMAX = None
MEDIAN_FILTER_SIZE = 3
CORRECT_MASTER = False
PLOT_DEPTH_MAPS = True
MASK_BACKGROUND_PIXELS = True
SIMULATED_CORRELATIONS = False
USE_FULL_CORRELATIONS = False
SIGMA_SIZE = None
SHIFT_SIZE = 150
CORRECT_DEPTH_DISTORTION = False

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    def apply_decode_defaults(cfg: DecodeConfig) -> DecodeConfig:
        defaults = dict(
            exp_path=EXP_PATH,

            n_tbins=N_TBINS,

            vmin=VMIN,
            vmax=VMAX,
            median_filter_size=MEDIAN_FILTER_SIZE,

            correct_master=CORRECT_MASTER,
            plot_depth_maps=PLOT_DEPTH_MAPS,
            mask_background_pixels=MASK_BACKGROUND_PIXELS,

            simulated_correlations=SIMULATED_CORRELATIONS,
            use_full_correlations=USE_FULL_CORRELATIONS,

            smooth_sigma=SIGMA_SIZE,
            shift=SHIFT_SIZE,

            correct_depth_distortion=CORRECT_DEPTH_DISTORTION,

            normalize_depth_maps=False,
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
        coded_vals_path = os.path.join(capture_folder, coded_vals_name)
        capture_file = np.load(coded_vals_path, allow_pickle=True)
        params = capture_file['cfg'].item()
        coded_vals = capture_file['coded_vals']
        im_width = params['im_width']
        mA = params['current']
        mV = params['amplitude'] * 1000
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

        correlations_total = np.load(corr_path, allow_pickle=True)['correlations']

        if cfg.simulated_correlations:
            coding_matrix = get_simulated_coding_matrix(capture_type, args.n_tbins, k)
        else:
            coding_matrix = build_coding_matrix_from_correlations(
                correlations_total,
                cfg.use_full_correlations,
                cfg.smooth_sigma,
                cfg.shift,
                cfg.n_tbins,
            )


        (rep_tau, rep_freq,tbin_res,
         t_domain,max_depth,tbin_depth_res,)= calculate_tof_domain_params(cfg.n_tbins, rep_tau)

        # -----------------------------------------------------------------
        # decode to depth map
        # -----------------------------------------------------------------
        depth_map, zncc = decode_depth_map(
            coded_vals,
            coding_matrix,
            im_width,
            tbin_depth_res,
            cfg.use_full_correlations,
        )


        depth_map = filter_hot_pixels(depth_map, hot_mask)

        coded_vals_filt = np.zeros_like(coded_vals)
        for i in range(coded_vals.shape[-1]):
            coded_vals_filt[:, :, i] = filter_hot_pixels(coded_vals[..., i], hot_mask)



        try:
            coded_vals_gt = np.load(gt_coded_vals_path, allow_pickle=True)['coded_vals']
            gt_depth_map, zncc = decode_depth_map(
                coded_vals_gt,
                coding_matrix,
                im_width,
                tbin_depth_res,
                args.use_full_correlations,
            )
            gt_depth_map = filter_hot_pixels(gt_depth_map, hot_mask)
        except FileNotFoundError:
            print('GT Depth Map not found, Using Captured Depth Map Instead')
            gt_depth_map = np.copy(depth_map)


        # optional crop for master
        if cfg.correct_master is False:
            depth_map = depth_map[:, : im_width // 2]
            gt_depth_map = gt_depth_map[:, : im_width // 2]
            coded_vals = coded_vals[:, : im_width // 2]

        if cfg.mask_background_pixels:
            depth_map = depth_map[20:450, :]
            gt_depth_map = gt_depth_map[20:450, :]
            mask = None

        mae = np.nanmean(np.abs(depth_map - gt_depth_map))
        rmse = np.sqrt(np.nanmean((depth_map - gt_depth_map) ** 2))

        cfg_dict = asdict(cfg)
        cfg_dict.update({'depth_map': depth_map, 'gt_depth_map': gt_depth_map,
                         'rmse': rmse, 'mae': mae})
        cfg_dict.update(params)
        depth_map_dict[coded_vals_path] = cfg_dict

    if cfg.plot_depth_maps:
        plot_capture_comparison(depth_map_dict, vmin=cfg.vmin, vmax=cfg.vmax,
                                normalize_depth_maps=cfg.normalize_depth_maps,
                                median_filter_size=cfg.median_filter_size)