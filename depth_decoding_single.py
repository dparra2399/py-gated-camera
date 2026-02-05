from spad_lib.spad512utils import *
from utils.file_utils import *
from plot_scripts.plot_utils import *
from utils.global_constants import *
from utils.tof_utils import build_coding_matrix_from_correlations, get_simulated_coding_matrix, decode_depth_map, \
    calculate_tof_domain_params, filter_hot_pixels

# -----------------------------------------------------------------------------
# CONFIG (capitalized)
# -----------------------------------------------------------------------------
EXP_PATH = "exp_0" #Only use if inside folder otherwise none
N_TBINS = 1500
VMIN = None
VMAX = None
MEDIAN_FILTER_SIZE = 5
CORRECT_MASTER = False
MASK_BACKGROUND_PIXELS = True
SIMULATED_CORRELATIONS = False
USE_FULL_CORRELATIONS = False
SIGMA_SIZE = None
SHIFT_SIZE = 150
CORRECT_DEPTH_DISTORTION = False

# paths
"""
Format:
capture_type, k , freq_mhz , mV , mA , duty , int_time

Example:
ham,3,5,100,50,10
"""

DEFAULT_RUNS = [
    "ham,3,5, 5000,50,20, 200" ,
]

EPSILON = 1e-12

def parse_args():
    p = argparse.ArgumentParser(description="Decode captured coded vals using correlations (with sweep-friendly runs)")

    # repeated runs (or defaults)
    p.add_argument("--run", action="append", type=capture_parse_run,
                   help="capture_type,k,freq_mhz,mV,mA,duty,int_time (repeatable)")

    # experiment folder selector (your EXP variable)
    p.add_argument("--exp_path", type=str, default=str(EXP_PATH) if EXP_PATH is not None else None)

    # decode / processing params (CONFIG -> CLI)
    p.add_argument("--n_tbins", type=int, default=N_TBINS)

    p.add_argument("--vmin", type=float, default=VMIN)
    p.add_argument("--vmax", type=float, default=VMAX)
    p.add_argument("--median_filter_size", type=int, default=MEDIAN_FILTER_SIZE)

    p.add_argument("--correct_master", type=str2bool, default=CORRECT_MASTER)
    p.add_argument("--mask_background_pixels", type=str2bool, default=MASK_BACKGROUND_PIXELS)

    p.add_argument("--simulated_correlations", type=str2bool, default=SIMULATED_CORRELATIONS)
    p.add_argument("--use_full_correlations", type=str2bool, default=USE_FULL_CORRELATIONS)

    # smoothing/shift for coding matrix build
    # keep None defaults like your config (so your function can interpret "no smoothing")
    p.add_argument("--smooth_sigma", type=float, default=SIGMA_SIZE)
    p.add_argument("--shift", type=int, default=SHIFT_SIZE)

    p.add_argument("--correct_depth_distortion", type=str2bool, default=CORRECT_DEPTH_DISTORTION)

    args = p.parse_args()

    # default runs if user didn't provide any
    if args.run is None:
        args.run = [capture_parse_run(r) for r in DEFAULT_RUNS]

    return args
# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()

    hot_mask = load_hot_mask(get_data_folder(HOT_MASK_PATH_WINDOWS, HOT_MASK_PATH_MAC))
    correlation_folder = get_data_folder(READ_PATH_CORRELATIONS_MAC, READ_PATH_CORRELATIONS_WINDOWS)
    capture_folder = get_data_folder(READ_PATH_CAPTURE_MAC, READ_PATH_CAPTURE_WINDOWS)
    if args.exp_path is not None: capture_folder = os.path.join(capture_folder, args.exp_path)

    x1, y1 = (180, 40)
    x2, y2 = (180, 160)
    x3, y3 = (180, 330)
    points = [(x1, y1), (x2, y2), (x3, y3)]
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']

    for i, r in enumerate(args.run, 1):

        corr_path = os.path.join(correlation_folder, make_correlation_filename(r['capture_type'], r['k'], r['freq_mhz'],
                                                            r['mV'], r['mA'],  r['duty']))

        coded_vals_path = os.path.join(capture_folder, make_capture_filename(r['capture_type'], r['k'], r['freq_mhz'],
                                             r['mV'], r['mA'],  r['duty'], r['int_time'], False))

        gt_coded_vals_path = os.path.join(capture_folder, make_capture_filename(r['capture_type'], r['k'], r['freq_mhz'],
                                             r['mV'], r['mA'],  r['duty'], r['int_time'], True))

        correlations_total = np.load(corr_path, allow_pickle=True)['correlations']

        if args.simulated_correlations:
            coding_matrix = get_simulated_coding_matrix(r['capture_type'], args.n_tbins, r["k"])
        else:
            coding_matrix = build_coding_matrix_from_correlations(
                correlations_total,
                args.use_full_correlations,
                args.smooth_sigma,
                args.shift,
                args.n_tbins,
            )

        capture_file = np.load(coded_vals_path, allow_pickle=True)
        cfg = capture_file['cfg'].item()
        coded_vals = capture_file['coded_vals']
        im_width = cfg['im_width']

        (rep_tau, rep_freq,tbin_res,
         t_domain,max_depth,tbin_depth_res,)= calculate_tof_domain_params(args.n_tbins, cfg['rep_tau'])

        # -----------------------------------------------------------------
        # decode to depth map
        # -----------------------------------------------------------------
        depth_map, zncc = decode_depth_map(
            coded_vals,
            coding_matrix,
            im_width,
            tbin_depth_res,
            args.use_full_correlations,
        )


        depth_map = filter_hot_pixels(depth_map, hot_mask)

        coded_vals_filt = np.zeros_like(coded_vals)
        for i in range(coded_vals.shape[-1]):
            coded_vals_filt[:, :, i] = filter_hot_pixels(coded_vals[..., i], hot_mask)


        # optional crop for master
        if args.correct_master is False:
            depth_map = depth_map[:, : im_width // 2]
            coded_vals = coded_vals[:, : im_width // 2]

        if args.mask_background_pixels:
            depth_map = depth_map[20:450, :]
            mask = None

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
            print('Could not find ground truth depth map.')
            gt_depth_map = None




        # ------------------------------------------------------------------
        # Diagnostics plots (coding vs coded at a few points)
        # ------------------------------------------------------------------
        plot_gated_images(
                coded_vals_filt,
                depth_map,
                gt_depth_map,
                vmin=VMIN,
                vmax=VMAX,
                median_filter_size=MEDIAN_FILTER_SIZE,
        )

        plot_sample_points(
                coded_vals,
                coding_matrix,
                points,
                depth_map,
                tbin_depth_res,
                vmin=args.vmin,
                vmax=args.vmax,
                median_filter_size=args.median_filter_size,
        )

        plot_sample_points_simple(
                coded_vals,
                coding_matrix,
                points,
                depth_map,
                tbin_depth_res,
                args.use_full_correlations,
                colors,
        )



