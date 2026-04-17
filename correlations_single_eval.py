# Imports
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from plot_scripts.plot_utils import plot_coding_curve, plot_coding_error, plot_correlations_one_plot, \
    plot_results_summary, plot_depth_error_distribution
from utils.global_constants import *
from utils.file_utils import get_data_folder, make_correlation_filename, corr_parse_run
from utils.tof_utils import (
    build_coding_matrix_from_correlations,
    decode_from_correlations,
    get_simulated_coding_matrix, calculate_tof_domain_params,
)

# =============================
# Defaults
# =============================
PHOTON_COUNT = 300
SBR = 1.0
TRIALS = 100
N_TBINS = 999
SMOOTH_SIGMA = 1
SHIFT = None
DEPTH_MARGIN = 3.0
REP_RATE = 5 * 1e6
DEPTH_SAMPLE = 1 #0.01
"""
Format:
capture_type,k,freq_mhz,mV,mA,duty,simulated_correlations

Example:
ham,3,5,100,50,10,False
"""

DEFAULT_RUNS = [
    "ham,3,10,4000,50,20,False",
    "ham,3,10,4000,50,20,True",
    "coarse,3,10,3400,50,30,False",
    #"ham,3,5, 4000,50,20",
    #"coarse,3,5, 3400,50,30",
]



def parse_args():
    p = argparse.ArgumentParser(description="Decode correlation files from parameterized runs")

    # repeated runs
    p.add_argument(
        "--run",
        action="append",
        type=corr_parse_run,
        help="run spec: capture_type,k,freq_mhz,mV,mA,duty",
    )

    # scalar overrides
    p.add_argument("--photon_count", type=int, default=PHOTON_COUNT)
    p.add_argument("--sbr", type=float, default=SBR)
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--n_tbins", type=int, default=N_TBINS)
    p.add_argument("--rep_rate", type=int, default=REP_RATE)
    p.add_argument("--depth_sample", type=int, default=DEPTH_SAMPLE)
    p.add_argument("--smooth_sigma", type=float, default=SMOOTH_SIGMA)
    p.add_argument("--shift", type=int, default=SHIFT)
    p.add_argument("--depth_margin", type=int, default=DEPTH_MARGIN)
    #p.add_argument("--simulated_correlations", action="store_true", default=SIMULATED_CORRELATIONS)

    args = p.parse_args()

    # if no runs given, use defaults
    if args.run is None:
        args.run = [corr_parse_run(r) for r in DEFAULT_RUNS]

    return args


# =============================
# Main
# =============================
if __name__ == "__main__":
    args = parse_args()

    if args.rep_rate is not None:
        (rep_tau, rep_freq, tbin_res,
         t_domain, max_depth, tbin_depth_res,) = calculate_tof_domain_params(args.n_tbins, 1. / args.rep_rate)
        depths = np.arange(args.depth_margin, max_depth - args.depth_margin, args.depth_sample)
    else:
        depths = np.arange(args.depth_margin, args.n_tbins - args.depth_margin, 1)
    folder = get_data_folder(READ_PATH_CORRELATIONS_MAC, READ_PATH_CORRELATIONS_WINDOWS)
    #folder = os.path.join(folder, "feb10th_2026")

    results = []

    total_runs = len(args.run)

    for i, r in enumerate(args.run, 1):

        filename = make_correlation_filename(r['capture_type'], r['k'], r['freq_mhz'],
                                             r['mV'], r['mA'],  r['duty'])

        path = os.path.join(folder, filename)

        print(
            f"[{i:02d}/{total_runs}] " +
            f"type={r['capture_type']:6s} | " +
            f"k={r['k']} | " +
            f"{r['freq_mhz']}MHz | " +
            f"{r['mV']}mV | " +
            f"{r['mA']}mA | " +
            f"{r['duty']}%  -->  {filename}"
        )



        name = r["capture_type"]


        # coding matrix
        if r["simulated_correlations"]:
            coding_matrix = get_simulated_coding_matrix(name, args.n_tbins, r["k"])
        else:
            file = np.load(path, allow_pickle=True)
            cfg = file["cfg"].item()
            correlations_total = file["correlations"]
            coding_matrix = build_coding_matrix_from_correlations(
                correlations_total,
                False,
                args.smooth_sigma,
                args.shift,
                args.n_tbins,
            )

        photon_count = args.photon_count * 1.4 if r['capture_type'] == 'ham' else args.photon_count
        decoded_depths = decode_from_correlations(
            coding_matrix=coding_matrix,
            depths=depths,
            photon_count=photon_count,
            sbr=args.sbr,
            trials=args.trials,
            rep_rate=args.rep_rate,
        )

        #col_sums = coding_matrix.sum(axis=1, keepdims=True)  # shape: (1, n_cols)

        #
        mins = coding_matrix.min()
        maxs = coding_matrix.max()

        coding_matrix = (coding_matrix - mins) / (maxs - mins)
        coding_matrix = coding_matrix * photon_count + (photon_count / args.sbr) / r['k']

        depth_res = 1000 if args.rep_rate is not None else 1
        rmse = float(np.sqrt(np.mean((decoded_depths - depths) ** 2))) * depth_res
        mae = float(np.mean(np.abs(decoded_depths - depths))) * depth_res

        print(f"      → MAE={mae:.3f} | RMSE={rmse:.3f}")

        results.append(dict(
            coding_matrix=coding_matrix,
            decoded_depths=decoded_depths,
            rmse=rmse,
            mae=mae,
            name=name,
            depths=depths,
            simulated=r["simulated_correlations"]
        ))

    plot_results_summary(results)

    plot_correlations_one_plot(results)

    plot_depth_error_distribution(results)

    plot_coding_error(results)

    plot_coding_curve(results)




