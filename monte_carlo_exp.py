# Imports
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

from utils.global_constants import *
from utils.file_utils import get_data_folder, make_correlation_filename, corr_parse_run
from utils.tof_utils import (
    build_coding_matrix_from_correlations,
    decode_from_correlations,
    get_simulated_coding_matrix, calculate_tof_domain_params,
)
from mpl_toolkits.mplot3d import Axes3D

# =============================
# Defaults
# =============================
PHOTON_COUNTS = np.linspace(1000, 3000, 10).tolist()
SBRS = np.linspace(0.1, 10, 10).tolist()
TRIALS = 100
N_TBINS = 999
SMOOTH_SIGMA = 1
SHIFT = None
DEPTH_MARGIN = 3.0
REP_RATE = 5 * 1e6
DEPTH_SAMPLE = 0.01
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
    p.add_argument("--photon_counts", type=float, nargs="+", default=PHOTON_COUNTS)
    p.add_argument("--sbrs", type=float, nargs="+", default=SBRS)
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

    total_runs = len(args.run) * len(args.photon_counts) * len(args.sbrs)
    run_idx = 0

    mae_results = np.zeros((len(args.run), len(args.photon_counts), len(args.sbrs)))
    rmse_results = np.zeros((len(args.run), len(args.photon_counts), len(args.sbrs)))

    for idx, r in enumerate(args.run):

        print(
            f"[{run_idx:02d}/{total_runs}] " +
            f"type={r['capture_type']:6s} | " +
            f"k={r['k']} | " +
            f"{r['freq_mhz']}MHz | " +
            f"{r['mV']}mV | " +
            f"{r['mA']}mA | " +
            f"{r['duty']}% | "
        )

        filename = make_correlation_filename(r['capture_type'], r['k'], r['freq_mhz'],
                                             r['mV'], r['mA'], r['duty'])

        path = os.path.join(folder, filename)

        name = r["capture_type"]

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

        for x, photon_count_base in enumerate(args.photon_counts):
            for y, sbr in enumerate(args.sbrs):

                photon_count = photon_count_base * 1.4 if r['capture_type'] == 'ham' else photon_count_base
                decoded_depths = decode_from_correlations(
                    coding_matrix=coding_matrix,
                    depths=depths,
                    photon_count=photon_count,
                    sbr=sbr,
                    trials=args.trials,
                    rep_rate=args.rep_rate,
                )

                depth_res = 1000 if args.rep_rate is not None else 1
                rmse = float(np.sqrt(np.mean((decoded_depths - depths) ** 2))) * depth_res
                mae = float(np.mean(np.abs(decoded_depths - depths))) * depth_res

                print(f"\t Photon Count = {photon_count} | SBR = {sbr} | RMSE = {rmse} | MAE = {mae}")

                results_dict = dict(
                    coding_matrix=coding_matrix,
                    decoded_depths=decoded_depths,
                    rmse=rmse,
                    mae=mae,
                    name=name,
                    depths=depths,
                    photon_count=photon_count_base,
                    sbr=sbr,
                )

                mae_results[idx, x, y] = mae
                rmse_results[idx, x, y] = rmse

    # photon_counts = np.array(args.photon_counts)
    # sbrs = np.array(args.sbrs)
    # X, Y = np.meshgrid(photon_counts, sbrs, indexing='ij')

    # fig = plt.figure(figsize=(5, 4))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # for idx, r in enumerate(args.run):
    #     ax.plot_surface(X, Y, rmse_results[idx], label=r['capture_type'])
    #     ax.set_xlabel('photons')
    #     ax.set_ylabel('sbr')
    #     ax.set_zlabel('rmse')
    #     ax.set_title(r['capture_type'])
    #
    # ax.set_zlim(0, 1000)
    # ax.legend()
    # plt.show()

    photon_counts = np.array(args.photon_counts)
    sbrs = np.array(args.sbrs)
    X, Y = np.meshgrid(np.log10(photon_counts), np.log10(sbrs), indexing='ij')
    fig = go.Figure()
    colors = ['green', 'orange', 'red']

    for idx, r in enumerate(args.run):
        fig.add_trace(go.Surface(x=X, y=Y, z=mae_results[idx],
                                 showscale=False,
                                 colorscale=[[0, colors[idx]], [1, colors[idx]]],
                                 ))
        fig.update_layout(
            title=r['capture_type'],
            scene=dict(
                xaxis_title='log photons',
                yaxis_title='log sbr',
                zaxis_title='mae',
                zaxis=dict(range=[0, 300])
            ),
            width=500,
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
        )
    pio.renderers.default = 'browser'
    fig.show()

