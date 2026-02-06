# Imports
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from felipe_utils import tof_utils_felipe
from utils.global_constants import *
from utils.file_utils import get_data_folder, make_correlation_filename, corr_parse_run
from utils.tof_utils import (
    build_coding_matrix_from_correlations,
    decode_from_correlations,
    get_simulated_coding_matrix,
)

# =============================
# Defaults
# =============================
PHOTON_COUNT = 1000
SBR =0.1
TRIALS = 100
N_TBINS = 999
SIMULATED_CORRELATIONS = True
SMOOTH_SIGMA = None
SHIFT = None
DEPTH_MARGIN = 100
"""
Format:
capture_type,k,freq_mhz,mV,mA,duty

Example:
ham,3,5,100,50,10
"""

DEFAULT_RUNS = [
    "ham,3,5, 3000,50,20",
    "coarse,3,5, 2400,50,30",
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
    p.add_argument("--smooth_sigma", type=float, default=SMOOTH_SIGMA)
    p.add_argument("--shift", type=int, default=SHIFT)
    p.add_argument("--depth_margin", type=int, default=DEPTH_MARGIN)
    p.add_argument("--simulated_correlations", action="store_true", default=SIMULATED_CORRELATIONS)

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

    depths = np.arange(args.depth_margin, args.n_tbins - args.depth_margin, 1)
    folder = get_data_folder(READ_PATH_CORRELATIONS_MAC, READ_PATH_CORRELATIONS_WINDOWS)

    results_dict = {}

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
        if args.simulated_correlations:
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
            #if name == 'ham':
            #    coding_matrix[..., 0] = np.roll(coding_matrix[..., 0], 10)


        #coding_matrix /= np.max(np.abs(coding_matrix), axis=0, keepdims=True)
        mins = coding_matrix.min(axis=0, keepdims=True)
        maxs = coding_matrix.max(axis=0, keepdims=True)

        coding_matrix = (coding_matrix - mins) / (maxs - mins)

        decoded_depths = decode_from_correlations(
            coding_matrix=coding_matrix,
            depths=depths,
            photon_count=args.photon_count,
            sbr=args.sbr,
            trials=args.trials,
        )

        rmse = float(np.sqrt(np.mean((decoded_depths - depths) ** 2)))
        mae = float(np.mean(np.abs(decoded_depths - depths)))

        print(f"      → MAE={mae:.3f} | RMSE={rmse:.3f}")

        results_dict[filename] = dict(
            coding_matrix=coding_matrix,
            decoded_depths=decoded_depths,
            rmse=rmse,
            mae=mae,
            name=name,
        )

    all_coding_matrix = np.stack([result['coding_matrix'] for result in results_dict.values()])
    fig, axs = plt.subplots(len(results_dict), 3, figsize=(13, 4 * len(results_dict)), squeeze=False)

    for i, filename in enumerate(results_dict):
        result = results_dict[filename]
        coding_matrix = result['coding_matrix']
        rmse = result['rmse']
        mae = result['mae']
        name = result['name']

        axs[i, 0].imshow(np.repeat(np.transpose(coding_matrix), 100, axis=0), aspect='auto')
        axs[i, 0].set_title(name + ' Coding Matrix')
        axs[i, 1].plot(coding_matrix)
        #axs[i, 1].set_ylim(0, np.max(all_coding_matrix))
        axs[i, 1].set_title(name + ' Coding Matrix')
        if i < len(results_dict) - 2:
            axs[i, 2].set_axis_off()

        axs[-2, 2].bar([i], rmse, label=name)
        axs[-2, 2].text(
            i,  # x position
            rmse,  # y position (height of bar)
            f"{rmse:.2f}",  # displayed text, formatted
            ha='center', va='bottom'
        )
        axs[-2, 2].set_title('RMSE (Lower is better)')
        axs[-2, 2].set_ylabel('RMSE (time bins)')

        axs[-1, 2].bar([i], mae, label=name)
        axs[-1, 2].text(
            i,  # x position
            mae,  # y position (height of bar)
            f"{mae:.2f}",  # displayed text, formatted
            ha='center', va='bottom'
        )
        axs[-1, 2].set_title('MAE (Lower is better)')
        axs[-1, 2].set_ylabel('MAE (time bins)')


    names = [dic['name'] for dic in results_dict.values()]
    axs[-1, 2].set_xticks(np.arange(0, len(names)))
    axs[-1, 2].set_xticklabels(names)
    plt.subplots_adjust(wspace=0.5, hspace=0.4)
    plt.show()

    correlations_p =  [dic['coding_matrix'] for dic in results_dict.values()]
    # for idx, corr in enumerate(correlations_p):
    #     name = names[idx]
    #     if name == 'coarse':
    #         corr = np.roll(corr, 300, axis=0)
    #         plt.plot(corr, color='r')
    #     else:
    #         plt.plot(corr, color='b')
    # plt.show()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab10")
    type_colors = {}  # map capture type -> color
    for idx, corr in enumerate(correlations_p):  # (n_tbins, 3)
        cap_type = names[idx]

        if cap_type not in type_colors:
            type_colors[cap_type] = cmap(len(type_colors))

        color = type_colors[cap_type]

        diffs = np.diff(corr, axis=0)
        distance = np.linalg.norm(diffs, axis=1).sum()

        ax.plot(
            corr[:, 0], corr[:, 1], corr[:, 2],
            color=color,
            label=f"{cap_type} ({distance:.2f})"
        )

        ax.text(corr[0, 0], corr[0, 1], corr[0, 2], f"{distance:.2f}", color=color)

        print(f"{cap_type} correlation distance: {distance:.3f}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.legend(fontsize=8)
    plt.show()
    #
    # fig, axs = plt.subplots(len(results_dict), 1, figsize=(9, len(results_dict)))
    # for idx, dictionary in enumerate(results_dict.values()):
    #     decoded_depths = dictionary['decoded_depths']
    #     errors = np.mean(np.abs(depths - decoded_depths), axis=0)
    #     axs[idx].bar(np.arange(0, depths.shape[-1]), errors)
    #     axs[idx].set_ylim(0, 100)
    # plt.show()






