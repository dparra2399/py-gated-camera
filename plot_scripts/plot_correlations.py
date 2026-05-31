import os
import sys
sys.path.append('..')
sys.path.append('.')


from plot_scripts.plot_utils import plot_correlation_functions, plot_correlation_comparison, \
    plot_correlation_comparison_seperate, get_string_name
from utils.file_utils import make_correlation_filename, get_data_folder, load_correlation_npz
from utils.global_constants import *
from utils.tof_utils import build_coding_matrix_from_correlations, get_simulated_coding_matrix
import numpy as np
import matplotlib.pyplot as plt
import pprint



# Add as many filenames as you want — each becomes one column in the plot
FILENAMES = [
    'hamk3_10mhz_500mV_16mA_20duty_correlations.npz',
    'coarsek3_10mhz_420mV_16mA_30duty_correlations.npz',
    'trapcoarsek3_10mhz_420mV_16mA_30duty_correlations.npz',
    #'hamk4_10mhz_770mV_16mA_15duty_correlations.npz',
    #'coarsek4_10mhz_540mV_16mA_23duty_correlations.npz',
    #'trapcoarsek4_10mhz_540mV_16mA_23duty_correlations.npz',
]

SMOOTH_SIGMA = 1
N_TBINS_DEFAULT = 1500
SHIFT = None   # shift applied when building coding matrix from correlations
OTHER_SHIFTS = [-620, -190, -190]   # shift applied to ideal coding matrix for alignment

if __name__ == "__main__":
    folder = get_data_folder(READ_PATH_CORRELATIONS_MAC, READ_PATH_CORRELATIONS_WINDOWS)

    measured_list = []
    coding_list   = []
    labels        = []
    K_last        = None

    for filename in FILENAMES:
        path = os.path.join(folder, filename)
        file = load_correlation_npz(path)
        cfg  = file["cfg"].item()
        correlations_total = file["correlations"]
        K = cfg["k"]
        K_last = K

        measured = build_coding_matrix_from_correlations(
            correlations_total, False, SMOOTH_SIGMA, SHIFT, N_TBINS_DEFAULT,
        )
        ideal = get_simulated_coding_matrix(cfg['capture_type'], N_TBINS_DEFAULT, K)

        measured_list.append(measured)
        coding_list.append(ideal)
        labels.append(get_string_name(cfg['capture_type'], K, True))

    point_list = [(10, 10), (100, 200), (50, 200)]

    # plot_correlation_functions(
    #     point_list,
    #     correlations_total,   # pass a list to get one column per set
    #     N_TBINS_DEFAULT,
    #     SMOOTH_SIGMA,
    # )

    if K_last < 1:
        plot_correlation_comparison(
            measured_list[0],
            coding_list[0],
            OTHER_SHIFTS,
        )
    else:
        plot_correlation_comparison_seperate(
            measured_list,
            coding_list,
            OTHER_SHIFTS,
            save_fig=True,
            labels=labels,
        )