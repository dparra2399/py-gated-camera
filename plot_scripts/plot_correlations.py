import os
import sys
sys.path.append('..')
sys.path.append('.')


from plot_scripts.plot_utils import plot_correlation_functions, plot_correlation_comparison
from utils.file_utils import make_correlation_filename, get_data_folder
from utils.global_constants import *
from utils.tof_utils import build_coding_matrix_from_correlations, get_simulated_coding_matrix
import numpy as np



filename =  'coarsek3_5mhz_4000mV_50mA_30duty_correlations.npz'

#filename =  'hamk3_5mhz_4000mV_50mA_20duty_correlations.npz'

SMOOTH_SIGMA = None
N_TBINS_DEFAULT = 1000
SHIFT = None #-8 # -4
OTHER_SHIFT = 100

if __name__ == "__main__":
    folder = get_data_folder(READ_PATH_CORRELATIONS_MAC, READ_PATH_CORRELATIONS_WINDOWS)
    path = os.path.join(folder, filename)


    file = np.load(path, allow_pickle=True)
    cfg = file["cfg"].item()
    correlations_total = file["correlations"]
    K = cfg["k"]

    measured_coding_matrix = build_coding_matrix_from_correlations(
        correlations_total,
        False,
        SMOOTH_SIGMA,
        SHIFT,
        N_TBINS_DEFAULT,
    )

    coding_matrix = get_simulated_coding_matrix(cfg['capture_type'], N_TBINS_DEFAULT, cfg["k"])


    point_list = [(10, 10), (200, 200), (50, 200)]

    # plot_correlation_functions(
    #         point_list,
    #         correlations,
    #         coding_matrix,
    #         SMOOTH_SIGMA,
    #         SMOOTH_CORRELATIONS,
    #         N_TBINS_DEFAULT,
    # )

    plot_correlation_comparison(
            measured_coding_matrix,
            coding_matrix,
            OTHER_SHIFT
    )