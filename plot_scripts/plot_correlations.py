import os
import sys
sys.path.append('..')
sys.path.append('.')


from plot_scripts.plot_utils import plot_correlation_functions, plot_correlation_comparison
from utils.file_utils import make_correlation_filename, get_data_folder
from utils.global_constants import *
from utils.tof_utils import build_coding_matrix_from_correlations, get_simulated_coding_matrix
import numpy as np
import matplotlib.pyplot as plt
import pprint



#filename =  'coarsek3_10mhz_3400mV_50mA_30duty_correlations.npz'

filename =  'hamk3_10mhz_4000mV_50mA_20duty_correlations.npz'

SMOOTH_SIGMA = 1
N_TBINS_DEFAULT = 1000
SHIFT = None #-8 # -4
OTHER_SHIFT = -30 #-80

if __name__ == "__main__":
    folder = get_data_folder(READ_PATH_CORRELATIONS_MAC, READ_PATH_CORRELATIONS_WINDOWS)
    path = os.path.join(folder, filename)


    file = np.load(path, allow_pickle=True)
    cfg = file["cfg"].item()
    correlations_total = file["correlations"]
    K = cfg["k"]

    pprint.pprint(cfg)
    # plt.imshow(np.mean(np.mean(correlations_total, axis=-1), axis=-1))
    # plt.show()

    measured_coding_matrix = build_coding_matrix_from_correlations(
        correlations_total,
        False,
        SMOOTH_SIGMA,
        SHIFT,
        N_TBINS_DEFAULT,
    )

    coding_matrix = get_simulated_coding_matrix(cfg['capture_type'], N_TBINS_DEFAULT, K)
    #coding_matrix = get_simulated_coding_matrix('ham', N_TBINS_DEFAULT, K)


    point_list = [(10, 10), (145, 275), (50, 200)]

    plot_correlation_functions(
            point_list,
            correlations_total,
            N_TBINS_DEFAULT,
            SMOOTH_SIGMA
    )

    plot_correlation_comparison(
            measured_coding_matrix,
            coding_matrix,
            OTHER_SHIFT
    )