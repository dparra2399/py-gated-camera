import os
import sys
import glob
sys.path.append('..')
sys.path.append('.')


from spad_lib.spad512utils import *
from plot_scripts.plot_utils import plot_correlation_functions
import numpy as np
from felipe_utils import CodingFunctionsFelipe


#filename = 'coarsek4_10mhz_7.6v_25w_correlations.npz'
#filename = 'coarsek3_10mhz_7v_34w_correlations.npz'
filename = 'hamk4_10mhz_8.5v_20w_correlations.npz'
SMOOTH_CORRELATIONS = True
SMOOTH_SIGMA = 30

try:
    path = f'/home/ubi-user/David_P_folder/py-gated-camera/correlation_functions/{filename}'
    file = np.load(path)
except FileNotFoundError:
    path = f'/Users/davidparra/PycharmProjects/py-gated-camera/correlation_functions/{filename}'
    file = np.load(path)

correlations = file['correlations']
total_time = file["total_time"]
n_tbins = file['n_tbins']
im_width = file["im_width"]
bitDepth = file["bitDepth"]
iterations = file["iterations"]
overlap = file["overlap"]
timeout = file["timeout"]
pileup = file["pileup"]
gate_steps = file["gate_steps"]
gate_step_arbitrary = file["gate_step_arbitrary"]
gate_step_size = file["gate_step_size"]
gate_direction = file["gate_direction"]
gate_trig = file["gate_trig"]
freq = file["freq"]
voltage = file['voltage']
size = file['size']

(rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, 1 / freq)
mhz = int(freq * 1e-6)


if 'coarse' in filename:
    num_gates = file['num_gates']
    gate_width = file['gate_width']

    irf = get_voltage_function(mhz, voltage, size,'pulse', n_tbins)
    coding_matrix = get_coarse_coding_matrix(gate_width * 1e3, num_gates, 0, gate_width * 1e3, rep_tau * 1e12, n_tbins, irf)
    demodfs = get_coarse_coding_matrix(gate_width * 1e3, num_gates, 0, gate_width * 1e3, rep_tau * 1e12, n_tbins, None)

elif 'ham' in filename:
    K = correlations.shape[-2]
    illum_type = 'pulse' if 'pulse' in filename else 'square'

    coding_matrix = get_hamiltonain_correlations(K, mhz, voltage, size, illum_type, n_tbins=n_tbins)

    func = getattr(CodingFunctionsFelipe, f"GetHamK{K}")
    (_, demodfs) = func(N=n_tbins, modDuty=1/5)
    demodfs, demodfs_arr = decompose_ham_codes(demodfs)
else:
    assert False

point_list = [(10, 10), (200, 200), (50, 200)]


plot_correlation_functions(
        point_list,
        correlations,
        coding_matrix,
        SMOOTH_SIGMA,
        SMOOTH_CORRELATIONS,

)