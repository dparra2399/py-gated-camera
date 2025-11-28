import os
import sys
sys.path.append('..')
sys.path.append('.')
import glob
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from felipe_utils.CodingFunctionsFelipe import *
from depth_decoding_single import N_TBINS_DEFAULT




from spad_lib.spad512utils import *
from plot_scripts.plot_utils import plot_correlation_functions, plot_correlation_comparison
import numpy as np
from felipe_utils import CodingFunctionsFelipe


#filename = 'coarsek4_10mhz_7.6v_25w_correlations.npz'
#filename = 'coarsek3_10mhz_7v_34w_correlations.npz'
#filename = 'hamk3_10mhz_8.5v_20w_correlations.npz'
#filename = 'coarsek8_10mhz_10v_12w_correlations.npz'

filename =  'hamk3_5mhz_6.5v_20w_correlations.npz'
#ilename = 'coarsek3_9mhz_10v_12w_correlations_extended.npz'

SMOOTH_CORRELATIONS = False
SMOOTH_SIGMA = 5
N_TBINS_DEFAULT = 99
SHIFT = 190

try:
    path = f'/home/ubi-user/David_P_folder/{filename}'
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
try:
    K = file['K']
except:
    K = correlations.shape[-2]


(rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, 1 / freq)
mhz = int(freq * 1e-6)
mhz = 10


if 'ham' in filename:
    (modfs, demodfs) = GetHamK3(N=n_tbins)
    irf = gaussian_pulse(np.arange(n_tbins), 0, 1, circ_shifted=True)
    modfs = np.fft.ifft(np.fft.fft(irf[..., np.newaxis], axis=0).conj() * np.fft.fft(modfs, axis=0), axis=0).real
    coding_matrix = np.fft.ifft(np.fft.fft(modfs, axis=0).conj() * np.fft.fft(demodfs, axis=0), axis=0).real
    coding_matrix = np.fft.ifft(np.fft.fft(irf[..., np.newaxis], axis=0).conj() * np.fft.fft(coding_matrix, axis=0),
                                axis=0).real
elif 'coarse' in filename:
    coding_matrix = np.kron(np.eye(K), np.ones((1, n_tbins // K)))
    irf = gaussian_pulse(np.arange(coding_matrix.shape[-1]), 0, 180, circ_shifted=True)
    coding_matrix = np.fft.ifft(
        np.fft.fft(irf[..., np.newaxis], axis=0).conj() * np.fft.fft(np.transpose(coding_matrix), axis=0),
        axis=0).real
#
# if 'coarse' in filename:
#     gate_width = file['gate_width']
#
#     irf = get_voltage_function(mhz, voltage, size,'pulse', n_tbins)
#     coding_matrix = get_coarse_coding_matrix(gate_width * 1e3, K, 0, gate_width * 1e3, rep_tau * 1e12, n_tbins, irf)
#     demodfs = get_coarse_coding_matrix(gate_width * 1e3, K, 0, gate_width * 1e3, rep_tau * 1e12, n_tbins, None)
#
# elif 'ham' in filename:
#     #illum_type = 'pulse' if 'pulse' in filename else 'square'
#     illum_type = 'square'
#     coding_matrix = get_hamiltonain_correlations(K, mhz, voltage, size, illum_type, n_tbins=n_tbins)
#
#     func = getattr(CodingFunctionsFelipe, f"GetHamK{K}")
#     (_, demodfs) = func(N=n_tbins, modDuty=1/5)
#     demodfs, demodfs_arr = decompose_ham_codes(demodfs)
# else:
#     assert False

point_list = [(10, 10), (200, 200), (50, 200)]

# irf = get_voltage_function(mhz, voltage, size, 'pulse', n_tbins)
# plt.plot(irf)
# plt.show()

plot_correlation_functions(
        point_list,
        correlations,
        coding_matrix,
        SMOOTH_SIGMA,
        SMOOTH_CORRELATIONS,
        N_TBINS_DEFAULT,
)

plot_correlation_comparison(
        correlations,
        coding_matrix,
        SMOOTH_SIGMA,
        SMOOTH_CORRELATIONS,
        N_TBINS_DEFAULT,
        SHIFT,
)