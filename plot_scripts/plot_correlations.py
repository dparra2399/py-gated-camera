import os
import sys
import glob
sys.path.append('..')
sys.path.append('.')


from spad_lib.SPAD512S import SPAD512S
from spad_lib.spad512utils import *
import numpy as np
import time
import matplotlib.pyplot  as plt
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter, median_filter, gaussian_filter1d
from felipe_utils import CodingFunctionsFelipe
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from glob import glob
from PIL import Image
import math
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse

filename = 'coarsek4_10mhz_7.6v_25w_correlations.npz'
#filename = 'coarsek3_10mhz_7v_34w_correlations.npz'
#filename = 'hamk4_10mhz_8.5v_20w_correlations.npz'

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
voltage = file["voltage"]
freq = file["freq"]

(rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, 1 / freq)
mhz = int(freq * 1e-6)


if 'coarse' in filename:
    num_gates = file['num_gates']
    gate_width = file['gate_width']
    if num_gates == 3 and mhz == 10:
        voltage = 7
        size = 34
    elif num_gates == 3 and mhz == 5:
        voltage = 5.7
        size = 67
    elif num_gates == 4 and mhz == 10:
        voltage = 7.6
        size = 25
    elif num_gates == 4 and mhz == 5:
        voltage = 6
        size = 50
    else:
        voltage = 10
        size = 12

    irf = get_voltage_function(mhz, voltage, size,'pulse', n_tbins)
    coding_matrix = get_coarse_coding_matrix(gate_width * 1e3, num_gates, 0, gate_width * 1e3, rep_tau * 1e12, n_tbins, irf)
    demodfs = get_coarse_coding_matrix(gate_width * 1e3, num_gates, 0, gate_width * 1e3, rep_tau * 1e12, n_tbins, None)

elif 'ham' in filename:
    K = correlations.shape[-2]
    if 'pulse' in filename:
        illum_type = 'pulse'
        size = 12
        voltage = 10

    else:
        illum_type = 'square'
        size = 20

    coding_matrix = get_hamiltonain_correlations(K, mhz, voltage, size, illum_type, n_tbins=n_tbins)

    func = getattr(CodingFunctionsFelipe, f"GetHamK{K}")
    (_, demodfs) = func(N=n_tbins, modDuty=1/5)
    demodfs, demodfs_arr = decompose_ham_codes(demodfs)

average_correlation = np.mean(np.mean(correlations[200:400,  100:256], axis=0), axis=0)
average_correlation = np.transpose(gaussian_filter1d(average_correlation, sigma=10, axis=-1))

point_list = [(10, 10), (200, 200), (50, 200)]

fig, axs = plt.subplots(1, len(point_list)+3)
axs[-1].imshow(np.sum(correlations[:,:,:,0], axis=-1)[: im_width // 2, :im_width //2])
axs[-1].set_title('Intensity Image')

axs[-2].plot(coding_matrix)
axs[-2].set_title('Corrfs \n (APD Signal)')
axs[-3].plot(average_correlation)
axs[-3].set_title('Corrfs \n (averaged)')

colors = ['red', 'blue', 'orange', 'green', 'purple', 'brown']
correlations_tmp = correlations.swapaxes(-1, -2)
correlations_tmp = gaussian_filter(correlations_tmp, sigma=(1, 1, 1, 0))
for i, item in enumerate(point_list):
    x, y = item

    axs[i].plot(gaussian_filter1d(correlations_tmp[y, x, :, :], sigma=10, axis=0))
    axs[-1].plot(x, y, 'o', color=colors[i % len(colors)])
    axs[i].set_title(f'Corrfs \n ({colors[i % len(colors)]} pixel)')
plt.show()

#
# if 'ham' in filename:
#     fig, axs = plt.subplots(2, demodfs.shape[-1])
#
#     colors = ['royalblue', 'darkorange', 'green', 'purple', 'brown', 'red', 'black']
#     counter = 0
#     for i, item in enumerate(demodfs_arr):
#         for j in range(item.shape[-1]):
#             axs[0][counter].plot(item[:, j], color=colors[i % len(colors)])
#             if counter < average_correlation.shape[-1]:
#                 axs[1][counter].plot(average_correlation[:, counter], color=colors[i % len(colors)])
#             else:
#                 axs[1][counter].set_axis_off()
#             counter += 1
#
#
#     plt.show()
# else:
#     fig, axs = plt.subplots(2, demodfs.shape[-1])
#
#     colors = ['royalblue', 'darkorange', 'green', 'purple', 'brown', 'red', 'black']
#     for i in range(demodfs.shape[-1]):
#         axs[0][i].plot(demodfs[:, i], color=colors[i % len(colors)])
#         if i < average_correlation.shape[-1]:
#             axs[1][i].plot(average_correlation[:, i], color=colors[i % len(colors)])
#         else:
#             axs[1][i].set_axis_off()
#     plt.show()