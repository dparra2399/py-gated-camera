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


path = '/home/ubi-user/David_P_folder/py-gated-camera/correlation_functions/coarsek3_10mhz_10v_correlations.npz'

file = np.load(path)

correlations = file['correlations']
total_time = file["total_time"]
n_tbins = file['n_tbins']
num_gates = file["num_gates"]
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
gate_width = file['gate_width']

(rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, 1 / freq)


mhz = int(freq * 1e-6)
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

average_correlation = np.mean(np.mean(correlations, axis=0), axis=0)
average_correlation = gaussian_filter1d(average_correlation, sigma=0.5, axis=-1)

point_list = [(10, 10), (200, 200), (100, 200)]

fig, axs = plt.subplots(1, len(point_list)+3)
axs[-1].imshow(np.sum(correlations[:,:,:,0], axis=-1)[: im_width // 2, :im_width //2])
axs[-1].set_title('Intensity Image')
axs[-2].plot(coding_matrix)
axs[-2].set_title('Correlations using APD Signal')
axs[-3].plot(np.transpose(average_correlation))
axs[-3].set_title('Correlations averaged across pixels')

colors = ['red', 'blue', 'orange', 'green', 'purple', 'brown']
for i, item in enumerate(point_list):
    x, y = item
    axs[i].plot(np.transpose(correlations[y, x, :, :]))
    axs[-1].plot(x, y, 'o', color=colors[i % len(colors)])
    axs[i].set_title(f'Correlation at {colors[i % len(colors)]} pixel')
    
plt.show()