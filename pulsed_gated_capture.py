# Libraries
import os
import glob
from spad_lib.SPAD512S import SPAD512S
from spad_lib.spad512utils import *
import numpy as np
import time
import matplotlib.pyplot  as plt
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter, median_filter
import math


port = 9999 # Check the command Server in the setting tab of the software and change it if necessary
SPAD1 = SPAD512S(port)

# Get informations on the system used
info = SPAD1.get_info()
print("\nGeneral informations of the camera :")
print(info)
temp = SPAD1.get_temps() # Current temperatures of FPGAs, PCB and Chip
print("\nCurrent temperatures of FPGAs, PCB and Chip :")
print(temp)
freq = SPAD1.get_freq() # Operating frequencies (Laser and frame)
print("\nOperating frequencies (Laser and frame) :")
print(freq)

# # # Set the voltage to the maximum value
Vex = 7
SPAD1.set_Vex(Vex)


# Editable parameters
intTime = 1000000 #integration time
num_gates = 15 #number of time bins
im_width = 512 #image width
bitDepth = 12
n_tbins = 640

#Don't edit
iterations = 1
overlap = 0
timeout = 0
pileup = 1
gate_steps =  num_gates
gate_step_arbitrary = 0
gate_width = math.ceil((((1/float(freq[0]))*1e12) // num_gates) * 1e-3 )
gate_step_size = gate_width * 1e3
gate_offset = 0
gate_direction = 1
gate_trig = 0

print(f'gate steps: {gate_steps}')
print(f'gate width: {gate_width}')
print(f'gate step size: {gate_step_size}')

coded_vals = SPAD1.get_gated_intensity(bitDepth, intTime, iterations, gate_steps, gate_step_size, gate_step_arbitrary, gate_width, 
                                    gate_offset, gate_direction, gate_trig, overlap, 1, pileup, im_width)

print(coded_vals.shape)
unit = "ms"
factor_unit = 1e-3

correct_master = False
decode_depths = True


if correct_master:
    coded_vals[:, im_width//2:, :] = np.roll(coded_vals[:, im_width//2:, :], shift=1)

if decode_depths:
    (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, 1./ float(freq[0]))
    # print(rep_freq, rep_tau, tbin_res)
    # print(rep_tau * 1e12)
    #print(gate_step_size,gate_steps, gate_offset, gate_width)
    mhz = int(freq[0][:2])
    irf = get_voltage_function(mhz, 10, 'pulse')
    coding_matrix = get_coarse_coding_matrix(gate_width * 1e3, num_gates, 0, gate_width * 1e3, rep_tau * 1e12, n_tbins, irf)

    #fig, axs = plt.subplots(1, 2)
    #axs[0].imshow(np.repeat(get_coarse_coding_matrix(gate_width * 1e3, num_gates, 0, gate_width * 1e3, rep_tau * 1e12, n_tbins).transpose(), 100, axis=0), aspect='auto')
    #axs[1].imshow(np.repeat(coding_matrix.transpose(), 100, axis=0), aspect='auto')
    #plt.show()
    #exit(0)

    norm_coding_matrix = zero_norm_t(coding_matrix)


    norm_coded_vals = zero_norm_t(coded_vals)

    print(norm_coded_vals.shape)
    print(norm_coding_matrix.shape)

    zncc = np.matmul(norm_coding_matrix, norm_coded_vals[..., np.newaxis]).squeeze(-1)
    
    depths = np.argmax(zncc, axis=-1)

    depth_map = np.reshape(depths, (512, 512)) * tbin_depth_res

    fig, axs = plt.subplots(3, figsize=(10, 10))

    x1, y1 = (78, 420)
    x2, y2 = (220, 220)

    axs[0].bar(np.arange(0, num_gates), coded_vals[y1, x1, :], color='red')
    axs[1].bar(np.arange(0, num_gates), coded_vals[y2, x2, :], color='blue')
    #axs[0].set_xticks(np.arange(0, metadata['Gate steps'])[::3])
    #axs[0].set_xticklabels(np.round(gate_starts, 1)[::3])

    axs[2].imshow(median_filter(depth_map, size=1))
    axs[2].plot(x1, y1, 'ro')
    axs[2].plot(x2, y2, 'bo')
    plt.show()
    print('done')

