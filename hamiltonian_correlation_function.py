# Libraries
import os
import glob
from spad_lib.SPAD512S import SPAD512S
from spad_lib.spad512utils import *
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter, median_filter
import math
from PIL import Image

port = 9999  # Check the command Server in the setting tab of the software and change it if necessary
SPAD1 = SPAD512S(port)

# Get informations on the system used
info = SPAD1.get_info()
print("\nGeneral informations of the camera :")
print(info)
temp = SPAD1.get_temps()  # Current temperatures of FPGAs, PCB and Chip
print("\nCurrent temperatures of FPGAs, PCB and Chip :")
print(temp)
freq = SPAD1.get_freq()  # Operating frequencies (Laser and frame)
print("\nOperating frequencies (Laser and frame) :")
print(freq)

# # # Set the voltage to the maximum value
Vex = 7
SPAD1.set_Vex(Vex)


# Editable parameters
intTime = 100  # integration time
K = 3  # number of time bins
im_width = 512  # image width
bitDepth = 12
#n_tbins = 640
shift = 100 # shift in picoseconds...
voltage = 10

#Don't edit thesee pelase:
iterations = 1
overlap = 0
timeout = 0
pileup = 0
gate_steps = 1
gate_step_arbitrary = 0
gate_step_size = 0
gate_direction = 1
gate_trig = 0

tau = ((1/float(freq[-2])) * 1e12) #Tau in picoseconds
n_tbins = int(tau // shift)

print(f'Number of effective bins: {n_tbins}')
print(f'Shift: {shift}')

save_into_file = True

save_path = '/home/ubi-user/David_P_folder'
save_name = f'hamk{k}_{freq*1e6}mhz_{voltage}v_correlations'

func = getattr(CodingFunctionsFelipe, f"GetHamK{K}")
(modfs, demodfs) = func(N=n_tbins)
gated_demodfs_np, gated_demodfs_arr = decompose_ham_codes(demodfs)


(rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, 1. / float(freq))

print(f'Time bin depth resolution {tbin_depth_res * 1000:.3f} mm')

correlations = np.zeros((im_width, im_width, K, n_tbins))

for i, item in enumerate(gated_demodfs_arr):
    for k in range(item.shape[-1]):
        gate = item[:, k]
        gate_width, gate_start = get_offset_width_spad512(gate, float(freq[-2]))


        for j in range(n_tbins):
            gate_start_tmp = gate_start + j * shift
            gate_start_tmp = gate_start_tmp % tau

            if (gate_start_tmp + (gate_width * 1e3)) > tau:
                gate_start_one = gate_start_tmp
                gate_start_two = 0
                gate_one_width = tau - gate_start_tmp
                gate_two_width = (gate_width * (1e3)) - gate_one_width
                gate_starts_helper = [gate_start_one, gate_start_two]
                gate_widths_helper = [gate_one_width, gate_two_width]

            else:
                gate_starts_helper = [gate_start_tmp]
                gate_widths_helper = [gate_width]

            counts = np.zeros((im_width, im_width))
            for p, gate_start_input in enumerate(gate_starts_helper):
                gate_width_help = gate_widths_helper[p]
                current_intTime = intTime
                while current_intTime > 4800:
                    print(f'starting current time {current_intTime}')
                    counts += SPAD1.get_gated_intensity(bitDepth, 4800, iterations, gate_steps, gate_step_size,
                                                        gate_step_arbitrary, gate_width_help,
                                                        gate_start_input, gate_direction, gate_trig, overlap, 1, pileup, im_width)[:, :, 0]
                    current_intTime -= 4800

                counts += SPAD1.get_gated_intensity(bitDepth, current_intTime, iterations, gate_steps, gate_step_size,
                                                    gate_step_arbitrary, gate_width_help,
                                                    gate_start_input, gate_direction, gate_trig, overlap, 1, pileup, im_width)[:, :,  0]

            correlations[:, :, i, :] = counts

unit = "ms"
factor_unit = 1e-3


if save_into_file:
    import os

    np.savez(os.path.join(save_path, save_name),
             total_time=intTime,
             K=K,
             im_width=im_width,
             bitDepth=bitDepth,
             n_tbins=n_tbins,
             iterations=iterations,
             overlap=overlap,
             timeout=timeout,
             pileup=pileup,
             gate_steps=gate_steps,
             gate_step_arbitrary=gate_step_arbitrary,
             gate_step_size=gate_step_size,
             gate_direction=gate_direction,
             gate_trig=gate_trig,
             freq=float(freq[-2]),
             voltage=voltage,
             correlations=correlations)

