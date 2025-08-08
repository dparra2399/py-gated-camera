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
from felipe_utils import CodingFunctionsFelipe
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
total_time = 500 #integration time
num_gates = 1 #number of time bins
im_width = 512 #image width
bitDepth = 12
K = 5
n_tbins = 640
correct_master = False
decode_depths = True
save_into_file = True
voltage = 10
save_path = '/mnt/researchdrive/research_users/David/gated_project_data'
save_name = f'hamK{K}_exp1'


#Get demodulation functions and split for use with Gated SPAD
func = getattr(CodingFunctionsFelipe, f"GetHamK{K}")
(modfs, demodfs) = func(N=n_tbins)
gated_demodfs_np, gated_demodfs_arr = decompose_ham_codes(demodfs)

#For each demod function we make a gate sequence
intTimes = [int(total_time // K)] * K
coded_vals = np.zeros((im_width, im_width, K))
for i, item in enumerate(gated_demodfs_arr):
    for j in range(item.shape[-1]):
        gate = item[:, j]
        gate_width, gate_offset = get_offset_width_spad512(gate, float(freq[0]))
        print(f'gate width = {gate_width}, gate offset = {gate_offset}')
                
        #Don't edit
        iterations = 1
        overlap = 1
        timeout = 0
        pileup = 0
        gate_steps =  1
        gate_step_arbitrary = 0
        gate_step_size = 0
        gate_direction = 1
        gate_trig = 0
        intTime = int(intTimes[i] // item.shape[-1])
        #intTime = int(total_time // gated_demodfs_np.shape[-1])

        #print(f'gate steps: {gate_steps}')

        #print(f'gate step size: {gate_step_size}')

        counts = SPAD1.get_gated_intensity(bitDepth, intTime, iterations, gate_steps, gate_step_size, gate_step_arbitrary, gate_width, 
                                            gate_offset, gate_direction, gate_trig, overlap, 1, pileup, im_width)
        
        coded_vals[:, :, i] += counts[:, :, 0]

print(coded_vals.shape)
unit = "ms"
factor_unit = 1e-3
        


if decode_depths:
    (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, 1./ float(freq[0]))
    # print(rep_freq, rep_tau, tbin_res)
    # print(rep_tau * 1e12)
    #print(gate_step_size,gate_steps, gate_offset, gate_width)

    mhz = int(freq[0][:2])
    #print(mhz)
    correlations = get_hamiltonain_correlations(demodfs, mhz, voltage, interpolate=True)

    # fig, axs = plt.subplots(1, 3)
    # axs[0].plot(demodfs)
    # axs[1].plot(get_voltage_function(mhz, 10))
    # axs[2].plot(correlations)
    # plt.show()
    # exit()

    norm_coding_matrix = zero_norm_t(correlations)

    norm_coded_vals = zero_norm_t(coded_vals)

    print(norm_coded_vals.shape)
    print(norm_coding_matrix.shape)

    zncc = np.matmul(norm_coding_matrix, norm_coded_vals[..., np.newaxis]).squeeze(-1)
    
    if correct_master:
        zncc[:, im_width//2:, :] = np.roll(zncc[:, im_width//2:, :], shift=-5)


    depths = np.argmax(zncc, axis=-1)

    depth_map = np.reshape(depths, (512, 512)) * tbin_depth_res

    fig, axs = plt.subplots(3, figsize=(10, 10))

    x1, y1 = (335, 71)
    x2, y2 = (320, 220)

    axs[0].bar(np.arange(0, K), coded_vals[y1, x1, :], color='red')
    axs[1].bar(np.arange(0, K), coded_vals[y2, x2, :], color='blue')
    #axs[0].set_xticks(np.arange(0, metadata['Gate steps'])[::3])
    #axs[0].set_xticklabels(np.round(gate_starts, 1)[::3])

    axs[2].imshow(depth_map)
    #axs[2].imshow(depth_map[:, :im_width//2])
    #axs[2].plot(x1, y1, 'ro')
    #axs[2].plot(x2, y2, 'bo')
    plt.show()
    print(f'min depth map: {np.min(depth_map)}, max depth map: {np.max(depth_map)}')



if save_into_file:
    import os
    np.savez(os.path.join(save_path, save_name),
         total_time=total_time,
         num_gates=num_gates,
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
         gate_offset=gate_offset,
         gate_direction=gate_direction,
         gate_trig=gate_trig,
         voltage=voltage,
         coded_vals=coded_vals,
         irf=get_voltage_function(mhz, voltage, 'square'))

    