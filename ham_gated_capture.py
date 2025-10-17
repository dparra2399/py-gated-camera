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
split_measurements = False
num_gates = 1 #number of time bins
im_width = 512 #image width0
bitDepth = 12
K = 4
n_tbins = 640
correct_master = False
decode_depths = True
save_into_file = True

duty=20
vmin = 21
vmax = 27

exp_num = 12
save_path = '/home/ubi-user/David_P_folder'
#save_path = '/mnt/researchdrive/research_users/David/gated_project_data'
save_name = f'hamK{K}_exp{exp_num}'

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
        #plt.plot(gate)
        #plt.show()
        gate_width, gate_offset = get_offset_width_spad512(gate, float(freq[-2]))
        print(f'gate width = {gate_width}, gate offset = {gate_offset}')
                
        #Don't edit
        iterations = 1
        overlap = 0
        timeout = 0
        pileup = 0
        gate_steps =  1
        gate_step_arbitrary = 0
        gate_step_size = 0
        gate_direction = 1
        gate_trig = 0
        #intTime = int(intTimes[i] // item.shape[-1])
        if split_measurements:
            intTime = int(total_time // gated_demodfs_np.shape[-1])
        else:
            intTime = total_time

        print(f'Integration time for hamiltonian row #{i+1}.{j+1}: {intTime}')

        current_intTime = intTime
        counts = np.zeros((im_width, im_width))
        while current_intTime > 4800:
            print(f'starting current time {current_intTime}')
            counts += SPAD1.get_gated_intensity(bitDepth, 4800, iterations, gate_steps, gate_step_size,
                                                gate_step_arbitrary, gate_width,
                                                gate_offset, gate_direction, gate_trig, overlap, 1, pileup, im_width)[:, :, 0]
            current_intTime -= 4800

        counts += SPAD1.get_gated_intensity(bitDepth, current_intTime, iterations, gate_steps, gate_step_size,
                                            gate_step_arbitrary, gate_width,
                                            gate_offset, gate_direction, gate_trig, overlap, 1, pileup, im_width)[:, :, 0]

        coded_vals[:, :, i] += counts

print(coded_vals.shape)
unit = "ms"
factor_unit = 1e-3
        


if decode_depths:
    (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, 1./ float(freq[-2]))
    # print(rep_freq, rep_tau, tbin_res)
    # print(rep_tau * 1e12)
    #print(gate_step_size,gate_steps, gate_offset, gate_width)

    mhz = int(float(freq[-2]) * 1e-6)
    if duty == 20 and mhz == 10:
        voltage = 8.5
    elif duty == 20 and mhz == 5:
        voltage = 6.5
    else:
        voltage = 10
    #print(mhz)

    if 'pulse' in save_name:
        illum_type = 'pulse'
        duty = 12
        voltage = 10
    else:
        illum_type = 'square'
        duty = 20

    correlations = get_hamiltonain_correlations(K, mhz, voltage, duty, illum_type, n_tbins=n_tbins)

    # fig, axs = plt.subplots(1, 3)
    # axs[0].plot(demodfs)
    # axs[1].plot(get_voltage_function(mhz, 10))
    # axs[2].plot(correlations)
    # plt.show()
    # exit()

    norm_coding_matrix = zero_norm_t(correlations)

    norm_coded_vals = zero_norm_t(coded_vals)

    #print(norm_coded_vals.shape)
    #print(norm_coding_matrix.shape)

    zncc = np.matmul(norm_coding_matrix, norm_coded_vals[..., np.newaxis]).squeeze(-1)
    
    if correct_master:
        zncc[:, im_width//2:, :] = np.roll(zncc[:, im_width//2:, :], shift=-5)


    depths = np.argmax(zncc, axis=-1)

    depth_map = np.reshape(depths, (512, 512)) * tbin_depth_res

    fig, axs = plt.subplots(3, figsize=(10, 10))

    x1, y1 = (70, 70)
    x2, y2 = (220, 330)

    axs[0].bar(np.arange(0, K), coded_vals[y1, x1, :], color='red')
    axs[1].bar(np.arange(0, K), coded_vals[y2, x2, :], color='blue')
    #axs[0].set_xticks(np.arange(0, metadata['Gate steps'])[::3])
    #axs[0].set_xticklabels(np.round(gate_starts, 1)[::3])

    axs[2].imshow(median_filter(depth_map, size=1), vmin=vmin, vmax=vmax)
    #axs[2].imshow(depth_map[:, :im_width//2])
    axs[2].plot(x1, y1, 'ro')
    axs[2].plot(x2, y2, 'bo')

    x, y = 20, 170
    width, height = 220, 320
    box = depth_map[y:y+height, x:x+width]
    wall = depth_map[:x, :y-20]

    print(f'box mean depth: {np.mean(box):.3f} \nwall mean depth: {np.mean(wall):.3f} \
          \nmean depth between wall and box: {np.mean(wall) - np.mean(box):.3f}')

    plt.show()


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
         freq=float(freq[-2]),
         voltage=voltage,
         coded_vals=coded_vals,
         split_measurements=split_measurements,
         duty=duty,
         irf=get_voltage_function(mhz, voltage, duty,illum_type))

    