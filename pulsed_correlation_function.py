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
intTime = 4000  # integration time
num_gates = 4 # number of time bins
im_width = 512  # image width
bitDepth = 12
#n_tbins = 640
shift = 300 # shift in picoseconds...
voltage = 7.6

save_into_file = True
plot_correlations = True

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

print('--------------------Parameters---------------')
print(f'Number of effective bins: {n_tbins}')
print(f'Shift: {shift}')
print(f'Frequency: {int(float(freq[-2]) * 1e-6)}MHZ')
print('---------------------------------------------')



save_path = '/home/ubi-user/David_P_folder/py-gated-camera/correlation_functions'
save_name = f'coarsek{num_gates}_{int(float(freq[-2]) * 1e-6)}mhz_{voltage}v_correlations'

gate_width = math.ceil((((1/float(freq[-2]))*1e12) // num_gates) * 1e-3 )
gate_starts = np.array([(gate_width * (gate_step)) for gate_step in range(num_gates)]) * 1e3

(rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, 1. / float(freq[-2]))

print(f'Time bin depth resolution {tbin_depth_res * 1000:.3f} mm')

correlations = np.zeros((im_width, im_width, num_gates, n_tbins))
for i in range(num_gates):
    gate_start = gate_starts[i]

    print('-------------------------------------------------------')
    print(f'Starting to measure correlations for gate number {i}')
    print('-------------------------------------------------------')

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
            
        if j % 20 == 0:
            print(f'Measuring gate shift number {j}')

        correlations[:, :, i, j] = counts
    print('-------------------------------------------------------')
    print(f'Finished to measure correlations for gate number {i}')
    print('-------------------------------------------------------')

correlations = np.flip(correlations, axis=-1)



unit = "ms"
factor_unit = 1e-3

if plot_correlations:

    mhz = int(float(freq[-2]) * 1e-6)
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

    point_list = [(100, 100), (200, 200), (100, 200)]

    fig, axs = plt.subplots(1, 5)
    axs[-1].imshow(np.sum(correlations[:,:,:,0], axis=-1)[: im_width // 2, :im_width //2])
    axs[-2].plot(coding_matrix)
    for i, item in enumerate(point_list):
        x, y = item
        axs[i].plot(np.transpose(correlations[y, x, :, :]))
        axs[-1].plot(x, y, 'ro')
        
    plt.show()


if save_into_file:
    import os

    np.savez(os.path.join(save_path, save_name),
             total_time=intTime,
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
             gate_direction=gate_direction,
             gate_trig=gate_trig,
             gate_width=gate_width,
             freq=float(freq[-2]),
             voltage=voltage,
             correlations=correlations)

