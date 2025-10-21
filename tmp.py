# import numpy as np
# import os
# from glob import glob
#
# # Path to directory
# folder = "/Volumes/velten/Research_Users/David/gated_project_data/exp5"
#
# # Get all .npz files in the directory
# npz_files = glob(os.path.join(folder, "*.npz"))
#
# for path in npz_files:
#     # if path.endswith("exp1.npz"):
#     #     continue
#     try:
#         # Load the .npz file
#         file = np.load(path)
#
#         # Convert to dict
#         params = {k: v for k, v in file.items()}
#
#         # Add freq
#         params["gate_width"] = 13 # 10 MHz
#         #params.pop("rep_freq")
#         #params["freq"] = 10_000_000
#
#         # Resave (overwrite original)
#         np.savez(path, **params)
#
#         print(f"Updated: {os.path.basename(path)}")
#     except Exception as e:
#         print(f"Failed to update {path}: {e}")

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
from matplotlib.animation import FuncAnimation, PillowWriter


# Editable parameters
intTime = 100  # integration time
num_gates = 3  # number of time bins
im_width = 512  # image width
bitDepth = 12
#n_tbins = 640
shift = 100 # shift in picoseconds...
voltage = 10

freq = 5*1e6
tau = ((1/float(freq)) * 1e12) #Tau in picoseconds
n_tbins = int(tau // shift)

print(f'Number of effective bins: {n_tbins}')
print(f'Shift: {shift}')

save_into_file = True

save_path = '/home/ubi-user/David_P_folder'
save_name = f'coarsek{num_gates}_{freq*1e6}mhz_{voltage}v_correlations'

gate_width = math.ceil((((1/float(freq))*1e12) // num_gates) * 1e-3 )
gate_starts = np.array([(gate_width * (gate_step)) for gate_step in range(num_gates)]) * 1e3

(rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, 1. / float(freq))

print(f'Time bin depth resolution {tbin_depth_res * 1000:.3f} mm')



plot_gates_animation = True

gates = {i: [] for i in range(num_gates)}
for i in range(num_gates):
    gate_start = gate_starts[i]

    for j in range(n_tbins):
        gate_start_tmp = gate_start + j * shift
        gate_start_tmp = gate_start_tmp % tau
        if (gate_start_tmp + (gate_width * 1e3)) > tau:
            gate_start_one = gate_start_tmp
            gate_start_two = 0
            gate_one_width = tau - gate_start_tmp
            gate_two_width = (gate_width * (1e3)) - gate_one_width

            gate_start_one_bin = int((gate_start_one * 1e-12) / tbin_res)
            gate_end_one_bin = int(((gate_start_one + (gate_one_width)) * 1e-12) / tbin_res)

            gate_start_two_bin = int((gate_start_two * 1e-12) / tbin_res)
            gate_end_two_bin = int(((gate_start_two + (gate_two_width)) * 1e-12) / tbin_res)

            gate_one = np.zeros((n_tbins))
            gate_two = np.zeros((n_tbins))

            gate_one[gate_start_one_bin:gate_end_one_bin] = 1
            gate_two[gate_start_two_bin:gate_end_two_bin] = 1

            gates[i].append([gate_one, gate_two])
            print('split gate in half')
        else:
            gate = np.zeros((n_tbins))
            gate_start_bin = int((gate_start_tmp * 1e-12) / tbin_res)
            gate_end_bin = int(((gate_start_tmp + (gate_width *(1e3))) * 1e-12) / tbin_res)
            gate[gate_start_bin:gate_end_bin] = 1
            gates[i].append(gate)


#gates = np.stack(gates, axis=-1)

#y = np.asarray(gates)

# fig, ax = plt.subplots()
# colors = ['blue', 'green','red', 'orange']
#
# lines = []
# seqs = []
#
# for k, gate_tmp in enumerate(gates.values()):
#     x = np.arange(n_tbins)
#
#     if type(gate_tmp[0]) == list:
#         (line,) = ax.plot(x, gate_tmp[0][0] + gate_tmp[0][1], animated=True,
#                           color=colors[k], linewidth=2)
#     else:
#         (line,) = ax.plot(x, gate_tmp[0], animated=True,
#                           color=colors[k], linewidth=2)
#
#     lines.append(line)          # <-- keep the artist
#     seqs.append(gate_tmp)       # <-- keep the data sequence
#
# def update(f):
#     for line, gate_tmp in zip(lines, seqs):
#         if type(gate_tmp[f]) == list:
#             line.set_ydata(gate_tmp[f][0] + gate_tmp[f][1])
#         else:
#             line.set_ydata(gate_tmp[f])
#     return tuple(lines)
#
# ani = FuncAnimation(fig, update, frames=len(seqs[0]), interval=5, blit=True)
# plt.show()