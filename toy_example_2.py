# Python imports
# Library imports
import os
import time

import numpy as np
from IPython.core import debugger

from felipe_utils import tof_utils_felipe
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from felipe_utils.CodingFunctionsFelipe import *
from spad_lib.global_constants import *
from felipe_utils import CodingFunctionsFelipe
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def get_voltage_function(mhz, voltage, size, illum_type, n_tbins=None):
    try:
        function = np.genfromtxt(os.path.join(READ_PATH_VOLTAGE_FUNCTIONS_WINDOWS,
                                              f'{illum_type}_{mhz}mhz_{voltage}v_{size}w.csv'),delimiter=',')[:, 1]
    except FileNotFoundError:
        function = np.genfromtxt(os.path.join(READ_PATH_VOLTAGE_FUNCTIONS_MAC,
                                              f'{illum_type}_{mhz}mhz_{voltage}v_{size}w.csv'),delimiter=',')[:, 1]
    modfs = function[2:]
    if illum_type == 'pulse':
        if size == 12:
            modfs[150:600] = 0
        else:
            modfs[modfs < 0] = 0
    elif illum_type == 'square':
        if size == 20:
            modfs[180:600] = 0
        else:
            modfs[modfs < 0] = 0

    if n_tbins is not None:
        f = interp1d(np.linspace(0, 1, len(modfs)), modfs, kind='cubic')
        modfs = f(np.linspace(0, 1, n_tbins))

    modfs /= np.sum(modfs, keepdims=True)
    return modfs

#matplotlib.use('QTkAgg')
breakpoint = debugger.set_trace

photon_count = 3000
sbr = 1.0
trials = 100
n_tbins = 1998

simulated_correlations = False
apd_correlations = False

smooth_correlations = True
smooth_sigma = 5

shift_correlations = True
shift = 110

depths = np.arange(200, n_tbins-200, 1)

#coarse_filename = 'coarsek4_10mhz_7.6v_25w_correlations.npz'
coarse_filename = 'coarsek3_5mhz_5.7v_67w_correlations.npz'
#coarse_filename = 'coarsek3_10mhz_7v_34w_correlations.npz'
#coarse_filename = 'coarsek8_10mhz_10v_12w_correlations.npz'
#coarse_filename = 'coarsek3_9mhz_10v_12w_correlations_extended.npz'

#ham_filename = 'hamk3_10mhz_8.5v_20w_correlations.npz'
ham_filename = 'hamk3_5mhz_6.5v_20w_correlations.npz'
#ham_filename = 'hamk4_10mhz_8.5v_20w_correlations.npz'
#ham_filename = 'hamk3_9mhz_10v_20w_correlations_extended.npz'



filenames = [coarse_filename, ham_filename]
rng = np.random.default_rng()
names = []
fig, axs = plt.subplots(len(filenames), 4, figsize=(13, 8), squeeze=False)
for i, filename in enumerate(filenames):
    try:
        path = os.path.join(READ_PATH_CORRELATIONS_WINDOWS, filename)
        file = np.load(path)
    except FileNotFoundError:
        path = os.path.join(READ_PATH_CORRELATIONS_MAC, filename)
        file = np.load(path)

    correlations = file['correlations']
    total_time = file["total_time"]
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
    mhz = int(freq * 1e-6)

    (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = tof_utils_felipe.calc_tof_domain_params(n_tbins, 1 / freq)

    #if 'ham' in filename:
    #    coding_matrix += np.flip(coding_matrix, axis=0)

    coding_matrix = np.roll(np.transpose(
        np.sum(np.sum(correlations[:, :correlations.shape[1] // 2, :], axis=0), axis=0)
    ), 0)

    if 'ham' in filename and simulated_correlations:
        func = getattr(CodingFunctionsFelipe, f"GetHamK{K}")
        (modfs, demodfs) = func(N=n_tbins)
        irf = gaussian_pulse(np.arange(n_tbins), 0, 40, circ_shifted=True)
        modfs = np.fft.ifft(np.fft.fft(irf[..., np.newaxis], axis=0).conj() * np.fft.fft(modfs, axis=0), axis=0).real
        coding_matrix = np.fft.ifft( np.fft.fft( modfs, axis=0 ).conj() * np.fft.fft( demodfs, axis=0 ), axis=0 ).real
        coding_matrix =  np.fft.ifft( np.fft.fft(irf[..., np.newaxis], axis=0 ).conj() * np.fft.fft( coding_matrix, axis=0 ), axis=0 ).real
    elif 'coarse' in filename and simulated_correlations:
        coding_matrix = np.kron(np.eye(K), np.ones((1, n_tbins // K)))
        irf = gaussian_pulse(np.arange(coding_matrix.shape[-1]), 0, n_tbins // (K+2), circ_shifted=True)
        coding_matrix = np.fft.ifft(np.fft.fft(irf[..., np.newaxis], axis=0).conj() * np.fft.fft(np.transpose(coding_matrix), axis=0),
                                    axis=0).real
    elif 'coarse' in filename and apd_correlations:
        coding_matrix = np.kron(np.eye(K), np.ones((1, n_tbins // K)))
        irf = get_voltage_function(mhz, voltage, size, 'pulse', n_tbins=n_tbins).squeeze()
        coding_matrix = np.fft.ifft(np.fft.fft(irf[..., np.newaxis], axis=0).conj() * np.fft.fft(np.transpose(coding_matrix), axis=0),
                                    axis=0).real

    elif 'ham' in filename and apd_correlations:
        func = getattr(CodingFunctionsFelipe, f"GetHamK{K}")
        (modfs2, demodfs) = func(N=n_tbins)
        modfs2 = np.tile(get_voltage_function(mhz, voltage, size, 'square', n_tbins=n_tbins)[..., np.newaxis],
                         modfs2.shape[-1])
        coding_matrix = np.fft.ifft(np.fft.fft(modfs2, axis=0).conj() * np.fft.fft(demodfs, axis=0),
                                    axis=0).real

    original_len = coding_matrix.shape[0]
    f = interp1d(
        np.linspace(0, 1, original_len),
        coding_matrix,
        kind='cubic',
        axis=0,
        fill_value='extrapolate'
    )
    coding_matrix = f(np.linspace(0, 1, n_tbins))


    if smooth_correlations:
        coding_matrix = gaussian_filter1d(coding_matrix, sigma=smooth_sigma, axis=0)

    if shift_correlations:
        coding_matrix = np.roll(coding_matrix, shift=shift, axis=0)

    coding_matrix /= np.max(np.abs(coding_matrix), axis=0, keepdims=True)

    if 'ham' in filename:
       coding_matrix[:, 1] = np.roll(coding_matrix[:, 1], 20)
       coding_matrix[:, 0] = np.roll(coding_matrix[:, 0], 20)


    clean_coded_vals = coding_matrix[depths, :]

    clean_coded_vals = (clean_coded_vals / np.sum(clean_coded_vals, axis=-1, keepdims=True)) * photon_count
    clean_coded_vals = clean_coded_vals + ((photon_count / sbr) / K)

    coded_vals = rng.poisson(clean_coded_vals, size=(trials, clean_coded_vals.shape[0], clean_coded_vals.shape[1]))
    #coded_vals = clean_coded_vals[np.newaxis, ...]

    norm_coding_matrix = tof_utils_felipe.zero_norm_t(coding_matrix)

    norm_coded_vals = tof_utils_felipe.zero_norm_t(coded_vals)

    zncc = np.matmul(norm_coding_matrix, norm_coded_vals[..., np.newaxis]).squeeze(-1)

    decoded_depth = np.argmax(zncc, axis=-1)

    if 'coarse' in filename:
        name = 'coarse'
    elif 'ham' in filename:
        name = 'ham'
    else:
        name = ''

    axs[i, 0].imshow(np.repeat(np.transpose(coding_matrix), 100, axis=0), aspect='auto')
    axs[i, 0].set_title('Coding Matrix')
    axs[i, 1].plot(coding_matrix - np.mean(coding_matrix, axis=0, keepdims=True))
    axs[i, 1].set_title('Coding Matrix')
    axs[i, 2].plot(zncc[0, 0, :])
    axs[i, 2].axvline(x=depths[0], c='r', linestyle='--')
    #axs[i, 3].axvline(x=decoded_depth[0, 0], c='g', linestyle='--')
    axs[i, 2].set_title('ZNCC Reconstruction')
    if i < len(filenames) - 2:
        axs[i, 3].set_axis_off()

    axs[-2, 3].bar([i],np.sqrt(np.mean((decoded_depth - depths) ** 2)), label=name)
    axs[-2, 3].text(
        i,  # x position
        np.sqrt(np.mean((decoded_depth - depths) ** 2)),  # y position (height of bar)
        f"{np.sqrt(np.mean((decoded_depth - depths) ** 2)):.2f}",  # displayed text, formatted
        ha='center', va='bottom'
    )
    axs[-2, 3].set_title('RMSE (Lower is better)')
    axs[-2, 3].set_ylabel('RMSE (time bins)')

    axs[-1, 3].bar([i], np.mean(np.abs(decoded_depth - depths)), label=name)
    axs[-1, 3].text(
        i,  # x position
        np.mean(np.abs(decoded_depth - depths)),  # y position (height of bar)
        f"{np.mean(np.abs(decoded_depth - depths)):.2f}",  # displayed text, formatted
        ha='center', va='bottom'
    )
    axs[-1, 3].set_title('MAE (Lower is better)')
    axs[-1, 3].set_ylabel('MAE (time bins)')

    names.append(name)




    print(f'{name}: \n\t MAE: {np.mean(np.abs(decoded_depth - depths)): .3f} \n\t RMSE: {np.sqrt(np.mean((decoded_depth - depths) ** 2)): .3f}')

axs[-1, 3].set_xticks(np.arange(0, len(names)))
axs[-1, 3].set_xticklabels(names)
plt.subplots_adjust(wspace=0.5, hspace=0.4)
plt.show()


