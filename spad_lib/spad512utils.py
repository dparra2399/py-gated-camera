import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
EPILSON = 1e-8
SPEED_OF_LIGHT = 3e8

def get_coarse_coding_matrix(gate_step_size, gate_steps, gate_offset, gate_width, laser_time, n_tbins=2000, irf=None):

    gate_starts = np.array([gate_offset + (gate_step_size * (gate_step)) for gate_step in range(gate_steps)])
    #print(gate_starts)
    #print(gate_width)

    time_resolution = np.linspace(0, laser_time, n_tbins)
    coding_matrix = np.zeros((n_tbins, gate_steps))
    for gate_step in range(gate_steps):
        gate_start = gate_starts[gate_step]
        gate_end = gate_start + gate_width
        
        #Make gate start and end 
        start_idx = np.searchsorted(time_resolution, gate_start, side='left')
        end_idx = np.searchsorted(time_resolution, gate_end, side='right')

        indices = np.arange(start_idx, end_idx)

        gate = np.zeros_like(time_resolution)
        gate[indices] = 1

        coding_matrix[:, gate_step] = gate

    if irf is not None:
        irf = irf.squeeze()
        irf = irf[..., np.newaxis]
        coding_matrix = np.fft.ifft(np.fft.fft(irf, axis=0).conj() * np.fft.fft(coding_matrix, axis=0), axis=0).real
    return coding_matrix

def get_hamiltonain_correlations(demodfs, mhz, voltage, interpolate=False):
    modfs = get_voltage_function(mhz, voltage, 'square')
    if interpolate:
        f = interp1d(np.linspace(0, 1, len(modfs)), modfs, kind='cubic')
        modfs= f(np.linspace(0, 1, demodfs.shape[0]))

    assert modfs.shape[0] == demodfs.shape[0], f'modfs shape: {modfs.shape}, demodfs shape: {demodfs.shape}'

    if modfs.ndim == 1: modfs = modfs[:, np.newaxis]
    if modfs.shape[1] != demodfs.shape[1]:
        modfs = np.tile(modfs, (1, demodfs.shape[1]))

    correlations = np.fft.ifft(np.fft.fft(modfs, axis=0).conj() * np.fft.fft(demodfs, axis=0), axis=0).real
    return correlations

def norm_t(C, axis=-1):
    norm = np.linalg.norm(C, ord=2, axis=axis, keepdims=True)
    return C / (norm + EPILSON)

def zero_norm_t(C, axis=-1):
    mean = np.mean(C, axis=axis, keepdims=True)
    return norm_t(C - mean, axis=axis)

def time2depth(time):
    return (SPEED_OF_LIGHT * time) / 2

def depth2time(depth):
    return (2 * depth) / SPEED_OF_LIGHT

def get_time_domain(rep_tau, n_tbins):
    tbins_res = rep_tau / n_tbins
    time_domain = np.arange(0, n_tbins) * tbins_res
    tbin_bounds = (np.arange(0, n_tbins + 1) * tbins_res) - 0.5 * tbins_res
    return (time_domain, tbins_res, tbin_bounds)


def calculate_tof_domain_params(n_tbins, rep_tau=None, max_depth=None):

    if (not (rep_tau is None)):
        max_depth = time2depth(rep_tau)
    elif (not (max_depth is None)):
        rep_tau = depth2time(max_depth)

    else:
        rep_tau = 1

    (t_domain, tbin_res, tbin_bounds) = get_time_domain(rep_tau, n_tbins)

    rep_freq = 1. / rep_tau
    tbin_depth_res = time2depth(tbin_res)
    return (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res)

def get_offset_width_spad512(gate, freq):
    (t_domain, tbin_res, tbin_bounds) = get_time_domain(1./freq, gate.shape[0])
    gate_width = np.sum(gate) * tbin_res
    gate_offset = np.argmax(gate == 1) * tbin_res
    return int(gate_width * 1e9), int(gate_offset * 1e12)

def decompose_ham_codes(demodfs):
    gates = []
    gates_two = []
    for i in range(demodfs.shape[-1]):
        gate_tmp = gated_ham(demodfs[:, i])
        for j in range(gate_tmp.shape[-1]):
            gates.append(gate_tmp[:, j])
        gates_two.append(gate_tmp)
    return np.stack(gates, axis=-1), gates_two
        
def gated_ham(demod):
    n_tbins = demod.shape[0]
    gates = np.zeros((n_tbins, 1))
    squares = split_into_indices(demod)
    for j in range(len(squares)):
        pair = squares[j]
        single_square = np.zeros((n_tbins, 1))
        single_square[pair[0]:pair[1] + 1, :] = 1
        gates = np.concatenate((gates, single_square), axis=-1)

    gates = gates[:, 1:]
    return gates


def split_into_indices(square_array):
    indices = []
    start_index = None
    for i, num in enumerate(square_array):
        if num == 1:
            if start_index is None:
                start_index = i
        elif num == 0 and start_index is not None:
            indices.append((start_index, i - 1))
            start_index = None
    if start_index is not None:
        indices.append((start_index, len(square_array) - 1))
    return indices


def get_voltage_function(mhz, voltage, illum_type):
    function = np.genfromtxt(f'/home/ubilaptop8/2025-Q2-David-P-captures/gated_project_code/voltage_functions/{illum_type}_{mhz}mhz_{voltage}v.csv',delimiter=',')[:, 1]
    modfs = function[2:] + 100
    modfs /= np.sum(modfs, keepdims=True)
    return modfs