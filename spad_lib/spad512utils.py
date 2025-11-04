import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.cluster import KMeans
from felipe_utils import CodingFunctionsFelipe
from felipe_utils.tof_utils_felipe import zero_norm_t

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
        assert irf.shape[0] == n_tbins
        irf = irf.squeeze()
        irf = irf[..., np.newaxis]
        coding_matrix = np.fft.ifft(np.fft.fft(irf, axis=0).conj() * np.fft.fft(coding_matrix, axis=0), axis=0).real
    return coding_matrix

def get_hamiltonain_correlations(K, mhz, voltage, size, illum_type, n_tbins=2000):
    func = getattr(CodingFunctionsFelipe, f"GetHamK{K}")
    (modfs, demodfs) = func(N=n_tbins, modDuty=1/5)
    modfs = get_voltage_function(mhz, voltage, size, illum_type, n_tbins, K, hamiltonian=True)
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


def get_voltage_function(mhz, voltage, size, illum_type, n_tbins=None ,K=3, hamiltonian=False):
    try:
        function = np.genfromtxt(f'/home/ubi-user/David_P_folder/py-gated-camera/voltage_functions/{illum_type}_{mhz}mhz_{voltage}v_{size}w.csv',delimiter=',')[:, 1]
    except FileNotFoundError:
        function = np.genfromtxt(f'/Users/davidparra/PycharmProjects/py-gated-camera/voltage_functions/{illum_type}_{mhz}mhz_{voltage}v_{size}w.csv',delimiter=',')[:, 1]

    modfs = function[2:]
    if illum_type == 'pulse': 
        if size == 12:
            modfs[150:600] = 0
            if hamiltonian:
                modfs = np.roll(modfs, -76, axis=0)
        elif size == 34 and mhz == 10:
            modfs[modfs < 0] = 0
            modfs = np.roll(modfs, 38, axis=0)
            modfs = gaussian_filter(modfs, sigma=10)
        elif size == 25 and mhz == 10:
            modfs[modfs < 0] = 0
            modfs = np.roll(modfs, 18, axis=0)
            modfs = gaussian_filter(modfs, sigma=10)
        elif size == 67 and mhz == 5:
            modfs[modfs < 0] = 0
            modfs = np.roll(modfs, -30, axis=0)
            modfs = gaussian_filter(modfs, sigma=10)
        elif size == 50 and mhz == 5:
            modfs[modfs < 0] = 0
            #modfs = np.roll(modfs, -21, axis=0)
            modfs = np.roll(modfs, -21, axis=0)
            modfs = gaussian_filter(modfs, sigma=10)
    elif illum_type == 'square':
        if size == 20:
            modfs[180:600] = 0
        else:
            modfs[modfs < 0] = 0
        if mhz == 10:
            if K == 3:
                modfs = np.roll(modfs, 9, axis=0)
            elif K == 4:
                modfs = np.roll(modfs, 13, axis=0)
        elif mhz == 5:
            if K == 3:
                modfs = np.roll(modfs, -33, axis=0)
            elif K == 4:
                modfs = np.roll(modfs, -29, axis=0)
                pass
            elif K == 5:
                modfs = np.roll(modfs, -55, axis=0)

        modfs = gaussian_filter(modfs, sigma=10)

    if n_tbins is not None:
        f = interp1d(np.linspace(0, 1, len(modfs)), modfs, kind='cubic')
        modfs = f(np.linspace(0, 1, n_tbins))
        #print(modfs)

    modfs /= np.sum(modfs, keepdims=True)

    return modfs

def cluster_kmeans(decoded, n_clusters=2):
    siz = decoded.shape
    decoded_depths = np.copy(decoded)
    decoded_depths.flatten()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto", max_iter=100_000, tol=1e-20,
                    algorithm='elkan').fit(decoded_depths.reshape(-1, 1))
    all_means = kmeans.cluster_centers_

    nice_nice = kmeans.predict(decoded_depths.reshape(-1, 1)).astype(float)
    for i in range(n_clusters):
        nice_nice[nice_nice == i] = all_means[i]
    nice_depth_map = np.reshape(nice_nice, siz)
    return nice_depth_map



def build_coding_matrix_from_correlations(
    correlations_total: np.ndarray,
    im_width: int,
    n_tbins: int,
    freq: float,
    use_full: bool,
    sigma_size: int,
    shift_size: int,
) -> np.ndarray:
    if use_full:
        # correlations_total: (H, W, n_tbins, K)
        corr_tmp = gaussian_filter(
            correlations_total.swapaxes(-1, -2),  # -> (H,W,K,n_tbins)
            sigma=(0.5, 0.5, 0.5, 0),
        )
        # roll along time
        corr_tmp = np.roll(corr_tmp, shift=shift_size, axis=-2)
        # smooth along time again
        coding_matrix = gaussian_filter1d(corr_tmp, sigma=sigma_size, axis=-2)
        return coding_matrix  # (H,W,n_tbins,K)

    # legacy path: spatial-sum correlations
    coding_matrix = np.transpose(
        np.sum(np.sum(correlations_total[: im_width // 2], axis=0), axis=0)
    )  # (n_tbins,K)
    coding_matrix = np.roll(coding_matrix, shift=shift_size, axis=0)
    coding_matrix = gaussian_filter1d(coding_matrix, sigma=sigma_size, axis=0)
    return coding_matrix


def decode_depth_map(
    coded_vals: np.ndarray,
    coding_matrix: np.ndarray,
    im_width: int,
    n_tbins: int,
    tbin_depth_res: float,
    use_correlations: bool,
    use_full_correlations: bool,
):
    # normalize per-pixel coded vals → (H,W,K)
    norm_coded_vals = zero_norm_t(coded_vals, axis=-1)

    if use_correlations and use_full_correlations:
        # coding_matrix: (H,W,n_tbins,K)
        norm_coding_matrix = zero_norm_t(coding_matrix, axis=-1)
        # (H,W,n_tbins,K) @ (H,W,K,1) -> (H,W,n_tbins)
        zncc = np.matmul(norm_coding_matrix, norm_coded_vals[..., None]).squeeze(-1)
    else:
        # coding_matrix: (n_tbins,K) or (333,K)
        norm_coding_matrix = zero_norm_t(coding_matrix, axis=-1)
        # (H,W,K) @ (K,n_tbins) -> (H,W,n_tbins)
        zncc = np.matmul(norm_coding_matrix, norm_coded_vals[..., np.newaxis]).squeeze(-1)

    depths = np.argmax(zncc, axis=-1)
    depth_map = depths.reshape((im_width, im_width)) * tbin_depth_res
    return depth_map, zncc


def intrinsics_from_fov(W, H, fov_x_deg, fov_y_deg):
    fx = (W/2) / np.tan(np.deg2rad(fov_x_deg/2))
    fy = (H/2) / np.tan(np.deg2rad(fov_y_deg/2))
    cx, cy = W/2, H/2
    return fx, fy, cx, cy

def intrinsics_from_sensor(W, H, f_mm, sensor_w_mm, sensor_h_mm):
    fx = f_mm * W / sensor_w_mm
    fy = f_mm * H / sensor_h_mm
    cx, cy = W/2, H/2
    return fx, fy, cx, cy

def intrinsics_from_pixel_pitch(W, H, f_mm, pitch_um):
    p_mm = pitch_um * 1e-3
    fx = fy = f_mm / p_mm
    cx, cy = W/2, H/2
    return fx, fy, cx, cy

def range_to_z(R, fx, fy, cx, cy):
    H, W = R.shape
    u = np.arange(W)[None, :]
    v = np.arange(H)[:, None]
    xu = (u - cx) / fx
    yv = (v - cy) / fy
    cos_theta = 1.0 / np.sqrt(1.0 + xu**2 + yv**2)
    return R * cos_theta