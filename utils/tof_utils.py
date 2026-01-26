import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from felipe_utils.tof_utils_felipe import zero_norm_t
from utils.global_constants import EPILSON, SPEED_OF_LIGHT

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



def build_coding_matrix_from_correlations(
    correlations_total: np.ndarray,
    use_full,
    sigma_size,
    shift_size,
    n_tbins=None,
) -> np.ndarray:
    if use_full:
        # correlations_total: (H, W, n_tbins, K)
        coding_matrix = gaussian_filter(
            correlations_total.swapaxes(-1, -2),  # -> (H,W,K,n_tbins)
            sigma=(1, 1, 1, 0),
        )
        # roll along time
        if shift_size is not None:
            coding_matrix = np.roll(coding_matrix, shift=shift_size, axis=-2)
        # smooth along time again
        if sigma_size is not None:
            coding_matrix = gaussian_filter1d(coding_matrix, sigma=sigma_size, axis=-2)
        return coding_matrix  # (H,W,n_tbins,K)

    # legacy path: spatial-sum correlations
    coding_matrix = np.transpose(
        np.sum(np.sum(correlations_total[:, :correlations_total.shape[1]//2, :], axis=0), axis=0)
    )  # (n_tbins,K)
    if shift_size is not None:
        coding_matrix = np.roll(coding_matrix, shift=shift_size, axis=0)
    if sigma_size is not None:
        coding_matrix = gaussian_filter1d(coding_matrix, sigma=sigma_size, axis=0)
    if n_tbins is not None:
        original_len = coding_matrix.shape[0]
        f = interp1d(
            np.linspace(0, 1, original_len),
            coding_matrix,
            kind='cubic',
            axis=0,
            fill_value='extrapolate'
        )
        coding_matrix = f(np.linspace(0, 1, n_tbins))
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
