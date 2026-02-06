import numpy as np
from felipe_utils import CodingFunctionsFelipe
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from utils.tof_utils import calculate_tof_domain_params
from felipe_utils.tof_utils_felipe import zero_norm_t
import matplotlib.pyplot as plt

k = 3
n_tbins = 999
trials = 100
photon_count = 1000
sbr = 0.1
rep_rate = 5 * 1e6
rep_tau = float(1/rep_rate)

rng = np.random.default_rng()

(rep_tau, rep_freq, tbin_res,
 t_domain, max_depth, tbin_depth_res,) = calculate_tof_domain_params(n_tbins, rep_tau)

depth_sample = 0.5
depths = np.arange(3.0, max_depth-3.0, depth_sample)


func = getattr(CodingFunctionsFelipe, f"GetHamK{k}")
(modfs, demodfs) = func(N=n_tbins)
irf = gaussian_pulse(np.arange(n_tbins), 0, 1, circ_shifted=True)
modfs = np.fft.ifft(np.fft.fft(irf[..., np.newaxis], axis=0).conj() * np.fft.fft(modfs, axis=0), axis=0).real
ham_cm = np.fft.ifft(np.fft.fft(modfs, axis=0).conj() * np.fft.fft(demodfs, axis=0), axis=0).real

total_modfs = np.zeros((depths.shape[0], n_tbins, k))
ham_cv = np.zeros((depths.shape[0], k))
for i in range(depths.shape[0]):
    for j in range(k):
        tmp = (modfs[:, j] / np.sum(modfs[:, j])) * photon_count
        clean_modf = tmp + ((photon_count / sbr) / n_tbins)
        total_modfs[i, :, j] = np.roll(clean_modf, int(depths[i] / tbin_depth_res))
        ham_cv[i, j] = np.inner(total_modfs[i, :, j], demodfs[:, j])

ham_noisy_cv = rng.poisson(ham_cv, size=(trials, ham_cv.shape[0], ham_cv.shape[1]))

ham_norm_cm= zero_norm_t(ham_cm, axis=-1)

ham_norm_cv = zero_norm_t(ham_noisy_cv, axis=-1)

ham_zncc = np.matmul(ham_norm_cm, ham_norm_cv[..., np.newaxis]).squeeze(-1)

ham_decoded_depth = np.argmax(ham_zncc, axis=-1) * tbin_depth_res

ham_rmse = float(np.sqrt(np.mean((ham_decoded_depth - depths) ** 2)))
ham_mae = float(np.mean(np.abs(ham_decoded_depth - depths)))

coarse_cm = np.kron(np.eye(k), np.ones((1, n_tbins // k)))
illum = gaussian_pulse(np.arange(coarse_cm.shape[-1]), 0, n_tbins // (k + 3), circ_shifted=True)
coarse_cm = np.fft.ifft(
    np.fft.fft(illum[..., np.newaxis], axis=0).conj() * np.fft.fft(np.transpose(coarse_cm), axis=0),
    axis=0).real

total_illums = np.zeros((depths.shape[0], n_tbins, k))
coarse_cv = np.zeros((depths.shape[0], k))
for i in range(depths.shape[0]):
    for j in range(k):
        tmp = (illum / np.sum(illum)) * photon_count
        clean_illum = tmp + ((photon_count / sbr) / n_tbins)
        total_illums[i, :, j] = np.roll(clean_illum, int(depths[i] / tbin_depth_res))
        coarse_cv[i, j] = np.inner(total_illums[i, :, j], coarse_cm[:, j])

coarse_noisy_cv = rng.poisson(coarse_cv, size=(trials, coarse_cv.shape[0], coarse_cv.shape[1]))

coarse_norm_cm= zero_norm_t(coarse_cm, axis=-1)

coarse_norm_cv = zero_norm_t(coarse_noisy_cv, axis=-1)

coarse_zncc = np.matmul(coarse_norm_cm, coarse_norm_cv[..., np.newaxis]).squeeze(-1)

coarse_decoded_depth = np.argmax(coarse_zncc, axis=-1) * tbin_depth_res

coarse_rmse = float(np.sqrt(np.mean((coarse_decoded_depth - depths) ** 2)))
coarse_mae = float(np.mean(np.abs(coarse_decoded_depth - depths)))

print(f"HAM    → MAE={ham_mae*1000:.3f} | RMSE={ham_rmse*1000:.3f}")
print(f"COARSE → MAE={coarse_mae*1000:.3f} | RMSE={coarse_rmse*1000:.3f}")
