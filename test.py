import numpy as np
import matplotlib.pyplot as plt

from plot_scripts.plot_utils import plot_results_summary, plot_coding_curve, plot_correlations_one_plot, \
    plot_coding_error
from utils.tof_utils import get_ham_code, get_coarse_code, simulate_counts, simulate_counts_shared_illum, decode_simulation_depths
from utils.tof_utils import calculate_tof_domain_params


# =========================
# Parameters
# =========================
K = 3
N_TBINS = 999
TRIALS = 100
PHOTON_COUNT = 100
SBR = 10.0

REP_RATE = 5e6
REP_TAU = float(1 / REP_RATE)

DEPTH_SAMPLE = 0.01

rng = np.random.default_rng()


# =========================
# ToF domain
# =========================
(
    rep_tau,
    rep_freq,
    tbin_res,
    t_domain,
    max_depth,
    tbin_depth_res,
) = calculate_tof_domain_params(N_TBINS, REP_TAU)

depths = np.arange(3.0, max_depth - 3.0, DEPTH_SAMPLE)





def print_example_counts(name, depths, coded_values):
    print(f"{name}")
    print(f"depth: {depths[0]}")
    print(f"counts: {coded_values[0]}")
    print(f"total counts: {np.sum(coded_values[0])}")
    print()



# =========================
# HAM
# =========================
ham_modfs, ham_demodfs, ham_cm = get_ham_code(K, N_TBINS)

_, ham_cv = simulate_counts(
    waveform=ham_modfs,
    demodfs=ham_demodfs,
    depths=depths,
    photon_count=PHOTON_COUNT,
    sbr=SBR,
    tbin_depth_res=tbin_depth_res,
    n_tbins=N_TBINS,
    k=K,
)

print_example_counts("HAM", depths, ham_cv)

ham_decoded_depth, ham_rmse, ham_mae = decode_simulation_depths(
    coding_matrix=ham_cm,
    coded_values=ham_cv,
    depths=depths,
    trials=TRIALS,
    tbin_depth_res=tbin_depth_res,
)


# =========================
# COARSE
# =========================
illum, coarse_demodfs, coarse_cm = get_coarse_code(K, N_TBINS)

_, coarse_cv = simulate_counts_shared_illum(
    illum=illum,
    coding_matrix=coarse_demodfs,
    depths=depths,
    photon_count=PHOTON_COUNT,
    sbr=SBR,
    tbin_depth_res=tbin_depth_res,
    n_tbins=N_TBINS,
    k=K,
)


print_example_counts("COARSE", depths, coarse_cv)

coarse_decoded_depth, coarse_rmse, coarse_mae = decode_simulation_depths(
    coding_matrix=coarse_cm,
    coded_values=coarse_cv,
    depths=depths,
    trials=TRIALS,
    tbin_depth_res=tbin_depth_res,
)

print(ham_cv.shape)
print(np.mean(ham_cv / coarse_cv, axis=0))

# =========================
# Summary
# =========================
print(f"HAM    → MAE={ham_mae * 1000:.3f} | RMSE={ham_rmse * 1000:.3f}")
print(f"COARSE → MAE={coarse_mae * 1000:.3f} | RMSE={coarse_rmse * 1000:.3f}")


results = [
    {
        "name": "ham",
        "coding_matrix": ham_cv,
        "waveform": ham_modfs,
        "rmse": ham_rmse * 1000,
        "mae": ham_mae * 1000,
        "depths": depths
    },
    {
        "name": "coarse",
        "coding_matrix": coarse_cv,
        "waveform": np.tile(illum[:, None], (1, K)),
        "rmse": coarse_rmse * 1000,
        "mae": coarse_mae * 1000,
        "depths": depths

    },
]
plot_correlations_one_plot(results)
plot_coding_error(results)
#plot_results_summary(results)
#plot_coding_curve(results)
