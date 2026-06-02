import numpy as np
import matplotlib.pyplot as plt

from plot_scripts.plot_utils import plot_results_summary, plot_coding_curve, plot_correlations_one_plot, \
    plot_coding_error
from utils.tof_utils import (
    get_code,
    simulate_counts,
    simulate_counts_shared_illum,
    decode_simulation_depths,
    calculate_tof_domain_params, scale_photon_count,
)

# =============================================================================
# Global parameters
# =============================================================================
N_TBINS      = 1500
TRIALS       = 100
PHOTON_COUNT = 2000
SBR          = 1.0
SPLIT_ACQUISITION = True
REP_RATE = 5e6
REP_TAU  = float(1 / REP_RATE)

DEPTH_SAMPLE = 0.01

# =============================================================================
# Coding schemes to evaluate
# Add / remove / reorder entries freely — everything else is automatic.
# Each entry:
#   type          : 'ham' | 'coarse' | 'rect' | 'trapcoarse' | 'traprect'
#   k             : number of codes
#   photon_count  : photons per measurement (tune per scheme as needed)
#   simulated     : passed through to results dict (used by some plot helpers)
# =============================================================================
RUNS = [
    {'type': 'ham',      'k': 4, 'photon_count': PHOTON_COUNT, 'simulated': True},
    {'type': 'coarse',   'k': 4, 'photon_count': PHOTON_COUNT, 'simulated': True},
    # coarsepw — one entry per pulse width you want to test
    {'type': 'coarsepw', 'k': 12, 'photon_count': PHOTON_COUNT, 'pulse_width': (N_TBINS // (8)) / (2 * np.sqrt(np.log(2))) * 0.86,  'simulated': True},
    {'type': 'coarsepw', 'k': 12, 'photon_count': PHOTON_COUNT, 'pulse_width': (N_TBINS // (12)) / (2 * np.sqrt(np.log(2))) * 0.86,  'simulated': True},
    # {'type': 'coarsepw', 'k': 4, 'photon_count': PHOTON_COUNT // 4, 'pulse_width': N_TBINS // 4,  'simulated': True},
    # {'type': 'coarsepw', 'k': 4, 'photon_count': PHOTON_COUNT // 4, 'pulse_width': N_TBINS // 2,  'simulated': True},
    # {'type': 'trapcoarse','k': 8, 'photon_count': PHOTON_COUNT // 8, 'simulated': True},
]

# =============================================================================
# ToF domain
# =============================================================================
(rep_tau, rep_freq, tbin_res,
 t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(N_TBINS, REP_TAU)

depths = np.arange(3.0, max_depth - 3.0, DEPTH_SAMPLE)


def print_example_counts(name, depths, coded_values):
    print(f"{name}")
    print(f"  depth:        {depths[0]}")
    print(f"  counts:       {coded_values[0]}")
    print(f"  total counts: {np.sum(coded_values[0])}")
    print()


# =============================================================================
# Run each scheme
# =============================================================================
results = []

for run in RUNS:
    cap_type     = run['type']
    k            = run['k']
    photon_count_base = run['photon_count']
    photon_count = scale_photon_count(photon_count_base, cap_type, k) if SPLIT_ACQUISITION else photon_count_base
    pulse_width  = run.get('pulse_width', None)
    label        = f"{cap_type}_k{k}" + (f"_pw{pulse_width}" if pulse_width is not None else "")

    print(f"--- {label} ---")

    # get_code returns (modfs_or_illum, demodfs_or_coding_matrix_T, coding_matrix)
    waveform_or_illum, demodfs_or_cm, coding_matrix = get_code(cap_type, k, N_TBINS, pulse_width=pulse_width)

    if cap_type == 'ham':
        _, coded_values = simulate_counts(
            waveform=waveform_or_illum,
            demodfs=demodfs_or_cm,
            depths=depths,
            photon_count=photon_count,
            sbr=SBR,
            tbin_depth_res=tbin_depth_res,
            n_tbins=N_TBINS,
            k=k,
        )
    else:
        _, coded_values = simulate_counts_shared_illum(
            illum=waveform_or_illum,
            coding_matrix=demodfs_or_cm,
            depths=depths,
            photon_count=photon_count,
            sbr=SBR,
            tbin_depth_res=tbin_depth_res,
            n_tbins=N_TBINS,
            k=k,
        )

    print_example_counts(label, depths, coded_values)

    decoded_depth, rmse, mae = decode_simulation_depths(
        coding_matrix=coding_matrix,
        coded_values=coded_values,
        depths=depths,
        trials=TRIALS,
        tbin_depth_res=tbin_depth_res,
    )

    print(f"  MAE={mae * 1000:.3f} mm  |  RMSE={rmse * 1000:.3f} mm\n")

    results.append({
        'name':          label,
        'coding_matrix': coded_values,
        'waveform':      waveform_or_illum if waveform_or_illum.ndim > 1
                         else np.tile(waveform_or_illum[:, None], (1, k)),
        'rmse':          rmse * 1000,
        'mae':           mae  * 1000,
        'depths':        depths,
        'simulated':     run.get('simulated', True),
    })

# =============================================================================
# Plot
# =============================================================================
# plot_correlations_one_plot(results)
# plot_coding_error(results)
plot_results_summary(results)
# plot_coding_curve(results)
