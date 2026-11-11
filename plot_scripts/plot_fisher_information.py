
import numpy as np
import matplotlib.pyplot as plt

from plot_scripts.plot_utils import get_string_name
from utils.tof_utils import get_code, calculate_tof_domain_params, scale_photon_count

# =========================
# CONFIG
# =========================
RUNS = [
    {'type': 'ham',        'pulse_width': 40},
    {'type': 'coarse'},
    {'type': 'trapcoarse'},
    # {'type': 'coarsepw', 'pulse_width': 40},
]
KS = [3, 4]   # one column per K
N_TBINS = 999

REP_RATE = 5e6
REP_TAU = float(1 / REP_RATE)   # depth range is [0, c * REP_TAU / 2)

PHOTON_BUDGET = 1000       # signal photons
SBRS = np.logspace(-2, 2, 50)   # signal-to-background ratios to sweep
SPLIT_ACQUISITION = True   # if True, PHOTON_BUDGET is shared across the acquisitions of each scheme
SINGLE_GATE_PER_FRAME = False   # if True, ham K>3 is split over 6 frames (one gate opening per frame); if False, over K
UNJENSENED = False          # if True, also report/plot the un-Jensened average CRLB

SPEED_OF_LIGHT = 3e8

# =========================
# COLORS
# =========================
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


# =========================
# Fisher information
# =========================
def expected_counts(modfs, demodfs, photon_count, sbr):
    """mu_i(t_0) for every circular shift t_0. Returns (n_tbins, k).
    Same scaling as simulate_counts / simulate_counts_shared_illum in tof_utils."""
    if modfs.ndim == 1:
        modfs = modfs[..., np.newaxis]   # one illumination shared by every gate
    n_tbins = modfs.shape[0]

    signal = (modfs / np.sum(modfs, axis=0, keepdims=True)) * photon_count
    clean = signal + ((photon_count / sbr) / n_tbins)

    return np.fft.ifft(
        np.fft.fft(clean, axis=0).conj()
        * np.fft.fft(demodfs, axis=0),
        axis=0,
    ).real


def fisher_information(mu, tbin_res):
    """I(t_0) = sum_i mu_i'(t_0)^2 / mu_i(t_0). mu is (n_tbins, k); returns (n_tbins,)."""
    # circular central difference, so the period edges wrap onto the neighboring pulse copy
    dmu = (np.roll(mu, -1, axis=0) - np.roll(mu, 1, axis=0)) / (2 * tbin_res)
    # a measurement with mu_i = 0 carries no information
    per_gate = np.divide(dmu ** 2, mu, out=np.zeros_like(mu), where=mu > 1e-12)
    return np.sum(per_gate, axis=-1)


def crlb_jensen(fisher):
    """(c / 2) / sqrt(mean I) in meters."""
    return (SPEED_OF_LIGHT / 2) / np.sqrt(np.mean(fisher))


def crlb_unjensened(fisher):
    """(c / 2) * sqrt(mean(1 / I)) in meters. inf if I(t_0) = 0 at any depth."""
    with np.errstate(divide='ignore'):
        return (SPEED_OF_LIGHT / 2) * np.sqrt(np.mean(1. / fisher))


def get_label(r):
    return get_string_name(r['type'], None, short=True)


# =========================
# MAIN
# =========================
(_, _, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(N_TBINS, REP_TAU)
to_mm = 1e3

labels = [get_label(r) for r in RUNS]
jensen     = np.zeros((len(KS), len(RUNS), len(SBRS)))
unjensened = np.zeros((len(KS), len(RUNS), len(SBRS)))
for c, k in enumerate(KS):
    for i, r in enumerate(RUNS):
        modfs, demodfs, _ = get_code(r['type'], k, N_TBINS, pulse_width=r.get('pulse_width'),
                                     shift=r.get('shift'), gate_width=r.get('gate_width'))

        photon_count = (scale_photon_count(PHOTON_BUDGET, r['type'], k, single_gate_per_frame=SINGLE_GATE_PER_FRAME)
                        if SPLIT_ACQUISITION else PHOTON_BUDGET)

        for j, sbr in enumerate(SBRS):
            mu = expected_counts(modfs, demodfs, photon_count, sbr)
            fisher = fisher_information(mu, tbin_res)
            jensen[c, i, j]     = crlb_jensen(fisher) * to_mm
            unjensened[c, i, j] = crlb_unjensened(fisher) * to_mm

        print(f"K={k} {labels[i]:<24s} Jensen bound: {jensen[c, i].min():8.2f} - {jensen[c, i].max():8.2f} mm"
              + (f" | un-Jensened: {unjensened[c, i].min():8.2f} - {unjensened[c, i].max():8.2f} mm"
                 if UNJENSENED else ""))

# =========================
# PLOT
# =========================
plt.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 20,
    "figure.titlesize": 28,
    "axes.linewidth": 2,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.minor.width": 1.5,
    "ytick.minor.width": 1.5,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.minor.size": 5,
    "ytick.minor.size": 5,
})
fig, axes = plt.subplots(1, len(KS), figsize=(8 * len(KS), 7), sharex=True, sharey=True, squeeze=False)

for c, (ax, k) in enumerate(zip(axes[0], KS)):
    for i, label in enumerate(labels):
        color = COLORS[i % len(COLORS)]
        ax.plot(SBRS, jensen[c, i], linewidth=5, color=color, label=label + f" (K={k})")
        if UNJENSENED:
            ax.plot(SBRS, unjensened[c, i], linewidth=4, linestyle='--', color=color)

    if UNJENSENED:
        # line-style key: solid = Jensen bound, dashed = un-Jensened
        ax.plot([], [], linewidth=5, color='k', label='Jensen bound')
        ax.plot([], [], linewidth=4, linestyle='--', color='k', label='Un-Jensened')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('SBR')
    ax.grid(True, which='major', color='0.9', lw=1.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)

axes[0, 0].set_ylabel('Average CRLB on depth std. (mm)')
fig.suptitle('CRLB on depth vs SBR', y=1.0)
fig.tight_layout()
plt.savefig('figures/fisher_information.pdf', bbox_inches='tight')
plt.show()
