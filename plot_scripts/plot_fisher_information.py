
import os
import numpy as np
import matplotlib.pyplot as plt

from plot_scripts.plot_utils import get_string_name, get_cap_color
from utils.tof_utils import get_code, calculate_tof_domain_params, scale_photon_count, decode_simulation_depths

# =========================
# CONFIG
# =========================
N_TBINS = 10000
RUNS = [
    {'type': 'ham',        'pulse_width': int((N_TBINS // (8)) / (2 * np.sqrt(np.log(2))))},
    {'type': 'coarse'},
    {'type': 'trapcoarse'},
    # {'type': 'coarsepw', 'pulse_width': 40},
]
KS = [3, 4]   # one column per K
REP_RATE = 5e6
REP_TAU = float(1 / REP_RATE)   # depth range is [0, c * REP_TAU / 2)

PHOTON_BUDGET = 5000       # signal photons
SBRS = np.logspace(-2, 2, 50)   # signal-to-background ratios to sweep
SPLIT_ACQUISITION = True   # if True, PHOTON_BUDGET is shared across the acquisitions of each scheme
SINGLE_GATE_PER_FRAME = False   # if True, ham K>3 is split over 6 frames (one gate opening per frame); if False, over K
UNJENSENED = False          # if True, also report/plot the un-Jensened average CRLB

RUN_MC = True               # if True, overlay Monte Carlo RMSE (Poisson counts, ZNCC decoding)
RECOMPUTE_MC = False        # if True, ignore the cached .npz and rerun the simulation
MC_SBRS = np.logspace(-2, 2, 9)   # SBRs to simulate (coarser than SBRS, each one is a full Monte Carlo run)
MC_TRIALS = 1000             # noisy realizations per depth
MC_SHIFT_STEP = 5           # simulate every MC_SHIFT_STEP-th time bin as a true depth
MC_TRIAL_CHUNK = 50         # trials decoded at once, keeps memory bounded
MC_CACHE = 'data/fisher_information_mc.npz'

SPEED_OF_LIGHT = 3e8

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


def mc_rmse(mu, coding_matrix, shifts):
    """Depth RMSE (m) from Poisson samples of mu at the given shifts, decoded with ZNCC.
    mu[t_0] equals the coded values simulate_counts / simulate_counts_shared_illum give for shift t_0."""
    true_depths = shifts * tbin_depth_res
    period = N_TBINS * tbin_depth_res
    sum_sq_err, n_total, remaining = 0.0, 0, MC_TRIALS
    while remaining > 0:
        chunk = min(MC_TRIAL_CHUNK, remaining)
        decoded, _, _ = decode_simulation_depths(coding_matrix, mu[shifts], true_depths, chunk, tbin_depth_res)
        # circular error, the same wrap-around the Fisher information assumes
        err = (decoded - true_depths + period / 2) % period - period / 2
        sum_sq_err += np.sum(err ** 2)
        n_total += err.size
        remaining -= chunk
    return np.sqrt(sum_sq_err / n_total)


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

# Monte Carlo results are cached; the key holds every setting they depend on
mc_key = repr([RUNS, KS, N_TBINS, REP_TAU, PHOTON_BUDGET, SPLIT_ACQUISITION, SINGLE_GATE_PER_FRAME,
               MC_TRIALS, MC_SHIFT_STEP, MC_SBRS.tolist()])
mc = None
if RUN_MC and not RECOMPUTE_MC and os.path.exists(MC_CACHE):
    saved = np.load(MC_CACHE)
    if str(saved['key']) == mc_key:
        mc = saved['rmse']
compute_mc = RUN_MC and mc is None
if compute_mc:
    mc = np.zeros((len(KS), len(RUNS), len(MC_SBRS)))
    mc_shifts = np.arange(0, N_TBINS, MC_SHIFT_STEP)

for c, k in enumerate(KS):
    for i, r in enumerate(RUNS):
        modfs, demodfs, coding_matrix = get_code(r['type'], k, N_TBINS, pulse_width=r.get('pulse_width'),
                                                 shift=r.get('shift'), gate_width=r.get('gate_width'))

        # ideal=True splits ham over K frames, ideal=False over 6 frames for K > 3 (one gate opening per frame)
        photon_count = (scale_photon_count(PHOTON_BUDGET, r['type'], k, ideal=not SINGLE_GATE_PER_FRAME)
                        if SPLIT_ACQUISITION else PHOTON_BUDGET)

        for j, sbr in enumerate(SBRS):
            mu = expected_counts(modfs, demodfs, photon_count, sbr)
            fisher = fisher_information(mu, tbin_res)
            jensen[c, i, j]     = crlb_jensen(fisher) * to_mm
            unjensened[c, i, j] = crlb_unjensened(fisher) * to_mm

        if compute_mc:
            for j, sbr in enumerate(MC_SBRS):
                mu = expected_counts(modfs, demodfs, photon_count, sbr)
                mc[c, i, j] = mc_rmse(mu, coding_matrix, mc_shifts) * to_mm

        print(f"K={k} {labels[i]:<24s} Jensen bound: {jensen[c, i].min():8.2f} - {jensen[c, i].max():8.2f} mm"
              + (f" | un-Jensened: {unjensened[c, i].min():8.2f} - {unjensened[c, i].max():8.2f} mm"
                 if UNJENSENED else "")
              + (f" | Monte Carlo: {mc[c, i].min():8.2f} - {mc[c, i].max():8.2f} mm" if RUN_MC else ""))

if compute_mc:
    os.makedirs(os.path.dirname(MC_CACHE), exist_ok=True)
    np.savez(MC_CACHE, key=mc_key, rmse=mc)

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
    "axes.titleweight": "bold",
    "figure.titleweight": "bold",
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

# bounds are thick solid lines, simulations thin dotted lines with dots in the matching color
MC_STYLE = dict(linewidth=3.5, linestyle=':', marker='o', markersize=10,
                markeredgecolor='white', markeredgewidth=1.5, zorder=3)

for c, (ax, k) in enumerate(zip(axes[0], KS)):
    for i, label in enumerate(labels):
        # same per-code colors as plot_dark_bright_single_pixel.py
        color = get_cap_color(RUNS[i]['type'], k)
        ax.plot(SBRS, jensen[c, i], linewidth=5, color=color, label=label)
        if UNJENSENED:
            ax.plot(SBRS, unjensened[c, i], linewidth=4, linestyle='--', color=color)
        if mc is not None:
            ax.plot(MC_SBRS, mc[c, i], color=color, **MC_STYLE)

    if UNJENSENED:
        # line-style key: solid = Jensen bound, dashed = un-Jensened
        ax.plot([], [], linewidth=5, color='k', label='Jensen bound')
        ax.plot([], [], linewidth=4, linestyle='--', color='k', label='Un-Jensened')
    if mc is not None:
        # line-style key: solid = CRLB, dotted = Monte Carlo (the Jensen entry above already covers solid)
        if not UNJENSENED:
            ax.plot([], [], linewidth=5, color='k', label='CRLB')
        ax.plot([], [], color='k', label='Monte Carlo (ZNCC)', **MC_STYLE)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(rf'$SBR_W$')
    ax.set_title(rf'$\mathbf{{K={k}}}$')
    ax.grid(True, which='major', color='0.9', lw=1.2)
    ax.spines[['top', 'right']].set_visible(False)

axes[0, 0].set_ylabel('Depth RMSE (mm)' if mc is not None else 'Average CRLB on depth std. (mm)')

# one legend under the figure: colors name the codes, black entries are the line-style key
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.0),
           ncol=len(handles), frameon=False)
fig.suptitle('Binning vs Hamiltonian: depth error lower bound vs SBR', y=1.0)
fig.tight_layout()
plt.savefig('figures/fisher_information.pdf', bbox_inches='tight')
plt.show()
