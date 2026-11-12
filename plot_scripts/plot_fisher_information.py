
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from plot_scripts.plot_utils import get_string_name, get_cap_color
from utils.tof_utils import get_code, calculate_tof_domain_params, scale_photon_count, decode_simulation_depths

# =========================
# CONFIG
# =========================
N_TBINS = 10000
RUNS = [
    {'type': 'ham'},           # pulse width comes from PANELS, not from here
    {'type': 'coarse'},
    {'type': 'trapcoarse'},
    # gate sweeping, as in plot_gate_sweeping_bounds.py: fixed gate width W = tau / 3, shift = N_TBINS // K
    {'type': 'sliding',    'gate_width': N_TBINS // 3},
    # {'type': 'coarsepw', 'pulse_width': 40},
]
KS = np.array([3, 4, 5, 6, 8, 10, 12, 16])   # x-axis: number of gates
HAM_KS = (3, 4, 5, 6)   # GetHamK* only exists for these K; ham is drawn only here
REP_RATE = 5e6
REP_TAU = float(1 / REP_RATE)   # depth range is [0, c * REP_TAU / 2)

PHOTON_BUDGET = 5000       # signal photons
SBR = 0.1                  # global SBR, same scene in both panels (np.inf = no background)
K0 = 8                     # band-limited laser: pulse FWHM fixed at tau / K0 (the K8Bandlimited runs)
# one panel each, like the matched / mismatched pulses in plot_coarse_binning_bounds.py
#   'No band-limit': pulse FWHM = tau / K, narrows as K grows
#   'Band-limit':    pulse FWHM = tau / K0 for every K (the two panels coincide at K = K0)
PANELS = {'No band-limit': None, 'Band-limit': K0}
SPLIT_ACQUISITION = True   # if True, PHOTON_BUDGET is shared across the acquisitions of each scheme
SINGLE_GATE_PER_FRAME = False   # if True, ham K>3 is split over 6 frames (one gate opening per frame); if False, over K
UNJENSENED = False          # if True, also report/plot the un-Jensened average CRLB

RUN_MC = True               # if True, overlay Monte Carlo RMSE (Poisson counts, ZNCC decoding)
RECOMPUTE_MC = False        # if True, ignore the cached .npz and rerun the simulation
MC_TRIALS = 1000            # noisy realizations per depth
MC_SHIFT_STEP = 5           # simulate every MC_SHIFT_STEP-th time bin as a true depth
MC_TRIAL_CHUNK = 50         # trials decoded at once, keeps memory bounded
MC_CACHE = 'data/fisher_information_vs_K_mc.npz'

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

    mu = np.fft.ifft(
        np.fft.fft(clean, axis=0).conj()
        * np.fft.fft(demodfs, axis=0),
        axis=0,
    ).real
    # FFT round-off leaves ~-1e-17 where mu should be 0 (no background); Poisson needs lam >= 0
    return np.clip(mu, 0, None)


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


def k_values(r):
    """K values a scheme is evaluated at."""
    return [k for k in KS if r['type'] != 'ham' or k in HAM_KS]


def build_code(r, k, k_pulse):
    """modfs, demodfs, coding_matrix with pulse FWHM = N_TBINS // k_pulse bins (k_pulse=None -> the gate count k).
    Same FWHM convention as get_code('coarse'), so 'No band-limit' binning is exactly the built-in code."""
    pulse_width = (N_TBINS // min(k_pulse or k, k)) / (2 * np.sqrt(np.log(2)))
    if r['type'] in ('coarse', 'trapcoarse'):
        if k_pulse is None:
            return get_code(r['type'], k, N_TBINS)
        return get_code(r['type'] + 'pw', k, N_TBINS, pulse_width=pulse_width)
    elif r['type'] == 'sliding':
        # same pulse as coarse binning in both panels: tau / K without band-limit, tau / min(K, K0) with it
        return get_code('sliding', k, N_TBINS, pulse_width=pulse_width,
                        shift=N_TBINS // k, gate_width=r['gate_width'])
    else:
        if k_pulse is None:
            return get_code(r['type'], k, N_TBINS, pulse_width=1)
        # ham keeps the fixed tau / K0 pulse for every K, as in the original plot (no min() with K)
        return get_code(r['type'], k, N_TBINS, pulse_width=(N_TBINS // k_pulse) / (2 * np.sqrt(np.log(2))))


def get_label(r):
    return get_string_name(r['type'], None, short=True)


# =========================
# MAIN
# =========================
(_, _, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(N_TBINS, REP_TAU)
to_mm = 1e3

labels = [get_label(r) for r in RUNS]
panel_items = list(PANELS.items())
# nan where a scheme has no code at that K (ham for K > 5), so its line just stops
jensen     = np.full((len(panel_items), len(RUNS), len(KS)), np.nan)
unjensened = np.full((len(panel_items), len(RUNS), len(KS)), np.nan)

# Monte Carlo results are cached; the key holds every setting they depend on
mc_key = repr([RUNS, KS.tolist(), HAM_KS, N_TBINS, REP_TAU, PHOTON_BUDGET, SPLIT_ACQUISITION,
               SINGLE_GATE_PER_FRAME, MC_TRIALS, MC_SHIFT_STEP, SBR, panel_items])
mc = None
if RUN_MC and not RECOMPUTE_MC and os.path.exists(MC_CACHE):
    saved = np.load(MC_CACHE)
    if str(saved['key']) == mc_key:
        mc = saved['rmse']
compute_mc = RUN_MC and mc is None
if compute_mc:
    mc = np.full((len(panel_items), len(RUNS), len(KS)), np.nan)
    mc_shifts = np.arange(0, N_TBINS, MC_SHIFT_STEP)

for i, r in enumerate(RUNS):
    for k in k_values(r):
        j = int(np.flatnonzero(KS == k)[0])

        # ideal=True splits ham over K frames, ideal=False over 6 frames for K > 3 (one gate opening per frame)
        photon_count = (scale_photon_count(PHOTON_BUDGET, r['type'], k, ideal=not SINGLE_GATE_PER_FRAME)
                        if SPLIT_ACQUISITION else PHOTON_BUDGET)

        for p, (title, k_pulse) in enumerate(panel_items):
            modfs, demodfs, coding_matrix = build_code(r, k, k_pulse)
            #if r['type'] == 'ham': plt.plot(modfs); plt.show()
            mu = expected_counts(modfs, demodfs, photon_count, SBR)
            fisher = fisher_information(mu, tbin_res)
            jensen[p, i, j]     = crlb_jensen(fisher) * to_mm
            unjensened[p, i, j] = crlb_unjensened(fisher) * to_mm
            if compute_mc:
                mc[p, i, j] = mc_rmse(mu, coding_matrix, mc_shifts) * to_mm

            print(f"{title:<16s} K={k:<3d} {labels[i]:<24s} Jensen bound: {jensen[p, i, j]:8.2f} mm"
                  + (f" | un-Jensened: {unjensened[p, i, j]:8.2f} mm" if UNJENSENED else "")
                  + (f" | Monte Carlo: {mc[p, i, j]:8.2f} mm" if RUN_MC else ""))

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
fig, axes = plt.subplots(1, len(panel_items), figsize=(8 * len(panel_items), 7), sharex=True, sharey=True, squeeze=False)

# bounds are thick solid lines, simulations thin dotted lines with dots in the matching color
MC_STYLE = dict(linewidth=3.5, linestyle=':', marker='o', markersize=10,
                markeredgecolor='white', markeredgewidth=1.5, zorder=3)

for p, (ax, (title, k_pulse)) in enumerate(zip(axes[0], panel_items)):
    for i, label in enumerate(labels):
        # same per-code colors as plot_dark_bright_single_pixel.py; k=None gives one color across all K
        color = get_cap_color(RUNS[i]['type'], None)
        ax.plot(KS, jensen[p, i], linewidth=5, color=color, label=label)
        if UNJENSENED:
            ax.plot(KS, unjensened[p, i], linewidth=4, linestyle='--', color=color)
        if mc is not None:
            ax.plot(KS, mc[p, i], color=color, **MC_STYLE)

    if UNJENSENED:
        # line-style key: solid = Jensen bound, dashed = un-Jensened
        ax.plot([], [], linewidth=5, color='k', label='Jensen bound')
        ax.plot([], [], linewidth=4, linestyle='--', color='k', label='Un-Jensened')
    if mc is not None:
        # line-style key: solid = CRLB, dotted = Monte Carlo (the Jensen entry above already covers solid)
        if not UNJENSENED:
            ax.plot([], [], linewidth=5, color='k', label='CRLB')
        ax.plot([], [], color='k', label='Monte Carlo (ZNCC)', **MC_STYLE)

    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(KS, labels=[str(k) for k in KS])
    ax.tick_params(axis='x', which='minor', bottom=False, labelbottom=False)
    ax.set_xlabel('Number of gates $K$')
    ax.set_title(rf'{title}  ($\mathbf{{w_p=\tau/K}}$)' if k_pulse is None
                 else rf'{title}  ($\mathbf{{w_p=\tau/K_0}}$)')
    if k_pulse is not None:
        # the band-limited pulse equals the scaled one at K = K0, as in the coarse binning plot
        ax.axvline(k_pulse, color='0.6', lw=2.5, ls=':')
        ax.text(k_pulse * 1.02, 0.02, rf'$K_0={k_pulse}$', color='0.4', fontsize=20,
                transform=ax.get_xaxis_transform())
    ax.grid(True, which='major', color='0.9', lw=1.2)
    ax.spines[['top', 'right']].set_visible(False)

axes[0, 0].set_ylabel('Depth RMSE (mm)' if mc is not None else 'Average CRLB on depth std. (mm)')

# one legend under the figure: colors name the codes, black entries are the line-style key
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.0),
           ncol=len(handles), frameon=False)
#fig.suptitle(rf'Gating Schemes: depth error lower bound vs $\mathbf{{K}}$  ($\mathbf{{SBR={SBR:g}}}$)', y=1.0)
fig.tight_layout()
plt.savefig('/Users/davidparra/PycharmProjects/py-gated-camera/plot_scripts/figures/fisher_information_vs_K.pdf', bbox_inches='tight')
plt.show()
