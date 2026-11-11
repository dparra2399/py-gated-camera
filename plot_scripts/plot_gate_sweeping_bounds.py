"""Plot the depth-error lower bounds for gate sweeping (Lemma 5, Corollaries 3 and 4).

The gate width W is fixed (W >> s) and does not depend on K; only the shift
Delta = tau / K changes. The pulse FWHM is w_p = kappa * s with kappa = 2 sqrt(2 ln 2).

Scaled pulse:  w_p = tau / K,  i.e. s = tau / (kappa K), shrinks as K grows   (Corollary 3)
Fixed pulse:   w_p = tau / K0, i.e. s = tau / (kappa K0), the same for every K (Corollary 4)

Monte Carlo lines follow test.py: get_code('sliding') with shift = n_tbins // K and gate width
int(W / tau * n_tbins), pulse_width = int((n_tbins // K0) / (2 sqrt(ln 2))) like test.py's PULSE_WIDTH
(n_tbins // K for the scaled pulse), counts from simulate_counts_shared_illum, photons split with
scale_photon_count, ZNCC decoding with decode_simulation_depths, error against the true depths.
SBR is the global SBR passed to the simulator, as in test.py; the bounds use the matching
background rate a = n / (SBR tau), the same scene as plot_coarse_binning_bounds.py.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
from scipy.integrate import quad
from scipy.stats import norm

from utils.tof_utils import (get_code, simulate_counts_shared_illum, decode_simulation_depths,
                             scale_photon_count, calculate_tof_domain_params)

# ---------------- parameters (edit these) ----------------
c = 3e8             # speed of light (m/s)
tau = 100e-9        # laser period (s)
K0 = 8              # gate count the fixed pulse is matched to
kappa = 2 * np.sqrt(2 * np.log(2))  # FWHM / standard deviation of a Gaussian pulse
s_fixed = tau / (kappa * K0)  # fixed pulse standard deviation (s)
W = tau / 3        # fixed gate width (s); needs W >> s and tau - W >> s
N = 5000            # total expected signal photons over the exposure
T = 1e-3            # total exposure time (s)
SBR = 0.1           # global signal-to-background ratio over the whole period, same definition as test.py
K_STEP = 48
K = np.arange(K0, 400, K_STEP)

n = N * tau / T     # signal photons per laser cycle
a = n / (SBR * tau) # background count rate that gives the global SBR (counts/s), same as coarse binning

# ---------------- Monte Carlo (edit these) ----------------
run_mc = True        # overlay simulated RMSE on the bounds
recompute_mc = False # ignore the cached .npz and rerun the simulation
n_tbins = 10000       # time bins per laser period (bin = tau / n_tbins)
trials = 1000         # noisy realizations per depth (large K is slow, the decoder scales with K)
depth_step = 0.2     # spacing of simulated depths (m)
depth_margin = 0.5   # keep depths away from the period edges (m)
trial_chunk = 50     # trials decoded at once, keeps memory bounded

# C = int g(x)^2 / (1 - Phi(x)) dx, g = unit-width Gaussian (Lemma 5)
C = quad(lambda x: np.exp(2 * norm.logpdf(x) - norm.logsf(x)), -30, 30, limit=500)[0]


# ---------------- Lemma 5 ----------------
def sigma_no_background(s):
    """Lemma 5(i): sigma_d >= (c/2) sqrt(s tau / (2 C N)). No K."""
    return (c / 2) * np.sqrt(s * tau / (2 * C * N))


def sigma_background(s):
    """Lemma 5(ii), first inequality (Theorem 1(ii) with fixed W). No K."""
    return (c / 2) * np.sqrt(np.sqrt(np.pi) * s * a * tau * W
                             / (N * n * (1 - np.exp(-W**2 / (4 * s**2)))))


def sigma_background_simple(s):
    """Lemma 5(ii), second inequality using 1 - e^{-x} <= 1 (Corollaries 3(ii) and 4(ii))."""
    return (c / 2) * np.sqrt(np.sqrt(np.pi) * s * a * tau * W / (N * n))


# ---------------- evaluate ----------------
s_scaled = tau / (kappa * K)
ones = np.ones_like(K, dtype=float)
to_mm = 1e3

curves = {
    "No background": {
        "fixed": sigma_no_background(s_fixed) * ones * to_mm,
        "scaled": sigma_no_background(s_scaled) * to_mm,
    },
    "With background": {
        "fixed": sigma_background(s_fixed) * ones * to_mm,
        "scaled": sigma_background(s_scaled) * to_mm,
    },
}


# ---------------- Monte Carlo (same pipeline as test.py) ----------------
(_, _, _, _, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, tau)
mc_depths = np.arange(depth_margin, max_depth - depth_margin, depth_step)
mc_sbrs = {"No background": np.inf, "With background": SBR}
gate_width = int(W / tau * n_tbins)   # like test.py's int(N_TBINS * .6)


def pulse_width_bins(k):
    """test.py's PULSE_WIDTH formula: FWHM of n_tbins // k bins, in gaussian_pulse width units."""
    return int((n_tbins // k) / (2 * np.sqrt(np.log(2))))


def mc_rmse(Kval, sbr, fixed):
    """Simulated depth RMSE (mm) for one K, global SBR and pulse mode, run like test.py."""
    shift = n_tbins // Kval   # test.py sets k = n_tbins // shift; here K is given, so invert it
    illum, cm, coding_matrix = get_code("sliding", Kval, n_tbins,
                                        pulse_width=pulse_width_bins(K0 if fixed else Kval),
                                        shift=shift, gate_width=gate_width)
    _, coded_values = simulate_counts_shared_illum(
        illum=illum, coding_matrix=cm, depths=mc_depths,
        photon_count=scale_photon_count(N, "sliding", Kval), sbr=sbr,
        tbin_depth_res=tbin_depth_res, n_tbins=n_tbins, k=Kval,
    )

    # same error as decode_simulation_depths in test.py, accumulated over trial chunks to bound memory
    sum_sq_err, n_total, remaining = 0.0, 0, trials
    while remaining > 0:
        chunk = min(trial_chunk, remaining)
        decoded, _, _ = decode_simulation_depths(coding_matrix, coded_values, mc_depths, chunk, tbin_depth_res)
        sum_sq_err += np.sum((decoded - mc_depths) ** 2)
        n_total += decoded.size
        remaining -= chunk
    return np.sqrt(sum_sq_err / n_total) * to_mm


def mc_curves():
    """Simulated RMSE for every panel and pulse mode, cached to data/ so restyling the plot is fast."""
    params = np.array([c, tau, K0, W, N, T, SBR, n_tbins, trials, depth_step, depth_margin, *K], dtype=float)
    cache = "data/gate_sweeping_mc_testpy.npz"
    if not recompute_mc and os.path.exists(cache):
        saved = np.load(cache)
        if saved["params"].shape == params.shape and np.allclose(saved["params"], params):
            return {title: {"fixed": saved[f"{title}_fixed"], "scaled": saved[f"{title}_scaled"]}
                    for title in mc_sbrs}

    out = {}
    for title, sbr in mc_sbrs.items():
        print(f"Monte Carlo {title}...")
        out[title] = {
            "fixed": np.array([mc_rmse(k, sbr, fixed=True) for k in K]),
            "scaled": np.array([mc_rmse(k, sbr, fixed=False) for k in K]),
        }
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, params=params,
             **{f"{title}_{mode}": v for title, cur in out.items() for mode, v in cur.items()})
    return out


# ---------------- plot ----------------
BLUE, ORANGE = "#2a78d6", "#eb6834"
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
    "ytick.minor.width": 1.5,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "ytick.minor.size": 5,
})
fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)

# bounds are thick solid lines, simulations thin dotted lines with dots in the matching color
mc_style = dict(lw=3.5, ls=":", marker="o", ms=10, mec="white", mew=1.5, zorder=3)
mc = mc_curves() if run_mc else None

for ax, (title, cur) in zip(axes, curves.items()):
    ax.plot(K, cur["fixed"], color=BLUE, lw=5, label=r"CRLB bound, fixed pulse ($w_p=\tau/K_0$)")
    ax.plot(K, cur["scaled"], color=ORANGE, lw=5, label=r"CRLB bound, scaled pulse ($w_p=\tau/K$)")
    if mc is not None:
        ax.plot(K, mc[title]["fixed"], color=BLUE, label="Monte Carlo, fixed pulse", **mc_style)
        ax.plot(K, mc[title]["scaled"], color=ORANGE, label="Monte Carlo, scaled pulse", **mc_style)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axvline(K0, color="0.6", lw=2.5, ls=":")
    ax.text(K0 * 1.02, 0.02, rf"$K_0={K0}$", color="0.4", fontsize=20,
            transform=ax.get_xaxis_transform())
    ax.set_title(title)
    ax.set_xticks(K, labels=[str(k) for k in K])
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.grid(True, which="major", color="0.9", lw=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Number of gates $K$")
axes[0].set_ylabel("Depth RMSE (mm)" if run_mc else "Lower bound on depth error (mm)")

# one legend under the figure so it never covers data
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)

fig.suptitle(rf"Gate sweeping: depth error lower bound vs $\mathbf{{K}}$", y=0.97)
fig.tight_layout()
filename = "figures/depth_bound_vs_K_sweeping.pdf"
fig.savefig(filename, dpi=200, bbox_inches="tight")
plt.show()