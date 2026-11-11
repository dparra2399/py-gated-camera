"""Plot the depth-error lower bounds from Theorem 1 and Corollaries 1 and 2.

The pulse FWHM is w_p = kappa * s with kappa = 2 sqrt(2 ln 2).
Fixed pulse:   w_p = tau / K0, i.e. s = tau / (kappa K0), the same for every K   (Corollary 2)
Scaled pulse:  w_p = tau / K,  i.e. s = tau / (kappa K), shrinks as K grows      (Corollary 1)

Both curves use the same Theorem 1 formulas; the only difference is the s you pass in.

Monte Carlo lines follow test.py: codes from get_code, counts from simulate_counts_shared_illum,
photons split with scale_photon_count, ZNCC decoding with decode_simulation_depths, error against
the true depths. Scaled pulse uses 'coarse' / 'trapcoarse' (built-in pulse, FWHM = n_tbins // K bins),
fixed pulse uses 'coarsepw' / 'trapcoarsepw' with the K = K0 width, like test.py's coarsepw runs.
SBR is the global SBR passed to the simulator, as in test.py; the bounds use the matching
background rate a = n / (SBR tau), so every scheme and every K sees the same scene.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from utils.tof_utils import (get_code, simulate_counts_shared_illum, decode_simulation_depths,
                             scale_photon_count, calculate_tof_domain_params)

# ---------------- parameters (edit these) ----------------
c = 3e8          # speed of light (m/s)
tau = 100e-9     # laser period (s)
K0 = 8           # gate count the fixed pulse is matched to
kappa = 2 * np.sqrt(2 * np.log(2))  # FWHM / standard deviation of a Gaussian pulse
s_fixed = tau / (kappa * K0)  # fixed pulse standard deviation (s)
rhos = [2]    # overlap factors, one row each: 1 = triangle coding, 2 = trapezoidal coding
N = 5000         # total expected signal photons over the exposure
T = 1e-3         # total exposure time (s)
SBR = 0.1        # global signal-to-background ratio over the whole period, same definition as test.py
K = np.array([8, 10, 12, 14, 16, 20, 24, 32])

n = N * tau / T             # signal photons per laser cycle
a = n / (SBR * tau)         # background count rate that gives the global SBR (counts/s), same for every K and rho

# ---------------- Monte Carlo (edit these) ----------------
run_mc = True        # overlay simulated RMSE on the bounds
recompute_mc = False # ignore the cached .npz and rerun the simulation
n_tbins = 10000       # time bins per laser period (bin = tau / n_tbins)
trials = 5000         # noisy realizations per depth
depth_step = 0.1     # spacing of simulated depths (m)
depth_margin = 0.5   # keep depths away from the period edges (m)
trial_chunk = 50     # trials decoded at once, keeps memory bounded


# ---------------- Theorem 1 ----------------
def sigma_no_background(K, s, rho):
    """Theorem 1(i): lower bound on depth error, no background."""
    return (c / 2) * s * np.sqrt(K / (rho * N))


def J2(W, s):
    """Lemma 3: J_2(W) = (1 - exp(-W^2 / (4 s^2))) / (sqrt(pi) s)."""
    return (1 - np.exp(-W**2 / (4 * s**2))) / (np.sqrt(np.pi) * s)


def sigma_background(K, s, rho):
    """Theorem 1(ii): lower bound on depth error with background.

    I_bar(K) <= (N / tau) * SBR_W * J_2(W),  SBR_W = n / (a W)
    sigma_d  >= (c / 2) / sqrt(I_bar bound)
             =  (c / 2) sqrt(sqrt(pi) s a tau W / (N n (1 - exp(-W^2 / (4 s^2)))))
    """
    W = rho * tau / K
    SBR_W = n / (a * W)
    I_bar = (N / tau) * SBR_W * J2(W, s)
    return (c / 2) / np.sqrt(I_bar)


def sigma_background_simple(K, s, rho):
    """Corollary 2(ii): weaker bound that grows as sqrt(K), fixed s."""
    return (c / 2) * np.sqrt(4 * np.sqrt(np.pi) * s**3 * a * K / (rho * N * n))


# ---------------- evaluate ----------------
s_scaled = tau / (kappa * K)
to_mm = 1e3


def bound_curves(rho):
    return {
        "No background": {
            "fixed": sigma_no_background(K, s_fixed, rho) * to_mm,
            "scaled": sigma_no_background(K, s_scaled, rho) * to_mm,
        },
        "With background": {
            "fixed": sigma_background(K, s_fixed, rho) * to_mm,
            "scaled": sigma_background(K, s_scaled, rho) * to_mm,
        },
    }


# ---------------- Monte Carlo (same pipeline as test.py) ----------------
(_, _, _, _, max_depth, tbin_depth_res) = calculate_tof_domain_params(n_tbins, tau)
mc_depths = np.arange(depth_margin, max_depth - depth_margin, depth_step)
mc_sbrs = {"No background": np.inf, "With background": SBR}
# fixed pulse: the width get_code('coarse') uses at K = K0, as in test.py's coarsepw runs
fixed_pulse_width = (n_tbins // K0) / (2 * np.sqrt(np.log(2)))


def mc_rmse(Kval, rho, sbr, fixed):
    """Simulated depth RMSE (mm) for one K, overlap rho, global SBR and pulse mode, run like test.py."""
    cap_type = "coarse" if rho == 1 else "trapcoarse"
    if fixed:
        cap_type += "pw"
        illum, cm, coding_matrix = get_code(cap_type, Kval, n_tbins, pulse_width=fixed_pulse_width)
    else:
        illum, cm, coding_matrix = get_code(cap_type, Kval, n_tbins)
    _, coded_values = simulate_counts_shared_illum(
        illum=illum, coding_matrix=cm, depths=mc_depths,
        photon_count=scale_photon_count(N, cap_type, Kval), sbr=sbr,
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


def mc_curves(rho):
    """Simulated RMSE for every panel and pulse mode, cached to data/ so restyling the plot is fast."""
    params = np.array([c, tau, K0, N, T, SBR, n_tbins, trials, depth_step, depth_margin, *K], dtype=float)
    cache = f"data/coarse_binning_mc_rho{rho}_testpy.npz"
    if not recompute_mc and os.path.exists(cache):
        saved = np.load(cache)
        if saved["params"].shape == params.shape and np.allclose(saved["params"], params):
            return {title: {"fixed": saved[f"{title}_fixed"], "scaled": saved[f"{title}_scaled"]}
                    for title in mc_sbrs}

    out = {}
    for title, sbr in mc_sbrs.items():
        print(f"Monte Carlo rho={rho}, {title}...")
        out[title] = {
            "fixed": np.array([mc_rmse(k, rho, sbr, fixed=True) for k in K]),
            "scaled": np.array([mc_rmse(k, rho, sbr, fixed=False) for k in K]),
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
fig, axes = plt.subplots(len(rhos), 2, figsize=(15, 7 * len(rhos)),
                         sharex=True, sharey=True, squeeze=False)

# bounds are thick solid lines, simulations thin dotted lines with dots in the matching color
mc_style = dict(lw=3.5, ls=":", marker="o", ms=10, mec="white", mew=1.5, zorder=3)

for row, rho in enumerate(rhos):
    mc = mc_curves(rho) if run_mc else None
    for ax, (title, cur) in zip(axes[row], bound_curves(rho).items()):
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
        ax.set_title(rf"{title}  ($\mathbf{{\rho={rho}}}$)")
        ax.set_xticks(K, labels=[str(k) for k in K])
        ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
        ax.grid(True, which="major", color="0.9", lw=1.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[row, 0].set_ylabel("Depth RMSE (mm)" if run_mc else "Lower bound on depth error (mm)")

for ax in axes[-1]:
    ax.set_xlabel("Number of gates $K$")

# one legend under the figure so it never covers data
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
fig.suptitle(rf"Gate Binning:  depth error lower bound vs $\mathbf{{K}}$", y=0.97)
fig.tight_layout()
if len(rhos) == 1:
    filename = f"figures/depth_coarse_binning_bound_vs_K_{rhos[0]}.pdf"
else:
    filename = f"figures/depth_coarse_binning_bound_vs_K.pdf"
fig.savefig(filename, dpi=200, bbox_inches="tight")
plt.show()
