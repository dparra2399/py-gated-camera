"""Plot the depth-error lower bounds for gate sweeping (Lemma 5, Corollaries 3 and 4).

The gate width W is fixed (W >> s) and does not depend on K; only the shift
Delta = tau / K changes. The pulse FWHM is w_p = kappa * s with kappa = 2 sqrt(2 ln 2).

Scaled pulse:  w_p = tau / K,  i.e. s = tau / (kappa K), shrinks as K grows   (Corollary 3)
Fixed pulse:   w_p = tau / K0, i.e. s = tau / (kappa K0), the same for every K (Corollary 4)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm

# ---------------- parameters (edit these) ----------------
c = 3e8             # speed of light (m/s)
tau = 100e-9        # laser period (s)
K0 = 10              # gate count the fixed pulse is matched to
kappa = 2 * np.sqrt(2 * np.log(2))  # FWHM / standard deviation of a Gaussian pulse
s_fixed = tau / (kappa * K0)  # fixed pulse standard deviation (s)
W = tau / 2         # fixed gate width (s); needs W >> s and tau - W >> s
N = 100            # total expected signal photons over the exposure
T = 1e-3            # total exposure time (s)
SBR_W = 0.1         # signal-to-background ratio of the fixed gate
K = np.arange(K0, 500, N)

n = N * tau / T     # signal photons per laser cycle
a = n / (SBR_W * W) # background count rate that gives SBR_W (counts/s)

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
    "axes.linewidth": 2,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "ytick.minor.width": 1.5,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "ytick.minor.size": 5,
})
fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)

for ax, (title, cur) in zip(axes, curves.items()):
    ax.plot(K, cur["fixed"], color=BLUE, lw=5, label=r"Fixed pulse width ($w_p=\tau/K_0$)")
    ax.plot(K, cur["scaled"], color=ORANGE, lw=5, label=r"Pulse scales with $K$ ($w_p=\tau/K$)")
    if "simple_fixed" in cur:
        ax.plot(K, cur["simple_fixed"], color=BLUE, lw=4, ls="--",
                label=r"Fixed $w_p$, Corollary 4(ii) (no $K$)")
        ax.plot(K, cur["simple_scaled"], color=ORANGE, lw=4, ls="--",
                label=r"Scaled $w_p$, Corollary 3(ii) ($\propto 1/\sqrt{K}$)")
    ax.axvline(K0, color="0.6", lw=2.5, ls=":")
    ax.text(K0 * 1.02, 0.02, rf"$K_0={K0}$", color="0.4", fontsize=20,
            transform=ax.get_xaxis_transform())
    ax.set_title(title)
    ax.set_xticks(K, labels=[str(k) for k in K])
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.grid(True, which="major", color="0.9", lw=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.set_xlabel("Number of gates $K$")
axes[0].set_ylabel("Lower bound on depth error (mm)")

fig.suptitle(rf"Gate sweeping: depth error lower bound vs $K$  ($W=\tau/2$, SBR$_W={SBR_W}$)", y=1.0)
fig.tight_layout()
filename = "figures/depth_bound_vs_K_sweeping.png"
fig.savefig(filename, dpi=200, bbox_inches="tight")
plt.show()