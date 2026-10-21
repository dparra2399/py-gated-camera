"""Plot the depth-error lower bounds from Theorem 1 and Corollaries 1 and 2.

The pulse FWHM is w_p = kappa * s with kappa = 2 sqrt(2 ln 2).
Fixed pulse:   w_p = tau / K0, i.e. s = tau / (kappa K0), the same for every K   (Corollary 2)
Scaled pulse:  w_p = tau / K,  i.e. s = tau / (kappa K), shrinks as K grows      (Corollary 1)

Both curves use the same Theorem 1 formulas; the only difference is the s you pass in.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------- parameters (edit these) ----------------
c = 3e8          # speed of light (m/s)
tau = 100e-9     # laser period (s)
K0 = 8           # gate count the fixed pulse is matched to
kappa = 2 * np.sqrt(2 * np.log(2))  # FWHM / standard deviation of a Gaussian pulse
s_fixed = tau / (kappa * K0)  # fixed pulse standard deviation (s)
rhos = [1]    # overlap factors, one row each: 1 = triangle coding, 2 = trapezoidal coding
N = 5000         # total expected signal photons over the exposure
T = 1e-3         # total exposure time (s)
SBR_K0 = 0.1     # signal-to-background ratio of one rho = 1 gate at K = K0
K = np.array([8, 10, 12, 14, 16, 20, 24, 32])

n = N * tau / T             # signal photons per laser cycle
W0 = tau / K0               # rho = 1 gate width at K = K0
a = n / (SBR_K0 * W0)       # background count rate that gives SBR_K0 (counts/s), same for every rho


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
            "simple": sigma_background_simple(K, s_fixed, rho) * to_mm,
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
fig, axes = plt.subplots(len(rhos), 2, figsize=(15, 7 * len(rhos)),
                         sharex=True, sharey=True, squeeze=False)

for row, rho in enumerate(rhos):
    for ax, (title, cur) in zip(axes[row], bound_curves(rho).items()):
        ax.plot(K, cur["fixed"], color=BLUE, lw=5, label=r"Fixed pulse width ($w_p=\tau/K_0$)")
        ax.plot(K, cur["scaled"], color=ORANGE, lw=5, label=r"Pulse scales with $K$ ($w_p=\tau/K$)")
        temp_label = r"Fixed $s$, Equation 4 ($\propto\sqrt{K}$)" if rho == 1 else r"Fixed $s$, Corollary 2 ($\propto\sqrt{K}$)"
        if "simple" in cur:
            ax.plot(K, cur["simple"], color=BLUE, lw=4, ls="--",
                      label=temp_label)
        ax.axvline(K0, color="0.6", lw=2.5, ls=":")
        ax.text(K0 * 1.02, 0.02, rf"$K_0={K0}$", color="0.4", fontsize=20,
                transform=ax.get_xaxis_transform())
        ax.set_title(rf"{title}  ($\rho={rho}$)")
        ax.set_xticks(K, labels=[str(k) for k in K])
        ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
        ax.grid(True, which="major", color="0.9", lw=1.2)
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax.legend(frameon=False)
    axes[row, 0].set_ylabel("Lower bound on depth error (mm)")

for ax in axes[-1]:
    ax.set_xlabel("Number of gates $K$")
fig.suptitle(rf"Coarse Binning: Depth error lower bound vs $K$  ($\mathrm{{SBR}}_W$ at $K_0={SBR_K0/rho}$)", y=1.0)
fig.tight_layout()
if len(rhos) == 1:
    filename = f"figures/depth_coarse_binning_bound_vs_K_{rhos[0]}.png"
else:
    filename = f"figures/depth_coarse_binning_bound_vs_K.png"
fig.savefig(filename, dpi=200, bbox_inches="tight")
plt.show()
