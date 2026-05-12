import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from plot_scripts.plot_utils import get_string_name
from utils.tof_utils import get_code, calculate_tof_domain_params

# =========================
# CONFIG
# =========================
TYPES = ['ham', 'coarse', 'trapcoarse']
KS    = [3,        3,            3]            # K per type
N_TBINS = 976

REP_RATE = 5e6
REP_TAU = float(1 / REP_RATE)

MODF_SHIFT = N_TBINS // 4   # roll modfs right so pulse isn't at t=0
SHOW_K     = 2              # when k > 2*SHOW_K, show only first/last SHOW_K with ⋮ in between

# =========================
# COLORS
# =========================
ILLUM_COLOR = 'tab:blue'
KI_COLORS   = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                'tab:brown',  'tab:pink',  'tab:gray', 'tab:olive', 'tab:cyan']

# =========================
# MAIN
# =========================
(_, _, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(N_TBINS, REP_TAU)

n_types = len(TYPES)
k_max = 2 * SHOW_K

fig = plt.figure(figsize=(5 + k_max, 2 * n_types))

# 3 main sections (modfs / demodfs / cm) are now columns; types are rows
gs_main = gridspec.GridSpec(n_types, 3, figure=fig,
                             width_ratios=[2, k_max, k_max],
                             hspace=0.1, wspace=0.05)

for j, (capture_type, k) in enumerate(zip(TYPES, KS)):
    modfs, demodfs, cm = get_code(capture_type, k, N_TBINS)

    # --- col 0: modfs ---
    ax = fig.add_subplot(gs_main[j, 0])
    if modfs.ndim == 1:
        ax.plot(np.roll(modfs, MODF_SHIFT), color=ILLUM_COLOR)
    else:
        ax.plot(np.roll(modfs[:, 0], MODF_SHIFT), color=ILLUM_COLOR)
    ax.axvline(x=MODF_SHIFT, color='k', linestyle='--', linewidth=1)
    ax.set_ylabel(get_string_name(capture_type, k, short=True), fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    if j == 0:
        ax.set_title(r'$s(t)$', fontsize=12)

    # --- col 1: demodfs — one sub-row per k, k=0 at bottom ---
    if k > 2 * SHOW_K:
        # truncated: top SHOW_K, ellipsis, bottom SHOW_K
        n_rows = 2 * SHOW_K + 1
        hr = [1] * SHOW_K + [0.4] + [1] * SHOW_K   # ellipsis row is shorter
        gs_d = gridspec.GridSpecFromSubplotSpec(n_rows, 1, subplot_spec=gs_main[j, 1],
                                                hspace=0.05, height_ratios=hr)
        # top group: ki=0 at row 0, ki=1 at row 1, ...
        for idx in range(SHOW_K):
            ki = idx
            ax = fig.add_subplot(gs_d[idx])
            ax.plot(demodfs[:, ki], color=KI_COLORS[ki % len(KI_COLORS)])
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0 and idx == 0:
                ax.set_title(r'$g(t)$', fontsize=12)
        # ellipsis
        ax = fig.add_subplot(gs_d[SHOW_K])
        ax.text(0.5, 0.5, '⋮', ha='center', va='center', fontsize=14,
                transform=ax.transAxes)
        ax.axis('off')
        # bottom group: ki=K-SHOW_K at row SHOW_K+1, ..., ki=K-1 at last row
        for idx in range(SHOW_K):
            ki = k - SHOW_K + idx
            ax = fig.add_subplot(gs_d[SHOW_K + 1 + idx])
            ax.plot(demodfs[:, ki], color=KI_COLORS[ki % len(KI_COLORS)])
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        gs_d = gridspec.GridSpecFromSubplotSpec(k, 1, subplot_spec=gs_main[j, 1], hspace=0.05)
        for ki in range(k):
            ax = fig.add_subplot(gs_d[ki])   # ki=0 at top
            ax.plot(demodfs[:, ki], color=KI_COLORS[ki % len(KI_COLORS)])
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0 and ki == 0:
                ax.set_title(r'$g(t)$', fontsize=12)

    # --- col 2: coding matrix — one sub-row per k, k=0 at bottom ---
    if capture_type == 'timeslicing':
        ax = fig.add_subplot(gs_main[j, 2])
        ax.axis('off')
    else:
        gs_c = gridspec.GridSpecFromSubplotSpec(k, 1, subplot_spec=gs_main[j, 2], hspace=0.05)
        for ki in range(k):
            ax = fig.add_subplot(gs_c[ki])   # ki=0 at top
            ax.plot(cm[:, ki], color=KI_COLORS[ki % len(KI_COLORS)])
            ax.set_yticks([])
            if ki == k - 1 and j == n_types - 1:
                ax.set_xlabel('Time Bins', fontsize=10)
            else:
                ax.set_xticks([])
            if j == 0 and ki == 0:
                ax.set_title(r'$F(d)$', fontsize=12)

plt.show()
