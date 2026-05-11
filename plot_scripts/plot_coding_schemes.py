import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from plot_scripts.plot_utils import get_string_name
from utils.tof_utils import get_code, calculate_tof_domain_params

# =========================
# CONFIG
# =========================
TYPES = ['ham', 'rect', 'traprect', 'timeslicing']
KS    = [3,        3,            16]            # K per type
N_TBINS = 976

REP_RATE = 5e6
REP_TAU = float(1 / REP_RATE)

MODF_SHIFT = N_TBINS // 4   # roll modfs right so pulse isn't at t=0
SHOW_K     = 2              # when k > 2*SHOW_K, show only first/last SHOW_K with ⋮ in between

# =========================
# MAIN
# =========================
(_, _, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(N_TBINS, REP_TAU)

n_types = len(TYPES)
k_max = 2 * SHOW_K

fig = plt.figure(figsize=(6 * n_types, 4 + k_max))

# 3 main sections (modfs / demodfs / cm), demodfs and cm sized to the largest K
gs_main = gridspec.GridSpec(3, n_types, figure=fig,
                             height_ratios=[2, k_max, k_max],
                             hspace=0.05, wspace=0.1)

for j, (capture_type, k) in enumerate(zip(TYPES, KS)):
    modfs, demodfs, cm = get_code(capture_type, k, N_TBINS)

    # --- row 0: modfs ---
    ax = fig.add_subplot(gs_main[0, j])
    if modfs.ndim == 1:
        ax.plot(np.roll(modfs, MODF_SHIFT))
    else:
        ax.plot(np.roll(modfs[:, 0], MODF_SHIFT))
    ax.axvline(x=MODF_SHIFT, color='k', linestyle='--', linewidth=1, label='t=0')
    ax.set_title(get_string_name(capture_type, k), fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    if j == 0:
        ax.set_ylabel(rf'$s(t)$', fontsize=12, labelpad=1)

    # --- row 1: demodfs — one sub-row per k, k=0 at bottom ---
    if k > 2 * SHOW_K:
        # truncated: top SHOW_K, ellipsis, bottom SHOW_K
        n_rows = 2 * SHOW_K + 1
        hr = [1] * SHOW_K + [0.4] + [1] * SHOW_K   # ellipsis row is shorter
        gs_d = gridspec.GridSpecFromSubplotSpec(n_rows, 1, subplot_spec=gs_main[1, j],
                                                hspace=0.05, height_ratios=hr)
        # top group: k=K-1 at gs row 0, k=K-2 at gs row 1, ...
        for idx in range(SHOW_K):
            ki = k - 1 - idx
            ax = fig.add_subplot(gs_d[idx])
            ax.plot(demodfs[:, ki])
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(rf'$g_{{{ki+1}}}(t)$', fontsize=12, labelpad=1)
        # ellipsis
        ax = fig.add_subplot(gs_d[SHOW_K])
        ax.text(0.5, 0.5, '⋮', ha='center', va='center', fontsize=14,
                transform=ax.transAxes)
        ax.axis('off')
        # bottom group: k=1 at gs row SHOW_K+1, k=0 at gs row SHOW_K+2
        for idx in range(SHOW_K):
            ki = SHOW_K - 1 - idx
            ax = fig.add_subplot(gs_d[SHOW_K + 1 + idx])
            ax.plot(demodfs[:, ki])
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(rf'$g_{{{ki+1}}}(t)$', fontsize=12, labelpad=1)
    else:
        gs_d = gridspec.GridSpecFromSubplotSpec(k, 1, subplot_spec=gs_main[1, j], hspace=0.05)
        for ki in range(k):
            ax = fig.add_subplot(gs_d[k - 1 - ki])   # k-1-ki → k=0 in bottom sub-row
            ax.plot(demodfs[:, ki])
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(rf'$g_{{{ki+1}}}(t)$', fontsize=12, labelpad=1)

    # --- row 2: coding matrix — one sub-row per k, k=0 at bottom ---
    if capture_type == 'timeslicing':
        ax = fig.add_subplot(gs_main[2, j])
        ax.axis('off')
    else:
        gs_c = gridspec.GridSpecFromSubplotSpec(k, 1, subplot_spec=gs_main[2, j], hspace=0.05)
        for ki in range(k):
            ax = fig.add_subplot(gs_c[k - 1 - ki])
            ax.plot(cm[:, ki])
            ax.set_yticks([])
            if ki == 0:
                ax.set_xlabel('Time Bins', fontsize=12)
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(rf'$F_{{{ki+1}}}(d)$', fontsize=12, labelpad=1)

plt.show()
