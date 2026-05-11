import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.tof_utils import get_code, calculate_tof_domain_params

# =========================
# CONFIG
# =========================
TYPES = ['coarse', 'ham', 'traprect']
K = 3
N_TBINS = 990

REP_RATE = 5e6
REP_TAU = float(1 / REP_RATE)

MODF_SHIFT = N_TBINS // 4   # roll modfs right so pulse isn't at t=0

# =========================
# MAIN
# =========================
(_, _, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(N_TBINS, REP_TAU)

n_types = len(TYPES)

fig = plt.figure(figsize=(3 * n_types, 4 + 1 * K))

# 3 main sections (modfs / demodfs / cm), demodfs and cm each get K sub-rows
gs_main = gridspec.GridSpec(3, n_types, figure=fig,
                             height_ratios=[1.5, K, K],
                             hspace=0.05, wspace=0.1)

for j, capture_type in enumerate(TYPES):
    modfs, demodfs, cm = get_code(capture_type, K, N_TBINS)

    # --- row 0: modfs ---
    ax = fig.add_subplot(gs_main[0, j])
    if modfs.ndim == 1:
        ax.plot(np.roll(modfs, MODF_SHIFT))
    else:
        ax.plot(np.roll(modfs[:, 0], MODF_SHIFT))
    ax.axvline(x=MODF_SHIFT, color='k', linestyle='--', linewidth=1, label='t=0')
    ax.set_title(capture_type, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    if j == 0:
        ax.set_ylabel('modfs', fontsize=12)

    # --- row 1: demodfs — one sub-row per k, k=0 at bottom ---
    gs_d = gridspec.GridSpecFromSubplotSpec(K, 1, subplot_spec=gs_main[1, j], hspace=0.05)
    #ax = fig.add_subplot(gs_d[0, 0])
    #ax.plot(demodfs)
    for ki in range(K):
        #ax.plot(demodfs[:, ki])

        ax = fig.add_subplot(gs_d[K - 1 - ki])   # K-1-ki → k=0 in bottom sub-row
        ax.plot(demodfs[:, ki])
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0 and ki == 0:
            #ax.set_ylabel(f'k={ki}', fontsize=9, rotation=0, labelpad=22)
            ax.set_ylabel(f'demodfs', fontsize=9, labelpad=22)
        # if ki == K - 1 and j == 0:
        #     ax.set_title('demodfs', fontsize=11, loc='left')

    # --- row 2: coding matrix — one sub-row per k, k=0 at bottom ---
    gs_c = gridspec.GridSpecFromSubplotSpec(K, 1, subplot_spec=gs_main[2, j], hspace=0.05)
    for ki in range(K):
        ax = fig.add_subplot(gs_c[K - 1 - ki])
        ax.plot(cm[:, ki])
        ax.set_yticks([])
        if ki == 0:
            ax.set_xlabel('t-bin', fontsize=9)
        else:
            ax.set_xticks([])
        if j == 0:
            ax.set_ylabel(f'k={ki}', fontsize=9, rotation=0, labelpad=22)
        if ki == K - 1 and j == 0:
            ax.set_title('coding matrix', fontsize=11, loc='left')

plt.show()
