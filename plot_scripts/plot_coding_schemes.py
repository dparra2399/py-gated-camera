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
# STYLE (matches plot_3d_teaser_fig.py's timing diagrams)
# =========================
FONT_SIZE   = 14
LW          = 2.5
PULSE_COLOR = 'red'
GATE_FILL   = '0.75'
ROW_AMP     = 0.7    # waveform height within its row (rows are spaced 1 apart)
X_MARGIN    = 0.1    # gate baselines drawn past each end of the period, as a fraction of tau


def to_unit(wave):
    """Rescale to [0, 1], so every waveform sits between its row's baseline and top."""
    return (wave - wave.min()) / max(np.ptp(wave), 1e-12)


def shown_rows(k):
    """Indices of the waveforms to draw, top to bottom; None marks the ⋮ row."""
    if k > 2 * SHOW_K:
        return list(range(SHOW_K)) + [None] + list(range(k - SHOW_K, k))
    return list(range(k))


def extend_period(wave, n_margin):
    """`wave` extended n_margin bins past both ends of the period, with its bin indices.

    A gate on at both t=0 and tau wraps around the period, so it continues periodically
    and no edge is drawn there. Anything else is 0 outside the period, so a gate that
    starts at t=0 or ends at tau still gets its vertical edge drawn.
    """
    idx = np.arange(-n_margin, len(wave) + n_margin)
    if wave[0] > 0.5 * wave.max() and wave[-1] > 0.5 * wave.max():
        return idx, wave[idx % len(wave)]
    return idx, np.where((idx >= 0) & (idx < len(wave)), wave[idx % len(wave)], 0)


def finish_axis(ax, row_y, row_labels, x_ticks, x_labels, bottom, xlabel):
    """Teaser-style axis: rows labelled on the left, only the bottom spine kept."""
    ax.set_yticks(row_y + ROW_AMP / 2, labels=row_labels, fontsize=FONT_SIZE - 2)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.3, row_y.max() + ROW_AMP + 0.2)
    ax.set_xticks(x_ticks, labels=x_labels if bottom else [''] * len(x_ticks),
                  fontsize=FONT_SIZE - 2)
    if bottom:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.spines[['top', 'right', 'left']].set_visible(False)


def plot_illum(ax, modf):
    # the pulse in red on a black baseline; one sample either side of the pulse stays
    # red so a rectangular pulse keeps red edges
    pulse = to_unit(np.roll(modf, MODF_SHIFT))
    on = pulse > 1e-3
    on |= np.roll(on, 1) | np.roll(on, -1)
    ax.plot([0, N_TBINS - 1], [0, 0], color='k', lw=LW)
    ax.plot(np.where(on, ROW_AMP * pulse, np.nan), color=PULSE_COLOR, lw=LW)
    ax.axvline(MODF_SHIFT, color='k', linestyle=':', lw=1.5, zorder=5)   # emission, t = 0
    ax.set_xlim(0, N_TBINS - 1)


def plot_gates(ax, demodfs, rows):
    n_margin = int(np.ceil(X_MARGIN * N_TBINS))
    row_y = len(rows) - 1 - np.arange(len(rows))   # top row highest
    for y, ki in zip(row_y, rows):
        if ki is None:
            continue
        idx, gate = extend_period(to_unit(demodfs[:, ki]), n_margin)
        ax.fill_between(idx, y, y + ROW_AMP * gate, color=GATE_FILL, lw=0)
        ax.plot(idx, y + ROW_AMP * gate, color='k', lw=LW)
    ax.axvline(0, color='k', linestyle=':', lw=1.5, zorder=5)   # emission, t = 0
    ax.set_xlim(-n_margin, N_TBINS + n_margin)
    return row_y


def plot_correlations(fig, subplot_spec, cm, rows, bottom):
    """One small x-y axis per correlation, so each F_i reads against 0/1 and depth.

    Unlike the gates, correlations are smooth curves with no flat level to anchor
    them, so stacked rows without axes are hard to read. Returns the axes, top first.
    """
    gs = gridspec.GridSpecFromSubplotSpec(len(rows), 1, subplot_spec=subplot_spec, hspace=0.35,
                                          height_ratios=[0.4 if ki is None else 1 for ki in rows])
    axes = []
    for r, ki in enumerate(rows):
        ax = fig.add_subplot(gs[r])
        axes.append(ax)
        if ki is None:
            ax.text(0.5, 0.5, r'$\vdots$', ha='center', va='center', fontsize=FONT_SIZE,
                    transform=ax.transAxes)
            ax.axis('off')
            continue
        ax.plot(to_unit(cm[:, ki]), color='k', lw=LW)   # each correlation scaled to 0-1
        ax.set_xlim(0, N_TBINS - 1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1], labels=['0', '1'], fontsize=FONT_SIZE - 4)
        ax.set_ylabel(rf'$F_{{{ki + 1}}}$', fontsize=FONT_SIZE - 2, rotation=0,
                      labelpad=12, va='center')
        # depth ticks on every row; labels only under the very bottom one
        last = bottom and r == len(rows) - 1
        ax.set_xticks([0, N_TBINS - 1], labels=['0', r'$d_{max}$'] if last else ['', ''],
                      fontsize=FONT_SIZE - 2)
        if last:
            ax.set_xlabel('Depth', fontsize=FONT_SIZE)
        ax.spines[['top', 'right']].set_visible(False)
    return axes


# =========================
# MAIN
# =========================
(_, _, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(N_TBINS, REP_TAU)

n_types = len(TYPES)
k_max = 2 * SHOW_K

fig = plt.figure(figsize=(7 + k_max, 2.5 * n_types))

# 3 main sections (modfs / demodfs / cm) are columns; types are rows
gs_main = gridspec.GridSpec(n_types, 3, figure=fig,
                             width_ratios=[2, k_max, k_max],
                             hspace=0.15, wspace=0.25)

for j, (capture_type, k) in enumerate(zip(TYPES, KS)):
    modfs, demodfs, cm = get_code(capture_type, k, N_TBINS,
                                  pulse_width=1,
                                  shift=1,
                                  gate_width=1)
    bottom = j == n_types - 1
    rows = shown_rows(k)

    # --- col 0: modfs ---
    ax = fig.add_subplot(gs_main[j, 0])
    plot_illum(ax, modfs if modfs.ndim == 1 else modfs[:, 0])
    finish_axis(ax, np.array([0]), [''], [MODF_SHIFT], ['0'], bottom, 'Time')
    ax.set_ylabel(get_string_name(capture_type, None, short=True), fontsize=FONT_SIZE)
    if j == 0:
        ax.set_title('Illum. ' + r'$S(t)$', fontsize=FONT_SIZE, fontweight='bold')

    # --- col 1: demodfs, one row per gate ---
    ax = fig.add_subplot(gs_main[j, 1])
    row_y = plot_gates(ax, demodfs, rows)
    finish_axis(ax, row_y, [r'$\vdots$' if ki is None else rf'$G_{{{ki + 1}}}$' for ki in rows],
                [0, N_TBINS], ['0', r'$\tau$'], bottom, 'Time')
    if j == 0:
        ax.set_title('Gate Functions ' + r'$G(t)$', fontsize=FONT_SIZE, fontweight='bold')

    # --- col 2: coding matrix, one small x-y axis per gate, in the same order as col 1 ---
    if capture_type == 'timeslicing':
        ax = fig.add_subplot(gs_main[j, 2])
        ax.axis('off')
    else:
        ax = plot_correlations(fig, gs_main[j, 2], cm, rows, bottom)[0]
    if j == 0:
        ax.set_title('Correlations ' + r'$F(d)$', fontsize=FONT_SIZE, fontweight='bold')


plt.savefig('figures/coding_schemes.pdf', bbox_inches='tight')
plt.show()
