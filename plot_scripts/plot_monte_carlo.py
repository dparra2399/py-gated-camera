import math
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from plot_scripts.plot_utils import get_cap_color

# =============================================================================
# CONFIG
# =============================================================================
FILENAMES = [
   "/Users/davidparra/PycharmProjects/py-gated-camera/data/monte_carlo_exp/ntbins1000_trials5000_photons300-3000_sbr0.1-10.0_ham_k4_coarse_k4_trapcoarse_k4.npz"
]

METRIC     = 'mae'    # 'mae' or 'rmse'
GRID_SIZE  = 7        # number of tick marks on x/y axes
Z_MAX      = 500     # max z-axis value (mm)

# Slice edges off the results to zoom into the interesting region.
# Set to None to keep the full range on that side.
TRIM_PHOTON_LOW  = 1    # drop this many points from the low-photon end
TRIM_PHOTON_HIGH = 1    # drop this many points from the high-photon end
TRIM_SBR_LOW     = 1    # drop this many points from the low-SBR end
TRIM_SBR_HIGH    = 1    # drop this many points from the high-SBR end

# =============================================================================
# HELPERS
# =============================================================================
def parse_label(label):
    """Parse 'ham_k3', 'coarse_k4', 'coarsepw_k3_pw50' → (capture_type, k)."""
    parts = str(label).split('_')
    cap_type = parts[0]
    k = None
    for p in parts[1:]:
        if p.startswith('k') and p[1:].isdigit():
            k = int(p[1:])
            break
    return cap_type, k


def label_to_color(label):
    cap_type, k = parse_label(label)
    color = get_cap_color(cap_type, k)
    return color if color is not None else '#888888'

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    n_files = len(FILENAMES)
    n_cols  = min(2, n_files)
    n_rows  = math.ceil(n_files / n_cols)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=[[{'type': 'surface'}] * n_cols] * n_rows,
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
        subplot_titles=[f.split('/')[-1] for f in FILENAMES],
    )

    for idx, filename in enumerate(FILENAMES):
        data = np.load(filename, allow_pickle=True)

        mae_results  = data['mae_results']   # (n_runs, n_photons, n_sbrs)
        rmse_results = data['rmse_results']
        photon_counts = data['photon_counts']
        sbrs          = data['sbrs']
        run_labels    = data['run_labels']

        results = mae_results if METRIC == 'mae' else rmse_results

        # trim edges on photon (axis 1) and sbr (axis 2)
        p_lo = TRIM_PHOTON_LOW  or 0
        p_hi = -(TRIM_PHOTON_HIGH) if TRIM_PHOTON_HIGH else None
        s_lo = TRIM_SBR_LOW     or 0
        s_hi = -(TRIM_SBR_HIGH)  if TRIM_SBR_HIGH  else None

        results       = results[:, p_lo:p_hi, s_lo:s_hi]
        photon_counts = photon_counts[p_lo:p_hi]
        sbrs          = sbrs[s_lo:s_hi]

        X = np.log10(photon_counts)
        Y = np.log10(sbrs)

        row = idx // n_cols + 1
        col = idx % n_cols + 1

        for j, label in enumerate(run_labels):
            color = label_to_color(label)
            fig.add_trace(go.Surface(
                z=results[j],
                x=X,
                y=Y,
                surfacecolor=np.ones_like(results[j]),
                colorscale=[[0, color], [1, color]],
                cmin=0, cmax=1,
                name=str(label),
                showscale=False,
                contours=dict(
                    x=dict(show=True, color='#4d4d4d', width=2),
                    y=dict(show=True, color='#4d4d4d', width=2),
                ),
            ), row=row, col=col)

    # axis ticks based on last loaded file
    xticks = np.round(np.linspace(np.min(X), np.max(X), num=GRID_SIZE), 2)
    yticks = np.round(np.linspace(np.min(Y), np.max(Y), num=GRID_SIZE), 2)

    gap   = 0.05
    width = (1 - (n_files - 1) * gap) / n_files

    scene_layouts = {}
    for i in range(1, n_files + 1):
        scene_key = 'scene' if i == 1 else f'scene{i}'
        start = (i - 1) * (width + gap)
        end   = start + width
        scene_layouts[scene_key] = dict(
            domain=dict(x=[start, end]),
            xaxis=dict(
                title=dict(text='Log Photon Count', font=dict(family='serif', size=20, color='black')),
                tickmode='array', tickvals=xticks,
                ticktext=[f'{v:.1f}' for v in xticks],
                tickfont=dict(family='serif', size=13, color='black'),
                showgrid=True, gridcolor='lightgray', backgroundcolor='white',
            ),
            yaxis=dict(
                title=dict(text='Log SBR', font=dict(family='serif', size=20, color='black')),
                tickmode='array', tickvals=yticks,
                ticktext=[f'{v:.1f}' for v in yticks],
                tickfont=dict(family='serif', size=13, color='black'),
                showgrid=True, gridcolor='lightgray', backgroundcolor='white',
            ),
            zaxis=dict(
                title=dict(text=f'{METRIC.upper()} (mm)', font=dict(family='serif', size=20, color='black')),
                tickfont=dict(family='serif', size=13, color='black'),
                showgrid=True, gridcolor='lightgray', backgroundcolor='white',
                range=[0, Z_MAX],
            ),
            camera=dict(eye=dict(x=2.0, y=-2.0, z=0.7)),
            bgcolor='white',
        )

    fig.update_layout(
        **scene_layouts,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=10, r=10, t=40, b=10),
        width=650 * n_files,
        height=700,
    )
    fig.write_image("monte_carlo_plot.svg")
    pio.renderers.default = 'browser'
    fig.show()
