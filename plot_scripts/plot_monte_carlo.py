import math
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from plot_scripts.plot_utils import get_cap_color, get_string_name

# =============================================================================
# CONFIG
# =============================================================================
FILENAMES = [
    #"/Users/davidparra/PycharmProjects/py-gated-camera/data/monte_carlo_exp/ntbins1000_trials5000_photons300-9000_sbr0.1-10.0_K3_split_ideal_2.npz",
    "/Users/davidparra/PycharmProjects/py-gated-camera/data/monte_carlo_exp/ntbins1000_trials5000_photons300-9000_sbr0.1-10.0_K4_split.npz",

    #"/Users/davidparra/PycharmProjects/py-gated-camera/data/monte_carlo_exp/ntbins1000_trials5000_photons300-9000_sbr0.1-10.0_K8_split.npz"
    #"/Users/davidparra/PycharmProjects/py-gated-camera/data/monte_carlo_exp/ntbins1000_trials5000_photons300-9000_sbr0.1-10.0_Sliding2_split_ideal.npz",
    #"/Users/davidparra/PycharmProjects/py-gated-camera/data/monte_carlo_exp/ntbins1000_trials5000_photons300-9000_sbr0.1-10.0_Sliding2_split.npz"

]

N_TBINS    = 1000     # time bins used in the sweep (from the filename, ntbins*)
METRIC     = 'mae'    # 'mae' or 'rmse'
GRID_SIZE  = 7        # number of tick marks on x/y axes
Z_MAX      = 500     # max z-axis value (mm)

# Slice edges off the results to zoom into the interesting region.
# Set to None to keep the full range on that side.
TRIM_PHOTON_LOW  = 1    # drop this many points from the low-photon end
TRIM_PHOTON_HIGH = None   # drop this many points from the high-photon end
TRIM_SBR_LOW     = 1    # drop this many points from the low-SBR end
TRIM_SBR_HIGH    = None    # drop this many points from the high-SBR end

# =============================================================================
# HELPERS
# =============================================================================
def parse_label(label):
    """Parse 'ham_k3', 'coarse_k4', 'coarsepw_k3_pw50', 'sliding_k5_sh10'
    → (capture_type, k, pw, shift)."""
    parts = str(label).split('_')
    cap_type = parts[0]
    k = None
    pw = None
    shift = None
    for p in parts[1:]:
        if p.startswith('k') and p[1:].isdigit():
            k = int(p[1:])
        if p.startswith('pw'):
            pw = int(float(p[2:]))
        if p.startswith('sh'):
            shift = int(float(p[2:]))
    return cap_type, k, pw, shift


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

    legend_seen = {}  # label -> color, so each entry is added to the legend once

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
            cap_type, k, pw, shift = parse_label(label)
            color = get_cap_color(cap_type, k, shift)
            color = color if color is not None else 'blue'

            print(cap_type, k, pw)


            #
            # if cap_type == 'coarse' or (cap_type == 'coarsepw' and k < 8):
            #    continue




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

            # Surfaces don't appear in the legend, so add a dummy legend-only
            # Scatter3d (renders nothing) once per unique label/color.
            legend_name = get_string_name(cap_type, None, True) + f" (K={k})"
            if pw is not None:
                # Express pulse width as a multiple of the "matched" width.
                if cap_type == 'coarsepw' and k:
                    ref_pw = (N_TBINS // k) / (2 * np.sqrt(np.log(2)))
                    print(pw); print(ref_pw); print(k)#; exit(0)

                    mult = pw / ref_pw
                    legend_name += f' ({mult:.1f}x)'
                else:
                    legend_name += ''#f' PW={pw}'
            if legend_name not in legend_seen:
                legend_seen[legend_name] = color
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=8, color=color, symbol='square'),
                    name=legend_name,
                    showlegend=True,
                ), row=1, col=1)

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
            camera=dict(eye=dict(x=2.0, y=-2.0, z=1.2)),
            bgcolor='white',
        )

    fig.update_layout(
        **scene_layouts,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=10, r=180, t=40, b=10),
        width=650 * n_files + 180,
        height=700,
        showlegend=True,
        legend=dict(
            font=dict(family='serif', size=14, color='black'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='lightgray', borderwidth=1,
            itemsizing='constant',
            x=1.0, y=0.5,
            xanchor='left', yanchor='middle',
        ),
    )
    fig.write_image("figures/monte_carlo_plot.svg", scale=1)
    fig.write_image("figures/monte_carlo_plot.pdf", scale=1)
    pio.renderers.default = 'browser'
    fig.show()
