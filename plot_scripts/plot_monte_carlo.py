import os
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# =============================================================================
# CONFIG
# =============================================================================
NPZ_PATH = "/Users/davidparra/PycharmProjects/py-gated-camera/data/monte_carlo_exp/ntbins999_trials100_photons300-3000_sbr0.1-10.0_ham_k3_coarse_k3_trapcoarse_k3.npz"

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    data = np.load(NPZ_PATH, allow_pickle=True)

    mae_results   = data['mae_results']    # (n_runs, n_photons, n_sbrs)
    rmse_results  = data['rmse_results']
    photon_counts = data['photon_counts']
    sbrs          = data['sbrs']
    run_labels    = data['run_labels']

    X, Y = np.meshgrid(np.log10(photon_counts), np.log10(sbrs), indexing='ij')

    colors = ['green', 'orange', 'red', 'blue', 'purple']
    fig = go.Figure()

    for idx, label in enumerate(run_labels):
        fig.add_trace(go.Surface(
            x=X, y=Y, z=mae_results[idx],
            name=str(label),
            showscale=False,
            colorscale=[[0, colors[idx % len(colors)]], [1, colors[idx % len(colors)]]],
        ))

    fig.update_layout(
        title="MAE vs Photon Count & SBR",
        scene=dict(
            xaxis_title='log10(photons)',
            yaxis_title='log10(SBR)',
            zaxis_title='MAE (mm)',
            #zaxis=dict(range=[0, 300]),
        ),
        width=800,
        height=600,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    pio.renderers.default = 'browser'
    fig.show()
