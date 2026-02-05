import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, median_filter, gaussian_filter1d
from felipe_utils.tof_utils_felipe import zero_norm_t, norm_t
from scipy.interpolate import interp1d


def plot_gated_images(
        coded_vals_save: np.ndarray,
        depth_map: np.ndarray,
        gt_depth_map: np.ndarray,
        vmin=None,
        vmax=None,
        median_filter_size: int = 3,
                   ) -> None:

    K = min(coded_vals_save.shape[-1], 4)
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(2, K, height_ratios=[1, 1])

    for i in range(K):
        ax1 = fig.add_subplot(gs[0, i])
        ax1.set_title('Gated Image {}'.format(i))
        im1 = ax1.imshow(coded_vals_save[:, :, i], vmin=0, vmax=np.nanmax(coded_vals_save))
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Photon Counts')

    x, y = 20, 170
    width, height = 220, 320
    patch = depth_map[y:y+height, x:x+width]

    vmin_local = np.nanmin(depth_map) if vmin is None else vmin
    vmax_local = np.nanmax(depth_map) if vmax is None else vmax

    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(median_filter(depth_map, size=median_filter_size), vmin=vmin_local, vmax=vmax_local)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Depth (meters)')
    rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='lime', facecolor='none')
    ax2.add_patch(rect)
    ax2.set_title('Full Depth Map')

    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(median_filter(patch, size=median_filter_size), vmin=vmin_local, vmax=vmax_local)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Depth (meters)')
    ax3.set_title('Depth Map Closeup')

    if gt_depth_map is not None:
        ax4 = fig.add_subplot(gs[1, 2])
        im4 = ax4.imshow(gt_depth_map, vmin=vmin_local, vmax=vmax_local)
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Depth (meters)')
        ax4.set_title('Ground Truth')

    fig.subplots_adjust(left=0.05, right=0.98, top=0.97, bottom=0.05, wspace=0.05, hspace=0.05)
    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
    plt.show()
    return



def plot_sample_points(
        coded_vals_save: np.ndarray,
        coding_matrix_save: np.ndarray,
        points: list,
        depth_map: np.ndarray,
        tbin_depth_res: float,
        use_full_correlations: bool = False,
        vmin=None,
        vmax=None,
        median_filter_size: int = 3,
                   ) -> None:

    K = coded_vals_save.shape[-1]
    #while len(points) < K:
    #    points.append((20, 330))

    norm_coded_vals_save = zero_norm_t(coded_vals_save)
    norm_coding_matrix_save = zero_norm_t(coding_matrix_save)

    fig, axs = plt.subplots(2, len(points)+1, figsize=(15, 8))
    axs[1, -1].imshow(median_filter(depth_map, size=median_filter_size), vmin=vmin, vmax=vmax)
    axs[0, -1].set_axis_off()
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, (x, y) in enumerate(points):
        if use_full_correlations:
            axs[1, i].plot(norm_coding_matrix_save[y, x, :, :])
        else:
            axs[1, i].plot(norm_coding_matrix_save)
        for coded_val in norm_coded_vals_save[y, x, :]:
            axs[1, i].plot(depth_map[y, x] / tbin_depth_res, coded_val, 'o', color=colors[i % len(colors)])

        axs[1, i].set_title(f'Correlations at {colors[i % len(colors)]} Dot')
        axs[1, -1].plot(x, y, 'o', color=colors[i % len(colors)])
        axs[1, i].axvline(depth_map[y, x] / tbin_depth_res, linestyle='--', color=colors[i % len(colors)])

        bars = axs[0, i].bar(np.arange(0, K), coded_vals_save[y, x, :], color=colors[i % len(colors)])
        # Label in compact form (e.g., 1.2k instead of 1200)
        axs[0, i].bar_label(
            bars,
            labels=[f"{val/1e3:.1f}k" if val > 999 else f"{val:.0f}" for val in coded_vals_save[y, x, :]],
            padding=2, fontsize=8)
        axs[0, i].ticklabel_format(style="sci", axis="y", scilimits=(3, 3))
        axs[0, i].set_title(f'Intensity Values \n Count = {np.sum(coded_vals_save[y, x, :]) / 1000:.1f}k')

    plt.show()
    return


def plot_sample_points_simple(
        coded_vals_save: np.ndarray,
        coding_matrix_save: np.ndarray,
        points: list,
        depth_map: np.ndarray,
        tbin_depth_res: float,
        use_full_correlations: bool = False,
        colors: list = None,
) -> None:

    norm_coded_vals_save = zero_norm_t(coded_vals_save)
    norm_coding_matrix_save = zero_norm_t(coding_matrix_save)

    fig, axs = plt.subplots(1, 1, figsize=(15, 8))
    if use_full_correlations:
        axs.plot(norm_coding_matrix_save[100, 100, :, :])
    else:
        axs.plot(norm_coding_matrix_save)
    for i, (x, y) in enumerate(points):
        for coded_val in norm_coded_vals_save[y, x, :]:
            axs.plot(depth_map[y, x] / tbin_depth_res, coded_val, 'o', color=colors[i % len(colors)])
        axs.axvline(depth_map[y, x] / tbin_depth_res, linestyle='--', color=colors[i % len(colors)])
    axs.set_title('Correlations with all points plotted')

    plt.show()
    return


def plot_correlation_functions(
        point_list: list,
        correlations: np.ndarray,
        n_tbins = None,
        smooth_sigma: float = None,
):
    average_correlation = np.transpose(np.mean(np.mean(correlations[20:-20, 20:correlations.shape[1]//2, :], axis=0), axis=0))
    if smooth_sigma is not None:
        average_correlation = gaussian_filter1d(average_correlation, sigma=smooth_sigma, axis=0)

    if n_tbins is not None:
        original_len = average_correlation.shape[0]
        f = interp1d(
            np.linspace(0, 1, original_len),
            average_correlation,
            kind='cubic',
            axis=0,
            fill_value='extrapolate'
        )
        average_correlation = f(np.linspace(0, 1, n_tbins))

    fig, axs = plt.subplots(1, len(point_list)+2, figsize=(15, 8))
    axs[-1].imshow(np.sum(correlations[:,:,:,0], axis=-1))
    axs[-1].set_title('Intensity Image')

    axs[-2].plot(average_correlation)
    axs[-2].set_title('Corrfs \n (averaged)')
    colors = ['red', 'blue', 'orange', 'green', 'purple', 'brown']
    correlations_tmp = correlations.swapaxes(-1, -2)
    if smooth_sigma is not None:
        correlations_tmp = gaussian_filter(correlations_tmp, sigma=(1, 1, 1, 0))
    for i, item in enumerate(point_list):
        x, y = item
        if smooth_sigma is not None:
            corr_func = gaussian_filter1d(correlations_tmp[y, x, :, :], sigma=smooth_sigma, axis=0)
        else:
            corr_func = correlations_tmp[y, x, :, :]
        axs[i].plot(corr_func)
        axs[-1].plot(x, y, 'o', color=colors[i % len(colors)])
        axs[i].set_title(f'Corrfs \n ({colors[i % len(colors)]} pixel)')
    plt.show()


def plot_correlation_comparison(
        measured_coding_matrix: np.ndarray,
        coding_matrix: np.ndarray,
):

    zero_mean_coding_matrix = coding_matrix - np.mean(coding_matrix, axis=0, keepdims=True)
    zero_mean_coding_matrix /=  np.max(np.abs(zero_mean_coding_matrix), axis=0, keepdims=True)
    zero_mean_correlations = measured_coding_matrix - np.mean(measured_coding_matrix, axis=0, keepdims=True)
    zero_mean_correlations /= np.max(np.abs(zero_mean_correlations), axis=0, keepdims=True)


    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    axs.plot(zero_mean_correlations, label='Measured')
    axs.plot(zero_mean_coding_matrix, linestyle='dashed', label='Ideal')
    axs.set_title('Measured vs. Ideal Gate Profiles (Correlations)')
    axs.legend()
    plt.show()

def plot_capture_comparison(depths_maps_dict, x=20, y=20, width=220, height=320,
                            vmin=None, vmax=None, normalize_depth_maps=False,
                            median_filter_size=3):
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(len(depths_maps_dict), 4, height_ratios=[1] * len(depths_maps_dict))

    for i, coded_vals_path in enumerate(depths_maps_dict):
        inner_dict = depths_maps_dict[coded_vals_path]

        depth_map = inner_dict['depth_map']
        gt_depth_map = inner_dict['gt_depth_map']
        name = inner_dict['capture_type']

        depth_map_plot = depth_map - np.mean(depth_map) if normalize_depth_maps else np.copy(depth_map)


        patch = depth_map_plot[y: y + height, x: x + width]

        # full map ---------------------------------------------------------
        ax = fig.add_subplot(gs[i, 0])
        im = ax.imshow(
            median_filter(depth_map_plot, size=median_filter_size), vmin=vmin, vmax=vmax
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Depth (meters)")
        ax.set_title(name)
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)

        # close-up ---------------------------------------------------------
        ax2 = fig.add_subplot(gs[i, 1])
        im2 = ax2.imshow(
            median_filter(patch, size=median_filter_size), vmin=vmin, vmax=vmax
        )
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Depth (meters)")
        ax2.set_title("Close-Up")

        # GT / error -------------------------------------------------------
        ax3 = fig.add_subplot(gs[i, 2])
        im3 = ax3.imshow(
            median_filter(gt_depth_map, size=median_filter_size), vmin=vmin, vmax=vmax
        )
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="Ground Truth")
        ax3.set_title("Ground Truth")
        ax3.set_xlabel(f"MAE: {inner_dict['mae'] * 1000: .3f} mm\nRMSE: {inner_dict['rmse'] * 1000: .3f} mm")

        # Error map -------------------------------------------------------
        ax4 = fig.add_subplot(gs[i, 3])
        error_map = np.abs(depth_map - gt_depth_map)
        im4 = ax4.imshow(error_map, vmin=0, vmax=0.5)
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label="Absolute Error (m)")
        ax4.set_title("Error Map")

    fig.subplots_adjust(left=0.05, right=0.98, top=0.97, bottom=0.05, wspace=0.05, hspace=0.05)
    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
    plt.show()


