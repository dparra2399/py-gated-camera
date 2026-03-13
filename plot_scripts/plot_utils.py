import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.pyplot import legend
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
        shift: int = None,
):
    mins = measured_coding_matrix.min(axis=0, keepdims=True)
    maxs = measured_coding_matrix.max(axis=0, keepdims=True)

    measured_coding_matrix = (measured_coding_matrix - mins) / (maxs - mins)

    mins = coding_matrix.min(axis=0, keepdims=True)
    maxs = coding_matrix.max(axis=0, keepdims=True)

    coding_matrix = (coding_matrix - mins) / (maxs - mins)

    #measured_coding_matrix = zero_norm_t(measured_coding_matrix, axis=-1)
    #coding_matrix = zero_norm_t(coding_matrix, axis=-1)

    if shift is not None:
        for i in range(coding_matrix.shape[-1]):
            coding_matrix[:, i] = np.roll(coding_matrix[:, i], shift)

    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    axs.plot(measured_coding_matrix, label='Measured')
    axs.plot(coding_matrix, linestyle='dashed', label='Ideal')
    axs.set_title('Measured vs. Ideal Gate Profiles (Correlations)')
    axs.legend()
    plt.show()

def plot_capture_comparison(depths_maps_dict, x=20, y=20, width=220, height=320,
                            vmins=None, vmaxs=None, normalize_depth_maps=False,
                            median_filter_size=3):


    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(len(depths_maps_dict), 4, height_ratios=[1] * len(depths_maps_dict))

    for i, coded_vals_path in enumerate(depths_maps_dict):
        if type(vmins) == list:
            assert len(vmins) == len(depths_maps_dict)
            vmin = vmins[i]
        elif vmins is not None:
            vmin = vmins
        else:
            vmin = None

        if type(vmaxs) == list:
            assert len(vmaxs) == len(depths_maps_dict)
            vmax = vmaxs[i]
        elif vmaxs is not None:
            vmax = vmaxs
        else:
            vmax = None


        inner_dict = depths_maps_dict[coded_vals_path]

        depth_map = inner_dict['depth_map']
        gt_depth_map = inner_dict['gt_depth_map']
        name = inner_dict['capture_type']


        depth_map_plot = depth_map - np.mean(depth_map) if normalize_depth_maps else np.copy(depth_map)
        gt_depth_map_plot = gt_depth_map - np.mean(gt_depth_map) if normalize_depth_maps else np.copy(gt_depth_map)


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
            median_filter(gt_depth_map_plot, size=median_filter_size), vmin=vmin, vmax=vmax
        )
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="Ground Truth")
        ax3.set_title("Ground Truth")
        ax3.set_xlabel(f"MAE: {inner_dict['mae'] * 1000: .3f} mm\nRMSE: {inner_dict['rmse'] * 1000: .3f} mm")

        # Error map -------------------------------------------------------
        ax4 = fig.add_subplot(gs[i, 3])
        error_map = np.abs(depth_map - gt_depth_map)
        im4 = ax4.imshow(error_map, vmin=0, vmax=0.1)
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label="Absolute Error (m)")
        ax4.set_title("Error Map")

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    #plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.show()

def plot_single_pixel_dist(depths_dict):
    fig, ax = plt.subplots(figsize=(7, 4))

    colors = ['green', 'red', 'orange', 'blue', 'purple', 'pink']

    keys = list(depths_dict.keys())
    n_methods = len(keys)

    # assume all share same depths
    first = depths_dict[keys[0]]
    gt_depths = first['gt_depths']

    x = np.arange(len(gt_depths))

    width = 0.8 / n_methods  # squish bars together

    all_errors = [
        np.abs(inner['depths'] - inner['gt_depths'])
        for inner in depths_dict.values()
    ]
    ymax = min(np.max(all_errors)+1, 5+1)

    for i, key in enumerate(keys):
        inner = depths_dict[key]

        error = np.abs(inner['depths'] - inner['gt_depths'])

        offset = (i - (n_methods - 1) / 2) * width

        label = (
            f"{inner['capture_type']} | "
            f"MAE={inner['mae'] * 1000:.2f}mm "
            f"RMSE={inner['rmse'] * 1000:.2f}mm"
        )

        bars = ax.bar(
            x + offset,
            error,
            width=width,
            color=colors[i % len(colors)],
            label=label
        )

        # add depth labels above each bar
        for xi, yi, d in zip(x + offset, error, inner['gt_depths']):
            ax.text(
                xi,
                yi + (0.02 * ymax),  # small offset above bar
                f"{d:.2f}",
                ha='center',
                va='bottom',
                rotation=0,
                fontsize=8
            )

    ax.set_ylim(0, ymax)
    ax.set_xticks(np.arange(len(first['phase_shifts'])))
    ax.set_xticklabels(first['phase_shifts'])

    ax.set_xlabel('Phase Shifts')
    ax.set_ylabel('Absolute Error (m)')
    ax.set_title('Single Pixel Absolute Error Comparison')

    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_single_pixel_corr(depths_dict):
    fig, ax = plt.subplots(len(depths_dict), 1, figsize=(7, 4))

    for i, inner_dict in enumerate(depths_dict.values()):

        coding_matrix = inner_dict['coding_matrix']

        tbin_depth_res = inner_dict['tbin_depth_res']

        depths = inner_dict['depths']
        gt_depths = inner_dict['gt_depths']

        depths_plot = (depths / tbin_depth_res).astype(int)
        gt_depths_plot = (gt_depths / tbin_depth_res).astype(int)

        ax[i].plot(coding_matrix)

        for j in range(depths_plot.shape[-1]):
            if j == 0:
                ax[i].axvline(gt_depths_plot[j], linestyle='--', color='blue', label='Ground Truth')
                ax[i].axvline(depths_plot[j],color='red',  label='Estimated')
            else:
                ax[i].axvline(gt_depths_plot[j], linestyle='--', color='blue')
                ax[i].axvline(depths_plot[j],color='red')

            ymax = ax[i].get_ylim()[1]

            if np.abs(depths[j] - gt_depths[j]) < 4:
                ax[i].plot([gt_depths_plot[j], depths_plot[j]],
                           [0.95 * ymax, 0.95 * ymax],
                           color='black',
                           linewidth=2)
        ax[i].legend()
        ax[i].set_title(f"{inner_dict['capture_type']}")


    plt.show()

def plot_single_pixel_depth_pairs(depths_dict):
    keys = list(depths_dict.keys())
    n = len(keys)

    fig, ax = plt.subplots(n, 1, figsize=(7, 3.2 * n), squeeze=False)

    for i, key in enumerate(keys):
        inner = depths_dict[key]

        gt = np.asarray(inner['gt_depths'])   # (N,)
        est = np.asarray(inner['depths'])     # (N,)
        phase_shifts = inner['phase_shifts']  # (N,) labels

        x = np.arange(len(gt))
        width = 0.35  # two bars next to each other

        # (optional) set y limits per subplot, or compute global if you want
        ymax = max(gt.max(), est.max()) + 0.05 * max(gt.max(), est.max(), 1e-12)

        a = ax[i, 0]
        bars_gt = a.bar(x - width/2, gt, width=width, label="GT")
        bars_est = a.bar(x + width/2, est, width=width, label="Estimated")

        # labels on bars (depth values)
        for b in bars_gt:
            y = b.get_height()
            a.text(b.get_x() + b.get_width()/2, y + 0.02 * ymax, f"{y:.2f}",
                   ha="center", va="bottom", fontsize=8)

        for b in bars_est:
            y = b.get_height()
            a.text(b.get_x() + b.get_width()/2, y + 0.02 * ymax, f"{y:.2f}",
                   ha="center", va="bottom", fontsize=8)

        a.set_ylim(0, ymax)
        a.set_xticks(x)
        a.set_xticklabels(phase_shifts)
        a.set_xlabel("Phase Shifts")
        a.set_ylabel("Depth (m)")

        title = inner.get("capture_type", str(key))
        a.set_title(title)

        a.legend()

    plt.tight_layout()
    plt.show()

def plot_results_summary(results):
    fig, axs = plt.subplots(len(results), 3, figsize=(13, 4 * len(results)), squeeze=False)

    for i, r in enumerate(results):
        cm = r["coding_matrix"]
        name = r["name"]
        rmse = r["rmse"]
        mae = r["mae"]

        # ---- heatmap ----
        axs[i, 0].imshow(np.repeat(cm.T, 100, axis=0), aspect="auto")
        axs[i, 0].set_title(f"{name} Coding Matrix")

        # ---- line plot ----
        axs[i, 1].plot(cm)
        axs[i, 1].set_title(f"{name} Coding Matrix")

        # ---- RMSE bar ----
        axs[-2, 2].bar([i], rmse)
        axs[-2, 2].text(i, rmse, f"{rmse:.2f}", ha="center", va="bottom")
        axs[-2, 2].set_title("RMSE (Lower is better)")
        axs[-2, 2].set_ylabel("RMSE (mm)")

        # ---- MAE bar ----
        axs[-1, 2].bar([i], mae)
        axs[-1, 2].text(i, mae, f"{mae:.2f}", ha="center", va="bottom")
        axs[-1, 2].set_title("MAE (Lower is better)")
        axs[-1, 2].set_ylabel("MAE (mm)")

    names = [r["name"] for r in results]
    axs[-1, 2].set_xticks(np.arange(len(names)))
    axs[-1, 2].set_xticklabels(names)

    plt.subplots_adjust(wspace=0.5, hspace=0.4)
    plt.show()

def plot_correlations_one_plot(results):
    correlations_p =  [dic['coding_matrix'] for dic in results]
    for idx, corr in enumerate(correlations_p):
        name = results[idx]['name']
        if name == 'coarse':
            corr = np.roll(corr, 300, axis=0)
            plt.plot(corr, color='r')
        else:
            plt.plot(corr, color='b')
    plt.show()

def plot_coding_curve(results):
    correlations_p =  [dic['coding_matrix'] for dic in results]
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab10")
    type_colors = {}  # map capture type -> color
    for idx, corr in enumerate(correlations_p):  # (n_tbins, 3)
        name = results[idx]['name']

        mins = corr.min(axis=0, keepdims=True)
        maxs = corr.max(axis=0, keepdims=True)

        corr = (corr - mins) / (maxs - mins)

        if name not in type_colors:
            type_colors[name] = cmap(len(type_colors))

        color = type_colors[name]

        diffs = np.diff(corr, axis=0)
        distance = np.linalg.norm(diffs, axis=1).sum()

        ax.plot(
            corr[:, 0], corr[:, 1], corr[:, 2],
            color=color,
            label=f"{name} ({distance:.2f})"
        )

        ax.text(corr[0, 0], corr[0, 1], corr[0, 2], f"{distance:.2f}", color=color)

        print(f"{name} correlation distance: {distance:.3f}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.legend(fontsize=8)
    plt.show()

def plot_coding_error(results):
    correlations_p =  [dic['coding_matrix'] for dic in results]
    fig, axs = plt.subplots(1, 1, figsize=(9, 7))

    cmap = plt.get_cmap("tab10")
    type_colors = {}  # map capture type -> color
    for idx, corr in enumerate(correlations_p):  # (n_tbins, 3)
        name = results[idx]['name']

        mins = corr.min(axis=0, keepdims=True)
        maxs = corr.max(axis=0, keepdims=True)

        coding_curve = (corr - mins) / (maxs - mins)

        if name not in type_colors:
            type_colors[name] = cmap(len(type_colors))

        color = type_colors[name]

        diffs = np.diff(coding_curve, axis=0)
        distance = np.linalg.norm(diffs, axis=1).sum()

        errors = []
        for i in range(corr.shape[0]):
            errors.append(np.sum(np.sqrt(corr[i, :])) / distance)

        axs.plot(errors, color=color, label=name)
        axs.set_xlabel("Depth")
        axs.set_ylabel("Upper Bound Error")
        axs.set_title("Upper Bound Error for {}".format(name))


    plt.legend(fontsize=8)
    plt.show()
