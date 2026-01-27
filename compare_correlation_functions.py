# Python imports
# Library imports
import os

from IPython.core import debugger

from felipe_utils import tof_utils_felipe
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from felipe_utils.CodingFunctionsFelipe import *
from utils.global_constants import *
from felipe_utils import CodingFunctionsFelipe
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from utils.file_utils import get_data_folder
from utils.tof_utils import build_coding_matrix_from_correlations

#matplotlib.use('QTkAgg')
breakpoint = debugger.set_trace

photon_count = 3000
sbr = 1.0
trials = 100
n_tbins = 1998
simulated_correlations = False

smooth_sigma = 1

shift = 110

depths = np.arange(200, n_tbins-200, 1)

coarse_filename = 'coarsek3_5mhz_correlations.npz'
ham_filename = 'hamk3_5mhz_correlations.npz'


if __name__ == "__main__":
    filenames = [coarse_filename, ham_filename]
    rng = np.random.default_rng()
    names = []

    fig, axs = plt.subplots(len(filenames), 4, figsize=(13, 8), squeeze=False)

    for i, filename in enumerate(filenames):
        folder = get_data_folder(READ_PATH_CORRELATIONS_MAC, READ_PATH_CORRELATIONS_WINDOWS)
        file = np.load(os.path.join(folder, filename), allow_pickle=True)
        cfg = file['cfg'].item()
        correlations_total = file['correlations']
        name = cfg['capture_type']
        (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = tof_utils_felipe.calc_tof_domain_params(n_tbins, cfg['rep_tau'])


        corr_path = (
            os.path.join(folder,
                         f"{cfg['capture_type']}k{cfg['k']}_{cfg['rep_rate'] * 1e-6:.0f}mhz_correlations.npz")
        )
        coding_matrix = build_coding_matrix_from_correlations(correlations_total, False,
                                                              smooth_sigma, shift, n_tbins)

        if 'ham' in filename and simulated_correlations:
            func = getattr(CodingFunctionsFelipe, f"GetHamK{cfg['k']}")
            (modfs, demodfs) = func(N=n_tbins)
            irf = gaussian_pulse(np.arange(n_tbins), 0, 40, circ_shifted=True)
            modfs = np.fft.ifft(np.fft.fft(irf[..., np.newaxis], axis=0).conj() * np.fft.fft(modfs, axis=0), axis=0).real
            coding_matrix = np.fft.ifft( np.fft.fft( modfs, axis=0 ).conj() * np.fft.fft( demodfs, axis=0 ), axis=0 ).real
            coding_matrix =  np.fft.ifft( np.fft.fft(irf[..., np.newaxis], axis=0 ).conj() * np.fft.fft( coding_matrix, axis=0 ), axis=0 ).real
        elif 'coarse' in filename and simulated_correlations:
            coding_matrix = np.kron(np.eye(cfg['k']), np.ones((1, n_tbins // cfg['k'])))
            irf = gaussian_pulse(np.arange(coding_matrix.shape[-1]), 0, n_tbins // (cfg['k']+2), circ_shifted=True)
            coding_matrix = np.fft.ifft(np.fft.fft(irf[..., np.newaxis], axis=0).conj() * np.fft.fft(np.transpose(coding_matrix), axis=0),
                                        axis=0).real


        coding_matrix /= np.max(np.abs(coding_matrix), axis=0, keepdims=True)

        # if 'ham' in filename:
        #    coding_matrix[:, 1] = np.roll(coding_matrix[:, 1], 20)
        #    coding_matrix[:, 0] = np.roll(coding_matrix[:, 0], 20)


        clean_coded_vals = coding_matrix[depths, :]

        clean_coded_vals = (clean_coded_vals / np.sum(clean_coded_vals, axis=-1, keepdims=True)) * photon_count
        clean_coded_vals = clean_coded_vals + ((photon_count / sbr) / cfg['k'])

        coded_vals = rng.poisson(clean_coded_vals, size=(trials, clean_coded_vals.shape[0], clean_coded_vals.shape[1]))
        #coded_vals = clean_coded_vals[np.newaxis, ...]

        norm_coding_matrix = tof_utils_felipe.zero_norm_t(coding_matrix)

        norm_coded_vals = tof_utils_felipe.zero_norm_t(coded_vals)

        zncc = np.matmul(norm_coding_matrix, norm_coded_vals[..., np.newaxis]).squeeze(-1)

        decoded_depth = np.argmax(zncc, axis=-1)


        axs[i, 0].imshow(np.repeat(np.transpose(coding_matrix), 100, axis=0), aspect='auto')
        axs[i, 0].set_title('Coding Matrix')
        axs[i, 1].plot(coding_matrix - np.mean(coding_matrix, axis=0, keepdims=True))
        axs[i, 1].set_title('Coding Matrix')
        axs[i, 2].plot(zncc[0, 0, :])
        axs[i, 2].axvline(x=depths[0], c='r', linestyle='--')
        #axs[i, 3].axvline(x=decoded_depth[0, 0], c='g', linestyle='--')
        axs[i, 2].set_title('ZNCC Reconstruction')
        if i < len(filenames) - 2:
            axs[i, 3].set_axis_off()

        axs[-2, 3].bar([i],np.sqrt(np.mean((decoded_depth - depths) ** 2)), label=name)
        axs[-2, 3].text(
            i,  # x position
            np.sqrt(np.mean((decoded_depth - depths) ** 2)),  # y position (height of bar)
            f"{np.sqrt(np.mean((decoded_depth - depths) ** 2)):.2f}",  # displayed text, formatted
            ha='center', va='bottom'
        )
        axs[-2, 3].set_title('RMSE (Lower is better)')
        axs[-2, 3].set_ylabel('RMSE (time bins)')

        axs[-1, 3].bar([i], np.mean(np.abs(decoded_depth - depths)), label=name)
        axs[-1, 3].text(
            i,  # x position
            np.mean(np.abs(decoded_depth - depths)),  # y position (height of bar)
            f"{np.mean(np.abs(decoded_depth - depths)):.2f}",  # displayed text, formatted
            ha='center', va='bottom'
        )
        axs[-1, 3].set_title('MAE (Lower is better)')
        axs[-1, 3].set_ylabel('MAE (time bins)')

        names.append(name)




        print(f'{name}: \n\t MAE: {np.mean(np.abs(decoded_depth - depths)): .3f} \n\t RMSE: {np.sqrt(np.mean((decoded_depth - depths) ** 2)): .3f}')

    axs[-1, 3].set_xticks(np.arange(0, len(names)))
    axs[-1, 3].set_xticklabels(names)
    plt.subplots_adjust(wspace=0.5, hspace=0.4)
    plt.show()


