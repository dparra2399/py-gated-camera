from felipe_utils.CodingFunctionsFelipe import *
import numpy as np
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse

(modfs, demodfs) = GetHamK4(1000)
N = modfs.shape[0]
L = 2 * N - 1

#modfs = modfs + 1000

irf = gaussian_pulse(np.arange(1000), 0, 70, circ_shifted=True)
modfs = np.fft.ifft(np.fft.fft(irf[..., np.newaxis], axis=0).conj() * np.fft.fft(modfs, axis=0), axis=0).real
correlations = np.fft.ifft(np.fft.fft(modfs, axis=0).conj() * np.fft.fft(demodfs, axis=0), axis=0).real

# # Zero-pad both signals to length L
# modfs_padded   = np.pad(modfs,   ((0, L - N), (0, 0)), mode='constant')
# demodfs_padded = np.pad(demodfs, ((0, L - N), (0, 0)), mode='constant')
#
# # Linear convolution using FFT
# correlations = np.fft.ifft(
#     np.fft.fft(demodfs_padded,   axis=0).conj() *
#     np.fft.fft(modfs_padded, axis=0),
#     axis=0
# ).real
# correlations = np.flip(correlations, axis=0)

fig, axs = plt.subplots(nrows=1, ncols=3)
axs[0].imshow(np.repeat(np.transpose(demodfs), 100, axis=0), aspect='auto')
axs[1].imshow(np.repeat(np.transpose(correlations), 100, axis=0), aspect='auto')
axs[2].plot(correlations)
plt.show()
