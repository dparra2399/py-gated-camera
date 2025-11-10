from felipe_utils.CodingFunctionsFelipe import *
import numpy as np

(modfs, demodfs) = GetHamK4(1000, 1000)


(modfs_tmp, demodfs_tmp) = GetHamK3(1000)

correlations = np.fft.ifft(np.fft.fft(modfs, axis=0).conj() * np.fft.fft(demodfs, axis=0), axis=0).real

correlations_tmp = np.fft.ifft(np.fft.fft(modfs_tmp, axis=0).conj() * np.fft.fft(demodfs_tmp, axis=0), axis=0).real

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(np.transpose(demodfs), aspect='auto')
axs[1].plot(correlations_tmp)
plt.show()

freq = ['10000000', '0']
NUM_GATES = 3
TAU = ((1 / float(freq[-2])) * 1e12)  # Tau in picoseconds
MHZ = int(float(freq[-2]) * 1e-6)
GATE_WIDTH = math.ceil(((TAU / 2) // NUM_GATES) * 1e-3)

print(TAU * 2)
print(GATE_WIDTH)