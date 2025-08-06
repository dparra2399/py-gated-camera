import os
import sys 

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from felipe_utils import CodingFunctionsFelipe
from spad_lib import spad512utils
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    n_tbins = 2000
    K = 4

    func = getattr(CodingFunctionsFelipe, f"GetHamK{K}")


    (modfs, demodfs) = func(N=n_tbins)

    gated_demodfs_np, gated_demodfs_arr = spad512utils.decompose_ham_codes(demodfs)

    print(gated_demodfs_np.shape)

    fig, axs = plt.subplots(1, 2)

    # for i in range(gated_demodfs.shape[-1]+1):
    #     if i == 0:
    #         axs[i].plot(demodfs)
    #     else:
    #         axs[i].plot(gated_demodfs[:, i-1])

    axs[0].imshow(np.repeat(demodfs.transpose(), 100, axis=0), aspect='auto', cmap='gray')
    axs[0].set_title(f'Hamiltonian K={demodfs.shape[-1]}')
    axs[1].imshow(np.repeat(gated_demodfs_np.transpose(), 100, axis=0), aspect='auto', cmap='gray')
    axs[1].set_title(f'Decomposed Hamiltonian K={demodfs.shape[-1]}')


    plt.show()