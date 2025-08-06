import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
from scipy import signal
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)


import socket
import time
from PIL import Image
import numpy as np
import gated_project_code.spad_lib.metadata_constants as metadata_constants



if __name__ == "__main__":
    cap_name = "acq00004"
    frame = 0
    img_num = 1
    name = f"IMG{img_num:05d}"
    base = f"/home/ubilaptop8/2025-Q2-David-P-captures/data/gated_images/{cap_name}"
    img_data = Image.open(f"{base}/{name}.PNG")
    img_data.load()

    metadata = metadata_constants.get_dict_metadata(img_data)

    gate_offset = metadata['Gate offset']
    gate_step_size = metadata['Gate step size']
    gate_steps = metadata['Gate steps']

    print(metadata)

    transient_img = np.zeros((512, 512, gate_steps))

    for i in range(gate_steps):
        name = f"IMG{i:05d}"
        img = Image.open(f"{base}/{name}.PNG")
        img.load()
        metadata_img = metadata_constants.get_dict_metadata(img)
        transient_img[:, :, i] = np.array(img)

    
    gate_starts = np.array([gate_offset + (gate_step_size * (gate_step)) for gate_step in range(gate_steps)])

    fig, axs = plt.subplots(2, figsize=(10, 10))

    x1, y1 = (340, 220)

    axs[0].bar(np.arange(0, metadata['Gate steps']), transient_img[y1, x1, :])
    axs[0].set_xticks(np.arange(0, metadata['Gate steps'])[::3])
    axs[0].set_xticklabels(np.round(gate_starts, 1)[::3])


    axs[1].imshow(np.sum(transient_img, axis=2))
    axs[1].plot(x1, y1, 'ro')
    axs[1].set_title(cap_name)

    plt.show()
    print('done')