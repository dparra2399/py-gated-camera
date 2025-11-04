import os
import numpy as np
from PIL import Image

def get_data_folder(data_folder_mac, data_folder_linux) -> str:
    if os.path.exists(data_folder_mac):
        return data_folder_mac
    return data_folder_linux

def load_hot_mask(path: str, threshold: int = 5000) -> np.ndarray:
    hot_mask = np.array(Image.open(path))
    hot_mask[hot_mask < threshold] = 0
    hot_mask[hot_mask > 0] = 1
    return hot_mask

def filter_npz_files(npz_files, k_list):
    filtered = []
    for f in npz_files:
        base_first = os.path.basename(f).split("_")[0]
        keep = any(
            (str(val) in base_first) or ("_gt_" in base_first and str(val) in base_first)
            for val in k_list
        )
        if keep:
            filtered.append(f)
    return filtered


def load_correlations_file(path: str):
    f = np.load(path)
    return f["correlations"], int(f["n_tbins"])

def get_scheme_name(path: str, K: int) -> str:
    if 'coarse' in path:
        name = f"Coarsek{K}"
    elif 'ham' in path:
        name = f"Hamk{K}"
    else:
        assert False, 'Path needs to be "coarse" or "ham"'
    if 'split' in path:
        name += '_Split'
    elif 'pulse' in path:
        name += '_Pulsed'
    elif 'gt' in path:
        name += '_GroundTruth'
    return name