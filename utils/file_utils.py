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

def save_capture_data(
    save_path,
    save_name,
    total_time,
    im_width,
    bit_depth,
    n_tbins,
    iterations,
    overlap,
    timeout,
    pileup,
    gate_steps,
    gate_step_arbitrary,
    gate_step_size,
    gate_direction,
    gate_trig,
    freq,
    voltage,
    coded_vals,
    split_measurements,
    size,
    gate_width,
    K,
    laser_freq = None,
):
    """Save SPAD capture data and metadata to a .npz file."""
    os.makedirs(save_path, exist_ok=True)
    np.savez(
        os.path.join(save_path, save_name),
        total_time=total_time,
        im_width=im_width,
        bitDepth=bit_depth,
        n_tbins=n_tbins,
        iterations=iterations,
        overlap=overlap,
        timeout=timeout,
        pileup=pileup,
        gate_steps=gate_steps,
        gate_step_arbitrary=gate_step_arbitrary,
        gate_step_size=gate_step_size,
        gate_direction=gate_direction,
        gate_trig=gate_trig,
        freq=freq,
        voltage=voltage,
        coded_vals=coded_vals,
        split_measurements=split_measurements,
        size=size,
        gate_width=gate_width,
        K = K,
        laser_freq=laser_freq,
    )
    print(f"✅ Saved capture data to {os.path.join(save_path, save_name)}.npz")



def save_correlation_data(
    save_path,
    save_name,
    total_time,
    im_width,
    bit_depth,
    n_tbins,
    iterations,
    overlap,
    timeout,
    pileup,
    gate_steps,
    gate_step_arbitrary,
    gate_step_size,
    gate_direction,
    gate_trig,
    freq,
    voltage,
    size,
    gate_width,
    K,
    correlations,
):
    """Save SPAD capture data and metadata to a .npz file."""
    os.makedirs(save_path, exist_ok=True)
    np.savez(
        os.path.join(save_path, save_name),
        total_time=total_time,
        im_width=im_width,
        bitDepth=bit_depth,
        n_tbins=n_tbins,
        iterations=iterations,
        overlap=overlap,
        timeout=timeout,
        pileup=pileup,
        gate_steps=gate_steps,
        gate_step_arbitrary=gate_step_arbitrary,
        gate_step_size=gate_step_size,
        gate_direction=gate_direction,
        gate_trig=gate_trig,
        freq=freq,
        voltage=voltage,
        size=size,
        gate_width=gate_width,
        K=K,
        correlations=correlations,
    )
    print(f"✅ Saved correlation data to {os.path.join(save_path, save_name)}.npz")


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("true", "1", "yes", "y")