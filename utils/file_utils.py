import os

import numpy as np
from PIL import Image
from dataclasses import dataclass, fields, asdict, is_dataclass
from typing import Optional, get_origin, get_args, Union
import argparse

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



def save_correlation_data(save_path, cfg, correlations):
    save_name = f'{cfg.capture_type}k{cfg.k}_{cfg.rep_rate * 1e-6:.0f}mhz_{cfg.amplitude * 1000:.0f}mV_{cfg.current:.0f}mA_{cfg.duty:.0f}duty_correlations.npz'
    os.makedirs(save_path, exist_ok=True)
    cfg_dict = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    out_file = os.path.join(save_path, save_name)
    np.savez(out_file, correlations=correlations, cfg=cfg_dict)
    print(f"✅ Saved correlation data to {out_file}.npz")


@dataclass
class Config:
    # Camera
    im_width: Optional[int] = None
    bit_depth: Optional[int] = None

    # Capture
    int_time: Optional[int] = None
    burst_time: Optional[int] = None
    k: Optional[int] = None
    shift: Optional[int] = None
    gate_shrinkage: Optional[int] = None
    capture_type: Optional[str] = None

    # Illumination
    amplitude: Optional[float] = None
    current: Optional[float] = None
    edge: Optional[float] = None
    duty: Optional[int] = None
    rep_rate: Optional[float] = None
    illum_type: Optional[str] = None

    # Plot
    plot_correlations: Optional[bool] = None

    # Save
    save_into_file: Optional[bool] = None
    save_path: Optional[str] = None

    # Parameters from previous parameters
    n_tbins: Optional[int] = None
    rep_tau: Optional[float] = None

    # Non-editable / control
    iterations: Optional[int] = None
    overlap: Optional[int] = None
    timeout: Optional[int] = None
    pileup: Optional[int] = None
    gate_steps: Optional[int] = None
    gate_step_arbitrary: Optional[int] = None
    gate_step_size: Optional[int] = None
    gate_direction: Optional[int] = None
    gate_trig: Optional[int] = None



def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("true", "1", "yes", "y")

def _base_type(annot):
    """
    Optional[int] -> int, Optional[float] -> float, etc.
    """
    origin = get_origin(annot)
    if origin is Union:
        args = [a for a in get_args(annot) if a is not type(None)]
        return args[0] if len(args) == 1 else annot
    return annot

def build_parser_from_config(config_cls, *, bool_parser=str2bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Correlation function capture")

    for f in fields(config_cls):
        t = _base_type(f.type)

        # argparse can't handle type=bool correctly, so use str2bool
        arg_type = bool_parser if t is bool else t

        parser.add_argument(f"--{f.name}", type=arg_type, default=None)

    return parser
