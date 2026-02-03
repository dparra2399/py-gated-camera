import os

import numpy as np
from PIL import Image
from dataclasses import fields, asdict, is_dataclass
from typing import  get_origin, get_args, Union
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



def save_capture_data(save_path, cfg_dict, coded_vals):

    save_name = make_capture_filename(cfg_dict['capture_type'],cfg_dict['k'], cfg_dict['rep_rate'] * 1e-6,
                              cfg_dict['amplitude'] * 1000, cfg_dict['current'],cfg_dict['duty'])

    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, save_name)
    np.savez(out_file, correlations=coded_vals, cfg=cfg_dict)
    print(f"✅ Saved Capture data to {out_file}.npz")

def save_correlation_data(save_path, cfg_dict, correlations):

    save_name = make_correlation_filename(cfg_dict['capture_type'],cfg_dict['k'], cfg_dict['rep_rate'] * 1e-6,
                              cfg_dict['amplitude'] * 1000, cfg_dict['current'],cfg_dict['duty'])

    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, save_name)
    np.savez(out_file, correlations=correlations, cfg=cfg_dict)
    print(f"✅ Saved correlation data to {out_file}.npz")


def make_filename(capture_type, k, freq_mhz, mV, mA, duty):
    return (
        f"{capture_type}k{k}_"
        f"{freq_mhz:.0f}mhz_"
        f"{mV:.0f}mV_"
        f"{mA:.0f}mA_"
        f"{duty:.0f}duty"
    )

def make_correlation_filename(capture_type, k, freq_mhz, mV, mA, duty):
    return make_filename(capture_type, k, freq_mhz, mV, mA, duty) + '_correlations.npz'

def make_capture_filename(capture_type, k, freq_mhz, mV, mA, duty):
    return make_filename(capture_type, k, freq_mhz, mV, mA, duty) + '_capture.npz'

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
