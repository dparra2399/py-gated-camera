import os
import re
import numpy as np
from PIL import Image
from dataclasses import fields, asdict, is_dataclass
from typing import  get_origin, get_args, Union
import argparse
import zipfile
import shutil
import atexit

def get_data_folder(data_folder_mac, data_folder_linux) -> str:
    if os.path.exists(data_folder_mac):
        return data_folder_mac
    return data_folder_linux

def load_hot_mask(path: str) -> np.ndarray:
    hot_mask = np.load(path)
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


def filter_capture_files(npz_files):
    filtered = []
    for f in npz_files:
        base_first = os.path.basename(f)
        keep = "gt" not in base_first
        if keep:
            filtered.append(f)
    return filtered

def load_correlation_npz(path: str):
    zip_path = os.path.splitext(path)[0] + '.zip'

    if os.path.exists(path):
        try:
            return load_npz(path)   # lazy; auto-recovers from a Bad CRC-32 on read
        except (zipfile.BadZipFile, ValueError, OSError, EOFError):
            # The .npz can't even be opened (e.g. a damaged zip directory). Fall
            # back to a pristine .zip if we have one; otherwise it's broken.
            if not os.path.exists(zip_path):
                raise

    if os.path.exists(zip_path):
        extract_dir = os.path.dirname(path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        atexit.register(os.remove, path)
        return load_npz(path)

    raise FileNotFoundError(f'Correlation file not found as .npz or .zip: {path}')

def recover_npz(path, repair=True, backup=True):
    """Recover arrays from a .npz that fails a CRC check (``BadZipFile: Bad CRC-32``).

    An ``.npz`` is a ZIP of ``.npy`` members; numpy verifies each member's
    CRC-32 when the array is read and raises ``zipfile.BadZipFile`` on a
    mismatch, throwing the bytes away even when they are actually intact. That
    is the common case here: the members are stored uncompressed, so a bad CRC
    usually means only the checksum / a few bytes were damaged by a write or
    copy that didn't finish atomically (a stopped run, a partial sync between
    machines) -- the array itself is still sitting there whole.

    Reads every member with the CRC check disabled and returns a dict mapping
    array name (without the ``.npy`` suffix) -> ``np.ndarray``.

    repair=True  also rewrites a clean, valid ``.npz`` in place so future
                 ``np.load()`` calls succeed normally.
    backup=True  first renames the corrupt original to ``<path>.corrupt`` so
                 nothing is lost (ignored when ``repair=False``).

    Raises RuntimeError if a member cannot be read at all (e.g. a compressed
    member whose deflate stream is genuinely damaged); in that case nothing is
    overwritten, so the original file is left untouched.
    """
    recovered = {}
    failed = {}

    # numpy validates member CRCs inside ZipExtFile._update_crc; disable it so a
    # bad checksum doesn't discard otherwise-readable bytes. Always restored.
    orig_update_crc = zipfile.ZipExtFile._update_crc
    zipfile.ZipExtFile._update_crc = lambda self, newdata: None
    try:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                key = name[:-4] if name.endswith('.npy') else name
                try:
                    with zf.open(name) as fp:            # stream: no full-file buffer
                        recovered[key] = np.lib.format.read_array(fp, allow_pickle=True)
                except Exception as e:                   # e.g. zlib error on a damaged stream
                    failed[key] = e
    finally:
        zipfile.ZipExtFile._update_crc = orig_update_crc

    if failed:
        detail = ', '.join(f'{k}: {type(e).__name__}: {e}' for k, e in failed.items())
        if repair:
            raise RuntimeError(
                f'Could not fully recover {path}; refusing to repair with missing '
                f'members ({detail}). Recovered: {list(recovered)}')
        print(f'[recover_npz] WARNING: {len(failed)} unreadable member(s) in {path}: {detail}')

    if repair and recovered:
        tmp = path + '.recovered'
        np.savez(tmp, **recovered)                       # np.savez appends .npz
        tmp = tmp + '.npz'
        with np.load(tmp, allow_pickle=True) as check:   # force the CRC check that was failing
            for key in recovered:
                _ = check[key]
        if backup:
            os.replace(path, path + '.corrupt')
        os.replace(tmp, path)

    return recovered

class _RecoveringNpz:
    """NpzFile-like wrapper that auto-repairs a corrupt .npz on member access.

    ``np.load`` opens an .npz lazily, so a ``Bad CRC-32`` only fires when a
    member is actually read. On that failure this recovers the file in place
    with :func:`recover_npz` and retries the read, so a corrupt-but-readable
    archive self-heals instead of crashing the caller.
    """
    def __init__(self, path, allow_pickle=True):
        self._path = path
        self._allow_pickle = allow_pickle
        self._npz = np.load(path, allow_pickle=allow_pickle)

    def __getitem__(self, key):
        try:
            return self._npz[key]
        except (zipfile.BadZipFile, ValueError, OSError, EOFError):
            print(f'[load_npz] Bad CRC in {self._path}; recovering in place...')
            self._npz.close()
            recover_npz(self._path, repair=True, backup=True)
            self._npz = np.load(self._path, allow_pickle=self._allow_pickle)
            return self._npz[key]

    @property
    def files(self):
        return self._npz.files

    def keys(self):
        return self._npz.files

    def __contains__(self, key):
        return key in self._npz.files

    def close(self):
        self._npz.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

def load_npz(path, allow_pickle=True):
    """Drop-in for ``np.load(path, allow_pickle=...)`` on .npz archives that
    auto-recovers from a ``Bad CRC-32``.

    Returns an object indexed by member name (``obj['coded_vals']``); if that
    read hits a CRC error the file is repaired in place (see :func:`recover_npz`)
    and the read retried, so an interrupted-write corruption self-heals.
    """
    return _RecoveringNpz(path, allow_pickle=allow_pickle)

def save_npz_atomic(path, **arrays):
    """Save an .npz atomically: write a temp file, then ``os.replace`` it into
    place. A stopped/killed run can't leave a half-written .npz that later fails
    a CRC check. ``path`` may be given with or without the .npz extension.
    """
    final = path if path.endswith('.npz') else path + '.npz'
    tmp = final + '.tmp.npz'          # ends in .npz so np.savez writes it verbatim
    np.savez(tmp, **arrays)
    os.replace(tmp, final)            # atomic swap into place
    return final

def get_capture_folder(path, delete_unzipped=True, return_cleanup=False):
    """Locate (and if needed unzip) a capture folder.

    By default the extracted folder is removed at program exit via atexit,
    which keeps every unzipped archive on disk until the process ends.

    Pass return_cleanup=True to instead get back (capture_folder, cleanup_dir):
    cleanup_dir is the extracted folder (or None if `path` was already a real
    directory), and the caller is responsible for shutil.rmtree-ing it as soon
    as it is done reading. This caps peak disk usage at one unzipped archive.
    """
    if os.path.isdir(path):
        return (path, None) if return_cleanup else path

    if zipfile.is_zipfile(path + ".zip"):
        path = path + ".zip"
        unzip_folder = os.path.splitext(path)[0]
        os.makedirs(unzip_folder, exist_ok=True)

        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(unzip_folder)

        if delete_unzipped and not return_cleanup:
            atexit.register(shutil.rmtree, unzip_folder, ignore_errors=True)

        inner_folder = os.path.join(unzip_folder, os.path.basename(unzip_folder))
        capture_folder = inner_folder if os.path.isdir(inner_folder) else unzip_folder

        return (capture_folder, unzip_folder) if return_cleanup else capture_folder

    raise FileNotFoundError(f'Capture path is not a directory or zip file: {path}')

def save_capture_and_gt_data(save_path, cfg_dict, coded_vals, gt_coded_vals):
    save_path = os.path.join(save_path, cfg_dict['exp_path']) \
            if cfg_dict['exp_path'] is not None else make_next_exp_folder(save_path)

    cfg_dict['ground_truth'] = False
    save_capture_data(save_path=save_path, cfg_dict=cfg_dict, coded_vals=coded_vals)
    if gt_coded_vals is not None:
        cfg_dict['ground_truth'] = True
        cfg_dict['int_time'] = cfg_dict['ground_truth_int_time']
        save_capture_data(save_path=save_path, cfg_dict=cfg_dict, coded_vals=gt_coded_vals)


def save_capture_data(save_path, cfg_dict, coded_vals):
    save_name = make_capture_filename(cfg_dict['capture_type'],cfg_dict['k'], cfg_dict['rep_rate'] * 1e-6,
                              cfg_dict['high_level_amplitude'] * 1000, cfg_dict['current'],cfg_dict['duty'],
                              cfg_dict['int_time'], cfg_dict['ground_truth'])

    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, save_name)
    save_npz_atomic(out_file, coded_vals=coded_vals, cfg=cfg_dict)
    print(f"✅ Saved Capture data to {out_file}")

def save_correlation_data(save_path, cfg_dict, correlations):

    save_name = make_correlation_filename(cfg_dict['capture_type'],cfg_dict['k'], cfg_dict['rep_rate'] * 1e-6,
                              cfg_dict['high_level_amplitude'] * 1000, cfg_dict['current'],cfg_dict['duty'])

    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, save_name)
    save_npz_atomic(out_file, correlations=correlations, cfg=cfg_dict)
    print(f"✅ Saved correlation data to {out_file}")


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

def make_capture_filename(capture_type, k, freq_mhz, mV, mA, duty, int_time, ground_truth):
    ground_truth_tag = '_gt' if ground_truth else f'_{int_time:.0f}ms'
    return (make_filename(capture_type, k, freq_mhz, mV, mA, duty) +
            ground_truth_tag +
            '_capture.npz')

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


def corr_parse_run(s: str):
    cap, k, f, mv, ma, duty, simulated_correlations = s.split(",")
    return dict(
        capture_type=cap,
        k=int(k),
        freq_mhz=float(f),
        mV=float(mv),
        mA=float(ma),
        duty=float(duty),
        simulated_correlations=str2bool(simulated_correlations),
    )

def capture_parse_run(s: str):
    cap, k, f, mv, ma, duty, int_time = s.split(",")
    return dict(
        capture_type=cap,
        k=int(k),
        freq_mhz=float(f),
        mV=float(mv),
        mA=float(ma),
        duty=float(duty),
        int_time=int(int_time),
    )

def capture_phase_shifts(s: str):
    phase_shifts = s.split(",")
    phase_shifts = [int(s) for s in phase_shifts]
    return phase_shifts


def make_next_exp_folder(base_dir, prefix="exp"):
    """
    Creates:
        exp_0, exp_1, exp_2, ...

    Returns the newly created folder path.
    """
    os.makedirs(base_dir, exist_ok=True)

    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    pattern = re.compile(rf"{prefix}_(\d+)$")

    nums = []
    for d in existing:
        m = pattern.match(d)
        if m:
            nums.append(int(m.group(1)))

    next_idx = 0 if len(nums) == 0 else max(nums) + 1

    new_path = os.path.join(base_dir, f"{prefix}_{next_idx}")
    os.makedirs(new_path)

    return new_path
