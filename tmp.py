import re
import os
import glob
import shutil
import numpy as np
from pathlib import Path

FOLDER = "/Users/davidparra/PycharmProjects/py-gated-camera/correlation_functions"
MAKE_BACKUP = True  # set to False if you don't want .bak backups
DRY_RUN = False     # set True to preview changes without writing files

# Example pattern: coarsek4_10mhz_7v_34w_correlations.npz
# We capture: voltage (float) before 'v_' and size (int) before 'w_'
FILENAME_RE = re.compile(
    r"""
    ^(?P<prefix>.+?)_               # anything up to first underscore
    (?P<mhz>\d+)mhz_                # frequency
    (?P<voltage>\d+(?:\.\d+)?)v_    # voltage like 7 or 7.6
    (?P<size>\d+)w_                 # size like 34
    correlations\.npz$              # suffix
    """,
    re.IGNORECASE | re.VERBOSE
)

def extract_voltage_size(filename: str):
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    voltage = float(m.group("voltage"))
    size = int(m.group("size"))
    return voltage, size

def process_npz(path: Path):
    """Read NPZ, set fields 'voltage' and 'size' from filename, keep all other fields."""
    vs = extract_voltage_size(path.name)
    if vs is None:
        print(f"Skip (pattern mismatch): {path.name}")
        return False
    voltage_val, size_val = vs

    # Load and copy all fields
    with np.load(path, allow_pickle=True) as f:
        data = {k: f[k] for k in f.files}

    # Decide if update needed
    already_ok = ("voltage" in data and np.allclose(data["voltage"], voltage_val)) and \
                 ("size"    in data and np.array_equal(np.array(data["size"]), np.array(size_val)))

    if already_ok:
        print(f"Unchanged (already has voltage={voltage_val}, size={size_val}): {path.name}")
        return False

    # Write/update fields
    data["voltage"] = np.array(voltage_val)  # store scalars explicitly
    data["size"] = np.array(size_val)

    # Backup
    if MAKE_BACKUP and not DRY_RUN:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)

    # Save in place
    if not DRY_RUN:
        np.savez(path, **data)

    print(f"Updated: {path.name}  -> voltage={voltage_val}, size={size_val}")
    return True

def main():
    npz_paths = [Path(p) for p in glob.glob(os.path.join(FOLDER, "*.npz"))]
    updated = 0
    skipped = 0
    mismatched = 0

    for p in sorted(npz_paths):
        vs = extract_voltage_size(p.name)
        if vs is None:
            mismatched += 1
            print(f"Skip (no match): {p.name}")
            continue
        changed = process_npz(p)
        if changed:
            updated += 1
        else:
            skipped += 1

    print("\nSummary")
    print(f"  Updated files:       {updated}")
    print(f"  Already correct:     {skipped}")
    print(f"  Name mismatches:     {mismatched}")

if __name__ == "__main__":
    main()