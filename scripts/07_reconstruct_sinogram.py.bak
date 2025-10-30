#!/usr/bin/env python3
import os, argparse, numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.transform import iradon

def recon_slice_from_sino(sino: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Reconstruct a 2D slice from a sinogram.
    - Ensures the sinogram is shaped (n_detectors, n_angles).
    - Clips tiny negatives and casts to float32.
    """
    # Auto-fix orientation if needed: angles should be the 2nd dim
    if sino.ndim != 2:
        raise ValueError(f"Sinogram must be 2D, got shape {sino.shape}")
    if sino.shape[1] != len(theta) and sino.shape[0] == len(theta):
        sino = sino.T
    sino = np.asarray(sino, dtype=np.float32)

    # Numerical hygiene
    sino = np.nan_to_num(sino, nan=0.0, posinf=0.0, neginf=0.0)
    sino[sino < 0] = 0.0

    # If everything is ~0, return zeros to avoid NaNs downstream
    if float(sino.max()) <= 0.0:
        return np.zeros((sino.shape[0], sino.shape[0]), dtype=np.float32)

    rec = iradon(sino, theta=theta, filter_name="ramp", circle=True)
    return rec.astype(np.float32)

def robust_png(vol: np.ndarray, out_png: str, title: str = "Reconstructed (mid slice)") -> None:
    """Save a PNG with robust percentile scaling to avoid 'all black' display."""
    mid = vol.shape[2] // 2
    sl = vol[:, :, mid].astype(np.float32)
    sl = np.nan_to_num(sl, nan=0.0, posinf=0.0, neginf=0.0)

    # Robust range
    lo, hi = np.percentile(sl, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(sl.min()), float(sl.max() if sl.max() > sl.min() else sl.min() + 1e-6)

    plt.figure(figsize=(6, 4))
    plt.imshow(sl, cmap="gray", vmin=lo, vmax=hi)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

def reconstruct_frame(sino_dir: str, theta: np.ndarray, out_nii: str, out_png: str | None = None) -> np.ndarray:
    files = sorted([f for f in os.listdir(sino_dir) if f.endswith(".npy")])
    if not files:
        raise FileNotFoundError(f"No sinograms found in {sino_dir}")

    rec_slices = []
    log_lines = []
    for f in files:
        path = os.path.join(sino_dir, f)
        sino = np.load(path)
        log_lines.append(f"{f}: shape={tuple(sino.shape)} min={float(np.nanmin(sino)):.3g} max={float(np.nanmax(sino)):.3g}")
        rec = recon_slice_from_sino(sino, theta)
        rec_slices.append(rec)

    vol = np.stack(rec_slices, axis=2).astype(np.float32)  # (H,W,Z)
    nib.Nifti1Image(vol, np.eye(4)).to_filename(out_nii)

    # Write a tiny log next to the NIfTI for QC
    with open(os.path.splitext(out_nii)[0] + "_qc.txt", "w") as fh:
        fh.write("\n".join(log_lines) + f"\nVOL: shape={tuple(vol.shape)} min={float(vol.min()):.3g} max={float(vol.max()):.3g}\n")

    if out_png:
        robust_png(vol, out_png)

    return vol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sino_dir", required=True, help="Folder with slice_###.npy sinograms")
    ap.add_argument("--out_nii", required=True)
    ap.add_argument("--out_png", default="")
    ap.add_argument("--angles", type=int, default=180, help="number of projection angles in [0,180)")
    args = ap.parse_args()

    theta = np.linspace(0.0, 180.0, args.angles, endpoint=False).astype(np.float32)
    os.makedirs(os.path.dirname(args.out_nii), exist_ok=True)
    reconstruct_frame(args.sino_dir, theta, args.out_nii, args.out_png or None)
    print("[OK] Reconstructed ->", args.out_nii)
    if args.out_png:
        print("[OK] PNG ->", args.out_png)

if __name__ == "__main__":
    main()

