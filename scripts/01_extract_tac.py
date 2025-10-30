#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless

from scipy.ndimage import binary_opening, binary_closing

def load_pet_4d(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError("PET image must be 4D (X,Y,Z,T).")
    return data

def load_pet_json(json_path):
    with open(json_path, "r") as f:
        meta = json.load(f)
    t0 = np.array(meta.get("FrameTimesStart", []), dtype=float)  # seconds
    dt = np.array(meta.get("FrameDuration", []), dtype=float)    # seconds
    if t0.size == 0 or dt.size == 0:
        raise ValueError("Missing FrameTimesStart/FrameDuration in sidecar JSON.")
    if t0.size != dt.size:
        raise ValueError("FrameTimesStart and FrameDuration have different lengths.")
    t_mid = t0 + 0.5 * dt
    return t0, dt, t_mid, meta

def build_brain_mask(pet4d, p=70):
    mean_img = pet4d.mean(axis=3)
    positive = mean_img[mean_img > 0]
    if positive.size == 0:
        raise ValueError("Temporal mean has no positive voxels; check input PET.")
    thresh = np.percentile(positive, p)
    mask = mean_img > (0.4 * thresh)
    # clean tiny islands
    mask = binary_opening(mask, iterations=2)
    mask = binary_closing(mask, iterations=2)
    return mask

def top_percent_roi(pet4d, t_frames=(0, 3), top_p=0.5):
    t0, t1 = t_frames
    t1 = min(t1, pet4d.shape[-1]-1)
    early = pet4d[..., t0:t1+1].mean(axis=3)
    flat = early[early > 0].ravel()
    if flat.size == 0:
        raise ValueError("No positive voxels in early frames; cannot form IDIF proxy.")
    thr = np.percentile(flat, 100 - top_p)  # e.g., top 0.5%
    roi = early >= thr
    return roi

def mean_tac(pet4d, roi):
    vox = pet4d[roi, :]
    return vox.mean(axis=0) if vox.size else np.zeros(pet4d.shape[-1], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pet", required=True, help="Path to *_pet.nii.gz (4D)")
    ap.add_argument("--json", required=True, help="Path to *_pet.json (PET-BIDS)")
    ap.add_argument("--outdir", default="results", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pet = load_pet_4d(args.pet)                       # (X,Y,Z,T)
    t0, dt, t_mid, _ = load_pet_json(args.json)       # time info

    if pet.shape[-1] != t_mid.size:
        raise ValueError(f"Mismatch: PET T={pet.shape[-1]} vs JSON frames={t_mid.size}")

    brain_mask = build_brain_mask(pet, p=70)
    idif_roi   = top_percent_roi(pet, t_frames=(0, 3), top_p=0.5)

    tac_brain = mean_tac(pet, brain_mask)
    tac_idif  = mean_tac(pet, idif_roi)

    pd.DataFrame({"time_mid_s": t_mid, "value": tac_brain}).to_csv(
        os.path.join(args.outdir, "tac_brain.csv"), index=False)
    pd.DataFrame({"time_mid_s": t_mid, "value": tac_idif}).to_csv(
        os.path.join(args.outdir, "tac_idif.csv"), index=False)

    print(f"[OK] Frames={pet.shape[-1]}  brainTAC[0/last]={tac_brain[0]:.4g}/{tac_brain[-1]:.4g}")
    print(f"[OK] Saved CSVs -> {args.outdir}/tac_brain.csv  {args.outdir}/tac_idif.csv")

if __name__ == "__main__":
    main()
