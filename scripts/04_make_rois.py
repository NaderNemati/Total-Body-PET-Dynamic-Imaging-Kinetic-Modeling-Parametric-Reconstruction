#!/usr/bin/env python3
import os, argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_opening, binary_closing

def brain_mask_from_temporal_mean(pet4d, p=70):
    mean_img = pet4d.mean(axis=3)
    pos = mean_img > 0
    thr = np.percentile(mean_img[pos], p) if pos.any() else 0.0
    m = mean_img > (0.4*thr)
    m = binary_opening(m, iterations=2)
    m = binary_closing(m, iterations=2)
    return m

def high_uptake_roi(pet4d, t_frames=(0,3), top_p=1.0):
    t0, t1 = t_frames
    t1 = min(t1, pet4d.shape[-1]-1)
    early = pet4d[..., t0:t1+1].mean(axis=3)
    flat = early[early>0].ravel()
    thr = np.percentile(flat, 100-top_p) if flat.size else np.inf
    return (early>=thr)

def center_cube(shape3d, cube=(32,32,32)):
    X,Y,Z = shape3d
    cx,cy,cz = cube
    sx = max((X-cx)//2,0); ex = min(sx+cx, X)
    sy = max((Y-cy)//2,0); ey = min(sy+cy, Y)
    sz = max((Z-cz)//2,0); ez = min(sz+cz, Z)
    m = np.zeros(shape3d, bool)
    m[sx:ex, sy:ey, sz:ez] = True
    return m

def low_uptake_roi(pet4d, p=20):
    mean_img = pet4d.mean(axis=3)
    pos = mean_img > 0
    thr = np.percentile(mean_img[pos], p) if pos.any() else 0.0
    return (mean_img>0) & (mean_img<=thr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pet", required=True, help="mini PET NIfTI 4D")
    ap.add_argument("--outdir", default="results/rois")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    img = nib.load(args.pet)
    pet4d = np.asarray(img.dataobj, dtype=np.float32)
    aff = img.affine

    brain = brain_mask_from_temporal_mean(pet4d, p=70)
    high  = high_uptake_roi(pet4d, (0,3), top_p=1.0)
    center= center_cube(pet4d.shape[:3], cube=(32,32,32))
    low   = low_uptake_roi(pet4d, p=20)

    for name, mask in [("brain",brain), ("high",high), ("center",center), ("low",low)]:
        nib.Nifti1Image(mask.astype(np.uint8), aff).to_filename(
            os.path.join(args.outdir, f"roi_{name}.nii.gz"))
        print(f"[OK] Saved ROI: {name} -> {os.path.join(args.outdir, f'roi_{name}.nii.gz')}")

if __name__=="__main__":
    main()

