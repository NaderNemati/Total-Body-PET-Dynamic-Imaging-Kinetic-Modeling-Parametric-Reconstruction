#!/usr/bin/env python3
"""
Create a smaller PET dataset from a large 4D PET NIfTI + PET-BIDS JSON:
- Temporal sub-sample (e.g., every 3rd frame)
- Spatial down-sample (e.g., step=2 in x,y,z)
- Optional center-crop to a target shape
Outputs: mini NIfTI + updated JSON next to it.
"""
import os, json, argparse
import numpy as np
import nibabel as nib

def center_crop3d(arr, target):
    """Center crop a 3D/4D array spatially to target (tx,ty,tz)."""
    x,y,z = arr.shape[:3]
    tx,ty,tz = target
    sx = max((x - tx)//2, 0); ex = sx + min(tx, x)
    sy = max((y - ty)//2, 0); ey = sy + min(ty, y)
    sz = max((z - tz)//2, 0); ez = sz + min(tz, z)
    if arr.ndim == 4:
        return arr[sx:ex, sy:ey, sz:ez, :]
    return arr[sx:ex, sy:ey, sz:ez]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pet_in", required=True, help="Input *_pet.nii.gz")
    ap.add_argument("--json_in", required=True, help="Input *_pet.json")
    ap.add_argument("--outdir", default="data/mini", help="Output directory")
    ap.add_argument("--t_step", type=int, default=3, help="Temporal step (keep every t_step frame)")
    ap.add_argument("--sp_step", type=int, default=2, help="Spatial step (stride in x,y,z)")
    ap.add_argument("--crop", default="", help="Center crop size 'X,Y,Z' (optional)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load lazily to save RAM
    img = nib.load(args.pet_in)
    dataobj = img.dataobj  # lazy
    hdr = img.header.copy()
    affine = img.affine

    # Load PET-BIDS timing
    with open(args.json_in, "r") as f:
        meta = json.load(f)
    t0 = np.array(meta.get("FrameTimesStart", []), dtype=float)
    dt = np.array(meta.get("FrameDuration", []), dtype=float)
    assert t0.size == dt.size and t0.size > 0, "Invalid PET-BIDS timing in JSON."

    # Build slicing
    sp = args.sp_step
    t = args.t_step
    # Spatial downsample by stride
    data_small = np.asarray(dataobj[::sp, ::sp, ::sp, ::t], dtype=np.float32)

    # Optional center crop
    if args.crop:
        tx,ty,tz = map(int, args.crop.split(","))
        data_small = center_crop3d(data_small, (tx,ty,tz))

    # Update header zooms if stepping (voxel size effectively increases by sp)
    zooms = list(img.header.get_zooms())
    if len(zooms) >= 3:
        zooms = list(zooms)
        zooms[0] *= sp; zooms[1] *= sp; zooms[2] *= sp
        # time zoom unchanged (index 3) because we are sub-sampling frames, not rescaling time
        hdr.set_zooms(tuple(zooms[:len(hdr.get_zooms())]))

    # Temporal metadata update
    t0_small = t0[::t]
    dt_small = dt[::t]
    assert data_small.shape[-1] == t0_small.size, "Mismatch after temporal sub-sampling."

    # Save NIfTI
    base = os.path.basename(args.pet_in).replace("_pet.nii.gz", "_petMINI.nii.gz")
    pet_out = os.path.join(args.outdir, base)
    nib.Nifti1Image(data_small, affine, hdr).to_filename(pet_out)

    # Save JSON
    meta_small = dict(meta)
    meta_small["FrameTimesStart"] = [float(x) for x in t0_small]
    meta_small["FrameDuration"]   = [float(x) for x in dt_small]
    json_out = pet_out.replace(".nii.gz", ".json")
    with open(json_out, "w") as f:
        json.dump(meta_small, f, indent=2)

    print("[OK] mini PET saved:", pet_out)
    print("[OK] mini JSON saved:", json_out)
    print("Shape MINI:", data_small.shape, "| Frames:", data_small.shape[-1])

if __name__ == "__main__":
    main()
