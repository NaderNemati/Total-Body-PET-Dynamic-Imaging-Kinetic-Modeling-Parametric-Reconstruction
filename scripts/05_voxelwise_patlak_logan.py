#!/usr/bin/env python3
"""
Voxelwise Patlak (Ki) and Logan (VT) maps with stability controls:
- Late-frame selection for Patlak/Logan
- Normalization to prevent overflow
- Frame-wise and voxel-wise Ct thresholds to avoid division by ~0
- R^2-based masking and clipping of outliers
- Chunked processing (low RAM)
Outputs:
  Ki_map.nii.gz, VT_map.nii.gz, R2_patlak.nii.gz, R2_logan.nii.gz, valid_mask.nii.gz
"""
import os, argparse, json
import numpy as np
import nibabel as nib
import pandas as pd

EPS = 1e-6

def r2_score(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2) + 1e-12
    return 1 - ss_res/ss_tot

def design_patlak(Cp, t):
    dt = float(np.mean(np.diff(t)))
    intCp = np.cumsum(Cp)*dt
    x = intCp/(Cp + EPS)
    y_factor = 1.0  # y = Ct/Cp -> handled later per-voxel
    return x, y_factor

def logan_xy(Ct, Cp, t):
    dt = float(np.mean(np.diff(t)))
    intCt = np.cumsum(Ct)*dt
    intCp = np.cumsum(Cp)*dt
    x = intCt/(Ct + EPS)        # denominator Ct
    y = intCp/(Ct + EPS)
    return x, y

def linfit(X, Y):
    # Fit Y = a*X + b
    A = np.vstack([X, np.ones_like(X)]).T
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    a, b = beta
    yhat = A @ beta
    return a, b, yhat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pet", required=True, help="mini PET 4D NIfTI")
    ap.add_argument("--json", required=True, help="PET-BIDS JSON (FrameTimesStart, FrameDuration)")
    ap.add_argument("--idif_csv", default="results/tac_idif.csv", help="time,value")
    ap.add_argument("--mask", default="", help="Brain mask NIfTI (optional)")
    ap.add_argument("--outdir", default="results/paramaps")
    ap.add_argument("--chunk", type=int, default=6000, help="voxels per chunk")
    # stability knobs
    ap.add_argument("--patlak_start_min", type=float, default=20.0,
                    help="use frames with mid-time >= this (minutes) for Patlak")
    ap.add_argument("--logan_start_min", type=float, default=20.0,
                    help="use frames with mid-time >= this (minutes) for Logan")
    ap.add_argument("--ct_mean_min", type=float, default=5e-3,
                    help="skip voxels whose mean Ct (normalized) < this")
    ap.add_argument("--r2_min", type=float, default=0.5,
                    help="mask voxels with R^2 below this for output maps")
    ap.add_argument("--vt_clip_max", type=float, default=10.0,
                    help="clip VT to [0, vt_clip_max] to suppress outliers in visualization")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load PET
    img = nib.load(args.pet)
    data = np.asarray(img.dataobj, dtype=np.float32)  # (X,Y,Z,T)
    aff  = img.affine
    X,Y,Z,T = data.shape

    # Load timing
    meta = json.load(open(args.json))
    t0  = np.array(meta["FrameTimesStart"], float)
    dtf = np.array(meta["FrameDuration"], float)
    tmid= t0 + 0.5*dtf
    assert tmid.size == T

    # Load Cp (IDIF)
    df = pd.read_csv(args.idif_csv).to_numpy()
    t_cp = df[:,0].astype(float)
    Cp   = df[:,1].astype(float)
    # ensure aligned length
    if Cp.size != T:
        raise ValueError(f"Cp length {Cp.size} != PET frames {T}")

    # late-frame indices
    pat_idx  = np.where((tmid/60.0) >= args.patlak_start_min)[0]
    log_idx  = np.where((tmid/60.0) >= args.logan_start_min)[0]
    if pat_idx.size < 8:  # need enough points
        pat_idx = np.arange(max(0, T-12), T)
    if log_idx.size < 8:
        log_idx = np.arange(max(0, T-12), T)

    # normalize Cp for stability (does not affect slopes in linear forms)
    Cp_n = Cp / (np.max(Cp) + EPS)

    # Build global regressors
    x_pat_full, _ = design_patlak(Cp_n, tmid)
    x_pat = x_pat_full[pat_idx]
    Cp_pat = Cp_n[pat_idx]

    # Brain mask
    if args.mask and os.path.exists(args.mask):
        m = nib.load(args.mask).get_fdata().astype(bool)
    else:
        m = (data.mean(axis=3) > 0)
    coords = np.array(np.where(m)).T
    nvox = coords.shape[0]
    print("[INFO] Voxels to process:", nvox)

    Ki_map  = np.zeros((X,Y,Z), np.float32)
    V0_map  = np.zeros((X,Y,Z), np.float32)
    R2_pat  = np.zeros((X,Y,Z), np.float32)

    VT_map  = np.zeros((X,Y,Z), np.float32)
    Cc_map  = np.zeros((X,Y,Z), np.float32)
    R2_log  = np.zeros((X,Y,Z), np.float32)

    valid   = np.zeros((X,Y,Z), np.uint8)

    chunk = int(args.chunk)
    for i in range(0, nvox, chunk):
        sub = coords[i:i+chunk]
        Ct  = data[sub[:,0], sub[:,1], sub[:,2], :].astype(np.float32)  # (n,T)

        # normalize Ct per-voxel to avoid tiny denominators
        Ct_max = np.maximum(Ct.max(axis=1, keepdims=True), EPS)
        Ct_n   = Ct / Ct_max

        # skip voxels with too small mean Ct in late frames
        mean_late = Ct_n[:, log_idx].mean(axis=1)
        keep = mean_late >= args.ct_mean_min
        if not np.any(keep):
            continue
        idx_keep = np.where(keep)[0]
        Ct_n = Ct_n[idx_keep]

        # --- Patlak (on late frames) ---
        Y_pat = (Ct_n[:, pat_idx] / (Cp_pat + EPS))  # (n_keep, T_late)
        A_pat = np.vstack([x_pat, np.ones_like(x_pat)]).T  # (T_late,2)
        # closed-form per-batch:
        XtX_inv = np.linalg.pinv(A_pat.T @ A_pat)
        betas = (XtX_inv @ A_pat.T @ Y_pat.T).T  # (n_keep,2)
        Ki = betas[:,0]; V0 = betas[:,1]
        Yhat = (A_pat @ betas.T).T
        r2p = np.array([r2_score(Y_pat[j], Yhat[j]) for j in range(Y_pat.shape[0])], np.float32)

        # --- Logan (on late frames) ---
        VT = np.zeros(len(idx_keep), np.float32)
        cc = np.zeros(len(idx_keep), np.float32)
        r2l= np.zeros(len(idx_keep), np.float32)
        Cp_log = Cp_n[log_idx]
        t_log  = tmid[log_idx]
        for j in range(len(idx_keep)):
            x_l, y_l = logan_xy(Ct_n[j, log_idx], Cp_n[log_idx], t_log)
            # drop frames where Ct very small even after norm
            good = (Ct_n[j, log_idx] > 5*EPS)
            if np.count_nonzero(good) < 8:
                VT[j] = 0; cc[j] = 0; r2l[j] = 0
                continue
            a,b,yhat = linfit(x_l[good], y_l[good])   # slope=VT
            VT[j], cc[j] = a, b
            r2l[j] = r2_score(y_l[good], yhat)

        # write back (only kept voxels)
        xyz = sub[idx_keep]
        Ki_map[xyz[:,0], xyz[:,1], xyz[:,2]] = Ki
        V0_map[xyz[:,0], xyz[:,1], xyz[:,2]] = V0
        R2_pat[xyz[:,0], xyz[:,1], xyz[:,2]] = r2p
        VT_clipped = np.clip(VT, 0, args.vt_clip_max)
        VT_map[xyz[:,0], xyz[:,1], xyz[:,2]] = VT_clipped
        Cc_map[xyz[:,0], xyz[:,1], xyz[:,2]] = cc
        R2_log[xyz[:,0], xyz[:,1], xyz[:,2]] = r2l

        # validity mask (both r2 above threshold)
        ok = (r2p >= args.r2_min) & (r2l >= args.r2_min)
        valid[xyz[:,0], xyz[:,1], xyz[:,2]] = ok.astype(np.uint8)

        if (i // chunk) % 10 == 0:
            print(f"[...] {i}/{nvox} voxels")

    # save
    def save(arr, name):
        nib.Nifti1Image(arr, aff).to_filename(os.path.join(args.outdir, name))

    save(Ki_map, "Ki_map.nii.gz")
    save(V0_map, "V0_map.nii.gz")
    save(R2_pat, "R2_patlak.nii.gz")
    save(VT_map, "VT_map.nii.gz")
    save(Cc_map, "c_map.nii.gz")
    save(R2_log, "R2_logan.nii.gz")
    save(valid,  "valid_mask.nii.gz")

    print("[OK] Saved parametric maps (+valid_mask) in", args.outdir)

if __name__ == "__main__":
    main()

