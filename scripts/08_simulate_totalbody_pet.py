#!/usr/bin/env python3
import os, argparse, json, numpy as np
import nibabel as nib
import pandas as pd
from skimage.transform import radon
from skimage.draw import ellipse

# ---------------- KINETICS ----------------
def one_tissue(Cp, t, K1, k2):
    dt = float(np.mean(np.diff(t)))
    Ct = np.zeros_like(Cp, dtype=np.float64)
    for i in range(len(t)):
        kernel = np.exp(-k2 * (t[i] - t[:i+1]))
        # (Cp[:i+1] * kernel) is causal; dx=dt gives Riemann approximation
        Ct[i] = K1 * np.trapz(Cp[:i+1] * kernel, dx=dt)
    return Ct

def make_cp(t, peak=1.0):
    # simple bolus + tail (seconds)
    Cp = peak * np.exp(-((t-60.0)/20.0)**2) + 0.1*np.exp(-t/600.0)
    Cp = Cp - Cp.min()
    Cp = Cp / (Cp.max() + 1e-9)
    return Cp

# ---------------- PHANTOM ----------------
def make_phantom(shape=(160,160,64), organs=()):
    H,W,Z = shape
    vol = np.zeros(shape, np.float32)
    rois = {}
    for name,(cy,cx,ry,rx,z0,z1,val) in organs:
        mask2d = np.zeros((H,W), bool)
        rr, cc = ellipse(cy, cx, ry, rx, shape=(H,W))
        mask2d[rr,cc] = True
        roi3d = np.zeros(shape, bool)
        roi3d[:, :, z0:z1] = mask2d[..., None]
        vol[roi3d] = val
        rois[name] = roi3d
    return vol, rois

def simulate_dynamic(rois, t, Cp, organ_params):
    """Return 4D dynamic volume (H,W,Z,T)."""
    T = len(t)
    shape3d = rois["full"].shape
    dyn = np.zeros(shape3d + (T,), np.float32)
    for name, mask in rois.items():
        if name == "full":
            continue
        K1, k2, scale = organ_params[name]
        Ct = one_tissue(Cp, t, K1, k2) * float(scale)
        if mask.any():
            dyn[mask, :] = Ct[None, :].astype(np.float32)
    return dyn

# ---------------- NOISE / PROJECTION ----------------
def poisson_noise(arr, scale=1e6):
    """Apply Poisson noise to normalized nonnegative activity."""
    arr = np.clip(arr, 0, None)
    lam = arr * float(scale)
    out = np.random.poisson(lam).astype(np.float32) / float(scale)
    return out

def make_slice_sinograms(frame, angles):
    """frame: (H,W,Z) -> list of sinograms per slice (detectors x angles)."""
    H,W,Z = frame.shape
    theta = np.linspace(0., 180., angles, endpoint=False)
    sinos = []
    for z in range(Z):
        img = frame[:, :, z].astype(np.float32)
        sino = radon(img, theta=theta, circle=True)  # (n_detectors, n_angles)
        sinos.append(sino.astype(np.float32))
    return sinos, theta

def fbp_stack_from_sinos(sinos, theta):
    from skimage.transform import iradon
    rec_slices = []
    for s in sinos:
        rec = iradon(s, theta=theta, filter_name="ramp", circle=True)
        rec_slices.append(rec.astype(np.float32))
    return np.stack(rec_slices, axis=2)  # (H,W,Z)

def save_rois(rois, outdir):
    os.makedirs(outdir, exist_ok=True)
    for k,v in rois.items():
        nib.Nifti1Image(v.astype(np.uint8), np.eye(4)).to_filename(
            os.path.join(outdir, f"roi_{k}.nii.gz")
        )

# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", default="results/totalbody_sim", help="root output folder")
    ap.add_argument("--angles",  type=int, default=180)
    ap.add_argument("--frames",  type=int, default=60)
    ap.add_argument("--dt",      type=float, default=120.0, help="frame duration seconds")
    ap.add_argument("--noise",   type=float, default=1.0, help="legacy scale factor (kept for compatibility)")
    # New: robust brightness controls
    ap.add_argument("--counts_scale", type=float, default=1e7,
                    help="Poisson λ scale (higher -> brighter).")
    ap.add_argument("--background", type=float, default=0.02,
                    help="Uniform background added before Poisson (fraction of max).")
    args = ap.parse_args()

    np.random.seed(0)
    H,W,Z = 160,160,64
    outroot   = args.outroot
    sino_dir  = os.path.join(outroot, "sinograms")
    recon_dir = os.path.join(outroot, "recon")
    rois_dir  = os.path.join(outroot, "rois")
    os.makedirs(sino_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(rois_dir, exist_ok=True)

    # organs: (name,(cy,cx,ry,rx,z0,z1,val_for_visual))
    organs = [
        ("brain",  (60,  80, 35, 45, 48, 63, 1.0)),
        ("liver",  (95,  70, 20, 35, 20, 35, 0.8)),
        ("kidneyL",(110, 55, 10, 15, 25, 33, 0.7)),
        ("kidneyR",(110,105, 10, 15, 25, 33, 0.7)),
        ("lungL",  (80,  55, 25, 30, 35, 48, 0.4)),
        ("lungR",  (80, 105, 25, 30, 35, 48, 0.4)),
    ]
    base, rois = make_phantom((H,W,Z), organs)
    rois["full"] = np.ones((H,W,Z), bool)
    save_rois(rois, rois_dir)

    # time axis + Cp
    T = args.frames
    t_start = np.arange(T) * args.dt        # FrameTimesStart
    t_mid   = t_start + 0.5*args.dt
    Cp = make_cp(t_mid, peak=1.0)

    # Organ kinetics (toy)
    organ_params = {
        "brain":   (0.0020, 0.0003, 1.0),
        "liver":   (0.0030, 0.0008, 0.9),
        "kidneyL": (0.0040, 0.0012, 1.1),
        "kidneyR": (0.0040, 0.0012, 1.1),
        "lungL":   (0.0015, 0.0010, 0.6),
        "lungR":   (0.0015, 0.0010, 0.6),
    }

    # --------- simulate 4D activity (noiseless) ----------
    dyn_true = simulate_dynamic(rois, t_mid, Cp, organ_params)  # (H,W,Z,T)
    dyn_min, dyn_max = float(dyn_true.min()), float(dyn_true.max())
    print(f"[INFO] dyn_true min/max before norm: {dyn_min:.3g} / {dyn_max:.3g}")

    # Normalize to [0,1] and add uniform background so Poisson won't zero it out
    if dyn_max > 0:
        dyn_norm = dyn_true / dyn_max
    else:
        dyn_norm = dyn_true.copy()
    dyn_norm += float(args.background)
    print(f"[INFO] dyn_norm min/max after +bg={args.background}: {float(dyn_norm.min()):.3g} / {float(dyn_norm.max()):.3g}")

    # Poisson noise with configurable counts
    dyn_noisy = poisson_noise(dyn_norm, scale=float(args.counts_scale))
    print(f"[INFO] dyn_noisy min/max (counts_scale={args.counts_scale:g}): {float(dyn_noisy.min()):.3g} / {float(dyn_noisy.max()):.3g}")

    # --------- forward projection + FBP reconstruction ----------
    theta = np.linspace(0., 180., args.angles, endpoint=False)
    recon4d = np.zeros((H,W,Z,T), np.float32)

    for k in range(T):
        frame = dyn_noisy[:, :, :, k]
        sinos, _ = make_slice_sinograms(frame, args.angles)

        # save sinograms
        fdir = os.path.join(sino_dir, f"frame_{k:03d}")
        os.makedirs(fdir, exist_ok=True)
        for z, s in enumerate(sinos):
            np.save(os.path.join(fdir, f"slice_{z:03d}.npy"), s)

        # reconstruct this frame
        rec = fbp_stack_from_sinos(sinos, theta)
        recon4d[:, :, :, k] = rec

        if k == 0:
            # quick sanity log on first frame
            s0 = sinos[0]
            print(f"[INFO] first sino shape={s0.shape} min/max={float(s0.min()):.3g}/{float(s0.max()):.3g}")
            print(f"[INFO] first recon frame min/max={float(rec.min()):.3g}/{float(rec.max()):.3g}")

    # --------- Save NIfTI 4D + JSON + IDIF -----------
    out_nii = os.path.join(outroot, "sub-tbpet_dyn.nii.gz")
    nib.Nifti1Image(recon4d, np.eye(4)).to_filename(out_nii)

    meta = {"FrameTimesStart": t_start.tolist(), "FrameDuration": [args.dt]*T}
    with open(os.path.join(outroot, "sub-tbpet_dyn.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # IDIF proxy
    pd.DataFrame({"time_s": t_mid, "value": Cp}).to_csv(os.path.join(outroot, "tac_idif.csv"), index=False)

    print(f"[OK] Simulated & reconstructed 4D -> {out_nii}")
    print("[OK] JSON & IDIF saved in", outroot)
    print("[OK] ROIs saved in", rois_dir)

if __name__ == "__main__":
    main()

