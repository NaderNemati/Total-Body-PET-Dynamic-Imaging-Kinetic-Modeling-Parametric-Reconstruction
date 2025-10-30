#!/usr/bin/env python3
import os, argparse, numpy as np

def have_cupy():
    try:
        import cupy as cp
        _ = cp.zeros((1,))
        return True
    except Exception:
        return False

def gpu_backproj(sino, theta):
    import cupy as cp
    sino_g = cp.asarray(sino, dtype=cp.float32)  # (detectors, angles)
    theta_g= cp.asarray(np.deg2rad(theta), dtype=cp.float32)
    n_det, n_ang = sino_g.shape
    N = n_det
    x = cp.linspace(-1,1,N)
    X,Y = cp.meshgrid(x,x)
    img = cp.zeros_like(X, dtype=cp.float32)
    # simple unfiltered backprojection (demo)
    for ia in range(n_ang):
        t = X*cp.cos(theta_g[ia]) + Y*cp.sin(theta_g[ia])
        # map t in [-1,1] to detector index
        u = (t + 1) * 0.5 * (N-1)
        u0 = cp.clip(cp.floor(u).astype(cp.int32), 0, N-1)
        val = sino_g[:, ia][u0]
        img += val
    img = img / n_ang
    return cp.asnumpy(img)

def cpu_backproj(sino, theta):
    from skimage.transform import iradon
    return iradon(sino, theta=theta, filter_name="ramp", circle=True).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sino", required=True, help="single slice sinogram .npy")
    ap.add_argument("--out",  required=True, help="reconstructed slice .npy")
    ap.add_argument("--angles", type=int, default=180)
    args = ap.parse_args()

    sino = np.load(args.sino)  # (detectors, angles)
    theta = np.linspace(0., 180., args.angles, endpoint=False)
    if have_cupy():
        rec = gpu_backproj(sino, theta)
        print("[OK] GPU backprojection")
    else:
        rec = cpu_backproj(sino, theta)
        print("[OK] CPU FBP (scikit-image)")
    np.save(args.out, rec.astype(np.float32))
    print("[OK] Saved ->", args.out)

if __name__ == "__main__":
    main()

