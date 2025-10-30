#!/usr/bin/env python3
import os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-6

def linfit(X, Y):
    A = np.vstack([X, np.ones_like(X)]).T
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    a,b = beta
    yhat = A @ beta
    return a,b,yhat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tac_csv", required=True, help="time_mid_s,value")
    ap.add_argument("--idif_csv", required=True, help="time,value")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--tag", default="", help="suffix for filenames")
    ap.add_argument("--metrics_csv", default="", help="append Ki,V0,VT,c to this CSV")
    # stability knobs
    ap.add_argument("--patlak_start_min", type=float, default=20.0)
    ap.add_argument("--logan_start_min",  type=float, default=20.0)
    ap.add_argument("--ct_min", type=float, default=0.0, help="drop frames with Ct <= ct_min")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tac = pd.read_csv(args.tac_csv).to_numpy()
    t   = tac[:,0].astype(float); Ct = tac[:,1].astype(float)
    Cp  = pd.read_csv(args.idif_csv).to_numpy()[:,1].astype(float)
    assert len(Ct)==len(Cp), "Length mismatch Ct vs Cp"

    # late frames
    tmin = t/60.0
    pat_idx = np.where(tmin >= args.patlak_start_min)[0]
    log_idx = np.where(tmin >= args.logan_start_min)[0]
    if pat_idx.size < 8: pat_idx = np.arange(max(0, len(t)-12), len(t))
    if log_idx.size < 8: log_idx = np.arange(max(0, len(t)-12), len(t))

    # normalize for stability (doesn't change slopes much)
    Ct_n = Ct / (np.max(Ct) + EPS)
    Cp_n = Cp / (np.max(Cp) + EPS)

    # --- Patlak ---
    dt = float(np.mean(np.diff(t)))
    intCp = np.cumsum(Cp_n)*dt
    Xp = intCp/(Cp_n + EPS)
    Yp = Ct_n/(Cp_n + EPS)
    sel = pat_idx[(Ct_n[pat_idx] > args.ct_min)]
    a,b,yhat = linfit(Xp[sel], Yp[sel])
    Ki, V0 = float(a), float(b)

    plt.figure(figsize=(6,5))
    plt.scatter(Xp[sel], Yp[sel], s=10, label="data")
    plt.plot(Xp[sel], yhat, label=f"fit: Ki={Ki:.3f}, V0={V0:.3f}")
    plt.xlabel("∫Cp dt / Cp"); plt.ylabel("Ct / Cp"); plt.title("Patlak plot")
    plt.legend(); plt.tight_layout()
    p_png = os.path.join(args.outdir, f"patlak{('_'+args.tag) if args.tag else ''}.png")
    plt.savefig(p_png, dpi=160); plt.close()

    # --- Logan ---
    intCt = np.cumsum(Ct_n)*dt
    intCp = np.cumsum(Cp_n)*dt
    Xl = intCt/(Ct_n + EPS)
    Yl = intCp/(Ct_n + EPS)
    sel = log_idx[(Ct_n[log_idx] > args.ct_min)]
    a,b,yhat = linfit(Xl[sel], Yl[sel])
    VT, c = float(a), float(b)

    plt.figure(figsize=(6,5))
    plt.scatter(Xl[sel], Yl[sel], s=10, label="data")
    plt.plot(Xl[sel], yhat, label=f"fit: VT={VT:.3f}")
    plt.xlabel("∫Ct dt / Ct"); plt.ylabel("∫Cp dt / Ct"); plt.title("Logan plot")
    plt.legend(); plt.tight_layout()
    l_png = os.path.join(args.outdir, f"logan{('_'+args.tag) if args.tag else ''}.png")
    plt.savefig(l_png, dpi=160); plt.close()

    if args.metrics_csv:
        import csv
        write_header = not os.path.exists(args.metrics_csv)
        with open(args.metrics_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header: w.writerow(["tag","Ki","V0","VT","c"])
            w.writerow([args.tag, f"{Ki:.6f}", f"{V0:.6f}", f"{VT:.6f}", f"{c:.6f}"])

    print(f"[OK] Saved -> {p_png}  {l_png}")
    print(f"[OK] Patlak: Ki={Ki:.4f}, V0={V0:.4f} | Logan: VT={VT:.4f}, c={c:.4f}")

if __name__ == "__main__":
    main()

