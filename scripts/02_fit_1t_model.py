#!/usr/bin/env python3
import os, argparse
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def one_tissue_model(Cp, t, K1, k2):
    t = np.asarray(t, dtype=float)
    Cp = np.asarray(Cp, dtype=float)
    dt = float(np.mean(np.diff(t)))
    Ct = np.zeros_like(t, dtype=np.float64)
    # causal convolution
    for i in range(len(t)):
        # kernel for tau in [0..i]
        kernel = np.exp(-k2 * (t[i] - t[:i+1]))
        # integrate Cp(tau)*kernel(tau) d tau
        # equivalently integrate reversed to align with increasing tau
        integrand = Cp[:i+1] * kernel
        Ct[i] = K1 * trapezoid(integrand, dx=dt)
    return Ct

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tac_csv", required=True, help="TAC CSV (time_mid_s,value), e.g. results/tac_brain.csv")
    ap.add_argument("--cp_csv",  default="", help="Plasma input CSV (time_s,value). Optional.")
    ap.add_argument("--idif_csv", default="results/tac_idif.csv", help="Fallback IDIF CSV if Cp missing.")
    ap.add_argument("--tag", default="", help="Optional tag for output filenames")
    ap.add_argument("--metrics_csv", default="", help="Optional CSV to append K1,k2")
    ap.add_argument("--outdir", default="results", help="Output directory")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tac = pd.read_csv(args.tac_csv).to_numpy()
    t   = tac[:,0].astype(float)
    Ct  = tac[:,1].astype(float)

    if args.cp_csv and os.path.exists(args.cp_csv):
        Cp = pd.read_csv(args.cp_csv).to_numpy()[:,1].astype(float)
    else:
        Cp = pd.read_csv(args.idif_csv).to_numpy()[:,1].astype(float)

    # Basic scaling (helps optimizer)
    Ct_max = float(np.max(Ct)) if np.max(Ct) > 0 else 1.0
    Cp_max = float(np.max(Cp)) if np.max(Cp) > 0 else 1.0
    Ct_n = Ct / Ct_max
    Cp_n = Cp / Cp_max

    def resid(p):
        K1, k2 = p
        Ct_hat = one_tissue_model(Cp_n, t, K1, k2)
        return Ct_hat - Ct_n

    p0 = [0.5, 0.1]
    bounds = ([0.0, 0.0], [5.0, 2.0])
    res = least_squares(resid, p0, bounds=bounds, max_nfev=3000)
    K1_hat, k2_hat = [float(x) for x in res.x]
    Ct_hat = one_tissue_model(Cp_n, t, K1_hat, k2_hat) * Ct_max

    # Plot
    tag = f"_{args.tag}" if args.tag else ""
    plt.figure(figsize=(6,4))
    plt.plot(t/60.0, Ct,  label="Observed TAC")
    plt.plot(t/60.0, Ct_hat, "--", label=f"1T fit (K1={K1_hat:.3f}, k2={k2_hat:.3f})")
    plt.xlabel("Time (min)")
    plt.ylabel("Activity (a.u.)")
    plt.title("1-Tissue Compartment Fit")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(args.outdir, f"fit_1t{tag}.png")
    plt.savefig(fig_path, dpi=160)

    # Optional metrics CSV append
    if args.metrics_csv:
        import csv
        metrics_path = args.metrics_csv
        write_header = not os.path.exists(metrics_path)
        with open(metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["tag","K1","k2"])
            w.writerow([args.tag, f"{K1_hat:.6f}", f"{k2_hat:.6f}"])

    print(f"[OK] Saved plot -> {fig_path}")
    print(f"[OK] K1={K1_hat:.4f}, k2={k2_hat:.4f}")

if __name__ == "__main__":
    main()

