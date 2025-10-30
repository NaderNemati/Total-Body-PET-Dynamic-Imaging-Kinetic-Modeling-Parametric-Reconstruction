#!/usr/bin/env python3
"""
Fit multiple ROIs end-to-end with stable options:
- Extract ROI TACs
- Run 1-Tissue fit (tagged outputs + metrics CSV)
- Run Patlak & Logan with late-frames, normalization, and Ct thresholds
- Produce a final merged summary CSV of all ROI kinetics

Outputs (under --outdir):
  tac_<roi>.csv
  fit_1t_<roi>.png, patlak_<roi>.png, logan_<roi>.png
  roi_1t_metrics.csv, roi_pl_metrics.csv, roi_params_summary.csv
  summary.txt (stdout/stderr logs)
"""
import os, sys, argparse, json, re
import numpy as np
import pandas as pd
import nibabel as nib
from subprocess import run, PIPE

def mean_tac(data, mask):
    vox = data[mask > 0]
    return vox.mean(axis=0) if vox.size else np.zeros(data.shape[-1], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pet", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--rois_dir", default="results/rois")
    ap.add_argument("--outdir",   default="results/roi_fits")
    ap.add_argument("--idif_csv", default="results/tac_idif.csv", help="IDIF CSV used for all ROI fits")

    # Stability knobs for Patlak/Logan (forwarded to scripts/03_patlak_logan.py)
    ap.add_argument("--patlak_start_min", type=float, default=20.0)
    ap.add_argument("--logan_start_min",  type=float, default=20.0)
    ap.add_argument("--ct_min",           type=float, default=0.001)

    # Extra options for 1T fit (just in case we want to pass Cp file)
    ap.add_argument("--cp_csv", default="", help="Optional plasma input CSV; if empty uses --idif_csv")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load PET & times
    pet_img = nib.load(args.pet)
    data = np.asarray(pet_img.dataobj, dtype=np.float32)
    with open(args.json, "r") as f:
        meta = json.load(f)
    t_mid = np.array(meta["FrameTimesStart"], float) + 0.5 * np.array(meta["FrameDuration"], float)

    # ROIs
    if not os.path.isdir(args.rois_dir):
        raise FileNotFoundError(f"ROIs directory not found: {args.rois_dir}")
    rois = sorted([f for f in os.listdir(args.rois_dir) if f.startswith("roi_") and f.endswith(".nii.gz")])
    if not rois:
        raise RuntimeError(f"No ROI masks found in {args.rois_dir}")

    # Metrics targets
    one_t_metrics = os.path.join(args.outdir, "roi_1t_metrics.csv")
    pl_metrics    = os.path.join(args.outdir, "roi_pl_metrics.csv")
    # clean previous to avoid mixing runs
    for p in (one_t_metrics, pl_metrics):
        if os.path.exists(p):
            os.remove(p)

    rows = []
    py_exec = sys.executable  # call child scripts with the active venv Python

    for r in rois:
        name = r.replace("roi_", "").replace(".nii.gz", "")
        mask = nib.load(os.path.join(args.rois_dir, r)).get_fdata().astype(bool)
        tac = mean_tac(data, mask)
        csv_tac = os.path.join(args.outdir, f"tac_{name}.csv")
        pd.DataFrame({"time_mid_s": t_mid, "value": tac}).to_csv(csv_tac, index=False)

        # --- 1-Tissue fit ---
        one_t_cmd = [
            py_exec, "scripts/02_fit_1t_model.py",
            "--tac_csv", csv_tac,
            "--outdir",  args.outdir,
            "--tag",     name,
            "--metrics_csv", one_t_metrics,
        ]
        if args.cp_csv:
            one_t_cmd += ["--cp_csv", args.cp_csv]
        else:
            one_t_cmd += ["--idif_csv", args.idif_csv]

        p1 = run(one_t_cmd, stdout=PIPE, stderr=PIPE, text=True)

        # --- Patlak & Logan (stable) ---
        pl_cmd = [
            py_exec, "scripts/03_patlak_logan.py",
            "--tac_csv", csv_tac,
            "--idif_csv", args.idif_csv,
            "--outdir", args.outdir,
            "--tag", name,
            "--metrics_csv", pl_metrics,
            "--patlak_start_min", str(args.patlak_start_min),
            "--logan_start_min",  str(args.logan_start_min),
            "--ct_min",           str(args.ct_min),
        ]
        p2 = run(pl_cmd, stdout=PIPE, stderr=PIPE, text=True)
        if p2.returncode != 0:
            # fallback without new flags (for backward-compat)
            pl_cmd_fallback = [
                py_exec, "scripts/03_patlak_logan.py",
                "--tac_csv", csv_tac,
                "--idif_csv", args.idif_csv,
                "--outdir", args.outdir,
                "--tag", name,
                "--metrics_csv", pl_metrics,
            ]
            p2 = run(pl_cmd_fallback, stdout=PIPE, stderr=PIPE, text=True)

        rows.append({
            "roi": name,
            "fit_1t_stdout": p1.stdout.strip(),
            "fit_1t_stderr": p1.stderr.strip(),
            "pl_stdout": p2.stdout.strip(),
            "pl_stderr": p2.stderr.strip()
        })

    # Save run logs
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "summary.txt"), index=False)

    # Merge metrics into a single CSV
    df1 = pd.read_csv(one_t_metrics) if os.path.exists(one_t_metrics) else pd.DataFrame(columns=["tag","K1","k2"])
    df2 = pd.read_csv(pl_metrics)    if os.path.exists(pl_metrics)    else pd.DataFrame(columns=["tag","Ki","V0","VT","c"])
    dfm = pd.merge(df1, df2, on="tag", how="outer")
    # Reorder columns for readability and add ROI column (same as tag)
    if "tag" in dfm.columns:
        dfm.insert(0, "roi", dfm["tag"])
    out_summary = os.path.join(args.outdir, "roi_params_summary.csv")
    dfm.to_csv(out_summary, index=False)

    print("[OK] ROI fits done ->", args.outdir)
    if os.path.exists(one_t_metrics): print(f"[OK] 1T metrics -> {one_t_metrics}")
    if os.path.exists(pl_metrics):    print(f"[OK] Patlak/Logan metrics -> {pl_metrics}")
    print(f"[OK] Final summary -> {out_summary}")

if __name__ == "__main__":
    main()

