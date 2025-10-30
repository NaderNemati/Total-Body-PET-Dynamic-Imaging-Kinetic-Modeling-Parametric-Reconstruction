#!/usr/bin/env bash
set -euo pipefail

echo "=== Activate environment ==="
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Virtual env not found. Create it with: python -m venv .venv"
  exit 1
fi

echo "=== Install dependencies ==="
pip install --upgrade pip
pip install -r env/requirements.txt

echo "=== Step 1: Simulate total-body PET data ==="
python scripts/08_simulate_totalbody_pet.py \
  --outroot results/totalbody_sim \
  --angles 360 \
  --frames 30 \
  --dt 60 \
  --counts_scale 1e7 \
  --background 0.02

SIM_NII=results/totalbody_sim/sub-tbpet_dyn.nii.gz
SIM_JSON=results/totalbody_sim/sub-tbpet_dyn.json
SIM_IDIF=results/totalbody_sim/tac_idif.csv
SIM_ROIS=results/totalbody_sim/rois

echo "=== Step 2: ROI-wise modelling ==="
python scripts/06_fit_rois.py \
  --pet "$SIM_NII" \
  --json "$SIM_JSON" \
  --rois_dir "$SIM_ROIS" \
  --tstar 1800 \
  --r2min 0.90 \
  --outdir results/roi_fits_sim

echo "=== Step 3: Voxel-wise Patlak/Logan parametric maps ==="
python scripts/05_voxelwise_patlak_logan.py \
  --pet "$SIM_NII" \
  --json "$SIM_JSON" \
  --idif_csv "$SIM_IDIF" \
  --mask "$SIM_ROIS/roi_full.nii.gz" \
  --tstar 1800 \
  --r2min 0.90 \
  --outdir results/paramaps_sim \
  --chunk 8000

echo "=== Step 4: Generate figures and report ==="
python - <<'PYCODE'
import os, nibabel as nib, numpy as np, matplotlib.pyplot as plt
def save_slices(nii, out_png_prefix):
    img = nib.load(nii).get_fdata().astype(np.float32)
    midz = img.shape[2]//2
    mids = [
        ("axial-z", img[:,:,midz]),
        ("axial-y", img[:,img.shape[1]//2,:]),
        ("axial-x", img[img.shape[0]//2,:,:])
    ]
    for tag, sl in mids:
        sl = np.nan_to_num(sl, 0.0, 0.0, 0.0)
        lo, hi = np.percentile(sl, [2,98])
        if hi<=lo:
            lo, hi = float(sl.min()), float(sl.max() or sl.min()+1e-6)
        plt.figure(figsize=(7,3))
        plt.imshow(sl, cmap="viridis", vmin=lo, vmax=hi, aspect="equal")
        plt.axis("off")
        plt.title(f"{os.path.basename(out_png_prefix)} {tag}")
        out_file = f"{out_png_prefix}_{tag}.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=160)
        plt.close()
        print("[OK] saved", out_file)

os.makedirs("results/figures_sim", exist_ok=True)
save_slices("results/paramaps_sim/Ki_map.nii.gz", "results/figures_sim/Ki")
save_slices("results/paramaps_sim/VT_map.nii.gz", "results/figures_sim/VT")
PYCODE

python scripts/10_generate_report.py \
  --roi_summary results/roi_fits_sim/roi_params_summary.csv \
  --fig_dir results/figures_sim \
  --out_md results/report_sim.md

echo "=== Pipeline complete. Results in results/roi_fits_sim/ results/paramaps_sim/ results/figures_sim/ ==="

