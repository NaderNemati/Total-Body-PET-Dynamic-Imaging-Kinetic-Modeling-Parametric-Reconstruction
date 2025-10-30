#!/usr/bin/env python3
import os, argparse, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def df_to_markdown_safe(df):
    """Return a markdown table if tabulate is available; otherwise a code block with CSV."""
    if df.empty:
        return "_No ROI summary found._"
    try:
        # pandas.to_markdown needs 'tabulate'
        return df.to_markdown(index=False)
    except Exception:
        # Fallback: CSV as code block
        from io import StringIO
        s = StringIO()
        df.to_csv(s, index=False)
        return "```\n" + s.getvalue() + "```\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi_summary", default="results/roi_fits_tb/roi_params_summary.csv")
    ap.add_argument("--fig_dir",     default="results/figures_tb")
    ap.add_argument("--out_md",      default="results/report_tb.md")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    if os.path.exists(args.roi_summary):
        df = pd.read_csv(args.roi_summary)
    else:
        df = pd.DataFrame(columns=["roi","K1","k2","Ki","V0","VT","c"])

    # simple bar charts
    for col in ["Ki","VT","K1","k2"]:
        if col in df.columns and not df.empty:
            plt.figure(figsize=(6,3.2))
            vals = pd.to_numeric(df[col], errors="coerce")
            plt.bar(df["roi"].astype(str), vals)
            plt.title(col); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
            outpng = os.path.join(args.fig_dir, f"{col}_bars.png")
            plt.savefig(outpng, dpi=140); plt.close()

    with open(args.out_md, "w") as f:
        f.write("# Total-Body PET: Reconstruction & Kinetic Modeling (Mini Demo)\n\n")
        f.write("## ROI Parameter Summary\n\n")
        f.write(df_to_markdown_safe(df))
        f.write("\n\n## Figures\n")
        for col in ["Ki","VT","K1","k2"]:
            png = os.path.join(args.fig_dir, f"{col}_bars.png")
            if os.path.exists(png):
                f.write(f"\n### {col}\n\n![]({png})\n")

    print("[OK] Report written ->", args.out_md)

if __name__ == "__main__":
    main()

