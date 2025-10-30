# Total-Body-PET-Dynamic-Imaging-Kinetic-Modeling-Parametric-Reconstruction

## Description

Positron Emission Tomography (PET) is traditionally used in a static mode: after injection of a radiotracer, a single scan is acquired (e.g., 60 min post-injection) over a limited axial field-of-view (AFOV). In contrast, total-body PET / large axial field of view (LAFOV-PET) devices now extend the axial span to ~1 m or more, enabling simultaneous imaging of nearly the whole body in one scan. This breakthrough enables several important advances:

1-**Dynamic whole-body acquisition:** Many organs and tissues can be imaged simultaneously over time, enabling extraction of time-activity curves (TACs) across the body.

2-**Kinetic modeling and parametric mapping:** With dynamic data across multiple organs, compartmental models (e.g., one-tissue, two-tissue, Patlak, Logan) can be applied at the voxel level to compute parametric maps (e.g., Ki, VT, K1, k2).

3-**Systemic physiology and multi-organ interplay:** With full‐body coverage, this technique supports investigations of tracer kinetics not just in isolated organs, but in networked systems (e.g., tumor-host interaction, pharmacokinetics, immunology).

4-**Higher sensitivity, lower dose, extended imaging times:** Long AFOV + advanced detectors yield much higher sensitivity, enabling ultra-fast scans, ultra-low tracer doses, and delayed imaging (many hours post-injection).

5-**Challenges:** Managing large 4D datasets (x,y,z,t), motion (whole body, respiratory/cardiac), deriving image-based input functions, selecting appropriate kinetic models, dealing with voxel-level noise, and implementing efficient reconstruction (FBP, OSEM, TOF) and parametric imaging pipelines (including deep learning).

6-**Clinical and research impact:** Potential applications include oncology (lesion dynamics, treatment response), cardiology, neurology, inflammation/infection, and drug development. The move from static SUV imaging to full‐body dynamic quantitation is a major shift.


In this mini-project, you are building a pipeline from simulation → reconstruction → ROI/voxel modeling → parametric map generation (Ki, VT), which aligns directly with this frontier. By generating your own 4D dynamic dataset, reconstructing frames, extracting TACs, performing voxelwise Patlak/Logan, and generating parametric maps, you demonstrate key skills and understanding for research in total-body PET.

## Key Recent Review Papers

Sun, Y., et al. Performance and application of the total-body PET/CT scanner: a literature review. [Open Access](https://ejnmmires.springeropen.com/articles/10.1186/s13550-023-01059-1?utm_source=chatgpt.com)

Sun, T., et al. Current progress and future perspectives in total‐body PET imaging, part I: Data processing and analysis. [Open Access](https://onlinelibrary.wiley.com/doi/epdf/10.1002/ird3.66)

## Background & Motivation

The development of LAFOV-PET systems has enabled whole-body dynamic imaging and voxel-wise parametric modeling, as reviewed in [(Sun 2024)](https://ejnmmires.springeropen.com/articles/10.1186/s13550-023-01059-1?utm_source=chatgpt.com) and [(Sun 2024 (Wiley))](https://onlinelibrary.wiley.com/doi/epdf/10.1002/ird3.66). This pipeline implements a simulation-to-parametrics workflow that aligns with this emerging modality: simulating dynamic total-body PET, reconstructing frames, extracting TACs for multiple organs and voxels, and computing Ki/VT maps.

## Literature

In recent years, long axial-field-of-view (LAFOV) or total-body PET scanners have unlocked exciting new opportunities for full-body dynamic imaging. Rather than imaging one organ at a time, these systems can capture nearly the entire body in one acquisition, allowing tracer kinetics across multiple organs to be studied simultaneously. For example, one review describes how modern total-body PET/CT devices enable ultra-low dose, ultra-fast, and delayed whole-body dynamic imaging, offering real-time insight into multi‐organ physiology.

While the hardware leap is clear, the modeling and analytical challenge is now central. Another review points out that the real question is no longer “can we image the whole body?” but rather “how do we process, reconstruct, and interpret these massive four-dimensional datasets?” This involves deriving reliable time‐activity curves, doing voxel-wise kinetic modeling, such as Ki and VT maps, and managing noise, motion, and input function issues that arise when moving from single-organ scans to whole-body dynamic data.

In response to this shift, many authors note that the field still needs robust pipelines for simulation, reconstruction, and modeling. Your pipeline – which simulates dynamic full-body PET data, reconstructs each frame, extracts TACs, fits models like Patlak, and generates parametric maps – aligns directly with the gap identified in the literature. By controlling the phantom, noise level, and modeling assumptions, you can explore bias, variance, and method development ahead of large clinical datasets.

Finally, the potential applications are very broad: full-body dynamic coverage enables studies of pharmacokinetics, systemic diseases, therapy response, and multi-organ physiology in ways that conventional PET cannot. The literature emphasizes that while the hardware is available, methodological standards, especially for voxel-wise dynamic modeling, are still emerging. By building this pipeline now, you position your work at the forefront of that methodological development.

## Repository Structure Tree
```python
pet-mini/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── run_demo.sh
├── env/
│   └── requirements.txt
├── scripts/
│   ├── 01_extract_tac.py
│   ├── 02_fit_1t_model.py
│   ├── 03_patlak_logan.py
│   ├── 04_make_rois.py
│   ├── 05_voxelwise_patlak_logan.py
│   ├── 06_fit_rois.py
│   ├── 07_reconstruct_sinogram.py
│   ├── 08_simulate_totalbody_pet.py
│   ├── 09_gpu_accel_recon.py
│   └── 10_generate_report.py
├── examples/
│   ├── Ki_axial-z.png
│   ├── VT_axial-z.png
│   └── recon_robust.png
├── docs/
│   └── LITERATURE.md
├── results/                  ← *ignored* by .gitignore (so large files are not tracked)
└── .github/
    └── workflows/
        └── smoke.yml
```

## Dataset Description

The dataset is known as the **Monash vis‑fPET‑fMRI Dataset (OpenNeuro accession ds003382)** and contains simultaneous functional PET and fMRI brain data from healthy human subjects.

#### Key attributes

**Subjects:** Subjects: 10 healthy adults (age range ~18-49, mean ~29 yrs) were scanned. 

**Acquisitions:** Participants underwent simultaneous BOLD-fMRI and dynamic PET with [^18F]-FDG radiotracer under a visual stimulation paradigm (checkerboard task) plus resting periods.

**PET data:** The dataset includes list-mode PET raw data, sinograms, and reconstructed dynamic PET images.

**fMRI & anatomical MRI:** Standard structural T1, field maps, and functional EPI scans are included.

**Format and metadata:** Data has been organized in a Brain Imaging Data Structure (BIDS)-like format, with sidecar JSON files, metadata, and a participants table.


It contains dynamic PET data with many individual time frames rather than just a single static image, which aligns smoothly with your pipeline’s steps of TAC extraction, kinetic modeling, and parametric mapping. The dataset spans an entire brain field of view—not just a small organ region—making it ideal for voxel-wise modeling and whole-volume analysis. It also offers both raw/list-mode and reconstructed data, giving you flexibility to simulate, reconstruct, benchmark, or compare methods. By leveraging this existing open dataset, you gain a real-data reference to compare your simulated results against, which is extremely helpful for demonstrating the validity and robustness of your pipeline.


#### Limitations/Considerations

Although dynamic, the field of view is still limited to the brain rather than full-body total-body PET, so if your project is aiming for whole-body simulation and modeling, you will need to simulate extra anatomy/organs beyond this dataset.

The list-mode data, sinograms, and reconstruction pipelines may require specific tools like STIR and SIRF, or compatibility; you must check the formats before using them.

The “whole body” aspect in your project requires additional modeling beyond what the brain dataset provides.
