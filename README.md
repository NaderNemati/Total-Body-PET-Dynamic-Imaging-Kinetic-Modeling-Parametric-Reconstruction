<div align="center">
  <table border=0 style="border: 0px solid #c6c6c6 !important; border-spacing: 0px; width: auto !important;">
    <tr>
      <td valign=top style="border: 0px solid #c6c6c6 !important; padding: 0px !important;">
        <div align=center valign=top>
          <img src="https://github.com/NaderNemati/Total-Body-PET-Dynamic-Imaging-Kinetic-Modeling-Parametric-Reconstruction/blob/main/image/F1.large.jpg" alt="Project Structure" style="margin: 0px !important; height: 400px !important;">
        </div>
      </td>
    </tr>
  </table>
</div>





# Total-Body-PET-Dynamic-Imaging-Kinetic-Modeling-Parametric-Reconstruction

## Description

Positron Emission Tomography (PET) has long been used in a static mode—after injection of a tracer, a scan is acquired (for example 60 min post-injection) typically over a limited axial field of view (AFOV). In contrast, modern total-body or large-axial-field-of-view (LAFOV) PET systems extend the axial span to ~1 m or more, allowing nearly the whole body to be imaged in one scan. This technological advancement enables significant opportunities:

1-**Dynamic whole-body acquisition:** Many organs and tissues can be imaged over time, enabling extraction of time-activity curves (TACs) across the body.

2-**Kinetic modeling and parametric mapping:** With dynamic data from many organs or voxels, compartment models (1-tissue, 2-tissue, Patlak, Logan) can be applied to compute parametric maps (e.g., Ki, VT, K1, k2).

3-**Systemic physiology and multi-organ interplay:** Full-body coverage supports studies of tracer kinetics beyond single organs—tumour-host interaction, pharmacokinetics, immunology.

4-**Higher sensitivity, lower dose, extended imaging times:** Long AFOV coupled with advanced detector design yields high sensitivity, enabling faster scans, lower tracer doses, or delayed imaging (many hours post-injection).

5-**Challenges:** Managing large 4D datasets (x, y, z, t), correcting for whole-body motion (respiratory/cardiac/subject), deriving image-based input functions, selecting appropriate models, dealing with voxel-level noise, and implementing efficient reconstruction and parametric workflows (e.g., FBP, OSEM, TOF, deep learning).

6-**Clinical and research impact:** Applications span oncology (lesion dynamics, response to therapy), cardiology, neurology, inflammation/infection, drug development. The transition from static SUV imaging to full-body dynamic quantification is a major shift.


In this mini-project, you are building a pipeline from simulation → reconstruction → ROI/voxel modeling → parametric map generation (Ki, VT), which aligns directly with this frontier. By generating your own 4D dynamic dataset, reconstructing frames, extracting TACs, performing voxelwise Patlak/Logan, and generating parametric maps, you demonstrate key skills and understanding for research in total-body PET.

## Key Recent Review Papers

Sun, Y., et al. Performance and application of the total-body PET/CT scanner: a literature review. [Open Access](https://ejnmmires.springeropen.com/articles/10.1186/s13550-023-01059-1?utm_source=chatgpt.com)

Sun, T., et al. Current progress and future perspectives in total‐body PET imaging, part I: Data processing and analysis. [Open Access](https://onlinelibrary.wiley.com/doi/epdf/10.1002/ird3.66)

## Background & Motivation

The development of LAFOV-PET systems has enabled whole-body dynamic imaging and voxel-wise parametric modeling, as described in the reviews above. This pipeline implements a simulation-to-parametric workflow that aligns with this emerging modality: simulating dynamic total-body PET, reconstructing frames, extracting TACs for multiple organs and voxels, and computing Ki and VT maps.

## Mathematical & Modelling Framework

Dynamic PET datasets, especially whole-body or full-organ coverage, involve several key mathematical features:

Time-Activity Curves (TACs): For each ROI/voxel we have Ct(t) = [Ct(t₁), Ct(t₂), …].

Input Function (Cp(t)): Plasma or blood tracer concentration vs time, required for compartmental/graphical modelling.

Compartmental models: Example: one-tissue model:

$$
C_t(t) = K_1 \int_{0}^{t} C_p(\tau) \; e^{-k_2 (t - \tau)} \; d\tau
$$


yielding parameters K₁, k₂.

Graphical modelling: Patlak and Logan analyses provide Ki (net influx) and VT (volume of distribution) after a threshold time t*.

Voxel-wise parametric mapping: Voxel-based fitting of the above models allows spatial maps of K₁, k₂, Ki, VT and goodness-of-fit (e.g., R²).

Statistical/fit metrics: Goodness-of-fit (R²), residuals, parameter uncertainties, frame-wise noise, count statistics matter.

Data dimensionality: Datasets are 4D (x,y,z,t), with many frames, often full-organ or whole-body coverage.

Noise & count-statistics: Modeling and processing must consider shot-noise, timing jitter, motion, partial volume effects.

Summary Table of Features

## Summary Table of Features

| Feature                     | Description                                   | Relevance for Pipeline                           |
| --------------------------- | --------------------------------------------- | ------------------------------------------------ |
| TACs                        | Activity vs time vectors for ROI/voxels       | Extract TACs, fit models                         |
| Input Function (Cp(t))      | Plasma tracer concentration vs time           | Essential for modelling                          |
| Compartment parameters      | K₁, k₂ (and possibly k₃/k₄)                   | Pipeline estimates K₁, k₂                        |
| Graphical metrics           | Ki, VT                                        | Pipeline computes Ki/VT maps                     |
| Voxel-wise maps             | Spatial maps of parameters and fit statistics | Pipeline outputs Ki_map, VT_map, R²_map          |
| Goodness-of-fit & residuals | R², residual error, parameter uncertainty     | Evaluate model robustness                        |
| Data dimensionality         | Large 4D volumes (x,y,z,t)                    | Pipeline simulates and handles 4D                |
| Noise & count statistics    | Variability due to low counts, timing, motion | Pipeline simulates noise, assesses bias/variance |


## Dataset Description

The dataset used for benchmarking and comparison is the Monash vis-fPET-fMRI Dataset (OpenNeuro accession ds003382). This dataset contains simultaneous FDG-fPET and BOLD-fMRI acquisitions from 10 healthy young adults.


#### Key attributes

**Subjects:** Subjects: 10 healthy adults (age range ~18-49, mean ~29 yrs) were scanned. 

**Acquisitions:** Participants underwent simultaneous BOLD-fMRI and dynamic PET with [^18F]-FDG radiotracer under a visual stimulation paradigm (checkerboard task) plus resting periods.

**PET data:** The dataset includes list-mode PET raw data, sinograms, and reconstructed dynamic PET images.

**fMRI & anatomical MRI:** Standard structural T1, field maps, and functional EPI scans are included.

**Format and metadata:** Data has been organized in a Brain Imaging Data Structure (BIDS)-like format, with sidecar JSON files, metadata, and a participants table.

## Dataset Features Table

| Metric                 | Value / Description                                                        |
| ---------------------- | -------------------------------------------------------------------------- |
| Number of participants | 10 healthy adults                                                          |
| Age range              | ~18-49 years                                                               |
| Mean age               | ~29 years                                                                  |
| Tracer                 | [¹⁸F]-FDG for functional PET                                               |
| Data coverage          | Whole brain dynamic PET (not yet full-body)                                |
| Data formats included  | List-mode PET, sinograms, reconstructed dynamic PET, fMRI & structural MRI |




## Repository Structure Tree
```python
pet-mini/
├── README.md
├── LICENSE
├── requirements.txt
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

```

## Steps to Run the Pipeline

1-Create and activate your virtual environment (e.g., python3 -m venv .venv; source .venv/bin/activate).

2-Install dependencies:

```python
pip install -r env/requirements.txt
```
3-Make the run script executable:

```python
chmod +x run_demo.sh

```
4-Execute the full pipeline:

```python
./run_demo.sh

```

5-After completion, inspect the outputs:
```bash
results/roi_fits_sim/roi_params_summary.csv

results/paramaps_sim/Ki_map.nii.gz, VT_map.nii.gz

results/figures_sim/ for PNG visualisations

results/report_sim.md for the markdown summary report
```

# LICENSE

#### Copyright (c) 2025 Nader Nemati
#### Licensed under the MIT License. See the LICENSE file in the project root.





















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
├── requirements.txt
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

```

## Dataset Description

The dataset is known as the **Monash vis‑fPET‑fMRI Dataset (OpenNeuro accession ds003382)** and contains simultaneous functional PET and fMRI brain data from healthy human subjects.

#### Key attributes

**Subjects:** Subjects: 10 healthy adults (age range ~18-49, mean ~29 yrs) were scanned. 

**Acquisitions:** Participants underwent simultaneous BOLD-fMRI and dynamic PET with [^18F]-FDG radiotracer under a visual stimulation paradigm (checkerboard task) plus resting periods.

**PET data:** The dataset includes list-mode PET raw data, sinograms, and reconstructed dynamic PET images.

**fMRI & anatomical MRI:** Standard structural T1, field maps, and functional EPI scans are included.

**Format and metadata:** Data has been organized in a Brain Imaging Data Structure (BIDS)-like format, with sidecar JSON files, metadata, and a participants table.

| Metric                 | Value                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------ |
| Number of subjects     | 10 healthy adults (age 18-49) ([PMC][1])                                             |
| Tracer used            | [^18F]-FDG (functional PET) ([PMC][1])                                               |
| Scanner acquisition    | Simultaneous PET + BOLD fMRI ([PMC][1])                                              |
| Data coverage          | Whole brain field of view ([PMC][1])                                                 |
| Data formats included  | List-mode PET, sinogram, reconstructed dynamic PET, fMRI & structural MRI ([PMC][1]) |
| Dynamic frame duration | ~1-min bins (typical) ([PMC][1])                                                     |


#### Summary Table of Features:

| Feature                        | Description                                                        | Implication for your pipeline                   |
| ------------------------------ | ------------------------------------------------------------------ | ----------------------------------------------- |
| Time-Activity Curve (TAC)      | Vector of activity vs time for ROI/voxel                           | You extract TACs and fit models                 |
| Input Function (Cp(t))         | Plasma or blood activity vs time                                   | Needed for compartmental or graphical modelling |
| Compartmental model parameters | K₁ (influx), k₂ (efflux), maybe k₃/k₄ depending on model           | Your 1-T model uses K₁ and k₂                   |
| Graphical model metrics        | Ki (net uptake), VT (volume of distribution)                       | You produce Ki and VT maps                      |
| Voxel-wise parametric maps     | Spatial maps of parameters (K₁, k₂, Ki, VT, R²)                    | Your pipeline outputs these                     |
| Goodness-of-fit / residuals    | Statistics like R², residual standard error                        | Important for assessing modelling robustness    |
| Data dimensionality            | 4D data (x,y,z,t), many frames, often full brain or body           | Your simulation replicates this large scale     |
| Noise / count statistics       | Variability due to tracer kinetics, counts/frame, timing precision | You simulate noise and assess bias/variance     |



This dataset consists of 10 healthy adults aged 18-49 who underwent simultaneous FDG-PET and BOLD-fMRI scanning under a visual stimulation protocol. The FDG tracer was used, and the dataset includes rich dynamic imaging: list-mode PET, sinograms, reconstructed PET frames in ~1-minute bins, along with full brain coverage. Because it offers both raw and reconstructed data across multiple modalities, it serves as a valuable benchmark for simulation, reconstruction, TAC extraction, and kinetic modeling workflows.





It contains dynamic PET data with many individual time frames rather than just a single static image, which aligns smoothly with your pipeline’s steps of TAC extraction, kinetic modeling, and parametric mapping. The dataset spans an entire brain field of view—not just a small organ region—making it ideal for voxel-wise modeling and whole-volume analysis. It also offers both raw/list-mode and reconstructed data, giving you flexibility to simulate, reconstruct, benchmark, or compare methods. By leveraging this existing open dataset, you gain a real-data reference to compare your simulated results against, which is extremely helpful for demonstrating the validity and robustness of your pipeline.


#### Limitations/Considerations

Although dynamic, the field of view is still limited to the brain rather than full-body total-body PET, so if your project is aiming for whole-body simulation and modeling, you will need to simulate extra anatomy/organs beyond this dataset.

The list-mode data, sinograms, and reconstruction pipelines may require specific tools like STIR and SIRF, or compatibility; you must check the formats before using them.

The “whole body” aspect in your project requires additional modeling beyond what the brain dataset provides.

# Steps to run the pipeline

1- Place the script above in your root folder, name it run_pipeline.sh

2- Make it executable:
```python
chmod +x run_pipeline.sh
```

3- Ensure your virtual environment exists (e.g., .venv) and you have env/requirements.txt.

4- Run it:

```python
./run_pipeline.sh
```

5- At the end, inspect:

```python
results/roi_fits_sim/roi_params_summary.csv

results/paramaps_sim/Ki_map.nii.gz, VT_map.nii.gz

results/figures_sim/ for PNGs

results/report_sim.md for the markdown report
```

# LICENSE

#### Copyright (c) 2025 Nader Nemati
#### Licensed under the MIT License. See the LICENSE file in the project root.

