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
