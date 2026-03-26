# MEPrep: A Robust Pipeline for Multi-Echo fMRI Denoising and Preprocessing 

## Zhishun Wang, Feng Liu, Rachel Marsh, Gaurav H. Patel, Jack Grinband

Multi-echo fMRI has emerged as a powerful strategy to mitigate head motion-related noise and minimize susceptibility-related signal loss in BOLD data.
Incorporating multi-echo independent component analysis (ME-ICA) to effectively distinguish between BOLD-related (TE-dependent) signals and non-BOLD (TE-independent)
noise yields substantial enhancements in performance compared to traditional echo-combination methods. We introduce a novel ICA-based denoising step applied to raw
multi-echo data (preICA), prior to optimal (i.e. T2*-weighted) echo combination and ME-ICA that yields substantial gains in data denoising. Our findings reveal that preICA
significantly enhances the efficacy of optimal echo combination and ME-ICA to reduce noise. To facilitate the reliable processing of multi-echo fMRI data, we integrated preICA
and ME-ICA into fMRIPrep, resulting in the creation of a robust multi-echo processing pipeline, called MEPrep, offering flexibility in preprocessing options (with or without preICA
and/or ME-ICA) beyond the echo combination approach offered by fMRIPrep. We validated MEPrep on an open resting-state multi-echo fMRI dataset, demonstrating that incorporating the preICA
step leads to statistically significant improvements in denoising efficacy, as evidenced by (1) enhanced T2* exponential model fitting accuracy; (2) reduced motion-related BOLD fluctuations;
(3) increased temporal signal-to-noise ratio; (4) improved spatial and temporal reliability of functional connectivity; and (5) increased Shannon entropy. MEPrep outperforms existing pipelines
by synergistically integrating preICA and ME-ICA, achieving superior noise suppression while preserving the neurobiological complexity of denoised BOLD signals. By automating multi-echo
preprocessing within a robust pipeline, MEPrep provides a scalable solution for high-quality multi-echo fMRI data preprocessing. The pipeline is openly available, ensuring reproducibility
and accessibility for the neuroimaging community.

<img width="3587" height="2250" alt="MEPrep Framework" src="https://github.com/user-attachments/assets/9f048ee6-79ce-4cd9-86f9-909b2d1ea13e" />

<img width="4000" height="2192" alt="Main Results A" src="https://github.com/user-attachments/assets/389ff754-692e-4dfc-91c3-f0ff5d04843c" />

<img width="4000" height="2250" alt="Main Results B" src="https://github.com/user-attachments/assets/1b109962-ef79-43ee-8d37-33257e9a4962" />

<img width="4000" height="2250" alt="Main Results C" src="https://github.com/user-attachments/assets/c9f4bf4d-3d2d-4c96-a892-3a2ae336c4ab" />

## Installation 
```bash
git clone https://github.com/zswang-brainimaging/MEPrep.git
cd MEPrep/fMRIPrep_Source
cp -r ../MEPrep_Source .
cp ./MEPrep_Source/Dockerfile .
cp ./MEPrep_Source/requirements.txt .
docker build -t  zswang2020/meprep_final:latest .
```
## Example
```bash
run_meprep.sh

#!/bin/bash
Input_Dir=/your/path/to/BIDS_Data_Folder_name
Output_Dir=/your/path/to/MEPrep_Output_Folder_Name
Work_Dir=/your/path/to/work_folder_name
Freesurfer_license_Dir=/your/path/to/freesurfer_license_folder
Predenoising_choice=preICA_scme   ## set it to Raw if choosing no pre-denoising
MEPrep_Image=zswang2020/meprep_final:latest
## replace the participant ID of 123456 with your participant ID
## run the Docker image:
docker run -v $Input_Dir:/input -v  $Output_Dir:/output -v $Work_Dir:/work  -v $Freesurfer_license_Dir:/fs_license_dir -it -d  $MEPrep_Image  /input  /output  participant  --participant-label 123456  --tedana-choice Both --preDenoising-choice $Predenoising_choice  --cifti-output 91k --output-spaces MNI152NLin2009cAsym:res-native T1w -w /work  --write-graph --output-layout legacy --fs-license-file /fs_license_dir/license.txt






