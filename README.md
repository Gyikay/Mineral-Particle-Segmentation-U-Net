# Mineral-Particle-Segmentation-U-Net
Semantic segmentation of mineral particles in conveyor-belt images using U-Net
# Mineral-Particle-Segmentation-U-Net

Semantic segmentation of mineral particles in conveyor-belt and tray images using a U-Net architecture.  
This repository accompanies the capstone project for the Master of Predictive Analytics (Resource Operations Analytics), Curtin University.

---

## 1. Project Overview

Real-time monitoring of rock fragmentation and particle size distribution (PSD) is critical for optimising blasting, crushing and downstream mineral processing. Traditional thresholding and watershed-based segmentation methods often fail under:

- Uneven illumination  
- Broad particle size distributions  
- Touching/overlapping fragments  
- Noisy plant environments  

This project develops and evaluates U-Net-based semantic segmentation models for mineral particles using three case studies that progressively improve label quality and dataset design:

1. **Case Study 1 – Auto-labelled images** (MATLAB Auto Labeller, Maerz 2001 data)  
2. **Case Study 2 – Manually labelled image** (MATLAB Image Labeller + Python U-Net)  
3. **Case Study 3 – Pre-labelled benchmark dataset** (Fu & Aldrich, 2023, LOOCV U-Net)

The core research questions are:

- How does **label quality** (auto vs manual vs benchmark) affect U-Net segmentation performance?
- How does **dataset type and size** influence model generalisation to new images and conditions?

---

## 2. Repository Structure

Suggested layout for this repository:

```text
Mineral-Particle-Segmentation-U-Net/
│

├── notebooks/
│   ├── case2.m
│   ├── UNet Code.ipynb
│   └── Case3_loocv.ipynb

│
├── data/
│   ├── case1/                           
│   ├── case2/
│   └── case3/
│   └── README.md                        # Notes on obtaining datasets
│
├── results/
│   ├── case1/
│   ├── case2/
│   └── case3/
│
├── models/
│   ├── case2_unet_rocks_manual.keras
│   └── case3_unet_best_fold3.keras
│
├── docs/
│   ├── Final_Report.pdf
│   └── Slides.pdf
│
├── requirements.txt
├── LICENSE
└── README.md
