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
Note: Raw images and large datasets are not necessarily stored in the repository.
Use data/README.md to explain where they can be obtained (e.g. Maerz, Fu & Aldrich datasets, internal plant images).

3. Methodology
3.1 Overall Approach

The project uses a mixed MATLAB–Python workflow:

MATLAB for data preparation, auto-labelling, and manual ground-truth creation.

Python (TensorFlow/Keras) for U-Net model training, cross-validation, and quantitative evaluation.

Across all cases, the core U-Net setup includes:

Input size: 512 × 512 pixels

Binary classes: rock (foreground) vs background

Optimiser: Adam, learning rate 1e-4

Loss: Binary Cross-Entropy + Dice loss (with variants such as Focal-Tversky in Case 3)

Early stopping based on validation loss / Dice score

3.2 Case Study 1 – Auto-Labelled Images (MATLAB Auto Labeller)

Objective:
Explore the feasibility of using automatically generated masks as pseudo-ground-truth for training a U-Net.

Key steps:

Dataset preparation

Conveyor-belt rock image selected from Maerz (2001).

RGB image converted to grayscale and paired with a binary mask.

Auto-mask generation

Adaptive local thresholding (adaptthresh, imbinarize).

Morphological opening/closing (imopen, imclose) and small-object removal (bwareaopen).

Optional distance transform + watershed to separate touching fragments.

Refinement and verification

Masks refined in MATLAB Image Segmenter: correction of edges, filling gaps.

Refined masks saved under structured images/ and masks/ folders.

Prototype U-Net attempt

Baseline U-Net implemented in MATLAB Deep Learning Toolbox.

Initial training halted due to format and label-quality issues; masks were corrected to binary, single-channel format.

Due to poor, noisy labels and merged fragments, no full U-Net training was pursued in Case 1; it serves as a qualitative baseline and motivation for improved labelling.

3.3 Case Study 2 – Manually Labelled Image (MATLAB + Python U-Net)

Objective:
Investigate how high-quality manual ground truth improves segmentation compared with auto-labelled watershed masks and assess generalisation to new images.

Pipeline:

Manual labelling in MATLAB

High-resolution rock-fragment image imported into MATLAB Image Labeller.

Each fragment manually outlined, producing an accurate pixel-level binary mask (1 = rock, 0 = background).

Masks exported as image2_gt.png and stored as dataset_case2/images/ and dataset_case2/masks_GT/.

Model training (Python)

U-Net re-implemented in TensorFlow/Keras with encoder–decoder structure and skip connections.

Configuration:

Optimiser: Adam (lr = 1e-4)

Loss: Binary Cross-Entropy + (1 – Dice)

Batch size: 8

Max epochs: 40 with early stopping (patience = 5)

Callbacks: ModelCheckpoint for best validation Dice

Images and masks:

Normalised to [0, 1]

Resized to 512 × 512

Augmented (rotation, flips, brightness/contrast changes).

Validation and testing

Cropped into overlapping tiles; 80% used for training and 20% for validation.

Inference performed on:

The training image (for sanity check).

Two unseen images (image_rock_on_a_conveyor_belt.png, rocks_on_lid.png) to test generalisation.

Metrics & post-processing

Accuracy, Precision, Recall, Dice, IoU, Specificity, and Boundary F1 (±3 px).

Post-processing: border clearing, morphological filtering, small-object removal to refine masks.

3.4 Case Study 3 – Pre-Labelled Benchmark Dataset (Fu & Aldrich, 2023)

Objective:
Benchmark the U-Net on a pre-labelled, high-quality dataset and evaluate robustness using Leave-One-Out Cross-Validation (LOOCV).

Dataset:

Four RGB images of rock fragments on circular trays with corresponding binary masks.

Provided by Fu & Aldrich (2023).

Stored in aligned images/ and masks/ folders (e.g., image1.png ↔ image1.png).

A strict pairing function (list_pairs_strict()) verifies one-to-one mapping.

Training configuration:

Input size: 512 × 512 × 3

Optimiser: Adam (lr = 1e-3)

Loss: Focal-Tversky / BCE + Dice (for class imbalance)

Batch size: 4

Epochs: 40 with early stopping after 6 epochs of no improvement

Cross-validation: 4-fold LOOCV (3 images train, 1 test per fold).

Data handling:

On-the-fly augmentation: random flips, small rotations, brightness/contrast changes.

Foreground-biased patch sampling ensures ≈60% of training patches contain rock pixels, preventing collapse to background.

Post-processing:

clear_border() removes rock regions connected to image edges (tray rim artefacts).

Predictions saved as binary PNGs and compared against ground truth using IoU and Dice.

4. Results Summary
4.1 Case 1 – Auto-Labelled Masks (No Training)

Auto-labels exhibited merged fragments, boundary loss, and heavy noise.

Qualitative assessment showed poor alignment with true rock fragments.

Conclusion: Auto-labelling alone was insufficient as ground truth for deep-learning segmentation in this dataset.

4.2 Case 2 – Manual Ground Truth vs Auto-Labels

Two experiments:

Baseline: U-Net trained on auto-labelled watershed masks.

Improved: U-Net retrained on manually labelled ground truth from MATLAB.

Quantitative comparison (example values):

Metric	Auto Mask (Baseline)	Manual GT (Retrained)
Accuracy	0.6337	0.7628
Precision	0.5970	0.7857
Recall	0.9996	0.9437
Dice	0.7476	0.8575
IoU	0.5969	0.7506
Boundary F1	0.1466	0.4824

Interpretation:

Manual labels significantly improved Dice, IoU and boundary quality, confirming that label fidelity is crucial.

On unseen conveyor and tray images, metrics were high (Dice ≈ 0.96, IoU ≈ 0.92), but qualitative overlays showed some broken fragments and incomplete regions, highlighting limited training diversity and domain shift.

4.3 Case 3 – Fu & Aldrich Benchmark Dataset

Using the optimised binary U-Net with BCE + Dice loss and foreground-biased sampling:

Fold	Test Image	IoU	Dice
1	image1.png	0.799	0.888
2	image2.png	0.830	0.907
3	image3.png	0.839	0.913
4	image4.png	0.798	0.888

Mean IoU ≈ 0.82, Mean Dice ≈ 0.90.

Qualitatively, the model achieved clear boundary delineation and accurate fragment separation across all folds.

Experiments confirmed that high-quality, pre-curated labels + targeted loss design yield the most stable and transferable performance.

5. Installation & Environment
5.1 Python Environment

Create and activate a virtual environment:

python -m venv rocks-seg
rocks-seg\Scripts\activate   # Windows
# source rocks-seg/bin/activate   # Linux / macOS
python -m pip install --upgrade pip

5.2 Dependencies

Install dependencies from requirements.txt:

pip install -r requirements.txt


requirements.txt includes (versions can be pinned as required):

tensorflow==2.15.0
numpy
pandas
scikit-learn
scikit-image
opencv-python
matplotlib
albumentations
tqdm
scipy
reportlab

5.3 Reproducibility

Random seeds are fixed (SEED = 42) for random, numpy, and tensorflow.

GPU availability can be checked via:

import tensorflow as tf
print("GPUs available:", tf.config.list_physical_devices('GPU'))


MATLAB R2023b is used for manual ground-truth generation and conversion of .mat label files to PNG masks.

6. How to Run

Note: Adjust script names and paths to match your actual repo.

6.1 Case 2 – Manual Ground Truth U-Net
python src/case2_train_unet.py \
  --data_root data/case2 \
  --epochs 40 \
  --batch_size 8 \
  --lr 1e-4


Outputs:

Trained model: models/unet_rocks_manual.keras

Metrics: results/case2/metrics_case2.csv

Overlays: results/case2/overlays/*.png

6.2 Case 3 – LOOCV on Fu & Aldrich Dataset
python src/case3_loocv_unet.py \
  --data_root data/case3_fu_aldrich \
  --epochs 40 \
  --batch_size 4 \
  --lr 1e-3


Outputs:

Per-fold models: models/case3_fold*.keras

Per-fold metrics: results/case3/metrics_loocv.csv

Qualitative overlays: results/case3/overlays/fold*_image*.png

7. Comparison with Fu & Aldrich (2023)

Fu & Aldrich (2023):

Objective: Real-time particle-size analysis on conveyor belts using dense CNNs.

Preprocessing: SLIC superpixels to simplify textures.

Labels: Three-class masks (particle, boundary, background).

Architecture: U-Net + ResNet-34 backbone.

Loss: Dice + Focal loss.

Metrics: Particle count and size-distribution accuracy.

This project (Case 3):

Objective: Benchmark segmentation quality and cross-dataset generalisation using U-Net.

Preprocessing: Direct training on binary masks (rock vs background), preserving texture detail.

Architecture: Modified vanilla U-Net with batch norm and dropout.

Loss: BCE + Dice with foreground-biased sampling.

Metrics: IoU, Dice, overlay visualisation.

Validation: 4-fold LOOCV with patch-based augmentation.

8. Key Takeaways

Label quality is the dominant factor in segmentation performance:

Auto-labels (Case 1) were insufficient for reliable training.

Manual labels (Case 2) substantially improved Dice, IoU, and boundary quality.

Benchmark labels (Case 3) delivered the most stable and accurate results.

Loss design and sampling strategy (BCE + Dice, Focal-Tversky, foreground-biased sampling) are crucial when working with small, imbalanced mining datasets.

High-quality, standardised datasets such as Fu & Aldrich (2023) provide strong baselines for future models and transfer-learning experiments in mineral image segmentation.

9. Acknowledgements

Supervisor: Prof. Chris Aldrich (Curtin University) for guidance, dataset access, and feedback.

Curtin University – Master of Predictive Analytics program and teaching staff.

Family, friends, and colleagues for their support throughout this project.

10. Citation

If you use this code or methodology in academic work, please cite the associated dissertation (once published) and the original dataset paper:

Fu, X., & Aldrich, C. (2023). [Full paper title]. [Journal], [Volume], [Pages].

Maerz, N. H. (2001). Automated online optical sizing analysis. [Publication details].
