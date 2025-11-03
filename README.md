# ag5003_sr4213_ssd2184 Iris Recognition Project

## Project Overview
This project implements an **iris recognition pipeline** based on classical methods from the literature, including Ma et al. (2003) and Daugman's Rubber Sheet Model. The pipeline processes iris images from the CASIA Iris Image Database (version 1.0) and performs both **identification** and **verification** tasks.  

The main steps of the pipeline are:

1. **Iris Localization**  
   - Detects the **pupil and iris boundaries** using a robust double-circle approach with refinement.  
   - Inspired by:  
     - [Ma et al., PAMI 2003](https://cse.msu.edu/~rossarun/BiometricsTextBook/Papers/Iris/Ma_IrisTexture_PAMI03.pdf)  
     - [A fast and robust iris localization method](https://www.researchgate.net/publication/235222151_A_fast_and_robust_iris_localization_method)  
     - [Additional reference: MDPI Sensors, 2023](https://www.mdpi.com/1424-8220/23/4/2238)  
   - Output: pupil circle `(x, y, r)` and iris circle `(x, y, r)`  

2. **Iris Normalization**  
   - Maps the iris from Cartesian coordinates to polar coordinates using **Daugman's Rubber Sheet Model**.  
   - Produces a fixed-size normalized iris image suitable for feature extraction.  

3. **Image Enhancement**  
   - Applies **CLAHE (Contrast Limited Adaptive Histogram Equalization)** for contrast enhancement.  
   - Performs median filtering to reduce noise and suppress eyelashes.  
   - Ensures robust normalization of intensity values.  

4. **Feature Extraction**  
   - Implements **Ma et al.'s spatial filters** for texture analysis.  
   - Extracts features using mean and average absolute deviation of filtered blocks.  
   - Produces a **1D feature vector** per iris sample.  

5. **Iris Matching**  
   - Uses **Linear Discriminant Analysis (LDA)** for dimensionality reduction and improved class separability.  
   - Matches features with a **nearest-center classifier** using multiple distance metrics (`l1`, `l2`, `cosine`).  
   - Rotation-invariant: computes features for 7 rotated versions of each test iris and selects the minimum distance.  

6. **Performance Evaluation**  
   - **Correct Recognition Rate (CRR)**: shows identification performance per distance metric.  
   - **ROC Curves** and **Equal Error Rate (EER)**: shows verification performance.  
   - All tables and figures are automatically saved as images.  

---

### Limitations
- **Computational cost**: Running the pipeline on multiple rotations per image makes it slower than ideal.  
- **Robustness to occlusions**: Eyelashes, eyelids, or reflections can still mess with iris localization and feature extraction.  
- **Fixed block size in feature extraction**: Using 8×8 blocks might miss some important texture details in irises that have unusual patterns.  
- **Dataset-specific tuning**: Right now, the system is set up for CASIA-Iris V1 and might need adjustments for other datasets.  

### Potential Improvements
- Explore **deep learning-based feature extraction** to improve accuracy and make the system more robust to variations in iris patterns.  
- Consider using **attention mechanisms** so the model focuses on the most informative parts of the iris.  
- Add **automatic handling for failed localizations**, so images where the pupil/iris detection fails are caught and managed without manual skipping.  
- Make the pipeline **faster and more efficient**, for example by reducing the number of rotations we check or by parallelizing feature extraction across multiple cores.

---

## Peer Evaluation Form
1. Iris Localization: ag5003
2. Iris Normalization: ssd2184
3. ImageEnhancement: ssd2184
4. Feature Extraction: sr4213
5. Iris Matching: sr4213
6. Performance Evaluation: sr4213, ssd2184
7. Iris Recognition main function: sr4213 & ag5003
8. Code Integration: ag5003

## Example Usage

To run the iris recognition pipeline, make sure you have downloaded and extracted the CASIA Iris Image Database (version 1.0). The `.rar` files need to be uncompressed so that the directory structure looks like this:

datasets/
|-------CASIA_Iris/
|-------CASIA Iris Image Database (version 1.0)/
|----------------------------------------------001/
|----------------------------------------------002/
...

```bash
#Make sure all required packages are installed
pip install numpy opencv-python matplotlib scikit-learn scipy

#Run the main pipeline
python IrisRecognition.py
