# ag5003_sr4213_ssd2184 Iris Recognition Project

## Project Overview
This project implements an **iris recognition pipeline** inspired by Ma et al. (2003). The pipeline processes iris images from the CASIA Iris Image Database (version 1.0) and performs both **identification** and **verification** tasks.  

The main steps of the pipeline are:

**Iris Localization**  
   - Detects the **pupil and iris boundaries** using a robust double-circle approach with refinement.  
   - Inspired by:  
     - [Ma et al., PAMI 2003](https://cse.msu.edu/~rossarun/BiometricsTextBook/Papers/Iris/Ma_IrisTexture_PAMI03.pdf)  
     - [A fast and robust iris localization method](https://www.researchgate.net/publication/235222151_A_fast_and_robust_iris_localization_method)  
     - [Additional reference: MDPI Sensors, 2023](https://www.mdpi.com/1424-8220/23/4/2238)  
   - Output: pupil circle `(x, y, r)` and iris circle `(x, y, r)`  

**Iris Normalization**  
   - Maps the iris from Cartesian coordinates to polar coordinates using **Daugman's Rubber Sheet Model**.  
   - Produces a fixed-size normalized iris image suitable for feature extraction.  

**Image Enhancement**  
   - Applies **CLAHE (Contrast Limited Adaptive Histogram Equalization)** for contrast enhancement.  
   - Performs median filtering to reduce noise and suppress eyelashes.  
   - Ensures robust normalization of intensity values.  

**Feature Extraction**  
   - Implements **Ma et al.'s feature extraction procedure** after the generation of a normalized and enhanced iris image.
   - Run two different spatial filters with different (delta x, delta y) values, (3, 1.5) and (4.5, 1.5)
   - Use the defined filter created by the paper to generated a modified Gabor filter with modulating sinusoidal function.
   - The spatial frequency (f) is defined as 1/delta_y and the filter kernel size is set to 35×35 pixels.
   - The region of interest (ROI) is extracted as the upper 48×512 portion of the normalized iris, corresponding to the region closest to the pupil (as described by the authors of the paper).
   - This is calculated by going through each kernel (gabor filter run for every channel) and each filter is applied to the ROI to obtain a magnitude response image.
   - The ROI is divided into 8×8 pixel blocks. For each block, two statistical measures are computed:
       - Mean (m) = average intensity of the filtered magnitude.
       - Average Absolute Deviation (sigma) = texture variability within the block.
   - Take the statistics for each block and concatenate across all filters to form a single 1D feature vector representing the iris (the output).
   - The final feature vector is normalized to help improve matching performance (to standardize scales).
   - This output provides distinctive features of an iris image to use for further processing and classification.

**Iris Matching**
   - Project features using Fisher Linear Discriminant Analysis (LDA) and then perform nearest-center classification under multiple distance metrics.
   - LDA is also performed for a variety of dimensions so that the CRR curve can be calculated across dimensions later. These were chosen from reading the curve on the paper and adding several more values in between for a smoother curve.
   - The distance measures used were the l1 distance (sum of absolute differences (city block distances), the l2 distance (squared euclidian norm), and cosine similarity.
   - The smallest or best distance tells us the predicted y value or identity the iris belongs to.
   - Computes features for 7 rotated versions of each test iris and selects the minimum distance.
   - Returns the predicted values for every single distance measure and also other intermeediate results like the actual class centers and distances for each metric so they can be used for the calculations for the CRR and ROC curves.

**Performance Evaluation**  
   - **Correct Recognition Rate (CRR)**: shows identification performance per distance metric.  
   - **ROC Curves** and **Equal Error Rate (EER)**: shows verification performance.  
   - All tables and figures are automatically saved as images.
  
**Helper Functions**
   - PreProcessing.py defines helper functions to run the pipeline.
   - complete_rotations: returns a list of rotated images to account for eye rotation. Uses circular horizontal shifts for rotation invariance.
   - load_iris_images: helps process the input data from the CASIA database by iterating through the directory and returning sorted paths to each image.
  
**Iris Recognition (main functionn)**
   - The main function of the code.
   - Calls the functions definind in the above steps (localization, image enhancement, feature extraction, iris matching, performance evaluation).
   - Splits dataset into training and test data (session 1 / folder 1 = train and session 2 / folder 2 = test).
   - Calls the function pipeline in order for training and test images.
   - Calls rotation function to rotate images when performing matching on the test dataset only.
   - There are no returns, but outputs CRR and ROC results in table and graph format after calling.
---

### Limitations
- Running the pipeline on multiple rotations per image improves rotation invariance but increases runtime. In this approach, each test image is expanded into seven rotated versions, all of which pass through the full iris matching process. For each rotation, the system computes the distance to every class center across all metrics, and then selects the smallest (best) distance as the final score. This effectively increases the workload by a factor of 7× in the test dataset, making the process computationally expensive and slowing performance.
- Eyelashes, eyelids, or reflections can affect iris localization and feature extraction by occluding the iris and pupil. This affects localization (and further downstream tasks) as this functions goal is to detect the pupil and iris boundaries, potentially causing important parts of the image to be missing or obscured. This affects how identifying the feature vector generated in feature extraction is and thus lowering recognition accuracy. 
- In the feature extraction stage, the normalized iris region is divided into 8×8 pixel blocks to compute the mean and average absolute deviation. This can cause over-smoothing or miss texture in the image, especially in irises with detailed patterns. While this implementation is faithful to the paper's methodology, using a more adaptive block-size could capture these variations and could be explored in further iterations of this project.
- Right now, the system is set up for CASIA-Iris V1 and might need adjustments for other datasets. Specific parameters such as the region of interest and kernel size are tuned to improve performance when identifying irises from this dataset and with respect to the papers definitions. Applying this recognition system to other datasets may require re-tuning these parameters and re-training the model to accommodate differences in image resolution, lighting conditions, and the contrast between the pupil and iris.

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
6. Performance Evaluation: sr4213 & ssd2184
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
