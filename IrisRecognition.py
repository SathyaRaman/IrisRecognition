#main function to call everything & build train and test datasets
import os
from PreProcessing import load_iris_images
import cv2
from IrisLocalization import IrisLocalization
from FeatureExtraction import feature_extraction
from ImageEnhancement import image_enhancement
from IrisMatching import iris_matching
from IrisNormalization import iris_norm
import matplotlib.pyplot as plt
from PreProcessing import complete_rotation
from PerformanceEvaluation import CRR_Result, ROC_Result
import numpy as np


def IrisRecognition(dataset_path):
    """
    Main function to perform the iris recognition pipeline.

    Steps:
    1. Iris Localization
    2. Iris Normalization
    3. Image Enhancement
    4. Feature Extraction
    5. Iris Matching
    6. Performance Evaluation (CRR and ROC)

    Parameters:
    - dataset_path: path to the root directory of the iris dataset

    Returns:
    - None (results are printed and saved via CRR_Result and ROC_Result)
    """

    #step 0: list all eye IDS folders in dataset for verification purposes
    eyes = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    #just picking one eye folder to check that image exist
    example_eye = os.path.join(dataset_path, eyes[0])
    image_files = load_iris_images(example_eye)

    if not image_files:
        print(f"No .bmp image files found in {example_eye}")
        return None, None

    img_path = os.path.join(example_eye, image_files[0])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    print(f"Processing image: {img_path}")

    #parameters for rottation augmentation
    initial_angles = [-9, -6, -3, 0, 3, 6, 9] #degrees for rotation invariance
    #training and testing arrays
    X_train, y_train = [], []
    X_test_groups, y_test = [], []

    #step 1: loop through each person's folder to build datsets
    for eye_id in eyes:
      label = eye_id #this is the label for this person

      #get training data (will be in the first folder / session 1)
      train_dir = os.path.join(dataset_path, eye_id, "1")
      train_imgs = load_iris_images(train_dir)
      for img_path in train_imgs:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        #Run step 1: Iris Localization
        pupil_circle, iris_circle, output_vis = IrisLocalization(img)
        #forced it to skip examples where iris localization failed
        if pupil_circle is None or iris_circle is None:
          print(f" Localization failed for image: {img_path}")
          continue

        #Run step 2: Iris Normalization
        normalized = iris_norm(img, pupil_circle, iris_circle)

        #Run step 3: Image Enhancement
        enhanced = image_enhancement(normalized)

        #Run step 4: Feature Extraction
        features = feature_extraction(enhanced)

        X_train.append(features) #added feature vector to training set
        y_train.append(label) #added corresponding label

      #get test data (second folder / session 2)
      test_dir = os.path.join(dataset_path, eye_id, "2")
      test_imgs = load_iris_images(test_dir)
      for i, img_path in enumerate(test_imgs, start=1):
        print(f"Processing image {i}/{len(test_imgs)}: {os.path.basename(img_path)}")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        #Run step 1: Iris Localization
        pupil_circle, iris_circle, output_vis = IrisLocalization(img)

        if pupil_circle is None or iris_circle is None:
          print(f" Localization failed for image: {img_path}")
          continue

        #Run step 2: Iris Normalization
        normalized = iris_norm(img, pupil_circle, iris_circle)
        # plt.imshow(output_vis)
        # plt.show()

        #Rotate the images and run the pipeline on all 7 rotations
        rotated_images = complete_rotation(normalized, initial_angles)
        rotated_features = []
        for rotation in rotated_images:

          #Run step 3: Image Enhancement
          enhanced_img = image_enhancement(rotation)
          print(f"enhanced")
        #   plt.imshow(output_vis)
        #   plt.show()

          #Run step 4: Feature Extraction
          feat = feature_extraction(enhanced_img)
          rotated_features.append(feat)

        #append all rotated features as a test group
        X_test_groups.append(rotated_features)
        y_test.append(label)

    X_train = np.vstack(X_train)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    #Run step 5: Iris Matching
    #returns predictions, LDA pipeline, class centers, labels, and distances
    y_pred, pipe, centers, class_labels, all_distances = iris_matching(X_train, y_train, X_test_groups, y_test)

    #Run step 6: performance evaluation to show CRR and ROC curve
    CRR_Result(y_test, y_pred, X_train, y_train, X_test_groups)
    ROC = ROC_Result(all_distances, y_test, class_labels, metrics=("l1","l2","cosine"))

if __name__ == "__main__":
    dataset_path = "datasets/CASIA_Iris/CASIA Iris Image Database (version 1.0)" 
    IrisRecognition(dataset_path)