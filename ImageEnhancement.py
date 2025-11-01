# Image enhancement for iris images
# Implements CLAHE-based contrast enhancement, median filtering, and robust normalization
# This version is similar to Ma et al.'s method but uses CLAHE for adaptive histogram equalization

import cv2 as cv
import numpy as np

def image_enhancement(img):
    """
    Enhance an iris image to improve feature extraction.
    
    Steps:
      1. Convert image to uint8 if not already.
      2. Apply CLAHE (adaptive histogram equalization).
      3. Apply median blur to reduce noise and eyelashes.
      4. Perform robust contrast normalization by clipping extreme pixel values.
    
    Parameters:
    - img: 2D numpy array (grayscale iris image)
    
    Returns:
    - enhanced: 2D numpy array (enhanced grayscale image, uint8)
    """

    #ensure the image is in uint8 format for OpenCV CLAHE
    if img.dtype != np.uint8:
        img = cv.convertScaleAbs(img)  # convert and scale image values

    #CLAHE: Contrast Limited Adaptive Histogram Equalization
    #clipLimit: max contrast enhancement per tile
    #tileGridSize: size of contextual regions (tiles)
    clahe = cv.createCLAHE(clipLimit=3.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)  # apply CLAHE

    #median blur with kernel size 3
    #reduces small noise and suppresses eyelashes
    enhanced = cv.medianBlur(enhanced, 3)

    #robust contrast normalization
    #clip extreme pixel values at 5th and 95th percentiles
    lo, hi = np.percentile(enhanced, (5, 95))
    enhanced = np.clip(enhanced, lo, hi)

    #scale clipped values to full [0, 255] range
    enhanced = cv.normalize(enhanced, None, 0, 255, cv.NORM_MINMAX)

    #return enhanced image as uint8
    return enhanced.astype(np.uint8)
