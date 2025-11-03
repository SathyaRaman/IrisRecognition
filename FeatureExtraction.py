import numpy as np
import cv2

def feature_extraction(enhanced_iris):
  '''
    Implements Ma et al. (2003) based feature extraction on a normalized & enhanced iris image.
    Steps:
      1. Create spatial filters (M1) with different δx values.
      2. Apply filters to the upper portion of the iris (ROI).
      3. Divide ROI into small blocks and extract statistical features (mean & average deviation).
      4. Concatenate all features into a 1D normalized feature vector.

    Parameters:
    - enhanced_iris: 2D numpy array (normalized and enhanced iris image).

    Returns:
    - v: 1D numpy array (final feature vector for matching).
    '''
  #describes the channel number, as defined in the paper
  #they run two different kinds of spatial filters (1 and 2)
  i = [1, 2]

  #std deviation of gaussian along x axis, one per channel
  #small std deviation = higher spread of frequency
  delta_xs = [3, 4.5]

  #gaussian along y axis, same value for both channels
  delta_y = 1.5

  #need to select this parameter value
  #defines the spatial frequency (cycles/pixel)
  f = 1 / delta_y

  #the width/height of the filter kernel in pixels
  #used this number as the size of the spatial kernel pixel (how many pixels width/height will the kernel be)
  #r is the is the radius (half-size) of the filter kernel.
  ksize = 35
  r = int(np.floor(ksize/2))

  #generate grid for kernel coordinates
  y, x = np.mgrid[-r:r+1, -r:r+1].astype(np.float32)
  #store filter kernels for each channel
  kernels = []

  #iterate through every channel to calculate the kernel for channel 1 and 2
  #the channel defines the delta values that will be used to calculate the modified Gabor filter
  for channel in range(len(i)):
    delta_x = delta_xs[channel]

    #modulating function of the defined filter
    M1 = np.cos(2 * np.pi * f * np.sqrt(x**2 + y**2))

    #filter as defined by Ma, modified Gabor filter with modulating sinusoidal functions
    G = (1 / (2 * np.pi * delta_x * delta_y)) * np.exp(-0.5 * ((x/delta_x)**2 + (y/delta_y)**2)) * M1
    kernels.append(G)

  #find region of interest (ROI) as defined by the paper
  #upper 48×512 region of the normalized iris image
  '''
  Paper defines ROI as: "the upper portion of a normalized iris image (corresponding to regions closer to
  the pupil) provides the most useful texture information for recognition"
  '''
  roi = enhanced_iris[:48, :].astype(np.float32)
  #normalize the image
  roi = (roi - roi.min())/(roi.max() - roi.min())

  #store all extracted features
  features = []

  #loop through each filter kernel
  for G in kernels:
      #filter the ROI and take magnitude
      response = cv2.filter2D(roi, cv2.CV_32F, G)
      magnitude = np.abs(response)

      #divide ROI into 8×8 blocks
      block_h, block_w = 8, 8
      H, W = magnitude.shape
      #iterate through every small block
      #capture each of the feature values: mean m and the average absolute
      #deviation of the magnitude of each filtered block (formulas from paper)
      for h in range(0, H, block_h):
          for w in range(0, W, block_w):
              Fxy = magnitude[h:h+block_h, w:w+block_w]
              m = np.mean(Fxy)
              sigma = np.mean(np.abs(Fxy - m))
              features.extend([m, sigma])

  #generated final 1D feature vector using feature values
  v = np.array(features, dtype=np.float32)

  #normalize vector to unit form for similarity comparison
  v = v / np.linalg.norm(v)
  return v
