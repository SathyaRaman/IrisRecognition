import numpy as np
import cv2 as cv

#mapping the iris from Cartesian coordinates to polar coordinates
def iris_norm(img, pupil, iris, radial_res = 64, angular_res = 512):
  """
    Implements Daugman's Rubber Sheet Model:
    - Transforms the circular iris region into a normalized rectangular polar image.
    - img: grayscale iris image
    - pupil: tuple (x, y, r) of pupil circle
    - iris: tuple (x, y, r) of iris circle
    - radial_res: number of radial samples
    - angular_res: number of angular samples
    """
  #convert image to float for precision in remapping
  img = img.astype(np.float32)

  #theta is angular coordinates around circle from 0 to 2pi
  theta = np.linspace(0, 2 * np.pi, angular_res, endpoint=False)
  r = np.linspace(0, 1, radial_res).reshape(-1, 1) #normalized radial coordinates where 0 is at pupil and 1 at iris boundary

  #making angle counterclockwise
  theta = -theta

  #coordinates of pupil boundary along theta
  x_p = pupil[0] + pupil[2] * np.cos(theta)
  y_p = pupil[1] + pupil[2] * np.sin(theta)

  #coordinates of iris boundary along theta
  x_s = iris[0] + iris[2] * np.cos(theta)
  y_s = iris[1] + iris[2] * np.sin(theta)

  #linear interpolation from pupil to iris boundary for each radial step
  X = (1 - r) * x_p + r * x_s #x coordinates for remapping
  Y = (1 - r) * y_p + r * y_s #y coodinates for remapping

  #converrt to float32 for opencv
  X = X.astype(np.float32)
  Y= Y.astype(np.float32)

  #remap image from circular region to rectangular polar coordinates
  polar = cv.remap(img, X, Y, interpolation = cv.INTER_LINEAR, borderMode = cv.BORDER_REFLECT)

  #normalize polar image to ange 0-255
  normalized = cv.normalize(polar, None, 0, 255, cv.NORM_MINMAX)

  return normalized