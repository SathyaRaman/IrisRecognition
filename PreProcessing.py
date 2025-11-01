import os
import numpy as np

#Code to load and read the images:
def load_iris_images(base_dir):
    """
    Traverses all subfolders of the given base directory and collects paths to BMP images.
    Parameters:
    - base_dir: root folder containing iris image dataset
    Returns:
    - sorted list of full paths to all BMP images
    """
    image_paths = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith('.bmp'):
                image_paths.append(os.path.join(root, f))
    return sorted(image_paths)

#rotate images before feature extraction
#take the minimum/smallest matching dist from each rotation of the 7 angles
def complete_rotation(img, angles):
    """
    Returns a list of rotated images to account for eye rotation.
    Uses circular horizontal shifts for rotation invariance.
    
    Parameters:
    - img: single iris image (2D numpy array)
    - angles: list of angles (degrees) to rotate
    
    Returns:
    - list of rotated images
    """
    rotated = []
    width = img.shape[1] #image width in pixels
    for degree in angles:
        #convert number of degrees to the number of pixels we need to shift
        #in unprocessed image
        shift = int(width * degree / 360)

        #apply circular horizontal shift
        rotated_img = np.roll(img, shift, axis=1)
        rotated.append(rotated_img)
    return rotated