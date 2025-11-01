import cv2
import numpy as np

def IrisLocalization(img):
    """
    Iris and pupil localization using projection-based initialization
    and circular gradient accumulation.
    Ideas and references:
    - https://www.mdpi.com/1424-8220/23/4/2238
    - Ma et al., "Iris Texture Analysis for Recognition", PAMI 2003
    - https://www.researchgate.net/publication/235222151_A_fast_and_robust_iris_localization_method

    This implements the 'double-circle' approach:
    1) First circle: approximate and refine pupil center and radius
    2) Second circle: detect iris boundary using radial intensity gradient analysis
    """

    #first checking image exists and preprocessing to gray color
    if img is None:
        raise ValueError("Input image is None.")
    
    #convert to gray scale if not already
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    h, w = gray.shape

    #step 1: Approximate pupil center using projections (first circle)
    #sum pixel intensities along columns and rows
    proj_x = np.sum(gray, axis=0)
    proj_y = np.sum(gray, axis=1)
    #lowest sum corresponds roughly to dark pupil area
    est_x = np.argmin(proj_x)
    est_y = np.argmin(proj_y)

    #step 2: Local refinement around dark region
    region = 60 #this is the poxels around estimated center to look for actual pupil
    xL, xR = max(est_x - region, 0), min(est_x + region, w)
    yT, yB = max(est_y - region, 0), min(est_y + region, h)
    patch = gray[yT:yB, xL:xR]

    #smooth patch and threshold to find pupil - helped to add gaussian blur
    patch = cv2.GaussianBlur(patch, (5, 5), 0)
    _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #compute centroid of binary blob for refined pupil center
    M = cv2.moments(binary)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else patch.shape[1] // 2
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else patch.shape[0] // 2

    pupil_x, pupil_y = xL + cx, yT + cy #this is the absolute coordinates in original image
    #need to estimate pupil radius from number of pixels inn binary blob
    pupil_pixels = np.sum(binary > 0)
    pupil_r = int(np.sqrt(pupil_pixels / np.pi)) if pupil_pixels > 0 else 30

    #step 3: Radial intensity analysis for iris boundary
    #sample 24 directions around pupil center, skip eyelid
    search_angles = np.linspace(0, 360, 24, endpoint=False)
    search_angles = [a for a in search_angles if 60 < a < 120 or 240 < a < 300]

    iris_candidates = [] #this will store radii form different directions
    grad_thresh = 6 #setting threshold for gradient magnitude to detect iris edges

    for a in search_angles:
        #here we scan radially from the pupil center along the angle a
        #we are trying to detect the iris boundary in this direction using gradient changes
        #only horizontal bands are considered to avoid eyelid interference
        theta = np.deg2rad(a)
        best_r, accum_grad = 0, 0.0
        prev_val = int(gray[pupil_y, pupil_x])

        #scan radially from just outside the pupil to middle image
        for r in range(pupil_r + 5, min(h, w) // 2 - 5, 3):
            x = int(pupil_x + r * np.cos(theta))
            y = int(pupil_y + r * np.sin(theta))
            if x < 0 or y < 0 or x >= w or y >= h:
                break
            val = int(gray[y, x])
            grad = val - prev_val
            #smoothing gradient to reduce noise
            accum_grad = 0.6 * accum_grad + 0.4 * grad 

            #if the accumulated gradient exceeds threshold, we consider this radius as the iris boundary
            if accum_grad > grad_thresh:
                best_r = r
                break
            prev_val = val

        #store candidate iris radius for this angle
        if best_r > 0:
            iris_candidates.append(best_r)

    if iris_candidates:
        #using 70th percentile of candidate radii - trial and error
        iris_r = int(np.percentile(iris_candidates, 70)) 
        iris_r = max(iris_r, int(pupil_r * 1.7)) #this to ensure radius is larger than pupil
    else:
        iris_r = int(pupil_r * 2.5)
    iris_r = min(iris_r, min(h, w) // 2 - 5) #this to pevent radius from exceeding image bounds

    #step 4: Construct iris mask and localized image
    mask = np.zeros_like(gray, np.uint8)
    cv2.circle(mask, (pupil_x, pupil_y), iris_r, 255, -1) #full iris
    cv2.circle(mask, (pupil_x, pupil_y), pupil_r, 0, -1) #remove pupil
    output_viz = cv2.bitwise_and(gray, gray, mask=mask) #final masked iris image

    #outputting tuples with circle parameters (x,y, radius)
    pupil_circle = (pupil_x, pupil_y, pupil_r)
    iris_circle = (pupil_x, pupil_y, iris_r)

    return pupil_circle, iris_circle, output_viz
