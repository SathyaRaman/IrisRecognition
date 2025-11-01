from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_distance(loc1, loc2, metric):
  """
    Compute distance between two sets of feature vectors.
    loc1: array of features (test or projected)
    loc2: array of features (training class centers)
    metric: "l1", "l2", or "cosine"
    """
  if metric == "l1":
    distance = cdist(loc1, loc2, metric="cityblock")
  #paper takes squared euclidian norm in formula
  elif metric == "l2":
    distance = cdist(loc1, loc2, metric="sqeuclidean")
  elif metric == 'cosine':
    distance = 1.0 - cosine_similarity(loc1, loc2)
  return distance


#when iris matching is called we recieve the image as a feature vector
#should be of length 1536
#using Fisher linear discriminant for dimension reduction and nearest center classifier for classification
#saving the lda dim to re-use this for CRR curve to re-run for every dimension
def iris_matching(X_train, y_train, X_test, y_test, lda_dim=None):
  """
    Iris recognition matching pipeline using:
    - LDA for dimensionality reduction
    - Nearest center classifier for matching
    - Multiple distance metrics for evaluation
    """
  #use fisher linear discriminant analysis
  #this reduces the dimensionality of features but also increases class
  #separability by considering both information of all samples and the underlying structure of each class.
  #f is the new feature vector post LDA and = WT * V

  #nearest center classifier for classification in a low-dimensional feature space
  #project requirements ask for this done for 3 different distance measures
  metrics = ["l1", "l2", "cosine"]

  #return predicted_label, distances
  #to offset for rotation invariance
  #minimum of 7 angle scores is taken for final matching distance

  #standardize + LDA (dims = n_classes - 1)
  class_labels = np.unique(y_train)

  #use dimension to factor into LDA computation if it is defined
  max_lda = min(len(class_labels) - 1, X_train.shape[1])
  d = max_lda if lda_dim is None else max(1, min(lda_dim, max_lda))

  #build a pipeline: standardize features, then apply LDA
  pipe = Pipeline([
      ("scaler", StandardScaler()),
      ("lda", LinearDiscriminantAnalysis(
          n_components=d,
          solver="eigen",
          shrinkage="auto"
      ))
  ])

  #fit pipeline on training data
  pipe.fit(X_train, y_train)

  #project to LDA space
  f_train = pipe.transform(X_train)

  #build class centers in LDA space
  centers = np.vstack([f_train[y_train == c].mean(axis=0) for c in class_labels])

  #get the predicted labels for every metric
  y_pred = {m: [] for m in metrics}
  all_distances = {m: [] for m in metrics}

  #go through the rotations of X_test
  for rotated_feats in X_test:
    f_test_rot = pipe.transform(np.vstack(rotated_feats))
    for metric in metrics:

      #find distance from rotation to training sample
      distances = get_distance(f_test_rot, centers, metric)

      #find min over the rotations
      #min_distances = np.min(distances, axis=0)
      min_distances = distances.min(axis=0)

      #saving for the ROC calculation later
      all_distances[metric].append(min_distances)

      #keep the best (smallest) distance per sample/class and rotation
      pred = class_labels[min_distances.argmin()]
      y_pred[metric].append(pred)


  for m in metrics:
          all_distances[m] = np.vstack(all_distances[m])

  #return predictions and other computation steps needed for ROC and CRR later
  return y_pred, pipe, centers, class_labels, all_distances

