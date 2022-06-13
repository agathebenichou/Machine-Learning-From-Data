import numpy as np

def get_random_centroids(X, k):

    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    number_of_pixels = X.shape[0]
    chosen_idx = np.random.choice(number_of_pixels, k, replace=False)
    centroids = X[chosen_idx, :]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float) 



def lp_distance(X, centroids, p=2):

    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    deltas = np.abs(X - centroids[:, np.newaxis])
    distances = ((((deltas) ** p).sum(axis=2))**(1/p))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    cents_new = np.zeros(centroids.shape)
    for i in range(max_iter):
        dist = lp_distance(X, centroids, p) 
        classes = np.argmin(dist, axis=0)  # get classes matrix for each image's pixel
        cents_new = np.array([X[classes == j].mean(axis=0) for j in range(k)])
        if np.array_equal(centroids, cents_new):
            return centroids, classes
        centroids = cents_new.copy()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes

def distance(p1, p2):
    return np.sum((p1 - p2)**2)

# initialization algorithm
def initialize(data, k):
    '''
    initialized the centroids for K-means++
    inputs:
        data - numpy array of data points having shape (200, 2)
        k - number of clusters
    '''
    ## initialize the centroids list and add
    ## a randomly selected data point to the list
    centroids = []
    centroids.append(data[np.random.randint(
            data.shape[0]), :])
  
    ## compute remaining k - 1 centroids
    for c_id in range(k - 1):
         
        ## initialize a list to store distances of data
        ## points from nearest centroid
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = np.inf
             
            ## compute distance of 'point' from each of the previously
            ## selected centroid and store the minimum distance
            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)
             
        ## select data point with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
    return np.array(centroids)

def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))



import random 
def initialize2(X_train,k):
  #from towards data science
  centroids = [random.choice(X_train)]
  for _ in range(k-1):
      # Calculate distances from points to the centroids
      dists = np.sum([euclidean(centroid, X_train) for centroid in centroids], axis=0)
      # Normalize the distances
      dists /= np.sum(dists)
      # Choose remaining points based on their distances
      new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
      centroids += [X_train[new_centroid_idx]]
  return np.array(centroids)




def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
  
    # call the initialize function to get the centroids
    centroids = initialize2(X, k)

    #kmeans    
    cents_new = np.zeros(centroids.shape)
    for i in range(max_iter):
        dist = lp_distance(X, centroids, p) 
        classes = np.argmin(dist, axis=0)  # get classes matrix for each image's pixel
        cents_new = np.array([X[classes == j].mean(axis=0) for j in range(k)])
        if np.array_equal(centroids, cents_new):
            return centroids, classes
        centroids = cents_new.copy()



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes
