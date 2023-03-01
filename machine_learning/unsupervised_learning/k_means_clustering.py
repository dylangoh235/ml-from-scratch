import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=2, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def _update_centroids(self, X, closest_centroids):
        centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            centroids[i, :] = np.mean(X[closest_centroids == i], axis=0) 

        return centroids
    
    def _compute_distances(self, X):
        distances = np.zeros((X.shape[0], self.k))
        for i, cluster in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(cluster - X, axis=1)

        return np.argmin(distances, axis=1)
        
    def transform(self, X):
        random_mask = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_mask, :]

        for _ in range(self.max_iter):
            prev_centroids = self.centroids
            closest_centroids = self._compute_distances(X)
            self.centroids = self._update_centroids(X, closest_centroids)
            if np.array_equal(prev_centroids, self.centroids):
                break
        
    def predict(self, X):
        closest_centroids = self._compute_distances(X)
        return closest_centroids