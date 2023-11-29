"""
K-medoids is a clustering algorithm similar to K-means but with a key distinction
in the way cluster centroids are defined. In K-medoids, each cluster is represented by the
actual data point (medoid) rather than the mean, making it more robust to outliers.
The algorithm aims to minimize the sum of dissimilarities between data points and
their respective cluster medoids. Unlike K-means, K-medoids does not require the calculation
of mean values, making it less sensitive to skewed or noisy data.
However, K-medoids can be computationally more expensive than K-means due to
the need to calculate dissimilarity metrics for all pairs of data points.
The choice between K-means and K-medoids depends on the dataset characteristics,
with K-medoids being favored in scenarios where robustness to outliers is crucial.
"""
import numpy as np
def kmedoids(X, k, max_iters=100):
    n, m = X.shape

    # Initialize medoids randomly
    medoids_idx = np.random.choice(n, k, replace=False)
    medoids = X[medoids_idx]

    labels = np.zeros(n, dtype=int)  # Initialize labels outside the loop

    for _ in range(max_iters):
        # Assign each data point to the nearest medoid
        distances = np.linalg.norm(X[:, np.newaxis] - medoids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update medoids
        new_medoids = []
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                medoid_index = np.argmin(np.sum(distances[labels == j], axis=0))
                new_medoids.append(cluster_points[medoid_index])

        # Check for convergence
        new_medoids = np.array(new_medoids)
        if np.array_equal(medoids, new_medoids):
            break

        medoids = new_medoids

    return labels, medoids
