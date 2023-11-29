"""
K-means is a popular clustering algorithm designed to group data points into K clusters based on similarity.
The process involves iteratively assigning data points to the nearest centroid and updating the centroids
to minimize the sum of squared distances.
This unsupervised learning algorithm is widely used for various applications,
such as customer segmentation and image analysis, relying on the concept of grouping similar items
into distinct clusters. The choice of K, the number of clusters, is a critical aspect,
often determined through techniques like the elbow method or domain expertise.
While simple and efficient, K-means is sensitive to initializations and may benefit from multiple
runs with different starting points to enhance robustness.
"""
import numpy as np


def kmeans(X, k, max_iters=100, tol=1e-4):
    # Step 1: Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    # Initialize labels
    labels = np.zeros(X.shape[0])

    for _ in range(max_iters):
        # Step 2: Assign each data point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        new_labels = np.argmin(distances, axis=1)

        # Step 3: Update centroids
        new_centroids = np.array([X[new_labels == j].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids
        labels = new_labels

    return labels, centroids
