import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans 
import time

# -----------------------------
# Exercise 1
# -----------------------------

def kmeans(X, k):
    """
    Perform K-Means clustering on a dataset.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features), input data to cluster.
    k : Number of clusters.

    Returns
    -------
    centroids : ndarray of shape (k, n_features), Coordinates of cluster centers.
    labels : ndarray of shape (n_samples,), luster assignment for each data point.
    """
    model = KMeans(n_clusters=k)
    model.fit(X)
    return model.cluster_centers_, model.labels_

# -----------------------------
# Exercise 2
# -----------------------------

#Load diamonds dataset 
diamonds_num = sns.load_dataset('diamonds').select_dtypes(include='number')

def kmeans_diamonds(n, k):
    """
    Apply K-Means clustering to the first n rows of the diamonds dataset.

    Parameters
    ----------
    n : int - Number of rows to use from the dataset.
    k : int - Number of clusters.

    Returns
    -------
    centroids : ndarray of shape (k, n_features), Cluster centers for the dataset subset.
    labels : ndarray of shape (n,), Cluster assignment for each observation.
    """
    X = diamonds_num.head(n)
    centroids, labels = kmeans(X, k=k)
    return centroids, labels



# -----------------------------
# Exercise 3 - Measures how long KMeans algorithm takes to run
# -----------------------------

def kmeans_timer(n, k, n_iter=5):
    """
    Measure the average runtime of K-Means clustering on the diamonds dataset.

    The function runs kmeans_diamonds(n, k) multiple times and computes
    the average execution time.

    Parameters
    ----------
    n : int - Number of rows from the dataset to use.
    k : int - Number of clusters.
    n_iter : int, default=5 - Number of times to repeat the experiment.

    Returns
    -------
    float
        Average runtime in seconds across all iterations.
    """
    times = []
    for _ in range(n_iter):
        start = time.time()
        kmeans_diamonds(n, k)
        end = time.time()
        times.append(end - start)
    return sum(times) / len(times)