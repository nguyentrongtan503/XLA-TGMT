from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess_image, flatten_image

def kmeans_cluster(image, n_clusters=2):
    # Flatten image for clustering
    flat_image = flatten_image(image)
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(flat_image)
    clustered = kmeans.labels_
    # Reshape clustered labels to original image shape
    clustered_image = clustered.reshape(image.shape[:2])
    return clustered_image

def display_clustered_image(clustered_image, title="K-means Clustered Image"):
    # Display the clustered image using matplotlib
    plt.imshow(clustered_image, cmap='viridis')
    plt.title(title)
    plt.axis('off')  # Hide axes
    plt.show()
