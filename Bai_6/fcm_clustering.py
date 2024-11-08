import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess_image, flatten_image

def fcm_cluster(image, n_clusters=2):
    # Flatten the image for clustering
    flat_image = flatten_image(image)
    # Transpose to match the input requirements of skfuzzy.cmeans
    flat_image = flat_image.T
    # Apply Fuzzy C-means clustering
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        flat_image, n_clusters, 2, error=0.005, maxiter=1000, init=None
    )
    # Extract the cluster with the highest membership for each pixel
    cluster_membership = np.argmax(u, axis=0)
    # Reshape to original image dimensions
    clustered_image = cluster_membership.reshape(image.shape[:2])
    return clustered_image

def display_fcm_image(clustered_image, title="FCM Clustered Image"):
    # Display the FCM clustered image using matplotlib
    plt.imshow(clustered_image, cmap='plasma')
    plt.title(title)
    plt.axis('off')  # Hide axes
    plt.show()
