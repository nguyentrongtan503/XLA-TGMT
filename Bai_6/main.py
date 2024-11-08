from preprocess import load_and_preprocess_image
from kmeans_clustering import kmeans_cluster
from fcm_clustering import fcm_cluster
import matplotlib.pyplot as plt

# Load and preprocess images
image1 = load_and_preprocess_image("./data/a1.png")
image2 = load_and_preprocess_image("./data/a2.png")
image3 = load_and_preprocess_image("./data/a3.png")

# Cluster each image using K-means and FCM
kmeans_clustered_images = [
    kmeans_cluster(image1, n_clusters=2),
    kmeans_cluster(image2, n_clusters=2),
    kmeans_cluster(image3, n_clusters=2)
]

fcm_clustered_images = [
    fcm_cluster(image1, n_clusters=2),
    fcm_cluster(image2, n_clusters=2),
    fcm_cluster(image3, n_clusters=2)
]

# Display images in a 3x3 grid for side-by-side comparison
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Titles for each column
columns = ["Image 1", "Image 2", "Image 3"]

# Display original images in the first row
original_images = [image1, image2, image3]
for i, ax in enumerate(axes[0]):
    ax.imshow(original_images[i])
    ax.set_title(f"Original - {columns[i]}")
    ax.axis('off')  # Hide axis

# Display K-means results in the second row
for i, ax in enumerate(axes[1]):
    ax.imshow(kmeans_clustered_images[i], cmap='viridis')
    ax.set_title(f"K-means - {columns[i]}")
    ax.axis('off')  # Hide axis

# Display FCM results in the third row
for i, ax in enumerate(axes[2]):
    ax.imshow(fcm_clustered_images[i], cmap='plasma')
    ax.set_title(f"FCM - {columns[i]}")
    ax.axis('off')  # Hide axis

# Adjust layout for clarity
plt.tight_layout()
plt.show()
