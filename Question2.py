import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from Constants import EIGHT_FILE, THREE_FILE
import matplotlib.pyplot as plt
from numpy.linalg import eigh

arr_three = np.loadtxt(THREE_FILE)
arr_eight = np.loadtxt(EIGHT_FILE)
height = width = 16
n_max_digits = 200

def show_gray_scale(img, title, fig_size = None, index=None, use_colorbar=False):
    if fig_size is not None:
        plt.figure(figsize=fig_size)
    if index is not None:
        plt.subplot(1, 2, index)
    img_cb = plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')

    if (use_colorbar):
        plt.colorbar(img_cb, fraction=0.046, pad=0.04)

# Question 2a
img_three = arr_three[0].reshape((height, width), order='F')
img_eight = arr_eight[0].reshape((height, width), order='F')

show_gray_scale(img_three, "Question 2a - three.txt", fig_size=(6,3), index=1)
show_gray_scale(img_eight, "Question 2a - eight.txt", index=2)
plt.show()


# Question 2b
X = np.vstack((arr_three, arr_eight))
mean = np.mean(X, axis=0)
img_mean = mean.reshape((height, width), order='F')

show_gray_scale(img_mean, "Question 2b", fig_size=(3,3))
plt.show()

# Question 2c
X_centered = X - mean
n = X.shape[0]
S = (X_centered.T @ X_centered) / (n - 1) 
S_matrix = S[:5, :5]
print("Question 2c: 5x5 matrix: \n", S_matrix)

# Question 2d
eigvals_all, eigvecs_all = eigh(S)   
eigvals = eigvals_all[::-1]          
eigvecs = eigvecs_all[:, ::-1]       

v1 = eigvecs[:, 0]   
v2 = eigvecs[:, 1]

# Visualize v1 and v2 as 16x16 images
# Used internet search for code.
def vec_to_image(v):
    vmin, vmax = v.min(), v.max()
    if vmax == vmin:
        return np.zeros_like(v).reshape((height, width), order='F')
    scaled = (v - vmin) / (vmax - vmin) * 255.0
    return scaled.reshape((height, width), order='F')

img_vec_1 = vec_to_image(v1)
img_vec_2 = vec_to_image(v2)

show_gray_scale(img_vec_1, "Question 2d - v1", fig_size=(6,3), index=1)
show_gray_scale(img_vec_2, "Question 2d - v2", index=2, use_colorbar=True)
plt.show()

# Question 2e - projections onto first two principal components
V = np.column_stack([v1, v2])  
coords = X_centered @ V                 

v_1 = coords[0]
v_2 = coords[n_max_digits]

print("Question 2e - v1:", v_1)
print("Question 2e - v2:", v_2)

# Question 2f - average reconstruction error
reconstructed = coords @ V.T   
residuals = X_centered - reconstructed
sq_errors = np.sum(residuals**2, axis=1)
avg_rec_error = np.mean(sq_errors)
print(f"2f:  {avg_rec_error:.6f}")

# Question 2g - 2D scatter plot using PCA
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)
kmeans = KMeans(n_clusters=2).fit(X_pca)
labels = kmeans.labels_
_, ax = plt.subplots()
ax.scatter(X_pca[:,0][labels == 0], X_pca[:,1][labels == 0], c='r')
ax.scatter(X_pca[:,0][labels == 1], X_pca[:,1][labels == 1], c='b')
plt.title("Question 2G - PCA Scatter Plot")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()