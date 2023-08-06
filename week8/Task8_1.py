#!/usr/bin/env python
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the breast cancer dataset
cancer = load_breast_cancer()

# Scale the cancer data using StandardScaler
scaler = StandardScaler()
scaled_cancer_data = scaler.fit_transform(cancer.data)
print("Scaled cancer data shape:", scaled_cancer_data.shape)

# Apply PCA with n_components=2
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(scaled_cancer_data)
print("PCA transformed cancer data shape:", pca_transformed.shape)

# Plot the first two principal components as a 2D scatter plot
mask = cancer.target == 0
plt.scatter(pca_transformed[mask, 0], pca_transformed[mask, 1], c='blue', marker='o', edgecolors='black')
plt.scatter(pca_transformed[~mask, 0], pca_transformed[~mask, 1], c='orange', marker='^', edgecolors='black')

plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()

# Print the PCA components and their shapes
print("PCA components shape:", pca.components_.shape)
print("PCA components:", pca.components_)

# Plot the first three features as a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
cmap = plt.cm.get_cmap("Spectral")
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=10, azim=10)
ax.scatter(scaled_cancer_data[:, 0], scaled_cancer_data[:, 1], scaled_cancer_data[:, 2], c=cancer.target, cmap=cmap)
ax.set_title("3D Plot of First Three Features")

# Plot the first two principal components as a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
cmap = plt.cm.get_cmap("Spectral")
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=10, azim=10)
ax.scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=cancer.target, cmap=cmap)
ax.set_title("3D Plot of First Two Principal Components")

plt.show()


# In[ ]:




