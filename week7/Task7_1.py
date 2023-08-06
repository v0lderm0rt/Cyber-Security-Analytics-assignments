#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


np.random.seed(0)


# In[3]:


X, y = make_blobs(n_samples=200, centers=[[3,2], [6,4], [10,5]], cluster_std=0.9)


# In[5]:


plt.scatter(X[:, 0], X[:, 1], marker='.')


# In[4]:


k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)


# In[6]:


k_means.fit(X)


# In[7]:


k_means_labels = k_means.labels_
print(k_means_labels)


# In[8]:


k_means_cluster_centers = k_means.cluster_centers_
print(k_means_cluster_centers)


# In[9]:


fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot with a black background (background is black because we can see the points
# connection to the centroid.
ax = fig.add_subplot(1, 1, 1, facecolor = 'black')

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.

#print(np.linspace(0, 1, len(set(k_means_labels))))
#print('colors:',colors)
#print(range(len([[2, 2], [-2, -1], [4, -3], [1, 1]])))


for k, col in zip(range(len([[2, 2], [-2, -1], [4, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col. 'w'- white
    #https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.axes.Axes.plot.html 
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    #'o' circle marker; 'k' - black
    #for detail of format string, https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.axes.Axes.plot.html 
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()

# Display the scatter plot from above for comparison.
plt.scatter(X[:, 0], X[:, 1], marker='.')


# In[10]:


import numpy as np 
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


X2, y2 = make_blobs(n_samples=200, centers=[[3,2], [6,4], [10,5]], cluster_std=0.9)


# In[14]:


agglom = AgglomerativeClustering(n_clusters = 3, linkage = 'average')
agglom.fit(X2,y2)


# In[16]:


plt.figure(figsize=(6,4))
x_min, x_max = np.min(X2, axis=0), np.max(X2, axis=0)
X2 = (X2 - x_min) / (x_max - x_min)
cmap = plt.cm.get_cmap("Spectral")
for i in range(X2.shape[0]):
    plt.text(X2[i, 0], X2[i, 1], str(y2[i]),
             color=cmap(agglom.labels_[i] / 10.), 
             fontdict={'weight': 'bold', 'size': 9})
    
plt.xticks([])
plt.yticks([])
plt.axis('off')

plt.show()

plt.scatter(X2[:, 0], X2[:, 1], marker='.')


# In[17]:


dist_matrix = distance_matrix(X2,X2) 
print(dist_matrix)
#condense the distance matrix using hierarchy.distance.pdisk 
#you should see the condensed distance matrix is a flat array. 
#It is the upper triangular of the distance matrix.
condensed_dist_matrix= hierarchy.distance.pdist(X2,'euclidean')
print()
print(condensed_dist_matrix)

#the following is another way to produce condensed_dist_matrix
#import scipy.spatial.distance as ssd
## convert the redundant n*n square matrix form into a condensed nC2 array     
## distArray[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j
##https://stackoverflow.com/questions/18952587/use-distance-matrix-in-scipy-cluster-hierarchy-linkage
#condensed_dist_matrix = ssd.squareform(dist_matrix)
#print()
#print(condensed_dist_matrix)


# In[18]:


Z = hierarchy.linkage(condensed_dist_matrix, 'complete')
dendro = hierarchy.dendrogram(Z)


# In[ ]:




