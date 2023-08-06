#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer

# Read the legitimate domains and DGA domain names
with open('dga-cryptolocke-50.txt') as f:
    dga_cryptolocke = f.read().splitlines()

with open('dga-post-tovar-goz-50.txt') as f:
    dga_post_tovar_goz = f.read().splitlines()

with open('Top-100.csv') as f:
    legit_domains = f.read().splitlines()

# Combine the legitimate domains and DGA domain names into a single list
all_domains = dga_cryptolocke + dga_post_tovar_goz + legit_domains

# Vectorize the domain names using CountVectorizer with 2-gram
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
X = vectorizer.fit_transform(all_domains)

# Apply kMeans clustering with n_clusters=2 and random_state=170
kmeans = KMeans(n_clusters=2, random_state=170)
y_pred = kmeans.fit_predict(X)

# Reduce dimensionality to 2 using t-SNE
tsne = TSNE(learning_rate=100, random_state=170)
X_tsne = tsne.fit_transform(X)

# Print the data shape before and after t-SNE
print(f"Data shape before t-SNE: {X.shape}")
print(f"Data shape after t-SNE: {X_tsne.shape}")

# Calculate and print clustering accuracy
y = np.concatenate((np.ones(len(dga_cryptolocke) + len(dga_post_tovar_goz)),
                    np.zeros(len(legit_domains))))
accuracy = np.mean(y_pred == y) * 100
print(f"Clustering accuracy: {accuracy:.2f}%")


# In[ ]:




