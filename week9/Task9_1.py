#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_digits

# Load the dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define the parameter grid
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

# Perform grid search with cross-validation
grid_search = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)

# Print the score of the test dataset
print('Test set score:', grid_search.score(X_test, y_test))

# Print the best parameters found by grid search
print('Best parameters:', grid_search.best_params_)

# Print the best score found by grid search
print('Best score:', grid_search.best_score_)

# Print the best estimator found by grid search
print('Best estimator:', grid_search.best_estimator_)


# In[ ]:





# In[ ]:




