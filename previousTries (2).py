# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 23:15:21 2019

@author: erikh
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:15:45 2019

@author: erikh
"""

# Data Preprocessing Template
"""

"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.utils import shuffle


def get_accuracy(pred, real):
    return len(pred[pred == real]) / len(pred)

# Importing the dataset
X = pd.read_csv('train_samples.csv', header=None)#, nrows = 5000)
y = pd.read_csv('train_labels.csv', header=None)#, nrows = 5000)
"""
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values
"""
print('Dataset loaded')

for i in range(len(y)):
    if y.iloc[i][0] == 5 or y.iloc[i][0] == 7:
        y = y.append(y.iloc[i])
        y = y.append(y.iloc[i])
        
        X = X.append(X.iloc[i])
        X = X.append(X.iloc[i])
    elif y.iloc[i][0] == 6:
        y = y.append(y.iloc[i])
        
        X = X.append(X.iloc[i])
        
        

print ('Classes 5 and 7 doubled')

X = X.append(X)
y = y.append(y)

X, y = shuffle(X, y)
print('Dataset doubled and shuffled')

mu, sigma = 0, 0.07
# creating a noise with the same dimension as the dataset (2,2) 
noise = np.random.normal(mu, sigma, [X.shape[0],X.shape[1]]) 
X = X + noise

print('Noise generated')



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#X_train, X_test2, y_train, y_test2 = train_test_split(X_train2, y_train2, test_size = 0.2)

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Fitting Random forrest to the Training set
# Create classifier
"""
# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 99, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
"""

# Classic SVM
"""
from sklearn.svm import SVC
classifier = SVC(C = 0.9, kernel = 'linear')
#classifier.fit(X_train, y_train)
"""

"""
from sklearn.model_selection import GridSearchCV
parameters =[ {'C': [0.01, 0.1, 1, 10], #so called `eta` value
              'kernel': ['linear'],
              'gamma': [0.001, 0.01, 0.1, 1],
              'random_state': [0]
              },
              {'C': [0.01, 0.1, 1, 10], #so called `eta` value
              'kernel': ['sigmoid'],
              'coef0': [0.0, 0.1, 0.3, 0.4],
              'gamma': [0.001, 0.01, 0.1, 1],
              'random_state': [0]
              }
    ]
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
"""
# Fine Tuned XGBoost
"""
from xgboost import XGBClassifier

classifier = XGBClassifier(n_estimators = 100, learning_rate = 0.05, max_depth = 2, min_child_weight = 2, gamma = 0.05, subsample = 0.7, colsample_bytree = 0.9, n_jobs = -1)
"""

# Perceptron neural network

from sklearn.neural_network import MLPClassifier # importul clasei
from sklearn.linear_model import Perceptron

classifier = MLPClassifier(hidden_layer_sizes=((200, 150)),
                                    activation='relu', solver='adam', batch_size='auto',
                                    learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5,
                                    max_iter=100, shuffle=True, random_state=None, tol=0.0001,
                                    early_stopping=True, validation_fraction=0.2, verbose = True)

#perceptron_model.fit(X, y)
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)
x_pred = classifier.predict(X_train)
"""
y_pred2 = classifier.predict(X_test2)
x_pred2 = classifier.predict(X_train2)
"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm2 = confusion_matrix(y_train, x_pred)

y_pred.shape = (len(y_pred), y_test.shape[1])
x_pred.shape = (len(x_pred), y_train.shape[1])
print('Accuracy: ', get_accuracy(y_pred, y_test))
print('Accuracy: ', get_accuracy(x_pred, y_train))
# PERCEPTRON GRID SEARCH
"""
from sklearn.model_selection import GridSearchCV
parameters = {'hidden_layer_sizes': [(200, 200), (150, 150)], #so called `eta` value
              'learning_rate': ['adaptive'],
              'max_iter': [100],
              'early_stopping': [True]
              }
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
"""

# Applying 10-Fold Cross Validation
"""
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, n_jobs = -1, cv = 5)
avg_accuracy = accuracies.mean()
accuracies.std()
"""

# Applying Grid Search to find the best model and the best parameters
"""
from sklearn.model_selection import GridSearchCV
parameters = {'learning_rate': [0.05], #so called `eta` value
              'max_depth': [2],
              'min_child_weight': [2],
              'gamma': [0.05],
              'subsample': [0.7],
              'colsample_bytree': [0.9],
              'n_estimators': [100]
              }
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 3,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

"""
 

print('----- CREATING KAGGLE SUBMISSION FORMAT ----')
to_predict = pd.read_csv('test_samples.csv', header=None)
results = pd.DataFrame(columns = ['Id', 'Prediction'])
sample_predictions = classifier.predict(to_predict)
for i in range(len(to_predict)):
    results = results.append({'Id': i+1, 'Prediction':sample_predictions[i]}, ignore_index=True)

results.to_csv('perceptron-15k-noise-07-2-layers.csv', encoding='utf-8', index=False)
