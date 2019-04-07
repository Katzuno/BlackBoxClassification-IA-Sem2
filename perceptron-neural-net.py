# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 23:15:21 2019

@author: erikh
"""

# Import librariile
import numpy as np
import pandas as pd

from sklearn.utils import shuffle

def get_accuracy(pred, real):
    return len(pred[pred == real]) / len(pred)

# Import dataset-ul
X = pd.read_csv('train_samples.csv', header=None, nrows = 5000)
y = pd.read_csv('train_labels.csv', header=None, nrows = 5000)

print('Dataset loaded')

# Pentru clasele 5 si 7 triplez observatiile, iar pentru clasa 6 dublez
for i in range(len(y)):
    if y.iloc[i][0] == 5 or y.iloc[i][0] == 7:
        y = y.append(y.iloc[i])
        y = y.append(y.iloc[i])
        
        X = X.append(X.iloc[i])
        X = X.append(X.iloc[i])
    elif y.iloc[i][0] == 6:
        y = y.append(y.iloc[i])
        
        X = X.append(X.iloc[i])
        
print ('Classes 5 and 7 tripled, 6 doubled')

X = X.append(X)
y = y.append(y)

X, y = shuffle(X, y)
print('Dataset doubled and shuffled')

mu, sigma = 0, 0.07
# Creez un noise de maxim 0.07, de acelasi shape ca variabila X pe care il adaug pe date
noise = np.random.normal(mu, sigma, [X.shape[0],X.shape[1]]) 
X = X + noise

print('Noise generated')



# Impart dataset-ul in 80 - 20
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Implementez retea de perceptroni

from sklearn.neural_network import MLPClassifier # importul clasei


classifier = MLPClassifier(hidden_layer_sizes=((200, 150)),
                                    activation='relu', solver='adam', batch_size='auto',
                                    learning_rate='adaptive', max_iter=100, shuffle=True, 
                                    random_state=None, tol=0.0001,
                                    early_stopping=True, validation_fraction=0.2, verbose = True)

classifier.fit(X_train, y_train)

# Prezic pe datele de antrenare si testare
y_pred = classifier.predict(X_test)
x_pred = classifier.predict(X_train)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, x_pred)

y_pred.shape = (len(y_pred), y_test.shape[1])
x_pred.shape = (len(x_pred), y_train.shape[1])
print('Accuracy TEST: ', get_accuracy(y_pred, y_test))
print('Accuracy TRAIN: ', get_accuracy(x_pred, y_train))


# Aplic 3-Fold Cross Validation si afisez media de acuratete si deviatia standard
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, n_jobs = -1, cv = 3)
avg_accuracy = accuracies.mean()
std_dev = accuracies.std()
print('Avg accuracy: ', avg_accuracy)
print('Std_dev: ', std_dev)

# Creez fisierul csv de output

print('----- CREATING KAGGLE SUBMISSION FORMAT ----')
to_predict = pd.read_csv('test_samples.csv', header=None)
results = pd.DataFrame(columns = ['Id', 'Prediction'])
sample_predictions = classifier.predict(to_predict)
for i in range(len(to_predict)):
    results = results.append({'Id': i+1, 'Prediction':sample_predictions[i]}, ignore_index=True)
results.to_csv('perceptron-15k-noise-07-2-layers.csv', encoding='utf-8', index=False)
