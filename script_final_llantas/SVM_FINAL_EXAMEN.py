# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:09:56 2021

@author: 2099791
"""
#=============================================================================
#
#       Autor: Carlos Martínez Cuéllar - Alyona Ivanova Araujo
#                
#
#=============================================================================
import random
import pandas as pd
import numpy
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

#=============================================================================
#      exportando datos de train y test

data1 = pd.read_csv('flat.class_train.csv', sep = " ")
data2 = pd.read_csv('full.class_train.csv', sep = " ")
data3 = pd.read_csv('flat.class_test.csv', sep = " ")
data4 = pd.read_csv('full.class_test.csv', sep = " ")
data1 = numpy.array(data1)
data2 = numpy.array(data2)
data3 = numpy.array(data3)
data4 = numpy.array(data4)
data_train = numpy.concatenate((data1, data2))
data_test = numpy.concatenate((data3, data4))

#desordenamos datos
numpy.random.shuffle(data_train)
numpy.random.shuffle(data_test)

# separamos
X_train = data_train[:,0:57600]
y_train = data_train[:,-1]
X_test = data_test[:,0:57600]
y_test = data_test[:,-1]
numpy.random.shuffle(y_test)

#=============================================================================
#               PCA

#base train y test
desc_pca = PCA(n_components = 50, whiten = True)
X_train = desc_pca.fit(X_train).transform(X_train)
X_test = desc_pca.fit(X_test).transform(X_test)

#=============================================================================
# Evaluación de la red neural tuneando todos los parámetros
# usa Cross validation

param_grid = [
        {
            'C': [0.1, 1, 10, 100, 1000],
            'activation' : ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            'gamma' : [1e-2, 1e-4, 1e-6, 1e-8]
        }
        ]

prueba_net = GridSearchCV(SVC(), param_grid, cv = 5, scoring = 'accuracy')

prueba_net.fit(X_train, y_train)
prueba_net.cv_results_
prueba_net.best_params_
prueba_net.best_score_

#=============================================================================

model_svm = SVC(kernel = 'rbf',  gamma = 1e-6, C = 1)
model_svm.fit(X_train, y_train)
    
#CLAVE REDUCIRLE EL PARAMETRO GAMA A 1e-5
#preguntar si es necesario cambiar los 0 a-1
    
pred_y = model_svm.predict(X_test)
# print(model_svm.score(X_train, y_train))
    
conf_m = confusion_matrix(y_test, pred_y)
clasf_m = classification_report(y_test, pred_y)
fscore = f1_score(y_test, pred_y)

s = y_test

count = 0

for i in range(len(pred_y)):
    if pred_y[i] == s[i]:
        count = count + 1

#=============================================================================

print('=====================================================================')
print('\n')
print("score del modelo :", model_svm.score(X_test, y_test))
print('=====================================================================')
print('\n')
print('matriz de confusion :', conf_m)
print('=====================================================================')
print('\n')
print('Metricas del modelo :', clasf_m)
print('======================================================================')
print('\n')
print('F Score :', fscore)
print('======================================================================')
print('\n')  
print('Accuracy test :', count/float(len(pred_y)))