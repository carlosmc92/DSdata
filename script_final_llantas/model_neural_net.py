#=============================================================================
#
#       Autor: Carlos Martínez Cuéllar - Alyona Ivanova Araujo
#                
#=============================================================================

import pandas as pd
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV


#=============================================================================
#  exportando datos de train y test

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

#separa datos
X_train = data_train[:,0:57600]
y_train = data_train[:,-1]
X_test = data_test[:,0:57600]
y_test = data_test[:,-1]


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
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [(1,),(2,),(3,)],
            'alpha' : [1e-2, 1e-4, 1e-6, 1e-8],
            'learning_rate_init' : [1e-2, 1e-4, 1e-6, 1e-8]
        }
        ]

prueba_net = GridSearchCV(MLPClassifier(), param_grid, cv = 5, scoring = 'accuracy')

prueba_net.fit(X_train, y_train)
prueba_net.cv_results_
prueba_net.best_params_
prueba_net.best_score_


#=============================================================================
# Red neuroral 

mlr = MLPClassifier(activation = 'relu',
                       solver = 'sgd',
                       alpha = 1e-2,
                       hidden_layer_sizes = (3,),
                       random_state = 1,
                       learning_rate_init = 1e-3, #entre menor sea LR mayor accuracy
                       batch_size = 32,
                       tol = 1e-4,)

mlr.fit(X_train, y_train)

pred_m = mlr.predict(X_test)
conf_m = confusion_matrix(y_test, pred_m)
clasf_m = classification_report(y_test, pred_m)
fscore = f1_score(y_test, pred_m)

s = y_test

count = 0

for i in range(len(pred_m)):
    if pred_m[i] == s[i]:
        count = count + 1

print('======================================================================')
print('\n')
print('Score train :', mlr.score(X_train, y_train))
print('======================================================================')
print('\n')
print('Matriz de confusión :', conf_m)
print('======================================================================')
print('\n')
print('Metricas de la red :', clasf_m)
print('======================================================================')
print('\n')
print('F Score :', fscore)
print('======================================================================')
print('\n')  
print('Accuracy test:', count/float(len(pred_m)))




    

