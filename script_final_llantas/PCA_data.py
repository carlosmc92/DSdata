#=============================================================================
#
#      Autor: Carlos Martinez Cuellar - Alyona Ivanova Araujo
#
#=============================================================================
# import random 
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

#=============================================================================
#           importar datos y separar las X y Y

data1 = pd.read_csv('flat.class_train.csv', sep = " ")
data2 = pd.read_csv('full.class_train.csv', sep = " ")
data1 = numpy.array(data1)
data2 = numpy.array(data2)
data = numpy.concatenate((data1, data2))
# data = numpy.random.shuffle(data)
X = data[:,0:57600]
y = data[:,-1]

#=============================================================================
#              PCA

desc_pca = PCA(n_components = 100, whiten = True)
x_desc = desc_pca.fit(X).transform(X)


#=============================================================================
#              Tree Classifier

X_train, X_test, y_train, y_test = train_test_split(X, y)
rf = RandomForestClassifier(n_estimators = 10)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
s = y_test

count = 0

for i in range(len(pred)):
    if pred[i] == s[i]:
        count = count + 1

model = BaggingClassifier(
    base_estimator = RandomForestClassifier(),
    n_estimators = 50,
    max_samples = 0.8,
#    oob_score = True,
    random_state = 1
    )

fit_bag = model.fit(X_train, y_train)
score_bag = model.score(X_test, y_test)

#=============================================================================
#                  PCA Tree Classifier

X_trainPCA, X_testPCA, y_trainPCA, y_testPCA = train_test_split(x_desc, y)
rf2 = RandomForestClassifier(n_estimators = 10)
rf2.fit(X_trainPCA, y_trainPCA)
pred2 = rf2.predict(X_testPCA)
s2 = y_testPCA

count2 = 0

for i in range(len(pred2)):
    if pred2[i] == s2[i]:
        count2 = count2 + 1

model2 = BaggingClassifier(
    base_estimator = RandomForestClassifier(),
    n_estimators = 50,
    max_samples = 0.8,
#    oob_score = True,
    random_state = 1
    )

fit_bag2 = model2.fit(X_trainPCA, y_trainPCA)
score_bag2 = model2.score(X_testPCA, y_testPCA)

#==============================================================================
#   Varianza
df = pd.DataFrame({'var': desc_pca.explained_variance_ratio_})

#=============================================================================
print('======================================================================')
print('\n')  
print('Accuracy all data:', count/float(len(pred)))
print('\n')  
print('Accuracy all data with bagging and PCA :', score_bag)
print('\n')        
print('======================================================================')
print('\n')  
print('Accuracy all data:', count2/float(len(pred2)))
print('\n')  
print('Accuracy all data with bagging and PCA :', score_bag2)
print('\n')
print('======================================================================')
print('\n') 
print('Varianza de cada componente :',df)
print('\n')
print('Los componentes de PCA explican la base en un :', sum(df.values))

    
        
    




