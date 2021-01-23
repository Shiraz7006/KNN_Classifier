#importing libraries 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#loading dataset
dataset = pd.read_csv('dataset.csv')

#finding the dataset data
print(len(dataset))
print(dataset.head())

#split dataset
X=dataset.iloc[:,2:32]
Y=dataset.iloc[:,1]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,test_size=0.3)

#faeture scaling
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
len(Y_test)

#define model class KNN
classifier=KNeighborsClassifier(n_neighbors=13,p=2,metric='euclidean')

#fit Model
classifier.fit(X_train,Y_train)

#predict Results
Y_pred=classifier.predict(X_test)
Y_pred

#Evaluate Model
cm=confusion_matrix(Y_test,Y_pred)
accuracy= classifier.score(X_test,Y_test)
print(accuracy)
