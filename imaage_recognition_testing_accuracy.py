import os
import pandas as pd
os.chdir("C:\\Users\\Sarthak\\Desktop\\project")
train=pd.read_csv("C:\\Users\\Sarthak\\Desktop\\project\\train-m.csv")
y=train.iloc[0:50000,1].values#takeing only 10 values

import numpy as np
train1=np.load("C:\\Users\\Sarthak\\Desktop\\project\\train_images.npy")
train_x = np.reshape(train1,(50000, 3072))

# Splitting the dataset into the Training set and Test set

df = pd.DataFrame(train_x)#converting to pandas from numpy
x=df.iloc[0:50000,:]#taking only 10 values
x.to_csv()#train data

# Splitting the dataset into the Training set and Test set

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


import time
start = time. time()

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

## Fitting Decision Tree Classification to the Training set
#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#classifier.fit(X_train, y_train)

## Fitting Naive Bayes to the Training set
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(X_train, y_train)

## Fitting Logistic Regression to the Training set
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X_train, y_train)
#

## Fitting SVM to the Training set
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state = 0)
#classifier.fit(X_train, y_train)


# Fitting K-NN to the Training set
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#classifier.fit(X_train, y_train)


## Fitting Kernel SVM to the Training set
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier.fit(X_train, y_train)

end = time. time()
print(end - start)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
