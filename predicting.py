# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('set.csv')
x = dataset.iloc[:, 4:5].values
y = dataset.iloc[:, 8].values #close


# Splitxing the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(X_train)
x_test = sc_x.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Visualising the train set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_test, y_pred, color = 'blue')
plt.title('open vs close (Training set)')
plt.xlabel('open')
plt.ylabel('close')
plt.show()