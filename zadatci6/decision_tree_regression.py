import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=25, random_state=0)

classifier = DecisionTreeRegressor(random_state=0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('MSE', mse)

r2 = r2_score(y_test, y_pred)
print('R2', r2)
