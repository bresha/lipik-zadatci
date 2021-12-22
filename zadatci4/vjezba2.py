from sklearn import datasets
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# nrows, ncols = 2, 5
# fig = plt.figure(figsize=(14, 8))
# for i in range(1, 11):
#     ax = fig.add_subplot(nrows, ncols, i)
#     ax.scatter(diabetes_X[:, i-1], diabetes_y)
    
# plt.show()

diabetes_X = diabetes_X[:, 2]
diabetes_X = diabetes_X[:, np.newaxis]

diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(diabetes_X, diabetes_y, test_size=0.2, random_state=42)

linear_model = lm.LinearRegression()
linear_model.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = linear_model.predict(diabetes_X_test)

mse = mean_squared_error(diabetes_y_test, diabetes_y_pred)
print(mse)

r2 = r2_score(diabetes_y_test, diabetes_y_pred)
print(r2)

plt.scatter(diabetes_X_test, diabetes_y_test)
plt.plot(diabetes_X_test, diabetes_y_pred, color='red', marker='o')
plt.show()
