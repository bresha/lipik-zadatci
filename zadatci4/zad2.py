from sklearn import datasets
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

housing_X, housing_y = datasets.fetch_california_housing(return_X_y=True)

# nrows, ncols = 2, 4
# fig = plt.figure(figsize=(14, 8))
# for i in range(1, 9):
#     ax = fig.add_subplot(nrows, ncols, i)
#     ax.scatter(housing_X[:, i-1], housing_y)
    
# plt.show()

# housing_X_1 = housing_X[:, 0]
# housing_X_1 = housing_X_1[:, np.newaxis]

# housing_X_4 = housing_X[:, 4]
# housing_X_4 = housing_X_4[:, np.newaxis]

# housing_X = np.stack((housing_X_1, housing_X_4), axis=1)
housing_X = housing_X[:, [0, 4]]

housing_X_train, housing_X_test, housing_y_train, housing_y_test = train_test_split(housing_X, housing_y, test_size=0.3, random_state=0)

linear_model = lm.LinearRegression()
linear_model.fit(housing_X_train, housing_y_train)
print(linear_model.coef_)
print(linear_model.intercept_)

housing_y_pred = linear_model.predict(housing_X_test)

mse = mean_squared_error(housing_y_test, housing_y_pred)
print(mse)

r2 = r2_score(housing_y_test, housing_y_pred)
print(r2)
