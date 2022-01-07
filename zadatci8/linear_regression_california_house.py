import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.min(), X_train.max())

scaler = MinMaxScaler(copy=False)

scaler.fit_transform(X_train)

print(X_train.min(), X_train.max())

scaler.transform(X_test)

linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)

y_pred = linear_regression.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print('Linear regression MSE', mse)

r2 = r2_score(y_test, y_pred)

print('Linear regression R2', r2)


ridge = Ridge(alpha=0.01)

ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)

mse_ridge = mean_squared_error(y_test, y_pred)

print('Ridge MSE', mse_ridge)

r2_ridge = r2_score(y_test, y_pred)

print('Ridge R2', r2_ridge)


lasso = Lasso(alpha=0.01)

lasso.fit(X_train, y_train)

y_pred = lasso.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_pred)

print('Lasso MSE', mse_lasso)

r2_lasso = r2_score(y_test, y_pred)

print('Lasso, R2', r2_lasso)