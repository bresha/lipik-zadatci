from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

knn_reg = KNeighborsRegressor(n_neighbors=10, p=1)

knn_reg.fit(X_train, y_train)

y_pred = knn_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print(mse)

r2 = r2_score(y_test, y_pred)

print(r2)
