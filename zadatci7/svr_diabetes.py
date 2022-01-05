from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

reg = svm.SVR(kernel='poly', C=10)

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('Mse', mse)

r2 = r2_score(y_test, y_pred)
print('R2', r2)

# najnizi mse sam dobio ovim parametrima
