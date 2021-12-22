from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

iris_X, iris_y = datasets.load_iris(return_X_y=True)

iris_X = StandardScaler().fit_transform(iris_X)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.2, random_state=0)

classifier = LogisticRegression()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cf = confusion_matrix(y_test, y_pred)
print(cf)

report = classification_report(y_test, y_pred)
print(report)
