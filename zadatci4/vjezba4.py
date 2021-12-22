import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

X_train = np.array([[1], [2], [3], [4], [5], [6]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# plt.scatter(X_train, y_train)
# plt.show()

classifier = LogisticRegression()

classifier.fit(X_train, y_train)

X_test = np.array([[2.5], [3.9]])
y_test = np.array([0, 1])

y_pred = classifier.predict(X_test)

cf = confusion_matrix(y_test, y_pred)

print(cf)

report = classification_report(y_test, y_pred)

print(report)
