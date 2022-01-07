from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X, y = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=0)


clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10000, learning_rate_init=0.001, alpha=0.01, random_state=0)

clf.fit(X_train, y_train)

print('Train accuracy', accuracy_score(y_train, clf.predict(X_train)))

print('Train confusion matrix')
print(confusion_matrix(y_train, clf.predict(X_train)))

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Test accuracy', acc)

cm = confusion_matrix(y_test, y_pred)

print('Test confusion matrix')
print(cm)


print('Validation accuracy', accuracy_score(y_val, clf.predict(X_val)))

print('Validation confusion matrix')
print(confusion_matrix(y_val, clf.predict(X_val)))