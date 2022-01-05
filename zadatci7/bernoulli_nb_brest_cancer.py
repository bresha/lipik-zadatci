from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix


X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = BernoulliNB()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


print('Traning accuracy', accuracy_score(y_train, clf.predict(X_train)))

score = accuracy_score(y_test, y_pred)

print('Testing accuracy', score)

train_cm = confusion_matrix(y_train, clf.predict(X_train))

print('Traning confusion matrix')
print(train_cm)

test_cm = confusion_matrix(y_test, y_pred)

print('Testing confusion matrix')
print(test_cm)