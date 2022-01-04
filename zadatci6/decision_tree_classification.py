import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)

classifier.fit(X_train, y_train)

fig = plt.figure(figsize=(10, 5))
_ = tree.plot_tree(classifier, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()

print('Preciznost na traning podatcima', accuracy_score(y_train, classifier.predict(X_train)))

y_pred = classifier.predict(X_test)

score = accuracy_score(y_test, y_pred)

print('Preciznost na testnim podatcima',score)

cm = confusion_matrix(y_test, y_pred)
print(cm)