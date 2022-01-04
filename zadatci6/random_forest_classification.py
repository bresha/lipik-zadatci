import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = load_wine()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=25, random_state=0)

classifier = RandomForestClassifier(n_estimators=30, random_state=0)

classifier.fit(X_train, y_train)


print('Preciznost na traning podatcima', accuracy_score(y_train, classifier.predict(X_train)))

y_pred = classifier.predict(X_test)

score = accuracy_score(y_test, y_pred)

print('Preciznost na testnim podatcima',score)

cm = confusion_matrix(y_test, y_pred)
print(cm)