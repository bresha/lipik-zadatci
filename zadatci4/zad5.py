import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('heart.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = KNeighborsClassifier(11)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cf = confusion_matrix(y_test, y_pred)
print(cf)

report = classification_report(y_test, y_pred)
print(report)
