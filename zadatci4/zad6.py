import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('Mall_Customers.csv')
print(data.head(5))

X = data.iloc[:, -2:]
print(X.shape)

inertia = list()
K = range(1, 10)
for k in K:
    kmeans = KMeans(k).fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10,6))
plt.plot(K, inertia)
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow method')
plt.show()
