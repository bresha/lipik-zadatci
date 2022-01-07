from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, _ = load_wine(return_X_y=True)

inertia = []
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

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
print(kmeans.cluster_centers_)
