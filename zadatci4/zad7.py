import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def show_clusters(points, cluster_labels):
    first_cluster_points = []
    second_cluster_points = []
    third_cluster_points = []
    fourth_cluster_points = []
    fifth_cluster_points = []
    
    for i in range(len(cluster_labels)):
        cluster_index = cluster_labels[i]
        if cluster_index == 0:
            first_cluster_points.append(points[i])
        elif cluster_index == 1:
            second_cluster_points.append(points[i])
        elif cluster_index == 2:
            third_cluster_points.append(points[i])
        elif cluster_index == 3:
            fourth_cluster_points.append(points[i])
        elif cluster_index == 4:
            fifth_cluster_points.append(points[i])
        
    first_cluster_points_x = [point[0] for point in first_cluster_points]
    first_cluster_points_y = [point[1] for point in first_cluster_points]

    second_cluster_points_x = [point[0] for point in second_cluster_points]
    second_cluster_points_y = [point[1] for point in second_cluster_points]

    third_cluster_points_x = [point[0] for point in third_cluster_points]
    third_cluster_points_y = [point[1] for point in third_cluster_points]

    fourth_cluster_points_x = [point[0] for point in fourth_cluster_points]
    fourth_cluster_points_y = [point[1] for point in fourth_cluster_points]

    fifth_cluster_points_x = [point[0] for point in fifth_cluster_points]
    fifth_cluster_points_y = [point[1] for point in fifth_cluster_points]

    plt.scatter(first_cluster_points_x, first_cluster_points_y, c='red')
    plt.scatter(second_cluster_points_x, second_cluster_points_y, c='blue')
    plt.scatter(third_cluster_points_x, third_cluster_points_y, c='green')
    plt.scatter(fourth_cluster_points_x, fourth_cluster_points_y, c='orange')
    plt.scatter(fifth_cluster_points_x, fifth_cluster_points_y, c='cyan')
    plt.show()


data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:, -2:]
X = X.to_numpy()
print(X.shape)

# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

kmeans = KMeans(n_clusters=5).fit(X)

show_clusters(X, kmeans.labels_)
