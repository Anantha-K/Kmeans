import numpy as np
import matplotlib.pyplot as plt

class KMeans :
    def __init__(self,k=3) :
        self.k=k
        self.X = np.array([
    [1, 2], [1, 4], [1, 0],
    [10, 2], [10, 4], [10, 0],
    [1, 3], [1, 5], [1, -1],
    [10, 3], [10, 5], [10, -1],
    [2, 2], [2, 4], [2, 0],
    [9, 2], [9, 4], [9, 0],
    [2, 3], [2, 5], [2, -1],
    [9, 3], [9, 5], [9, -1],
    [0, 2], [0, 4], [0, 0],
    [11, 2], [11, 4], [11, 0]
])
        
        self.clusters = [[] for _ in range(k)]
        self.centroids = self.initialize_centroids(self.X,self.k)
        

    def initialize_centroids(self,X, k):
        indices = np.random.choice(X.shape[0], k, replace=False)
        centroids = X[indices]
        return centroids     
    def clustering(self):
        i=0
        for i in range(len(self.X)):
            min_distance = float('inf')
            min_index = 0
            for j in range(len(self.centroids)):
                centroid_point=self.centroids[j]
                data_point=self.X[i]
                distance = np.linalg.norm(centroid_point - data_point)
                if(distance<min_distance):
                    min_distance=distance
                    min_index=j
            self.clusters[min_index].append(self.X[i])

    def plot_clusters(self):
        for i, cluster in enumerate(self.clusters):
            cluster = np.array(cluster)
            if cluster.size > 0:  
                plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i+1}')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='black', marker='x', label='Centroids')
        plt.legend()
        plt.show()    
  

    
kmeans = KMeans(k=3)
kmeans.clustering()
kmeans.plot_clusters()
