from sklearn import datasets
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data
y = iris.target

# DBSCAN
dbscan = DBSCAN(eps=0.9).fit(X)
pca = PCA(n_components=2).fit(X)
pca_2d=pca.transform(X)
for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
    elif dbscan.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
    elif dbscan.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')
plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Noise'])
plt.title('DBSCAN finds 2 clusters and Noise')
plt.show()




