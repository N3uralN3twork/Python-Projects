"""Source 1: https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
   Source 2: https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html"""

#Load and prepare libraries:
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths' : 0}

"k-Means:" \
"It's an algorithm that searches for a pre-determined number of clusters" \
"within an unlabeled data set" \
"Center = arithmetic mean of all the points in a cluster"

#Sample dataset:
X, y_true = make_blobs(n_samples = 300, centers = 4,
                       cluster_std = 0.50, random_state = 123)
plt.scatter(x = X[:,0], y = X[:, 1], s = 50)

#It's pretty simple to pick out the four clusters in our dataset rn.
#1. Initialize the algorithm and number of clusters you think
kmeans = KMeans(n_clusters = 4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
centers = kmeans.cluster_centers_

"Visualizing results:"
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(x = centers[:, 0], y = centers[:, 1],
            c = 'black', s = 200, alpha = 0.5)

"About k-Means:"
#A. k-Means uses an iterative approach called expectation-maximization
    #1. Guess some cluster centers
    #2. Repeat until converged
    #3. We maximize a fitness function (the mean) for each cluster
#B. It should be noted that the global optimal result may not be achieved
#C. The number of clusters must be selected beforehand
      #Look into the silhouette analysis for more information
#D. k-Means is limited to linear boundaries for clusters:
         #Not the best if the real clusters have complicated geometric boundaries
         #I.E. weird shapes
    X2, y2 = make_moons(200, noise = 0.05, random_state = 123)
    labels = KMeans(n_clusters = 2, random_state = 123).fit_predict(X2)
    plt.scatter(X2[:, 0], X2[:, 1], c=labels,
                s=50, cmap='viridis')

"For nonlinear clusters:"
Spectrum = SpectralClustering(n_clusters = 2,
                              affinity = 'nearest_neighbors',
                              assign_labels = 'kmeans')
labels = Spectrum.fit_predict(X2)
plt.scatter(X2[:, 0], y = X2[:, 1],
            c = labels, s = 50, cmap = 'viridis')
#Now you see that Spectral can be used to find nonlinear boundaries


"About Hierarchical Clustering:"
    #1. Treat each data point as one cluster, K
    #2. Form a cluster by joining the 2 closest data points, K-1
    #3. Form more clusters by joining the 2 closest clusters, K-2
    #4. Repeat until 1 large cluster is formed
    #5. Use a dendrogram to break up the 1 large cluster
        #a.

X = np.array([[5,3],
             [10,15],
             [15,12],
             [24,10],
             [30,30],
             [85,70],
             [71,80],
             [60,78],
             [70,55],
             [80,91],])
labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:, 0], y = X[:, 1], label = "True Position")
for label, x, y, in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()
#The data obviously form 2 clusters
#But in real life, data will not always be so neat, and there may be hundreds of clusters

cluster = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean',
                                  linkage = 'ward')
cluster.fit_predict(X)
#We believe the number of clusters is 2
#Affinity used is set to euclidean distance
#Linkage=ward means we minimize the variant between the clusters
#Fit_predict assigns a class to each data point

#Plotting the clusters
plt.scatter(X[:,0],X[:,1], c = cluster.labels_, cmap = 'rainbow')

"Real-world Dataset:"
#Unfortunately, you need to use Jupyter Notebook to use it
df = pd.read_csv("C:/Users/MatthiasQ.MATTQ/Downloads/shopping.csv")
pp.ProfileReport(df)

data = df.iloc[:, 3:5].values
cluster = AgglomerativeClustering(n_clusters = 5, linkage = 'ward')
cluster.fit_predict(data)
plt.scatter(x = data[:, 0], y = data[:, 1],
            c = cluster.labels_, cmap = 'rainbow')


"Gaussian Mixture Models:"
"Attempts to find a mixture of multi-dimensional Gaussian probability distributions that best" \
"model any input dataset."