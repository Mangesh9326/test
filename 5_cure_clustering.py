import pandas as pd
from pyclustering.cluster.cure import cure
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np  
  
url = "data/5_iris.csv"

df = pd.read_csv(url) 
df = df.drop(["variety"], axis=1) 
df = df.drop_duplicates()  
  
print(df.head())  
  
data = df.values.tolist()
cure_instance = cure(data, 3)
cure_instance.process()  
clusters = cure_instance.get_clusters()
cluster_labels = np.full(len(data), fill_value=-1, dtype=int)
for cluster_id, cluster in enumerate(clusters):  
    cluster_labels[cluster] = cluster_id  
    pca = PCA(n_components=2)  
    reduced_data = pca.fit_transform(df)  
  
plt.figure(figsize=(7.5, 5))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels)
plt.xlabel('Petal length, Petal width') 
plt.ylabel('Sepal length, Sepal width')
plt.title('Cure Clustering on Iris Dataset')
plt.show()

plt.figure(figsize=(10, 7))
for cluster_id in np.unique(cluster_labels):  
    plt.scatter(reduced_data[cluster_labels == cluster_id, 0],
                reduced_data[cluster_labels == cluster_id, 1],
                label=f'Cluster {cluster_id}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('CURE Clustering on Iris Dataset')
    plt.legend()
    plt.show()  
