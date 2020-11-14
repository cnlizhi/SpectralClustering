from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

k = 8

wfile = pd.read_csv("C:\\Users\\70743\\Desktop\\node_distance.txt")
data = np.array(wfile)[:,1:4]
datasize = np.round((1+np.sqrt(1+4*len(data)))/2).astype('int32')
w = np.zeros(datasize*datasize).reshape(datasize,datasize)
for val in range(len(data)):
    w[np.round(data[val,0]).astype('int32')][np.round(data[val,1]).astype('int32')] = data[val,2]
d = np.zeros(datasize*datasize).reshape(datasize,datasize)
for val in range(datasize):
    d[val][val] = sum(w[:,val])
l = d-w
eigenvalue,featurevector = np.linalg.eig(l)
u = np.zeros(datasize*k).reshape(k,datasize)
for val in range(k):
    index = np.argmin(eigenvalue)
    u[val] = featurevector[index]
    eigenvalue = np.delete(eigenvalue,index)
    featurevector = np.delete(featurevector,index,axis = 0)
kmeans = KMeans(n_clusters = k).fit(u.T)
print(kmeans.cluster_centers_)
