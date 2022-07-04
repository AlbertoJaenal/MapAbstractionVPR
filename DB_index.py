import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import geometry


def compute_db_index(poses, labels, cluster_centers, w=1):
    number_of_clusters = len(cluster_centers)
    
    indexes_, centroids, cluster_radius = [], [], []
    for cluster_n in range(number_of_clusters):
        centroid = geometry.SE2Poses(cluster_centers[cluster_n][:2],
                                     geometry.Rotation2.from_euler('xyz', cluster_centers[cluster_n] * [0, 0, 1]))
        centroids.append(centroid)
        indexes_.append(int(np.argmin([geometry.metric(centroid, pose, w=w) for pose in poses])))
        cluster_radius.append(np.sqrt(np.mean([geometry.metric(centroid, pose, w=w)**2 for pose in poses[cluster_n==labels]])))
     
    db_matrix = np.zeros([number_of_clusters, number_of_clusters])
    for i in range(number_of_clusters):
        for j in range(i+1, number_of_clusters):
            db_matrix[i, j] = (cluster_radius[i] + cluster_radius[j]) / np.array(geometry.metric(centroids[j], centroids[i], w=w))
            db_matrix[j, i] = db_matrix[i, j]
    
    return db_matrix.max(0).mean(), indexes_



def try_clusters(full_poses_, min_c, max_c, step_c, repetitions, wd=1, title='Unknown dataset'):
    ks, infs = [], []
    print('Using SE2 and w=1 for combined distance!')
    
    for k in np.arange(min_c, max_c, step_c):
        
        db_i, ch_i = [], []
        for i in range(repetitions):            
            kmeans = KMeans(n_clusters=k, 
                            tol=1e-5, max_iter=10000, algorithm='full', n_init=50).fit(full_poses_.as_numpy())

            db_ind, _ = compute_db_index(full_poses_, kmeans.labels_, kmeans.cluster_centers_, w=wd)

            # Davies-Bouldin Index
            db_i.append(db_ind)

        print(k, np.mean(db_i))

        ks.append(k)
        infs.append(np.mean(db_i))

    
    fig = plt.figure(figsize=(18, 6.5))
    fig.suptitle(title, fontsize=50)
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.tick_params(axis='x', labelsize=19)
    plt.tick_params(axis='y', labelsize=19)
    plt.xlabel("Number of clusters", fontsize=28)
    plt.ylabel("DB Index", fontsize=28)
    plt.plot(ks, infs)
    plt.show()