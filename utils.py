import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats._multivariate import _PSD,_LOG_2PI
from sklearn.cluster import KMeans

import geometry
from DB_index import compute_db_index


##############################################################################################
#########################              INITIALIZATION                #########################
##############################################################################################

def initialize(init_mode, number_of_clusters, poses, feats, repetitions=3):
    if init_mode == 'random':
        # Get the indexes
        indexes = np.sort(np.random.choice(np.arange(len(poses)),  number_of_clusters, replace=False))
        covariances = np.zeros([len(indexes), 4, 4])
        for cluster_n, ind_ in enumerate(indexes):
            covariances[cluster_n, :] = np.eye(4) * [0.5, 0.5, np.pi/18, 0.5]
        
    elif init_mode == 'kmeans':
        eval_dict = []
        # Using cosine and sine to codify the pose for the initialization
        new_poses = np.hstack([poses.as_numpy()[:, :2], 
                               np.cos(poses.as_numpy()[:, -1].reshape([-1, 1])), 
                               np.sin(poses.as_numpy()[:, -1].reshape([-1, 1]))])
        # Select the intialization with lower DB index
        for try_n in range(repetitions):
            kmeans = KMeans(n_clusters=number_of_clusters, tol=1e-5, max_iter=10000, algorithm='full', n_init=50).fit(new_poses)
            centers = np.hstack([kmeans.cluster_centers_[:, 0].reshape([-1, 1]),
                                 kmeans.cluster_centers_[:, 1].reshape([-1, 1]),
                                 np.arctan2(kmeans.cluster_centers_[:, 2], 
                                            kmeans.cluster_centers_[:, 3]).reshape([-1, 1])])
            
            db_index, indexes = compute_db_index(poses, kmeans.labels_, centers)
            eval_dict.append((db_index, indexes, np.copy(kmeans.labels_)))
            
            
        print('DBs:', [x[0] for x in eval_dict])
        indexes = eval_dict[int(np.argmin([x[0] for x in eval_dict]))][1]
        labels = eval_dict[int(np.argmin([x[0] for x in eval_dict]))][2]
        
        # Pre-compute the covariances
        covariances = np.zeros([len(indexes), 4, 4])
        for cluster_n, ind_ in enumerate(indexes):
            covariances[cluster_n, :3, :3] = np.cov(geometry.logSE2(poses[labels==cluster_n] / poses[ind_]).T)
            covariances[cluster_n,  3,  3] = np.mean(np.linalg.norm(feats[labels==cluster_n] - feats[ind_], axis=-1) ** 2)
        
    else:
        raise ValueError('{} not a known init method'.format(init_mode))
    
    return indexes, covariances, labels
    

####################################################################################################
#########################              2D POSE + APPEARANCE                #########################
####################################################################################################
    
def maximization(data, per_cluster_probs, previous_mean, dev=False):
    pose_data, feat_data = data
    norm_probs = per_cluster_probs.reshape([-1, 1]) / np.sum(per_cluster_probs)
    
    muPos = weighted_SE2_mean(previous_mean[0], pose_data, norm_probs)
    muApp = np.average(feat_data, weights=norm_probs.squeeze(), axis=0)
    
    pose_dev = geometry.logSE2(muPos / pose_data) # From mu to poses
    feat_dev = np.linalg.norm(feat_data - muApp, axis=1, keepdims=True)
    
    cov = np.identity(4)
    cov[:3, :3] = np.dot((norm_probs * pose_dev).T, pose_dev)
    cov[3,   3] = np.dot((norm_probs * feat_dev).T, feat_dev)
    return (muPos, muApp), cov
    
class distribution:
    # Distribution parameters
    def __init__(self, mean, cov):
        self.pose_mean, self.feat_mean = mean
        self.cov = cov

    # PDF evaluation 
    def pdf(self, data):
        pose_data, feat_data = data
        
        # Concatenating
        pose_dev = geometry.logSE2(self.pose_mean / pose_data) # From mu to poses
        total_dev = np.hstack([pose_dev,
                               np.linalg.norm(feat_data - self.feat_mean, axis=1, keepdims=True)])
        
        psd = _PSD(self.cov, allow_singular=False)
        rankp, prec_Up, log_det_covp = psd.rank, psd.U, psd.log_pdet
        total_maha = np.sum(np.square(np.dot(total_dev, prec_Up)), axis=-1)
        
        return np.exp(-0.5 * (rankp * _LOG_2PI + log_det_covp + total_maha))


def weighted_SE2_mean(prev_mean, poses, weights):
    """
    Weighted mean of several poses, given their weights.
    """
    imax = weights.argmax().squeeze()
    increments = geometry.logSE2(prev_mean / poses)
    rotation_mean = geometry.Rotation2.from_euler('z', increments[:, 2:]).mean(weights.squeeze()).as_rotvec()

    return prev_mean * geometry.expSE2(np.concatenate([np.average(increments[:, :2].squeeze(), 
                                                                  weights=weights.squeeze(), 
                                                                  axis=0), 
                                                       rotation_mean[-1].reshape([1])]))