import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

import geometry

def plot_ellipse(mean_p, cov_mat, color='b', alpha=0.3):
    cov00, cov11 = cov_mat[0, 0], cov_mat[1, 1]
    pearson = np.clip(cov_mat[0, 1] / np.sqrt(cov00 * cov11),
                      -1, 1)

    ell_radius_x, ell_radius_y = np.sqrt(1 + pearson), np.sqrt(1 - pearson)

    ell_path = patches.Ellipse((0, 0), 
                               width=ell_radius_x * 2, 
                               height=ell_radius_y * 2,
                               alpha=alpha, 
                               linewidth=2, 
                               fill=True, 
                               color=color)
    ell_tr = transforms.Affine2D() \
                       .rotate_deg(45) \
                       .scale(np.sqrt(cov00) * 2, np.sqrt(cov11) * 2) \
                       .translate(mean_p[0], mean_p[1])
    return ell_path, ell_tr

def generate_cov(mean, cov, N=1000):
    n = np.random.multivariate_normal([0, 0, 0], cov, N)
    poses = mean * geometry.expSE2(n)
    d = poses.as_numpy() - mean.as_numpy()
    return np.dot(d.T, d) / N

def plot_map(model, plot_text=False, path_to_save=None):
    model.get_cluster_params(drop=True)

    colors = plt.cm.rainbow(np.linspace(0, 1, model.number_of_clusters // 2 + 1))
    colors = np.concatenate([colors, colors, colors])

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.axis('Equal')

    base_positions = model.data[0].t()[:, :2]
    plt.scatter(base_positions[:, 0], base_positions[:, 1], c='k', s=2, alpha=0.3)
    
    nn = np.copy(colors)[:model.valid_clusters]
    
    for iseg, (cluster_, color) in enumerate(zip(model.clusters, colors)):
        mean = cluster_['mu_pose']
        
        # Re-calculate the matrix for positions
        aux_cov = generate_cov(mean, cluster_['cov_matrix'][:3, :3], N=10000)[:2, :2]
        arrow_size = np.sqrt(aux_cov[0, 0])
        
        ell_path, ell_tr = plot_ellipse(mean.t(), aux_cov, color=color)
                 
        ell_path.set_transform(ell_tr + ax.transData)
        if plot_text: ax.text(mean.t()[0], mean.t()[1], str(iseg), fontsize=15)
        ax.add_patch(ell_path)
        
        ax.arrow(mean.t()[0], 
                 mean.t()[1], 
                 arrow_size * np.cos(mean.R().as_rotvec().squeeze()), 
                 arrow_size * np.sin(mean.R().as_rotvec().squeeze()), 
                 color=color, width=arrow_size / 20)
               
    if path_to_save is not None:
        plt.savefig(path_to_save)
    else:
        plt.show()
    plt.close('all')