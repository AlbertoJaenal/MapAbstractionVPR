import numpy as np
import time

import geometry, utils

_COV_REG = 1e-6

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('\t%s  %.2f s' % (method.__name__.capitalize(), (te - ts)))
        return result
    return timed

class expectationMaximization:

    def __init__(self,
                 poses,
                 features,
                 number_of_clusters):
                 
        # Load data
        if features is not None:
            assert( len(poses) == len(features) )
        self.data = (poses, features)
        self.data_length = len(poses)
        
        # Expectation Maximization parameters 
        self.number_of_clusters = number_of_clusters
        self.valid_clusters = number_of_clusters
        
        # Cluster parameters
        self.cov_dim = 4
        self.mus, self.priors = None, None
        self.covariances = np.zeros((self.number_of_clusters, self.cov_dim, self.cov_dim))
        self.areas, self.pdf_matrix = None, np.zeros([self.data_length, self.number_of_clusters])
        
        # Regularization
        self.reg_cov = np.identity(self.cov_dim) * _COV_REG
        
         # EM output
        self.clusters = []


    # Initialize parameters
    def initialize(self, init_mode='random'):
        assert(init_mode in ['random', 'kmeans'])
        
        ################################
        # Normal random initialization #
        ################################
        # Mean
        indexes, covariances, _ = utils.initialize(init_mode, self.number_of_clusters, self.data[0], self.data[1])
        self.mus =  [(self.data[0][ind_], self.data[1][ind_]) for ind_ in indexes]
        
        # Covariances
        if covariances is None:
            for dim in range(self.number_of_clusters):
                np.fill_diagonal( self.covariances[dim, :, :], np.array(cov_diag))
        else:
            self.covariances[:] = covariances[:]
        
        self.covariances += self.reg_cov
        
        # Priors are set uniformly to 1 / number_of_clusters
        self.priors = np.ones(self.number_of_clusters) / self.number_of_clusters

    # Run the optimization
    def run(self, 
            max_iterations=100, 
            converged_epsilon=10, 
            verbose=True):
        
        log_likelihoods, converged = [], False
        
        ############
        # 1. EM loop 
        if verbose: print('Shapes:', len(self.mus), self.covariances.shape, self.priors.shape)
        
        for i in range(max_iterations):
            if verbose:
                print('Step %d' % i)
            
            # Expectation 
            self.expectation()
            
            # Likelihood calculation
            if i > 0: 
                log_likelihoods.append(np.sum(np.log(np.mean(self.pdf_matrix * self.pdf_matrix_norm[:, None] + np.finfo(self.pdf_matrix.dtype).eps, axis=1))))
                if verbose: print('\tLog-likelihood val: %.4f \t\t Valid clusters %d' % (log_likelihoods[-1], len(self.clusters)))
            
            # The loop has converged if the likelihood has not changed in 3 steps
            converged = (len(log_likelihoods) > 3) and np.all(np.abs(np.diff(log_likelihoods)[-3:]) < converged_epsilon)

            if converged: 
                if verbose: print('\nConverged at step %d' % i)
                break
            
            # Maximization 
            self.maximization()
            self.get_cluster_params(True)
            
        self.get_cluster_params(True)


    # Expectation function
    @timeit
    def expectation(self):
        self.pdf_matrix = np.zeros((self.data_length, self.number_of_clusters))
        
        # Iterate over all clusters
        for cluster_num in range(self.number_of_clusters):
            # Precompute pdf for the n-th cluster
            self.pdf_matrix[:, cluster_num] = self.priors[cluster_num] * \
                                              utils.distribution(mean=self.mus[cluster_num], 
                                                                 cov=self.covariances[cluster_num] + self.reg_cov).pdf(self.data)
        self.pdf_matrix_norm = self.pdf_matrix.sum(1)
        self.pdf_matrix /= self.pdf_matrix_norm[:, None]


    # Maximization function
    @timeit
    def maximization(self):
        # Iterate over all clusters to calculate its params
        for cluster_num in range(self.number_of_clusters):
            norm_probs = self.pdf_matrix[:, cluster_num] / (self.pdf_matrix[:, cluster_num].sum())
            mu, cov = utils.maximization(self.data, norm_probs, self.mus[cluster_num])
            self.mus[cluster_num] = mu
            self.covariances[cluster_num][:] = cov[:] + self.reg_cov
            
        self.priors = np.sum(self.pdf_matrix, axis=0)
        self.priors /= self.priors.sum()
            

    # Extract the params
    def get_cluster_params(self, drop=False):
        self.clusters.clear()
        data_indexes = self.pdf_matrix.argmax(1)
        for cluster_num in range(self.number_of_clusters):
            if drop and np.trace(self.covariances[cluster_num, :3, :3]) < 10*_COV_REG: 
                continue
            
            self.clusters.append({
                                    'mu_pose': self.mus[cluster_num][0],
                                    'mu_feat': self.mus[cluster_num][1],
                                    'cov_matrix': self.covariances[cluster_num],
                                    'likelihoods': self.pdf_matrix[:, cluster_num],
                                    'prior': self.priors[cluster_num]  / self.priors.sum()
                                 })
        self.valid_clusters = len(self.clusters)
