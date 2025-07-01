import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[-5.0, +0.0],
                        [+0.0, +5.0],
                        [+5.0, +0.0],
                        [+0.0, -5.0],
                        [+0.0, +0.0]])
group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]]])

# read data into memory
data_set = np.genfromtxt("hw05_data_set.csv", delimiter = ",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 5

# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    # your implementation starts below
    initial_centroids = np.genfromtxt("hw05_initial_centroids.csv", delimiter=",")
    means = initial_centroids

    distances = np.linalg.norm(X[:, np.newaxis] - means, axis=2)
    assignments = np.argmin(distances, axis=1)

    covariances = []
    priors = []
    for k in range(K):
        cluster_data = X[assignments == k]
        if cluster_data.shape[0] > 1:
            cov = np.cov(cluster_data, rowvar=False)
        else:
            cov = np.eye(X.shape[1]) * 1e-2
        covariances.append(cov)
        priors.append(cluster_data.shape[0] / X.shape[0])

    # your implementation ends above
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below
    
    N, D = X.shape

    for _ in range(100):
        responsibilities = np.zeros((N, K))
        for k in range(K):
            try:
                rv = stats.multivariate_normal(mean=means[k], cov=covariances[k], allow_singular=True)
                responsibilities[:, k] = priors[k] * rv.pdf(X)
            except np.linalg.LinAlgError:
                responsibilities[:, k] = 1e-8

        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities /= responsibilities_sum

        Nk = responsibilities.sum(axis=0)
        for k in range(K):
            rk = responsibilities[:, k].reshape(-1, 1)
            means[k] = (rk * X).sum(axis=0) / Nk[k]
            diff = X - means[k]
            covariances[k] = ((rk * diff).T @ diff) / Nk[k] + 1e-6 * np.eye(D)
            priors[k] = Nk[k] / N

    assignments = np.argmax(responsibilities, axis=1)
    
    
    # your implementation ends above
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below
    def draw_pdf_contour(mean, cov, color, linestyle):
        x_vals = np.linspace(-8, 8, 300)
        y_vals = np.linspace(-8, 8, 300)
        xx, yy = np.meshgrid(x_vals, y_vals)
        grid = np.column_stack((xx.ravel(), yy.ravel()))
        rv = stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True)
        zz = rv.pdf(grid).reshape(xx.shape)
        plt.contour(xx, yy, zz, levels=[0.01], colors=color, linestyles=linestyle)

    colors = ['red', 'green', 'blue', 'purple', 'orange']
    plt.figure(figsize=(8, 8))

    # draw solid contours (EM result)
    for k in range(K):
        cluster_points = X[assignments == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=colors[k], label=f"Cluster {k+1}")
        draw_pdf_contour(means[k], covariances[k], colors[k], linestyle='solid')

    # draw dashed contours (true Gaussians — always in order!)
    for k in range(5):
        draw_pdf_contour(group_means[k], group_covariances[k], colors[k], linestyle='dashed')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("EM Clustering Results")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # your implementation ends above
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)