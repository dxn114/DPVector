import numpy as np

def distance_l2(x, y):
    return np.linalg.norm(x - y, axis=1)

def distance1_l2(x, y):
    return np.linalg.norm(x - y)

def center_mean(data):
    return np.mean(data, axis=0)

def sample_uniform_ball(n, d, bound):
    u = np.random.normal(scale=1, size=(n, d + 2))
    u /= np.linalg.norm(u, axis=1)[:, None]
    return u[:, :d] * bound

def clip_norm(x, bound, order):
    return x / np.maximum(1, np.linalg.norm(x, ord=order, axis=1) / bound)[:, None]

def kmeans_lloyd_step(x, centroids, distance=distance_l2, center=center_mean):
    dist_to_cluster = np.stack([distance(x, c) for c in centroids], axis=1)
    centroid_idx = np.argmin(dist_to_cluster, axis=1)
    return np.stack([center(x[centroid_idx == i]) for i in range(len(centroids))])

def debias_reciprocal(shift, scale, dist=np.random.normal(size=10_000)):
    X = shift + scale * dist
    mb = shift * (1 / X).mean()
    return shift * np.clip(mb, 1, 2)

def release_dp_kmeans_lloyd_step(x, bound, epsilon2, centroids, distance=distance_l2):
    from opendp.meas import make_base_geometric, make_base_gaussian
    numer_scale = 2 / epsilon2 * bound
    denom_scale = 2 / epsilon2
    denom_base = make_base_geometric(scale=numer_scale)
    numer_base = make_base_gaussian(denom_scale, D="VectorDomain<AllDomain<f64>>")
    def center_dp_mean(x_cluster):
        dp_numer = np.array(numer_base(x_cluster.sum(axis=0)))
        dp_denom = np.array(denom_base(len(x_cluster)))
        center = dp_numer / np.maximum(debias_reciprocal(dp_denom, denom_scale), 1)
        center /= np.maximum(1, np.linalg.norm(center) / bound)
        return center
    return kmeans_lloyd_step(x, centroids, distance, center_dp_mean)

def release_dp_kmeans_lloyd(x, bound, epsilon2, centroids, steps, distance=distance_l2):
    step_epsilon = epsilon2 / steps
    for i in range(steps):
        centroids = release_dp_kmeans_lloyd_step(
            x, bound, step_epsilon, centroids, distance
        )
        min_val = np.min(centroids)
        max_val = np.max(centroids)
        centroids = (centroids - min_val) / (max_val - min_val)
    return centroids

def kmeans_with_dp(x, n_centroids, d, norm_bound, epsilon2, steps):
    cliped_vector_data = clip_norm(x, norm_bound,2)
    centroids = sample_uniform_ball(n_centroids, d, norm_bound)
    centroids_noisy = release_dp_kmeans_lloyd(cliped_vector_data, norm_bound, epsilon2, centroids, steps)
    return centroids_noisy,cliped_vector_data