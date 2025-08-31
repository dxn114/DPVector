import random
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn import preprocessing
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster import cluster_visualizer
import csv
import faiss
from metrics import compute_jk, compute_ncr
from cluster import kmeans_with_dp,clip_norm,distance1_l2,distance_l2,center_mean
from distance import compute_distances

import argparse
from opendp.mod import enable_features

plt.rcParams['font.sans-serif'] = ['SimHei']

enable_features("contrib", "floating-point")

plt.rcParams['font.sans-serif'] = ['SimHei']

enable_features("contrib", "floating-point")


# 使用 argparse 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="PNCS-DP clustering experiment")
    parser.add_argument('--n', type=int, required=True, help='向量数')
    parser.add_argument('--d', type=int, required=True, help='向量维度')
    parser.add_argument('--k', type=int, required=True, help='投影后维度')
    parser.add_argument('--epsilon0', type=float, required=True, help='隐私参数1')
    parser.add_argument('--delta', type=float, required=True, help='隐私参数2')
    parser.add_argument('--qnum', type=int, required=True, help='查询次数')
    parser.add_argument('--topk', type=int, required=True, help='选择的距离最近的项数')
    parser.add_argument('--k1', type=int, required=True, help='聚类数')
    parser.add_argument('--file_id', type=int, required=True, help='数据集编号')
    return parser.parse_args()

args = parse_args()
n = args.n
d = args.d
k = args.k
epsilon0 = args.epsilon0
delta = args.delta
qnum = args.qnum
topk = args.topk
k1 = args.k1
file_id = args.file_id
    
def debias_reciprocal(shift, scale, dist=np.random.normal(size=10_000)):
    X = shift + scale * dist
    mb = shift * (1 / X).mean()
    return shift * np.clip(mb, 1, 2)

def sample_uniform_ball(n, d, bound):
    u = np.random.normal(scale=1, size=(n, d + 2))
    u /= np.linalg.norm(u, axis=1)[:, None]
    return u[:, :d] * bound

if file_id==1:
    file_name = "data1.csv"
elif file_id==2:
    file_name = "data2.csv"
elif file_id==3:
    file_name = "data3.csv"
elif file_id==4:
    file_name = "data4.csv"


x1 = []

try:
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            line = []
            for cell in row:
                try:
                    value = float(cell)
                except ValueError:
                    value = 0
                line.append(value)
            x1.append(line)
except FileNotFoundError:
    print(f"Error: File {file_name} not found.")
except Exception as e:
    print(f"Error: {e}")

def sample_vectors(x1, n, d):
    indices = random.choices(range(len(x1)), k=n)
    x0 = [x1[idx][:d] for idx in indices]
    return x0

vector_data=sample_vectors(x1, n, d)
vector_data = np.array(vector_data)

vector_data=clip_norm(vector_data, d,1)

def laplace_function1(beta):
    x = np.random.random()
    return (1 / (2 * beta)) * np.exp(-np.abs(x) / beta)

def center(x, y):
    return [(xi + yi) / 2 for xi, yi in zip(x, y)]

def diam(x, y):
    return sum(abs(yi - xi) for xi, yi in zip(x, y))

class QuadTreeNode:
    def __init__(self, x, y, dim, dep, points,node_id=None):
        self.x = x[:]  
        self.y = y[:]  
        self.dim = dim
        self.dep = dep
        self.points = points[:] 
        self.w = 0  
        self.v = []  
        self.s = {}  
        self.id = node_id
        self.left = None
        self.right = None

def build_tree(x, y, dim, dep, points, maxdep,node_id=1):
    if dep == maxdep or len(points) == 0:
        print("this id is none:",node_id)
        return None

    node = QuadTreeNode(x, y, dim, dep, points,node_id)
    split_dim = dep % dim
    t1, t2 = x[split_dim], y[split_dim]
    t3 = random.uniform(t1 + (t2 - t1) / 3, t1 + 2 * (t2 - t1) / 3)

    points_left = [p for p in points if p[split_dim] < t3]
    points_right = [p for p in points if p[split_dim] >= t3]

    y_left = y[:]
    y_left[split_dim] = t3
    node.left = build_tree(x, y_left, dim, dep + 1, points_left, maxdep,2*node_id)

    x_right = x[:]
    x_right[split_dim] = t3
    node.right = build_tree(x_right, y, dim, dep + 1, points_right, maxdep,2*node_id+1)
    return node

def makeprivate(root, mindiam, para1):
    queue = [root]
    while queue:
        c = queue.pop(0)
        if diam(c.x, c.y) > mindiam:
            c.w = len(c.points) + laplace_function1(para1)
            if c.w > 2 * para1:
                if c.left:
                    queue.append(c.left)
                if c.right:
                    queue.append(c.right)

def DynamicProgramkMedian(k, c, para1):
    if c == None:
        return []
    if k == 0:
        c.v.append(c.w * diam(c.x, c.y))
        c.s[k] = []
        DynamicProgramkMedian(k, c.left, para1)
        DynamicProgramkMedian(k, c.right, para1)
        return []
    if c.w < 2 * para1:
        c.v.extend([0] * (k + 1 - len(c.v)))
        c.s[k] = [center(c.x, c.y)] * k
        return [center(c.x, c.y)] * k
    else:
        DynamicProgramkMedian(k, c.left, para1)
        DynamicProgramkMedian(k, c.right, para1)
        if c.left == None and c.right == None:
            c.v.extend([0] * (k + 1 - len(c.v)))
            c.s[k] = [center(c.x, c.y)] * k
            return c.s[k]
        if c.left == None or c.right == None:
            if c.left == None:
                c.v.extend([c.right.v[k]] * (k + 1 - len(c.v)))
                c.s[k] = c.right.s[k]
                return c.s[k]
            else:
                c.v.extend([c.left.v[k]] * (k + 1 - len(c.v)))
                c.s[k] = c.left.s[k]
                return c.s[k]
        minv = float("inf")
        best_i = 0
        for i in range(k + 1):
            if c.left.v[i] + c.right.v[k - i] < minv:
                minv = c.left.v[i] + c.right.v[k - i]
                best_i = i
        c.v.extend([minv] * (k + 1 - len(c.v)))
        c.s[k] = c.left.s[best_i]+c.right.s[k - best_i]
        return c.s[k]

def dpkmedian(x, n, d, k, para1):
    maxA = max(max(abs(val) for val in point) for point in x)
    maxdep = min(60, math.ceil(d * math.log(n, 2)))
    root = build_tree([-maxA] * d, [maxA] * d, d, 0, x, maxdep)
    mindiam = maxA / n
    makeprivate(root, mindiam, para1)
    for i in range(k):
        DynamicProgramkMedian(i, root, para1)
    dpkmedian_center= DynamicProgramkMedian(k, root, para1)
    return dpkmedian_center

epsilon1 = epsilon0*0.5
para1 = d * math.log(n / epsilon1, 2)
dpkmedian_center=dpkmedian(vector_data, n, d, k1, para1)


def auto_select_k(data, k1, k_range=2):
    best_k = k1
    best_score = -1
    best_labels = None
    best_centroids = None
    for k_try in range(max(2, k1 - k_range), k1 + k_range + 1):
        try:
            kmeans = KMeans(n_clusters=k_try, n_init=10, random_state=0).fit(data)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            score = calculate_silhouette_score(data, labels, centroids, distance1_l2, k_try)
        except Exception:
            score = -1
        if score > best_score:
            best_score = score
            best_k = k_try
            best_labels = labels
            best_centroids = centroids
    return best_k, best_labels, best_centroids, best_score

auto_k, auto_labels, auto_centroids, auto_score = auto_select_k(vector_data, k1, k_range=2)
print(f"自动选择的最佳聚类数: {auto_k}, silhouette_score: {auto_score}")
n_centroids = auto_k


# ~~~~ K-Means params ~~~~
norm_bound = 10
steps = 5
epsilon2 = 0.2*epsilon1

centroids_noisy,cliped_vector_data = kmeans_with_dp(vector_data, n_centroids, d, norm_bound, epsilon2, steps)

def assign_clusters_and_count(data_points, noisy_centroids):
    n = len(data_points)  
    num_clusters = len(noisy_centroids)  
    cluster_assignments = []  
    for i in range(n):
        min_distance = float('inf')  
        closest_cluster = 0  
        for j in range(num_clusters):
            distance = distance1_l2(data_points[i], noisy_centroids[j])
            if distance < min_distance:
                min_distance = distance
                closest_cluster = j
        cluster_assignments.append(closest_cluster)

    cluster_sizes = [0] * num_clusters  
    for cluster in cluster_assignments:
        cluster_sizes[cluster] += 1
    return cluster_assignments, cluster_sizes

cluster_assignments, cluster_sizes = assign_clusters_and_count(data_points=cliped_vector_data, noisy_centroids=np.array(centroids_noisy))
dpkmedian_cluster_assignments,dpkmedian_cluster_sizes = assign_clusters_and_count(np.array(vector_data), np.array(dpkmedian_center))

def calculate_silhouette_score(data, cluster_labels, noisy_centroids, distance_func, num_clusters):
    n = len(data) 
    silhouette_score = 0
    for i in range(n):
        intra_cluster_distance = 0
        intra_cluster_count = 0
        for j in range(n):
            if cluster_labels[i] == cluster_labels[j]:  # 同簇
                intra_cluster_distance += distance_func(data[i], data[j])
                intra_cluster_count += 1
        a_i = intra_cluster_distance / intra_cluster_count
        nearest_cluster_distance = float('inf')
        nearest_cluster_index = -1
        for cluster_idx in range(num_clusters):
            if cluster_idx == cluster_labels[i]:  # 跳过当前簇
                continue
            distance_to_centroid = distance_func(data[i], noisy_centroids[cluster_idx])
            if distance_to_centroid < nearest_cluster_distance:
                nearest_cluster_distance = distance_to_centroid
                nearest_cluster_index = cluster_idx
        inter_cluster_distance = 0
        inter_cluster_count = 0
        for j in range(n):
            if cluster_labels[j] == nearest_cluster_index:  # 属于最近簇
                inter_cluster_distance += distance_func(data[i], data[j])
                inter_cluster_count += 1
        if inter_cluster_count == 0:
            inter_cluster_distance += 1
            inter_cluster_count += 0.01
        b_i = inter_cluster_distance / inter_cluster_count
        s_i = (b_i - a_i) / max(b_i, a_i)
        silhouette_score += s_i
    silhouette_score /= n
    return silhouette_score
calculate_silhouette_score(cliped_vector_data, cluster_assignments, centroids_noisy, distance1_l2, k1)


def run_kmeans(data, num_clusters):
    kmeans_instance = KMeans(n_clusters=num_clusters,n_init=10).fit(data)
    labels = kmeans_instance.labels_
    centroids = kmeans_instance.cluster_centers_
    counts = [sum(labels == i) for i in range(num_clusters)]
    return labels, counts, centroids

def initialize_kmedians(data_dim, num_clusters):
    return [[random.random() for _ in range(data_dim)] for _ in range(num_clusters)]

def run_kmedians(data, initial_medians):
    kmedians_instance = kmedians(data, initial_medians)
    kmedians_instance.process()
    clusters = kmedians_instance.get_clusters()
    medians = kmedians_instance.get_medians()
    labels = kmedians_instance.predict(data)
    counts = [sum(labels == i) for i in range(len(initial_medians))]
    return clusters, medians, labels, counts

def calculate_sse(data, labels, centers, distance_func):
    return sum(distance_func(data[i], centers[labels[i]]) for i in range(len(data)))

def print_results(kmeans_counts, kmeans_centroids, kmedians_counts, kmedians_medians, sse_kmeans_noisy, sse_kmeans, sse_kmedians):
    """
    打印聚类结果和 SSE。

    参数:
        各类统计信息，具体见函数参数。
    """
    print("KMeans Counts:", kmeans_counts)
    kmeans_centroids_formatted = [[f"{coord:.4f}" for coord in centroid] for centroid in kmeans_centroids]
    print("KMeans Centroids:", kmeans_centroids_formatted)
    print("K-Medians Counts:", kmedians_counts)
    print("K-Medians Medians:", kmedians_medians)
    print("SSE (Noisy Centroids):", sse_kmeans_noisy)
    print("SSE (KMeans):", sse_kmeans)
    print("SSE (K-Medians):", sse_kmedians)

kmeans_labels,kmeans_counts,kmeans_centroids=run_kmeans(cliped_vector_data, k1)

data_samples = cliped_vector_data.tolist() 
initial_medians = initialize_kmedians(d, k1)
kmedians_clusters, kmedians_medians, kmedians_labels, kmedians_counts = run_kmedians(data_samples, initial_medians)

sse_kmeans_noisy=calculate_sse(cliped_vector_data, cluster_assignments, centroids_noisy, distance1_l2)
sse_kmeans=calculate_sse(cliped_vector_data, kmeans_labels, kmeans_centroids, distance1_l2)
sse_kmedians=calculate_sse(cliped_vector_data, kmedians_labels, kmedians_medians, distance1_l2)
print_results(kmeans_counts, kmeans_centroids, kmedians_counts, kmedians_medians, sse_kmeans_noisy, sse_kmeans, sse_kmedians)

def calculate_projecction(d, k, delta, epsilon1,matrixtype='normal'):
    if matrixtype=='normal':
        projection_matrix = np.random.normal(0, np.sqrt(1/k), (d, k))
    if matrixtype=='equal':
        projection_matrix = np.random.choice([-math.sqrt(1/k), math.sqrt(1/k)], size=(d, k), p=[1/2,1/2])
    if matrixtype=='chaos':
        projection_matrix = np.random.choice([-math.sqrt(3/k), 0, math.sqrt(3/k)], size=(d, k), p=[1/6, 2/3, 1/6])

    max_row_norm = math.sqrt(max(sum(value**2 for value in row) for row in projection_matrix))
    sigma = (max_row_norm) * math.sqrt(2 * (math.log(1 / (2 * delta)) + epsilon1)) / epsilon1
    return projection_matrix, max_row_norm, sigma

queries = sample_vectors(x1, qnum, d)

def compute_noisy_projections(x, p, query, sig, n, k, d, qnum):
    y = [[sum(x[i][l] * p[l][j] for l in range(d)) for j in range(k)] for i in range(n)]
    delt = [[np.random.normal(0, sig) for _ in range(k)] for _ in range(n)]
    z = [[y[i][j] + delt[i][j] for j in range(k)] for i in range(n)]
    qy = [[sum(query[i][l] * p[l][j] for l in range(d)) for j in range(k)] for i in range(qnum)]
    qz = [[qy[i][j] + np.random.normal(0, sig) for j in range(k)] for i in range(qnum)]
    return y, z, qy, qz

projection_matrix, max_row_norm, sigma = calculate_projecction(d, k, delta, epsilon1, matrixtype='equal')

projected_data,noisy_projected_data,query_projection,noisy_query_projection = compute_noisy_projections(vector_data, projection_matrix, queries, sigma, n, k, d, qnum)


def compute_distances_and_metrics(query, data, projected_query, noisy_projected_query,noisy_data, topk):
    dist_original = compute_distances(query,data)  
    dist_projected = compute_distances(projected_query, noisy_data)  
    dist_noisy_projected = compute_distances(noisy_projected_query, noisy_data)  
    ranked_original = np.argsort(dist_original, axis=1)[:, :topk]   
    ranked_projected = np.argsort(dist_projected, axis=1)[:, :topk] 
    ranked_noisy_projected = np.argsort(dist_noisy_projected, axis=1)[:, :topk] 
    return dist_original,ranked_original,ranked_projected,ranked_noisy_projected

dist_original,ranked_original,ranked_projected,ranked_noisy_projected = compute_distances_and_metrics(queries, vector_data, query_projection, noisy_query_projection, noisy_projected_data, topk)

jk_withoutdp=compute_jk(ranked_original,ranked_projected,qnum,topk)
jk_withdp=compute_jk(ranked_original,ranked_noisy_projected,qnum,topk)

ncr_withoutdp=compute_ncr(ranked_original,ranked_projected,qnum,topk)
ncr_withdp=compute_ncr(ranked_original,ranked_noisy_projected,qnum,topk)


def compute_distances_and_metrics_for_center(query, centroids_noisy,dpkmedian_center,kmeans_centroids, kmedians_medians):
    dist_center_dpkmeans = compute_distances(query, centroids_noisy)  
    dist_center_dpkmedian = compute_distances(query, dpkmedian_center)  
    dist_center_kmeans = compute_distances(query, kmeans_centroids)  
    dist_center_kmedian = compute_distances(query, kmedians_medians) 
    
    ranked_center_dpkmeans = np.argsort(dist_center_dpkmeans, axis=1) 
    ranked_center_dpkmedian = np.argsort(dist_center_dpkmedian, axis=1)
    ranked_center_kmeans = np.argsort(dist_center_kmeans, axis=1) 
    ranked_center_kmedian = np.argsort(dist_center_kmedian, axis=1)
    return ranked_center_dpkmeans,ranked_center_dpkmedian,ranked_center_kmeans,ranked_center_kmedian

ranked_center_dpkmeans,ranked_center_dpkmedian,ranked_center_kmeans,ranked_center_kmedian = compute_distances_and_metrics_for_center(queries, centroids_noisy,dpkmedian_center,kmeans_centroids, kmedians_medians)


def compute_scores(vector_data, centroids_noisy, n, k1,alpha):
    scores = np.zeros((n, k1))  
    for i in range(n):
        distances = []
        maxdist= float('-inf')
        mindist= float('inf')     
        for j in range(k1):
            dist = distance1_l2(vector_data[i], centroids_noisy[j])
            if(dist>maxdist):
                maxdist=dist
            if(dist<mindist):
                mindist=dist
            distances.append((dist, j)) 
        distances.sort(key=lambda x: x[0])
        for rank, (dist, j) in enumerate(distances): 
            distance_score = (dist-maxdist)/(mindist-maxdist) 
            rank_score = (k1-rank)/k1 
            scores[i, j] = alpha* distance_score + (1-alpha)*rank_score

    return scores

scores = compute_scores(vector_data, centroids_noisy, n, k1,alpha=0.3)
print("scores:",scores)
epsilon3 = 0.3*epsilon1

def assign_clusters(scores,epsilon3):
    exp_scores = np.exp(epsilon3*scores)
    sum_exp_scores = np.sum(exp_scores, axis=1)
    probabilities = exp_scores / sum_exp_scores[:, None]
    cluster_assignments_after_select = np.array([np.random.choice(probabilities.shape[1], p=prob) for prob in probabilities])
    cluster_sizes_after_select = np.bincount(cluster_assignments_after_select, minlength=probabilities.shape[1])
    return cluster_assignments_after_select, cluster_sizes_after_select

cluster_assignments_after_select, cluster_sizes_after_select = assign_clusters(scores,epsilon3)

def compute_candidate(noisy_projected_data, query_projection, ranked_center_dpkmeans, cluster_assignments, cluster_sizes, beta, qnum, topk):
    import faiss
    noisy_projected_data = np.array(noisy_projected_data, dtype=np.float32)
    query_projection = np.array(query_projection, dtype=np.float32)
    cluster_assignments_np = np.array(cluster_assignments)
    result = np.full((qnum, topk), -1)
    res = faiss.StandardGpuResources()

    for i in range(qnum):
        query = query_projection[i].reshape(1, -1)
        all_distances = []
        for j in range(cluster_sizes.shape[0]):
            cluster_id = ranked_center_dpkmeans[i][j]
            cluster_members = np.where(cluster_assignments_np == cluster_id)[0]
            if len(cluster_members) == 0:
                continue
            cluster_data = noisy_projected_data[cluster_members].astype(np.float32)
            d2 = faiss.pairwise_distance_gpu(res, np.ascontiguousarray(query), np.ascontiguousarray(cluster_data))
            dists = np.sqrt(d2[0])  # (n,)
            all_distances.extend(zip(cluster_members, dists))
            if len(cluster_members) >= beta * topk:
                break
        all_distances.sort(key=lambda x: x[1])
        result[i] = [idx for idx, _ in all_distances[:topk]]
    return result

ranked_projected_after_select = compute_candidate(noisy_projected_data, query_projection, ranked_center_dpkmeans, cluster_assignments_after_select, cluster_sizes_after_select, beta=3, qnum=qnum,topk=topk)

jk_PNCS_kmeans=compute_jk(ranked_original,ranked_projected_after_select,qnum,topk)
ncr_PNCS_kmeans=compute_ncr(ranked_original,ranked_projected_after_select,qnum,topk)

def compute_nafv(query,vector_data,n,d,epsilon0,delta):
    sig = math.sqrt(2 * (math.log(1 / (2 * delta)) + (epsilon0/d))) / (epsilon0/d)*math.sqrt(d)
    delt = np.random.normal(0, sig, (n, d))  
    vector_data_nafv = np.array(vector_data) + delt
    dist_nafv = compute_distances(query,vector_data_nafv)
    ranked_nafv = np.argsort(dist_nafv, axis=1)[:, :topk]
    return dist_nafv,ranked_nafv

dist_nafv,ranked_nafv = compute_nafv(queries,vector_data,n,d,epsilon0,delta)

jk_nafv=compute_jk(ranked_original,ranked_nafv,qnum,topk)
ncr_nafv=compute_ncr(ranked_original,ranked_nafv,qnum,topk)

def compute_nafd(query,vector_data,n,epsilon0,delta,qnum):
    sig = math.sqrt(qnum*2 * (math.log(1 / (2 * delta)) + epsilon0)) / epsilon0
    delt = np.random.normal(0, sig, (qnum, n)) 
    dist_nafd_pre = compute_distances(query,vector_data)
    dist_nafd = dist_nafd_pre + delt
    ranked_nafd = np.argsort(dist_nafd, axis=1)[:, :topk]
    return dist_nafd,ranked_nafd

dist_nafd,ranked_nafd= compute_nafd(queries, vector_data, n, epsilon0, delta, qnum)

jk_nafd=compute_jk(ranked_original,ranked_nafd,qnum,topk)
ncr_nafd=compute_ncr(ranked_original,ranked_nafd,qnum,topk)


def ldp_rr_two_endpoints(vector_data_array, epsilon):
    data_min = np.min(vector_data_array)
    data_max = np.max(vector_data_array)
    c = (data_min + data_max) / 2.0
    r = (data_max - data_min) / 2.0
    if r == 0.0:
        return vector_data_array.copy()
    alpha = (np.exp(epsilon) + 1.0) / (np.exp(epsilon) - 1.0)
    w_plus = c + alpha * r
    w_minus = c - alpha * r
    perturbed_array = np.zeros_like(vector_data_array, dtype=float)
    it = np.nditer(vector_data_array, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        w = vector_data_array[idx]
        numerator = (w - c) * (np.exp(epsilon) - 1.0) + r * (np.exp(epsilon) + 1.0)
        denominator = 2.0 * r * (np.exp(epsilon) + 1.0)
        p_plus = numerator / denominator
        p_plus = max(0.0, min(1.0, p_plus))
        if np.random.rand() < p_plus:
            perturbed_array[idx] = w_plus
        else:
            perturbed_array[idx] = w_minus
        
        it.iternext()
    
    return perturbed_array

def compute_rr(query,vector_data,epsilon0):
    flipped_vector_data = ldp_rr_two_endpoints(vector_data, epsilon0)
    dist_rr = compute_distances(query,flipped_vector_data)
    ranked_rr = np.argsort(dist_rr, axis=1)[:, :topk]
    return dist_rr,ranked_rr

dist_rr,ranked_rr=compute_rr(queries, vector_data,  epsilon0)

jk_rr=compute_jk(ranked_original,ranked_rr,qnum,topk)
ncr_rr=compute_ncr(ranked_original,ranked_rr,qnum,topk)

ncr_PNCS_kmeans_mean = np.mean(ncr_PNCS_kmeans)
ncr_rr_mean = np.mean(ncr_rr)
ncr_withoutdp_mean = np.mean(ncr_withoutdp)
ncr_withdp_mean = np.mean(ncr_withdp)
ncr_nafd_mean = np.mean(ncr_nafd)
ncr_nafv_mean = np.mean(ncr_nafv)

jk_PNCS_kmeans_mean = np.mean(jk_PNCS_kmeans)
jk_rr_mean = np.mean(jk_rr)
jk_withoutdp_mean = np.mean(jk_withoutdp)
jk_withdp_mean = np.mean(jk_withdp)
jk_nafd_mean = np.mean(jk_nafd)
jk_nafv_mean = np.mean(jk_nafv)

print(f"ncr_PNCS_kmeans 平均值: {ncr_PNCS_kmeans_mean}")
print(f"ncr_rr 平均值: {ncr_rr_mean}")
print(f"ncr_withoutdp 平均值: {ncr_withoutdp_mean}")
print(f"ncr_withdp 平均值: {ncr_withdp_mean}")
print(f"ncr_nafd 平均值: {ncr_nafd_mean}")
print(f"ncr_nafv 平均值: {ncr_nafv_mean}")  

print(f"jk_PNCS_kmeans 平均值: {jk_PNCS_kmeans_mean}")
print(f"jk_rr 平均值: {jk_rr_mean}")
print(f"jk_withoutdp 平均值: {jk_withoutdp_mean}")
print(f"jk_withdp 平均值: {jk_withdp_mean}")
print(f"jk_nafd 平均值: {jk_nafd_mean}")
print(f"jk_nafv 平均值: {jk_nafv_mean}") 


def add_gumbel_noise(h, epsilon):
    noise = -np.log(-np.log(np.random.uniform(0, 1, size=len(h)))) / epsilon
    h_noisy = h + noise
    return h_noisy

def top_k_gumbel(h, k, epsilon):
    h_noisy = add_gumbel_noise(h, epsilon)
    top_k_indices = np.argsort(h_noisy)[-k:][::-1]
    return top_k_indices,h[top_k_indices]

h = np.arange(n, 0, -1) 
t = np.arange(0, topk, 1)
t_repeated = np.tile(t, (qnum, 1))
top_k_indices_array = np.zeros((qnum, topk))
for i in range(qnum):
    top_k_indices,top_k_elements = top_k_gumbel(h, topk, epsilon0/max(qnum,100000))
    top_k_indices_array[i] = top_k_indices

jk_em=compute_jk(t_repeated,top_k_indices_array,qnum,topk)
ncr_em=compute_ncr(t_repeated,top_k_indices_array,qnum,topk)

jk_em_mean = np.mean(jk_em)
ncr_em_mean = np.mean(ncr_em)

print(f"jk_em 平均值: {jk_em_mean}")
print(f"ncr_em 平均值: {ncr_em_mean}")


