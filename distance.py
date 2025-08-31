import faiss
import numpy as np
def compute_distances(query, data, batch_size=2048):
    query = np.array(query)
    data = np.array(data)
    res = faiss.StandardGpuResources()  
    q = np.ascontiguousarray(query.astype('float32'))
    xb = np.ascontiguousarray(data.astype('float32'))
    squared_blocks = []
    nd = xb.shape[0]
    for i in range(0, nd, batch_size):
        block = xb[i : i + batch_size]     
        d2_block = faiss.pairwise_distance_gpu(res, q, block)
        squared_blocks.append(d2_block)
    D2 = np.hstack(squared_blocks)           
    return np.sqrt(D2)