import numpy as np
def compute_jk(a,b,qnum,topk):
    metrics=np.zeros(qnum)
    for i in range(qnum):
        common_topk = len(set(a[i]) & set(b[i])) 
        metrics[i]=common_topk/(2*topk-common_topk)
    return metrics

def compute_ncr(a,b,qnum,topk):
    metrics=np.zeros(qnum)
    weights = np.arange(topk, 0, -1)
    for i in range(qnum):
        indices_in_a = np.array([np.where(a[i] == x)[0][0] if x in a[i] else -1 for x in b[i]])
        for idx in indices_in_a:
            if idx != -1:  
                metrics[i] += weights[idx] 
        metrics[i] = float(metrics[i]*2)/float(topk*topk+topk)
    return metrics