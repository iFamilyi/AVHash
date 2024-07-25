import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

def CalcHammingDist_numpy(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - B1 @ B2.t())
    return distH

def chunked_hamming_distance(B1, B2, chunk_size=1000):
    q = B2.shape[1]
    distH = torch.zeros(B1.shape[0], B2.shape[0])

    epoch = B1.shape[0] // chunk_size
    rem = B1.shape[0] % chunk_size

    print(epoch, rem)
    for i in range(epoch):
        chunk_B1 = B1[i*chunk_size:i*chunk_size + chunk_size]
        chunk_distH = 0.5 * (q - chunk_B1 @ B2.t())
        distH[i*chunk_size:i*chunk_size + chunk_size] = chunk_distH

    if rem:
        chunk_B1 = B1[epoch*chunk_size:]
        chunk_distH = 0.5 * (q - chunk_B1 @ B2.t())
        distH[epoch*chunk_size:] = chunk_distH

    return distH


def CalcTopMap(qB, rB, queryL, retrievalL, topk):

    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap += topkmap_
    topkmap = topkmap / num_query
    return topkmap



def CalcMap(qB, rB, queryL, retrievalL):
    num_query = queryL.shape[0]
    map = 0


    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd).astype(int)
        if tsum == 0:
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query

    return map

def CalcTopAcc(qB, rB, queryL, retrievalL, device, topk):

    num_query = queryL.shape[0]
    topkacc = 0
    for iter in range(num_query):
        gnd = (queryL[iter, :] @ retrievalL.t() > 0).float()
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = torch.argsort(hamm)
        gnd = gnd[ind]
        tgnd = gnd[0:topk]
        tsum = torch.sum(tgnd)
        if tsum == 0:
            continue
        topkacc += tsum / topk
    topkacc = topkacc / num_query
    torch.cuda.empty_cache()
    return topkacc

def mean_average_precision(query_code,
                           database_code,
                           query_labels,
                           database_labels,
                           device,
                           topk=None,
                           ):

    if topk == None:
        topk = database_labels.shape[0]

    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1, as_tuple=False).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    torch.cuda.empty_cache()
    return mean_AP

#
def mean_average_precision_2(query_code,
                           database_code,
                           query_labels,
                           database_labels,
                           device,
                           topk=None,
                           ):

    if topk == None:
        topk = database_labels.shape[0]

    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1, as_tuple=False).squeeze() + 1.0).float()

        mean_AP += (score / index).sum() / topk

    mean_AP = mean_AP / num_query
    torch.cuda.empty_cache()
    return mean_AP

def precision_k(query_code, database_code, query_labels, database_labels, topk=None):

    num_query = query_labels.shape[0]

    precision_topk = 0

    for iter in range(num_query):
        gnd = (query_labels[iter, :] @ database_labels.t() > 0).float()
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[iter, :] @ database_code.t())
        ind = torch.argsort(hamming_dist)
        gnd = gnd[ind]
        tgnd = gnd[0:topk]
        tsum = tgnd.sum().int().item()
        if tsum == 0:
            continue
        precision_topk += tsum / topk
    res = precision_topk / num_query
    torch.cuda.empty_cache()
    return res

def precision_k_other(query_code, database_code, query_labels, database_labels, topk=None):

    num_query = query_labels.shape[0]

    precision_topk = 0

    for iter in range(num_query):
        gnd = (query_labels[iter, :] @ database_labels.t() > 0).float()
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[iter, :] @ database_code.t())
        ind = torch.sort(hamming_dist,stable=True)[1]
        gnd = gnd[ind]
        tgnd = gnd[0:topk]
        tsum = tgnd.sum().int().item()
        if tsum == 0:
            continue
        precision_topk += tsum / topk
    res = precision_topk / num_query
    torch.cuda.empty_cache()
    return res

def recall_k(query_code, database_code, query_labels, database_labels, topk=None):

    num_query = query_labels.shape[0]

    recall_topk = 0

    for iter in range(num_query):
        gnd = (query_labels[iter, :] @ database_labels.t() > 0).float()
        total = gnd.sum().int().item()
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[iter, :] @ database_code.t())
        ind = torch.argsort(hamming_dist)
        gnd = gnd[ind]
        tgnd = gnd[0:topk]
        tsum = tgnd.sum().int().item()
        if tsum == 0:
            continue
        recall_topk += tsum / total
    res = recall_topk / num_query
    torch.cuda.empty_cache()
    return res

def pr_curve_1(qF, rF, qL, rL, topK=-1):
    n_query = qF.shape[0]
    if topK == -1 or topK > rF.shape[0]:  # top-K 之 K 的上限
        topK = rF.shape[0]

    Gnd = (qL @ rL.t() > 0).float()
    Rank = torch.argsort(CalcHammingDist(qF,rF))

    P, R = [], []
    for k in range(1, topK + 1, 50):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = torch.zeros(n_query)  # 各 query sample 的 Precision@R
        r = torch.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            gnd_all = torch.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = torch.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k  # 求出所有查询样本的Percision@K
            r[it] = gnd_r / gnd_all  # 求出所有查询样本的Recall@K

        P.append(torch.mean(p))
        R.append(torch.mean(r))

    return P,R


def pr_curve_3(qF, rF, qL, rL, topK=-1):
    n_query = qF.shape[0]
    if topK == -1 or topK > rF.shape[0]:
        topK = rF.shape[0]

    Gnd = (qL @ rL.t() > 0).float()
    Rank = torch.argsort(chunked_hamming_distance(qF,rF))

    P, R = [], []
    for k in range(1, 100 + 1, 20):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = torch.zeros(n_query)  # 各 query sample 的 Precision@R
        r = torch.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            # gnd = (qL[it] @ rL.t() > 0).float()
            gnd_all = torch.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = torch.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k  # 求出所有查询样本的Percision@K
            r[it] = gnd_r / gnd_all  # 求出所有查询样本的Recall@K
        P.append(torch.mean(p))
        R.append(torch.mean(r))

    for k in range(101, 200 + 1, 30):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = torch.zeros(n_query)  # 各 query sample 的 Precision@R
        r = torch.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            gnd_all = torch.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = torch.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k  # 求出所有查询样本的Percision@K
            r[it] = gnd_r / gnd_all  # 求出所有查询样本的Recall@K
        P.append(torch.mean(p))
        R.append(torch.mean(r))

    for k in range(201, 400 + 1, 50):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = torch.zeros(n_query)  # 各 query sample 的 Precision@R
        r = torch.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            gnd_all = torch.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = torch.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k  # 求出所有查询样本的Percision@K
            r[it] = gnd_r / gnd_all  # 求出所有查询样本的Recall@K
        P.append(torch.mean(p))
        R.append(torch.mean(r))

    for k in range(401, topK + 1, 200):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = torch.zeros(n_query)  # 各 query sample 的 Precision@R
        r = torch.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            gnd_all = torch.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = torch.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k  # 求出所有查询样本的Percision@K
            r[it] = gnd_r / gnd_all  # 求出所有查询样本的Recall@K
        P.append(torch.mean(p))
        R.append(torch.mean(r))

    return P,R