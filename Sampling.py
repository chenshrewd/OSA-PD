import torch
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
import random
import pdb
from scipy import stats
from utils import *


def supervised_learning(args, unlabeledloader, full_data):
    n_classes = args.known_class
    k = args.query_batch
    quertIndex = []
    labelArr = []
    precision, recall = 0, 0

    for _, (index, (_, labels)) in enumerate(unlabeledloader):
        quertIndex += index
        labelArr += list(np.array(labels.data))

    known_indexs = [quertIndex[i].item() for i in range(len(labelArr)) if labelArr[i].item() < n_classes]
    if not full_data:
        sampled_data = random.sample(known_indexs, min(k, len(known_indexs)))
    else:
        sampled_data = known_indexs
    return sampled_data, [], precision, recall


def random_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    precision, recall = 0, 0
    for _, (index, (_, labels)) in enumerate(unlabeledloader):
        queryIndex += index
        labelArr += list(np.array(labels.data))

    tmp_data = np.vstack((queryIndex, labelArr)).T
    np.random.shuffle(tmp_data)
    tmp_data = tmp_data.T  # 未标记集所有数据
    queryIndex = tmp_data[0][:args.query_batch]
    labelArr = tmp_data[1]
    queryLabelArr = tmp_data[1][:args.query_batch]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def confidence_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu, method):
    model.eval()
    flag = 0
    n = args.query_batch
    if method == "MostConfidence":
        flag = 1
    unlabeled_index, unlabeled_label, unlabeled_confidence = get_my_confidence(model, unlabeledloader, use_gpu)

    if flag == 1:
        sorted_idx = torch.argsort(unlabeled_confidence, descending=True)
    else:
        sorted_idx = torch.argsort(unlabeled_confidence, descending=False)

    selected_index = unlabeled_index[sorted_idx][:n]
    selected_label = unlabeled_label[sorted_idx][:n]

    right_selected_index = selected_index[selected_label < args.known_class]
    wrong_selected_index = selected_index[selected_label >= args.known_class]

    precision = len(right_selected_index) / len(selected_label)

    recall = (right_selected_index.size(0) + Len_labeled_ind_train) / (
            len(np.where(unlabeled_label < args.known_class)[0]) + Len_labeled_ind_train)

    return right_selected_index.tolist(), wrong_selected_index.tolist(), precision, recall


def uncertainty_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    precision, recall = 0, 0
    for _, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        features, outputs = model(data)

        softmax_output = torch.softmax(outputs, 1)
        uncertaintyArr += list(np.array((-softmax_output * torch.log(softmax_output)).sum(1).cpu().data))
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    tmp_data = np.vstack((uncertaintyArr, queryIndex, labelArr)).T
    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T
    queryIndex = tmp_data[1][-args.query_batch:].astype(int)
    labelArr = tmp_data[2].astype(int)
    queryLabelArr = tmp_data[2][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def badge_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    torch.cuda.empty_cache()
    model.eval()
    n_classes = args.known_class
    K = args.query_batch

    gradEmbedding, unlabeled_index, unlabeled_label = get_grad_embed(model, unlabeledloader, n_classes, use_gpu)

    embs = torch.Tensor(gradEmbedding)
    ind = torch.argmax(torch.norm(embs, 2, 1)).item()
    # embs = embs.cuda()
    mu = [embs[ind]]
    indsAll = [ind]
    centInds = [0.] * len(embs)
    cent = 0
    while len(mu) < K:
        if len(mu) == 1:
            D2 = torch.cdist(mu[-1].view(1, -1), embs, 2)[0].numpy()
        else:
            newD = torch.cdist(mu[-1].view(1, -1), embs, 2)[0].numpy()
            for i in range(len(embs)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(embs[ind])
        indsAll.append(ind)
        cent += 1
        torch.cuda.empty_cache()

    indsAll_tensor = torch.tensor(indsAll)
    selected_index = unlabeled_index[indsAll_tensor]
    selected_label = unlabeled_label[indsAll_tensor]
    right_selected_index = selected_index[selected_label < args.known_class]
    wrong_selected_index = selected_index[selected_label >= args.known_class]

    precision = len(right_selected_index) / len(selected_label)

    recall = (right_selected_index.size(0) + Len_labeled_ind_train) / (
            len(np.where(unlabeled_label < args.known_class)[0]) + Len_labeled_ind_train)

    return right_selected_index.tolist(), wrong_selected_index.tolist(), precision, recall


def coreset_sampling(args, trainloader, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    labeled_data_label, labeled_embeddings = get_label_embed(model, trainloader, use_gpu)
    unlabeled_data_index, unlabeled_data_label, unlabeled_embeddings = get_my_label_embed(model, unlabeledloader,
                                                                                          use_gpu)
    m = unlabeled_embeddings.size(0)  # 样本数量
    if labeled_embeddings.size(0) == 0:
        min_dist = np.tile(float("inf"), m)
    else:
        dist_ctr = pairwise_distances(unlabeled_embeddings, labeled_embeddings)
        min_dist = np.amin(dist_ctr, axis=1)  # 求出所有未标记集数据与标记集数据最小的距离

    idxs = []
    n = args.query_batch

    for i in range(n):
        idx = min_dist.argmax()
        idxs.append(idx)
        dist_new_ctr = pairwise_distances(unlabeled_embeddings, unlabeled_embeddings[[idx], :])
        for j in range(m):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

    idxs_tensor = torch.tensor(idxs)

    selected_index = unlabeled_data_index[idxs_tensor]
    selected_label = unlabeled_data_label[idxs_tensor]
    right_selected_index = selected_index[selected_label < args.known_class]
    wrong_selected_index = selected_index[selected_label >= args.known_class]

    precision = len(right_selected_index) / len(selected_label)

    recall = (right_selected_index.size(0) + Len_labeled_ind_train) / (
            len(np.where(unlabeled_data_label < args.known_class)[0]) + Len_labeled_ind_train)

    return right_selected_index.tolist(), wrong_selected_index.tolist(), precision, recall


def lfpsa_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    S_ij = {}
    for _, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        _, outputs = model(data)
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))
        # activation value based
        v_ij, predicted = outputs.max(1)
        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i]
            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]
            if tmp_class not in S_ij:
                S_ij[tmp_class] = []
            S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])

    # fit a two-component GMM for each class
    tmp_data = []
    for tmp_class in S_ij:
        S_ij[tmp_class] = np.array(S_ij[tmp_class])
        activation_value = S_ij[tmp_class][:, 0]
        if len(activation_value) < 2:
            continue
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(np.array(activation_value).reshape(-1, 1))
        prob = gmm.predict_proba(np.array(activation_value).reshape(-1, 1))
        prob = prob[:, gmm.means_.argmax()]
        if tmp_class == args.known_class:
            prob = [0] * len(prob)
            prob = np.array(prob)

        if len(tmp_data) == 0:
            tmp_data = np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))
        else:
            tmp_data = np.vstack((tmp_data, np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))))

    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T
    queryIndex = tmp_data[2][-args.query_batch:].astype(int)
    labelArr = tmp_data[3].astype(int)
    queryLabelArr = tmp_data[3][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def osa_prd_sampling(args, trainloader, unlabeledloader, Len_labeled_ind_train, model_D, model_C, use_gpu):
    model_D.eval()
    model_C.eval()
    labeled_data_label, labeled_embeddings = get_label_embed(model_D, trainloader, use_gpu)
    anchor = torch.tensor([])

    for i in range(args.known_class):
        indices = torch.nonzero(labeled_data_label == i).squeeze()
        embeddings_class = labeled_embeddings[indices]
        anchor_i = embeddings_class.mean(dim=0).view(1, -1)
        anchor = torch.cat((anchor, anchor_i), dim=0)

    indices = torch.nonzero(labeled_data_label >= args.known_class).squeeze()
    embeddings_class = labeled_embeddings[indices]
    if torch.numel(embeddings_class) != 0:
        anchor_i = embeddings_class.mean(dim=0).view(1, -1)
        anchor = torch.cat((anchor, anchor_i), dim=0)

    unlabeled_data_index, unlabeled_data_label, unlabeled_probs_value_1, unlabeled_probs_value_2, unlabeled_probs_1, unlabeled_probs_2, unlabeled_embeddings, unlabeled_probs_1_2 = get_pro_embed(
        model_D, model_C, unlabeledloader, use_gpu)

    distance = torch.cdist(unlabeled_embeddings, anchor, p=2, compute_mode='donot_use_mm_for_euclid_dist')

    values, indices = torch.topk(distance, k=2, dim=1, largest=False)
    eur_pro_1 = indices[:, 0]
    eur_pro_2 = indices[:, 1]

    prob_sort_index = torch.argsort(unlabeled_probs_value_1, descending=True)

    sorted_unlabeled_probs_1 = unlabeled_probs_1[prob_sort_index]
    sorted_unlabeled_probs_2 = unlabeled_probs_2[prob_sort_index]
    sorted_unlabeled_probs_1_2 = unlabeled_probs_1_2[prob_sort_index]
    sorted_eur_pro_1 = eur_pro_1[prob_sort_index]
    sorted_eur_pro_2 = eur_pro_2[prob_sort_index]
    sorted_unlabeled_data_label = unlabeled_data_label[prob_sort_index]
    sorted_unlabeled_data_index = [unlabeled_data_index[i] for i in prob_sort_index]

    is_consistent = torch.nonzero(
        torch.ne(sorted_unlabeled_probs_1, args.known_class) & torch.ne(sorted_unlabeled_probs_2, args.known_class) &
        torch.ne(sorted_eur_pro_1, args.known_class) & torch.ne(sorted_eur_pro_2, args.known_class) &
        (torch.ne(sorted_unlabeled_probs_1, sorted_eur_pro_1) | torch.ne(sorted_unlabeled_probs_1_2, sorted_eur_pro_1) |
         torch.ne(sorted_unlabeled_probs_1, sorted_unlabeled_probs_1_2)))

    not_consistent_index = [sorted_unlabeled_data_index[i.item()] for i in is_consistent]

    selected_label = sorted_unlabeled_data_label[is_consistent]

    if is_consistent.shape[0] >= args.query_batch:
        selected_index = not_consistent_index[:args.query_batch]
        selected_label = selected_label[:args.query_batch].squeeze().tolist()
    else:
        consistent_index = [i for i in sorted_unlabeled_data_index if i not in not_consistent_index]
        selected_index = not_consistent_index + consistent_index[:(args.query_batch - is_consistent.shape[0])]
        selected_label = selected_label.squeeze().tolist()
        sorted_unlabeled_data_label_list = sorted_unlabeled_data_label.tolist()
        unselected_label = [i for i in sorted_unlabeled_data_label_list if i not in selected_label]
        selected_label = selected_label + unselected_label[:(args.query_batch - is_consistent.shape[0])]

    right_selected_index = [selected_index[i] for i in range(len(selected_label)) if
                            selected_label[i] < args.known_class]

    wrong_selected_index = [i for i in selected_index if i not in right_selected_index]

    precision = len(right_selected_index) / len(selected_label)

    recall = (len(right_selected_index) + Len_labeled_ind_train) / (
                len(np.where(sorted_unlabeled_data_label < args.known_class)[0]) + Len_labeled_ind_train)

    return right_selected_index, wrong_selected_index, precision, recall


def osa_pdd_sampling(args, trainloader, unlabeledloader, Len_labeled_ind_train, model_D, model_C, use_gpu):
    model_D.eval()
    model_C.eval()

    n = args.known_class
    k = args.query_batch

    labeled_data_label, labeled_embeddings = get_label_embed(model_D, trainloader, use_gpu)
    anchor = torch.tensor([])

    for i in range(n):
        indices = torch.nonzero(labeled_data_label == i).squeeze()
        embeddings_class = labeled_embeddings[indices]
        anchor_i = embeddings_class.mean(dim=0).view(1, -1)
        anchor = torch.cat((anchor, anchor_i), dim=0)

    indices = torch.nonzero(labeled_data_label >= n).squeeze()
    embeddings_class = labeled_embeddings[indices]
    if torch.numel(embeddings_class) != 0:
        anchor_i = embeddings_class.mean(dim=0).view(1, -1)
        anchor = torch.cat((anchor, anchor_i), dim=0)

    unlabeled_index, unlabeled_label, logits_D, logits_C, unlabeled_emb = get_probs_embed(model_D, model_C, unlabeledloader, use_gpu)

    distance = torch.cdist(unlabeled_emb, anchor, p=2, compute_mode='donot_use_mm_for_euclid_dist')

    values, indices = torch.topk(distance, k=2, dim=1, largest=False)

    eur_pro_1_indice = indices[:, 0]
    eur_pro_2_indice = indices[:, 1]

    values, indices = torch.topk(logits_D, k=2, dim=1, largest=True)
    pre_D_1_indice = indices[:, 0]
    pre_D_2_indice = indices[:, 1]

    unknown_index_d_1 = torch.nonzero(pre_D_1_indice == n).squeeze()
    unknown_index_d_2 = torch.nonzero(pre_D_2_indice == n).squeeze()
    unknown_index_e_1 = torch.nonzero(eur_pro_1_indice == n).squeeze()
    unknown_index_e_2 = torch.nonzero(eur_pro_2_indice == n).squeeze()
    unknown_index = torch.unique(torch.cat((unknown_index_d_1, unknown_index_d_2, unknown_index_e_1, unknown_index_e_2)))
    print("unknown_num", unknown_index.size(0))
    known_index = torch.tensor([i for i in range(len(unlabeled_index)) if i not in unknown_index])
    if known_index.numel() == 0:
        known_index = unknown_index
        print("known_index is a empty tensor!!!!!!")
    known_logits_D = logits_D[known_index]
    known_logits_C = logits_C[known_index]

    known_logits_D_n = known_logits_D[:, :-1]

    scores = DKL(args, known_logits_D_n, known_logits_C)
    if k <= scores.size(0):
        _, top_indices = torch.topk(scores, k=k, largest=True)
        known_selected_index = known_index[top_indices]
        unknown_selected_index = torch.tensor([], dtype=torch.int64)
    else:
        known_selected_index = known_index
        unknown_indices = torch.randperm(unknown_index.size(0))[:(k-scores.size(0))]
        unknown_selected_index = unknown_index[unknown_indices]
    select_index = torch.cat((known_selected_index, unknown_selected_index), dim=0)
    print("select_num", select_index.size(0))
    selected_index = [unlabeled_index[e] for e in select_index]
    selected_label = unlabeled_label[select_index]
    right_selected_index = [selected_index[i] for i in range(len(selected_label)) if selected_label[i] < n]

    wrong_selected_index = [i for i in selected_index if i not in right_selected_index]

    precision = len(right_selected_index) / len(selected_label)

    recall = (len(right_selected_index) + Len_labeled_ind_train) / (len(np.where(unlabeled_label < n)[0]) + Len_labeled_ind_train)

    return right_selected_index, wrong_selected_index, precision, recall

