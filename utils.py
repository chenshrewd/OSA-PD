import os
import copy
import sys
import errno
import shutil
from sklearn.manifold import TSNE
import os.path as osp
import torch.nn.functional as F
import numpy as np

import torch


def euclidean_func(mu, embs):
    D2 = []
    batch_size = 500
    for i in range(0, embs.shape[0], batch_size):
        batch_embs = embs[i:i + batch_size]
        dist = torch.cdist(mu.view(1, -1), batch_embs, 2).squeeze()
        D2.append(dist.cpu().numpy())

    D2 = np.concatenate(D2, axis=0)
    return D2


def get_dataentropy(args, dataloader, model, use_gpu):

    info = 0.0

    for _, (data, labels) in enumerate(dataloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        _, outputs = model(data)

        softmax_output = torch.softmax(outputs, 1)
        entropy_list = list(np.array((-softmax_output * torch.log(softmax_output)).sum(1).cpu().data))
        info_batch = sum(entropy_list)
        info += info_batch

    info /= len(dataloader.sampler)
    return info


def DKL(args, logits_D, logits_C):

    temperature = args.T
    alpha = args.alpha
    beta = args.beta
    target_D = logits_D.max(1)[1]
    target_C = logits_C.max(1)[1]
    gt_mask_D = _get_gt_mask(logits_D, target_D)
    gt_mask_C = _get_gt_mask(logits_C, target_C)
    gt_mask = torch.logical_or(gt_mask_D, gt_mask_C)
    other_mask_D = _get_other_mask(logits_D, target_D)
    other_mask_C = _get_other_mask(logits_C, target_C)
    other_mask = torch.logical_and(other_mask_D,other_mask_C)
    pred_D = F.softmax(logits_D / temperature, dim=1)
    pred_C = F.softmax(logits_C / temperature, dim=1)
    pred_D = cat_mask(pred_D, gt_mask, other_mask)
    pred_C = cat_mask(pred_C, gt_mask, other_mask)
    log_pred_D = torch.log(pred_D)

    tckd_score = F.kl_div(log_pred_D, pred_C, reduction='none') * (temperature ** 2) / target_C.shape[0]
    tckd_score = torch.sum(tckd_score, dim=1)

    pred_C_part2 = F.softmax(logits_C / temperature - 1000.0 * gt_mask, dim=1)
    log_pred_D_part2 = F.log_softmax(logits_D / temperature - 1000.0 * gt_mask, dim=1)
    nckd_score = F.kl_div(log_pred_D_part2, pred_C_part2, reduction='none') * (temperature ** 2) / target_C.shape[0]
    nckd_score = torch.sum(nckd_score, dim=1)

    scores = alpha * tckd_score + beta * nckd_score
    return scores


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def get_label_embed(model, dataloader, use_gpu):
    model.eval()
    embsize = model.module.linear2.in_features
    data_label = torch.zeros([len(dataloader.sampler), ], dtype=torch.int64)
    embeddings = torch.zeros([len(dataloader.sampler), embsize], dtype=torch.float32)
    with torch.no_grad():
        start = 0
        end = 0
        for _, (data, labels) in enumerate(dataloader):
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            batch_size = labels.shape[0]
            end += batch_size
            features, _ = model(data)

            data_label[start:end] = labels.cpu()
            embeddings[start:end] = features.cpu()
            start += batch_size
    return data_label, embeddings


def get_my_label_embed(model, dataloader, use_gpu):
    model.eval()
    embsize = model.module.linear2.in_features
    data_index = torch.zeros([len(dataloader.sampler), ], dtype=torch.int64)
    data_label = torch.zeros([len(dataloader.sampler), ], dtype=torch.int64)
    embeddings = torch.zeros([len(dataloader.sampler), embsize], dtype=torch.float32)
    with torch.no_grad():
        start = 0
        end = 0
        for _, (index, (data, labels)) in enumerate(dataloader):
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            batch_size = labels.shape[0]
            end += batch_size
            features, _ = model(data)

            data_index[start:end] = index
            data_label[start:end] = labels.cpu()
            embeddings[start:end] = features.cpu()
            start += batch_size
    return data_index, data_label, embeddings


def get_my_confidence(model, dataloader, use_gpu):
    model.eval()
    data_index = torch.zeros([len(dataloader.sampler), ], dtype=torch.int64)
    data_label = torch.zeros([len(dataloader.sampler), ], dtype=torch.int64)
    data_confidence = torch.zeros([len(dataloader.sampler), ], dtype=torch.float32)
    with torch.no_grad():
        start = 0
        end = 0
        for _, (index, (data, labels)) in enumerate(dataloader):
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            batch_size = labels.shape[0]
            end += batch_size
            _, outputs = model(data)
            confidence = torch.max(outputs, dim=1)[0]

            data_index[start:end] = index
            data_label[start:end] = labels.cpu()
            data_confidence[start:end] = confidence.cpu()
            start += batch_size
    return data_index, data_label, data_confidence


def get_pro_embed(model_D, model_C, dataloader, use_gpu):
    model_D.eval()
    model_C.eval()

    embsize = model_D.module.linear2.in_features
    data_index = list(range(len(dataloader.sampler)))
    data_label = torch.zeros([len(dataloader.sampler), ], dtype=torch.int64)
    probs_value_1 = torch.zeros([len(dataloader.sampler), ], dtype=torch.float32)
    probs_value_2 = torch.zeros([len(dataloader.sampler), ], dtype=torch.float32)
    probs_1 = torch.zeros([len(dataloader.sampler), ], dtype=torch.int64)
    probs_2 = torch.zeros([len(dataloader.sampler), ], dtype=torch.int64)
    probs_1_C = torch.zeros([len(dataloader.sampler), ], dtype=torch.int64)
    embeddings = torch.zeros([len(dataloader.sampler), embsize], dtype=torch.float32)
    with torch.no_grad():
        start = 0
        end = 0
        for _, (index, (data, labels)) in enumerate(dataloader):
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            batch_size = len(index)
            end += batch_size
            features_D, outputs_D = model_D(data)
            _, outputs_C = model_C(data)

            values, indices = torch.topk(outputs_D, k=2, dim=1, largest=True)
            predictions_D_1_values = values[:, 0]
            predictions_D_2_values = values[:, 1]
            predictions_D_1_indices = indices[:, 0]
            predictions_D_2_indices = indices[:, 1]
            predictions_C = outputs_C.max(1)
            data_index[start:end] = index.tolist()
            data_label[start:end] = labels.cpu()
            probs_value_1[start:end] = predictions_D_1_values[0].cpu()
            probs_value_2[start:end] = predictions_D_2_values[0].cpu()
            probs_1[start:end] = predictions_D_1_indices.cpu()
            probs_2[start:end] = predictions_D_2_indices.cpu()
            probs_1_C[start:end] = predictions_C[1].cpu()
            embeddings[start:end] = features_D.cpu()
            start += batch_size
    return data_index, data_label, probs_value_1, probs_value_2, probs_1, probs_2, embeddings, probs_1_C


def get_probs_embed(model_D, model_C, dataloader, use_gpu):
    model_D.eval()
    model_C.eval()

    embsize = model_D.module.linear2.in_features
    n_D = model_D.module.linear2.out_features
    n_C = model_C.module.linear2.out_features

    data_index = list(range(len(dataloader.sampler)))
    data_label = torch.zeros([len(dataloader.sampler), ], dtype=torch.int64)
    probs_D = torch.zeros([len(dataloader.sampler), n_D], dtype=torch.float32)
    probs_C = torch.zeros([len(dataloader.sampler), n_C], dtype=torch.float32)
    embeddings = torch.zeros([len(dataloader.sampler), embsize], dtype=torch.float32)

    with torch.no_grad():
        start = 0
        end = 0
        for _, (index, (data, labels)) in enumerate(dataloader):
            if use_gpu:
                data = data.cuda()
            batch_size = len(index)
            end += batch_size
            features_D, outputs_D = model_D(data)
            _, outputs_C = model_C(data)

            data_index[start:end] = index.tolist()
            data_label[start:end] = labels
            probs_D[start:end] = outputs_D.cpu()
            probs_C[start:end] = outputs_C.cpu()
            embeddings[start:end] = features_D.cpu()
            start += batch_size

    return data_index, data_label, probs_D, probs_C, embeddings


def get_grad_embed(model, dataloader, n_classes, use_gpu):
    model.eval()

    embsize = model.module.linear2.in_features
    embedding = np.zeros([len(dataloader.dataset.data), embsize * (n_classes)])
    data_index = torch.zeros([len(dataloader.sampler), ], dtype=torch.int64)
    data_label = torch.zeros([len(dataloader.sampler), ], dtype=torch.int64)

    # 获取梯度向量
    with torch.no_grad():
        start = 0
        end = 0
        for _, (index, (data, labels)) in enumerate(dataloader):

            batch_size = len(index)
            end += batch_size
            data_index[start:end] = index
            data_label[start:end] = labels
            start += batch_size

            if use_gpu:
                data, labels = data.cuda(), labels.cuda()

            features, outputs = model(data)

            features = features.data.cpu().numpy()
            batchProbs = F.softmax(outputs, dim=1).data.cpu().numpy()
            maxInds = np.argmax(batchProbs, 1)
            for j in range(len(labels)):
                for c in range(n_classes):
                    if c == maxInds[j]:
                        embedding[index[j]][embsize * c: embsize * (c + 1)] = copy.deepcopy(features[j]) * (
                                1 - batchProbs[j][c])
                    else:
                        embedding[index[j]][embsize * c: embsize * (c + 1)] = copy.deepcopy(features[j]) * (
                                -1 * batchProbs[j][c])

        embedding = embedding[~np.all(embedding == 0, axis=1)]
        return torch.Tensor(embedding), data_index, data_label


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    """
    Write console output to external text file.
    
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
