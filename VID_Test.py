from Dataloader import dataloader
from VID_Trans_model import VID_Trans,VID_TransVideo


from Loss_fun import make_loss
import random
import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import sys
import logging
import os
import time
import torch.nn as nn
import torch
import torch.autograd.profiler as profiler

from torch.cuda import amp
from utility import AverageMeter, optimizer,scheduler

from torch.autograd import Variable
from torch.nn import functional as F


from rerank import re_ranking


def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = mat1 + mat2
    distmat.addmm_(input1, input2.t(), beta=1, alpha=-2)
    return distmat

def compute_distance_matrix(input1, input2, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        torch.Tensor: distance matrix.

    Examples::
       # >>> from torchreid import metrics
       # >>> input1 = torch.rand(10, 2048)
       # >>> input2 = torch.rand(100, 2048)
       # >>> distmat = metrics.compute_distance_matrix(input1, input2)
       # >>> distmat.size() # (10, 100)
    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input1.dim()
    )
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input2.dim()
    )
    assert input1.size(1) == input2.size(1)

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat





def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


#823 测试的时候每次一个tracklets的数据，所以有长有短，这里面的b, s, c, h, w 中的b其实是指的是tracklets的长度
def extract_features(data_loader, model, use_gpu=True, pool='avg'):
    features_list, pids_list, camids_list = [], [], []
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids, _) in enumerate(tqdm(data_loader)):
        #for imgs, pids, camids, _ in tqdm(data_loader):
            if use_gpu:
                imgs = imgs.cuda(non_blocking=True)

            b, s, c, h, w = imgs.size()
            features = model(imgs, pids, cam_label=camids)
            features = features.view(b, -1)
            if pool == 'avg':
                features = torch.mean(features, 0)
            else:
                features, _ = torch.max(features, 0)
            features = features.data.cpu()

            features_list.append(features)

            # Ensure pids and camids are iterable (list or tensor) before extending
            if not isinstance(pids, (list, torch.Tensor)):
                pids = [pids]
            if not isinstance(camids, (list, torch.Tensor)):
                camids = [camids]

            pids_list.extend(pids)
            camids_list.extend(camids)
    #823这里的提取特征是一个tracklets提出来，每一个tracklets对应了一个feature
    features = torch.stack(features_list) # 一个1980场的list 变成  1980 * 13506的tensro
    pids = np.asarray(pids_list) #1980的list 变为array
    camids = np.asarray(camids_list) #110860

    return features, pids, camids


# def compute_distance_matrix(qf, gf):
#     m, n = qf.size(0), gf.size(0) # 1980 9330
#     distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     distmat.addmm_(1, -2, qf, gf.t())
#     return distmat


def test(model, queryloader, galleryloader, pool='avg', use_gpu=True):
    model.eval()
    #qf1980 13056 tensor  q_pids 1980  q_camids 110860 因为一个tracklets id一样，所以是1980，但是camerid不一定一样 所以每个tacklets下的每个图片的camid都得记录，是110860
    qf, q_pids, q_camids = extract_features(queryloader, model, use_gpu, pool)
    gf, g_pids, g_camids = extract_features(galleryloader, model, use_gpu, pool)

    print(f"Extracted features for query set, obtained {qf.size(0)}-by-{qf.size(1)} matrix")
    print(f"Extracted features for gallery set, obtained {gf.size(0)}-by-{gf.size(1)} matrix")

    print("Computing distance matrix")
    #distmat = compute_distance_matrix(qf, gf).numpy()
    metricchoose = 'euclidean'
    distmat = compute_distance_matrix(qf, gf, metricchoose)
    distmat = distmat.numpy()

    rerank = True
    if rerank:
        #print('Applying person re-ranking ...')
        distmat_qq = compute_distance_matrix(qf, qf, metricchoose)
        distmat_gg = compute_distance_matrix(gf, gf, metricchoose)
        distmat1 = re_ranking(distmat, distmat_qq, distmat_gg)
        print("Computing CMC and mAP with reranking,")
        cmc, mAP = evaluate(distmat1, q_pids, g_pids, q_camids, g_camids)

        print("Results ---------- ")
        print(f"mAP: {mAP:.1%}")
        print(f"CMC curve r1: {cmc[0]}")






    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ---------- ")
    print(f"mAP: {mAP:.1%}")
    print(f"CMC curve r1: {cmc[0]}")

    return cmc[0], mAP




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VID-Trans-ReID")
    parser.add_argument(
        "--Dataset_name", default="", help="The name of the DataSet", type=str)
    parser.add_argument(
        "--model_path", default="", help="pretrained model", type=str)
    parser.add_argument(
        '--batch_size', default=32, type=int, help='batch size of train')

    parser.add_argument(
        '--seq_len', default=4, type=int, help='seq len')
    args = parser.parse_args()
    Dataset_name=args.Dataset_name
    pretrainpath=args.model_path
    batch_size=args.batch_size
    seq_len=args.seq_len

    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")



    train_loader,  num_query, num_classes, camera_num, view_num,q_val_set,g_val_set = dataloader(Dataset_name,batch_size,seq_len)
    #model = VID_Trans(num_classes=num_classes, camera_num=camera_num,pretrainpath='/home/ma1/work/VID-Trans-ReID/jx_vit_base_p16_224-80ecf9dd.pth',seq_len=seq_len)
    model = VID_TransVideo(num_classes=num_classes, camera_num=camera_num,pretrainpath='/home/ma1/work/VID-Trans-ReID/jx_vit_base_p16_224-80ecf9dd.pth',seq_len=seq_len)

    device = "cuda"
    model=model.to(device)

    checkpoint = torch.load(pretrainpath)
    model.load_state_dict(checkpoint)


    model.eval()
    cmc,map = test(model, q_val_set,g_val_set)
    print('CMC: %.4f, mAP : %.4f'%(cmc,map))

