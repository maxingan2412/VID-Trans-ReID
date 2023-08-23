from Dataloader import dataloader
from VID_Trans_model import VID_Trans


from Loss_fun import make_loss
import random
import torch
import numpy as np
import os
import argparse
from tqdm import tqdm

import logging
import os
import time
import torch.nn as nn
import torch
import torch.autograd.profiler as profiler

from torch.cuda import amp
from utility import AverageMeter, optimizer,scheduler

from torch.autograd import Variable








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

# def test(model, queryloader, galleryloader, pool='avg', use_gpu=True, ranks=[1, 5, 10, 20]):
#     model.eval()
#     qf, q_pids, q_camids = [], [], []
#     with torch.no_grad():
#       #with profiler.profile(record_shapes=True, profile_memory=True, use_cuda=True) as prof:
#       for batch_idx, (imgs, pids, camids,_) in enumerate(tqdm(queryloader)): #1980
#
#         if use_gpu:
#             imgs = imgs.cuda(non_blocking=True)
#         #imgs = Variable(imgs, volatile=True)
#
#         b,  s, c, h, w = imgs.size()
#
#         features = model(imgs,pids,cam_label=camids )
#
#         features = features.view(b, -1)
#         features = torch.mean(features, 0)
#         features = features.data.cpu()
#         qf.append(features)
#
#         q_pids.append(pids)
#         q_camids.extend(camids)
#
#       #print(prof.key_averages().table(sort_by="self_cpu_time_total"))
#
#       qf = torch.stack(qf)
#       q_pids = np.asarray(q_pids)
#       q_camids = np.asarray(q_camids)
#       print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
#       gf, g_pids, g_camids = [], [], []
#       for batch_idx, (imgs, pids, camids,_) in enumerate(tqdm(galleryloader)): #9330
#         if use_gpu:
#             imgs = imgs.cuda(non_blocking=True)
#         #imgs = Variable(imgs, volatile=True)
#         b, s,c, h, w = imgs.size()
#         features = model(imgs,pids,cam_label=camids)
#         features = features.view(b, -1)
#         if pool == 'avg':
#             features = torch.mean(features, 0)
#         else:
#             features, _ = torch.max(features, 0)
#         features = features.data.cpu()
#         gf.append(features)
#         g_pids.append(pids)
#         g_camids.extend(camids)
#     gf = torch.stack(gf)
#     g_pids = np.asarray(g_pids)
#     g_camids = np.asarray(g_camids)
#     print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
#     print("Computing distance matrix")
#     m, n = qf.size(0), gf.size(0)
#     distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     distmat.addmm_(1, -2, qf, gf.t())
#     distmat = distmat.numpy()
#     gf = gf.numpy()
#     qf = qf.numpy()
#
#     print("Original Computing CMC and mAP")
#     cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
#
#     # print("Results ---------- {:.1%} ".format(distmat_rerank))
#     print("Results ---------- ")
#
#     print("mAP: {:.1%} ".format(mAP))
#     print("CMC curve r1:",cmc[0])
#
#     return cmc[0], mAP

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

    features = torch.stack(features_list)
    pids = np.asarray(pids_list)
    camids = np.asarray(camids_list)

    return features, pids, camids


def compute_distance_matrix(qf, gf):
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    return distmat


def test(model, queryloader, galleryloader, pool='avg', use_gpu=True):
    model.eval()

    qf, q_pids, q_camids = extract_features(queryloader, model, use_gpu, pool)
    gf, g_pids, g_camids = extract_features(galleryloader, model, use_gpu, pool)

    print(f"Extracted features for query set, obtained {qf.size(0)}-by-{qf.size(1)} matrix")
    print(f"Extracted features for gallery set, obtained {gf.size(0)}-by-{gf.size(1)} matrix")

    print("Computing distance matrix")
    distmat = compute_distance_matrix(qf, gf).numpy()

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
    args = parser.parse_args()
    Dataset_name=args.Dataset_name
    pretrainpath=args.model_path
    batch_size=args.batch_size

    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")



    train_loader,  num_query, num_classes, camera_num, view_num,q_val_set,g_val_set = dataloader(Dataset_name,batch_size)
    model = VID_Trans( num_classes=num_classes, camera_num=camera_num,pretrainpath='/home/ma1/work/VID-Trans-ReID/jx_vit_base_p16_224-80ecf9dd.pth')

    device = "cuda"
    model=model.to(device)

    checkpoint = torch.load(pretrainpath)
    model.load_state_dict(checkpoint)


    model.eval()
    cmc,map = test(model, q_val_set,g_val_set)
    print('CMC: %.4f, mAP : %.4f'%(cmc,map))

