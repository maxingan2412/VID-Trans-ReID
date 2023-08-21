from Dataloader import dataloader
from VID_Trans_model import VID_Trans


from Loss_fun import make_loss

import random
import torch
import numpy as np
import os
import argparse

import logging
import os
import time
import torch
import torch.nn as nn
from torch_ema import ExponentialMovingAverage
from torch.cuda import amp
import torch.distributed as dist

from utility import AverageMeter, optimizer,scheduler

import os
from datetime import datetime

   
        

       
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
#性能问题应该发生在这里面。 test里面
def test(model, queryloader, galleryloader, pool='avg', use_gpu=True, ranks=[1, 5, 10, 20]):
    model.eval()
    qf, q_pids, q_camids = [], [], []
    with torch.no_grad():
      for batch_idx, (imgs, pids, camids,_) in enumerate(queryloader):
       
        if use_gpu:
            imgs = imgs.cuda()
        #imgs = Variable(imgs, volatile=True)
        
        b,  s, c, h, w = imgs.size()
        
        
        features = model(imgs,pids,cam_label=camids )
       
        features = features.view(b, -1)
        features = torch.mean(features, 0)
        features = features.data.cpu()
        qf.append(features)
        
        q_pids.append(pids)
        q_camids.extend(camids)
      qf = torch.stack(qf)
      q_pids = np.asarray(q_pids)
      q_camids = np.asarray(q_camids)
      print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
      gf, g_pids, g_camids = [], [], []
      for batch_idx, (imgs, pids, camids,_) in enumerate(galleryloader):
        if use_gpu:
            imgs = imgs.cuda()
        #imgs = Variable(imgs, volatile=True)
        b, s,c, h, w = imgs.size()
        features = model(imgs,pids,cam_label=camids)
        features = features.view(b, -1)
        if pool == 'avg':
            features = torch.mean(features, 0)
        else:
            features, _ = torch.max(features, 0)
        features = features.data.cpu()
        gf.append(features)
        g_pids.append(pids)
        g_camids.extend(camids)
    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) +               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    gf = gf.numpy()
    qf = qf.numpy()
    
    print("Original Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    
    # print("Results ---------- {:.1%} ".format(distmat_rerank))
    print("Results ---------- ")
    
    print("mAP: {:.1%} ".format(mAP))
    print("CMC curve r1:",cmc[0])
    
    return cmc[0], mAP



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VID-Trans-ReID")
    parser.add_argument(
        "--Dataset_name", default="", help="The name of the DataSet", type=str)
    parser.add_argument(
        "--ViT_path", default="", help="The name of the vit pth", type=str)
    parser.add_argument(
        '--epochs', default=120, type=int, help='number of total epochs to run')
    parser.add_argument(
        '--batch_size', default=32, type=int, help='batch size of train')

    args = parser.parse_args()
    Dataset_name=args.Dataset_name
    pretrained_path = args.ViT_path
    epochs = args.epochs
    batch_size = args.batch_size

    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")


    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True    
    train_loader,  num_query, num_classes, camera_num, view_num,q_val_set,g_val_set = dataloader(Dataset_name,batch_size) #这里完成了 datloader的组合
    model = VID_Trans( num_classes=num_classes, camera_num=camera_num,pretrainpath=pretrained_path)
    #这里似乎都还没有实例化，是for循环才开始有各种值的
    loss_fun,center_criterion= make_loss( num_classes=num_classes) # return   loss_func,center_criterion
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr= 0.5)
    
    optimizer= optimizer(model)
    scheduler = scheduler(optimizer)
    scaler = amp.GradScaler()

    #Train
    device = "cuda"
    model=model.to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    cmc_rank1=0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        
        scheduler.step(epoch)
        model.train()
        #还是高搞清楚train_loader的数据结构，这里的train_loader是一个list，长度是epoch的长度，每个元素是一个list，长度是batch的长度，每个元素是一个tuple，长度是4，分别是img，pid，camid，target_cam
        for Epoch_n, (img, pid, target_cam,labels2) in enumerate(train_loader):
            #labels2 是噪声注入的标记，代表每张照片是否注入噪声,但还是没找着注入的地方
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            img = img.to(device) # 32 4 3 256 128
            pid = pid.to(device) # tensor 32  tensor([ 78,  78,  78,  78, 260, 260, 260, 260, 159, 159, 159, 159, 441, 441,441, 441, 535, 535, 535, 535,  69,  69,  69,  69, 237, 237, 237, 237,395, 395, 395, 395], device='cuda:0')
            target_cam = target_cam.to(device) # tensor 128 应该对应上面的32*4 从这里推测  每个batch 应该是128个图片，pid是32，也就是每个tracklets 对应的 id，
            #一个batch 128张图  每4张图是一个tracklets，所以32个tracklets， pid表示的就是这些tracklets的id，因为同一个tracklet的id一样所以只用32就可以表示完成，注意 pid里面可能有重复 因为不同的tracklets可能有相同的id
            labels2=labels2.to(device) # tensor 32 4
            with amp.autocast(enabled=True):
                target_cam=target_cam.view(-1)
                # [Global_ID, Local_ID1, Local_ID2, Local_ID3, Local_ID4 ], [global_feat, part1_f, part2_f, part3_f,part4_f],  a_vals
                score, feat ,a_vals= model(img, pid, cam_label=target_cam) #这里是模型的前向传播 score是一个list 长度是5，元素是 glbaol和local的特征 维度是 32 625
                
                labels2=labels2.to(device)
                attn_noise  = a_vals * labels2 # 俩都是 32 4，
                attn_loss = attn_noise.sum(1).mean() #是一个值了 tensor(1.3899, device='cuda:0', grad_fn=<MeanBackward0>)
                # ID_LOSS+ TRI_LOSS, center
                loss_id ,center= loss_fun(score, feat, pid, target_cam) # 14 2000+
                loss=loss_id+ 0.0005*center +attn_loss
            scaler.scale(loss).backward() #这里是反向传播

            scaler.step(optimizer) # 用计算出的梯度来更新模型的参数。但在实际更新参数之前，GradScaler会首先检查这些梯度是否存在不稳定的情况（例如太大或太小）。如果存在不稳定的情况，它会跳过参数更新，并调整尺度因子以防止将来再次出现这种情况。如果梯度稳定，它会将梯度除以尺度因子并执行参数更新。
            scaler.update() # 更新缩放因子。基于过去几步中梯度的稳定性来动态调整尺度因子。
            ema.update() #这不是混合精度训练的一部分，看起来像是对指数移动平均（Exponential Moving Average, EMA）的更新。EMA通常用于平滑模型参数的变化，使模型在训练的后期阶段更加稳定。

            #我们不但在前面更新了模型参数，这里还更新了center_criterion的参数，是的，CenterLoss中包含可学习参数。
            for param in center_criterion.parameters(): # parameters 625 768
                    param.grad.data *= (1. / 0.0005) #对每个参数的梯度进行缩放。具体来说，它将每个参数的梯度乘以1. / 0.0005，也就是乘以2000。这是梯度重加权的一种策略，可能用于调整特定损失函数对整体参数更新的影响。
            scaler.step(optimizer_center) #使用给定的optimizer_center（一个优化器实例）来更新模型的参数。之前的scaler.scale(loss).backward()代码（从之前的问题中看到）计算了模型的梯度。
            scaler.update() # 更新GradScaler的缩放因子，这是混合精度训练的一部分。


            if isinstance(score, list):
                acc = (score[0].max(1)[1] == pid).float().mean() #score[0].max(1)[1]取列表中的第一个元素并找出每行的最大值的索引。然后，这些索引与pid进行比较，结果是一个布尔张量。通过将其转化为浮点数并计算其均值，我们得到了准确率。
            else:
                acc = (score.max(1)[1] == pid).float().mean()

            loss_meter.update(loss.item(), img.shape[0]) #更新损失的度量。这可能是一个用于跟踪平均损失的实用工具。它使用了当前的损失值和批次中的图像数量（img.shape[0]）
            acc_meter.update(acc, 1)

            torch.cuda.synchronize() # 这是一个CUDA操作同步指令。当你执行一个CUDA操作时，例如GPU上的张量操作，它通常是异步的。这意味着CPU代码会继续执行，而不等待GPU操作完成。这条指令会使CPU等待直到所有CUDA流中的任务都完成。在性能分析、时间测量或确保特定操作前后的数据一致性时，这很有用
            if (Epoch_n + 1) % 50 == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (Epoch_n + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        if (epoch+1)%30 == 0 :
               
               model.eval()
               cmc,map = test(model, q_val_set,g_val_set)
               print('CMC: %.4f, mAP : %.4f'%(cmc,map))
               if cmc_rank1 < cmc:
                  cmc_rank1=cmc

                  save_path = 'VID-Trans-ReID_pth'
                  Dataset_name = args.Dataset_name
                  Batch_size = args.Batch_size
                  current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                  file_name = f"{Dataset_name}_Main_Model_Batch{Batch_size}_{current_time}.pth"
                  save_filename = os.path.join(save_path, file_name)

                  # 创建目录，如果它不存在
                  if not os.path.exists(save_path):
                      os.makedirs(save_path)

                  torch.save(model.state_dict(), save_filename)



               # if cmc_rank1 < cmc:
               #    cmc_rank1=cmc
               #
               #    save_path = 'VID-Trans-ReID'
               #    save_filename = os.path.join(save_path, Dataset_name + 'Main_Model.pth')
               #    # 创建目录，如果它不存在
               #    if not os.path.exists(save_path):
               #        os.makedirs(save_path)
               #
               #    torch.save(model.state_dict(), save_filename)

                  #torch.save(model.state_dict(),os.path.join('/VID-Trans-ReID',  Dataset_name+'Main_Model.pth'))
        
     
