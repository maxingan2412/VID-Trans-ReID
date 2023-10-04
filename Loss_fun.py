import torch.nn.functional as F
from loss.softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss


class LossMaker:
    def __init__(self, num_classes):
        self.feat_dim = 768
        self.center_criterion = CenterLoss(num_classes=num_classes, feat_dim=self.feat_dim, use_gpu=True)
        self.center_criterion2 = CenterLoss(num_classes=num_classes, feat_dim=3072, use_gpu=True)
        self.triplet = TripletLoss()
        self.xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def compute_id_loss(self, score, target):
        if isinstance(score, list):
            loss_list = [self.xent(scor, target) for scor in score[1:]]
            avg_loss = sum(loss_list) / len(loss_list)
            return 0.25 * avg_loss + 0.75 * self.xent(score[0], target)
        else:
            return self.xent(score, target)

    def compute_triplet_loss(self, feat, target):
        if isinstance(feat, list):
            loss_list = [self.triplet(feats, target)[0] for feats in feat[1:]]
            avg_loss = sum(loss_list) / len(loss_list)
            return 0.25 * avg_loss + 0.75 * self.triplet(feat[0], target)[0]
        else:
            return self.triplet(feat, target)[0]

    def compute_center(self, feat, target):
        if isinstance(feat, list):
            center1 = self.center_criterion(feat[0], target)
            center2_list = [self.center_criterion2(feats, target) for feats in feat[1:]]
            avg_center2 = sum(center2_list) / len(center2_list)
            return 0.25 * avg_center2 + 0.75 * center1
        else:
            return self.center_criterion(feat, target)

    def loss_func(self, score, feat, target, target_cam):
        id_loss = self.compute_id_loss(score, target)
        tri_loss = self.compute_triplet_loss(feat, target)
        center = self.compute_center(feat, target)
        return id_loss + tri_loss, center




def make_loss(num_classes):
    feat_dim = 768
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    center_criterion2 = CenterLoss(num_classes=num_classes, feat_dim=3072, use_gpu=True)

    triplet = TripletLoss()
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, target, target_cam):
        if isinstance(score, list):
            ID_LOSS = [xent(scor, target) for scor in score[1:]]
            ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
            ID_LOSS = 0.25 * ID_LOSS + 0.75 * xent(score[0], target)
        else:
            ID_LOSS = xent(score, target)

        if isinstance(feat, list):
            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
            TRI_LOSS = 0.25 * TRI_LOSS + 0.75 * triplet(feat[0], target)[0]

            center = center_criterion(feat[0], target)
            centr2 = [center_criterion2(feats, target) for feats in feat[1:]]
            centr2 = sum(centr2) / len(centr2)
            center = 0.25 * centr2 + 0.75 * center
        else:
            TRI_LOSS = triplet(feat, target)[0]

        return ID_LOSS + TRI_LOSS, center

    return loss_func, center_criterion


def make_loss_seq(num_classes,seq_len):
    
    feat_dim =768
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    # center_criterion2 = CenterLoss(num_classes=num_classes, feat_dim= seq_len * 768, use_gpu=True)
    center_criterion2 = CenterLoss(num_classes=num_classes, feat_dim= 1 * 768, use_gpu=True)
    triplet = TripletLoss()
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        
    #这里定义了各种损失 loss_id ,center= loss_fun(score, feat, pid, target_cam)
    def loss_func(score, feat, target, target_cam):
        if isinstance(score, list): # 交叉熵损失
                ID_LOSS = [xent(scor, target) for scor in score[1:]] #  scor tensor 32 625 target tensor 32,这俩是怎么做的损失？--是可以做的  target是pid  [tensor(6.4318, device='cuda:0', grad_fn=<SumBackward0>), tensor(6.4446, device='cuda:0', grad_fn=<SumBackward0>), tensor(6.4400, device='cuda:0', grad_fn=<SumBackward0>), tensor(6.4571, device='cuda:0', grad_fn=<SumBackward0>)]
                ID_LOSS = sum(ID_LOSS) / len(ID_LOSS) # tensor(6.4434, device='cuda:0', grad_fn=<DivBackward0>)
                ID_LOSS = 0.25 * ID_LOSS + 0.75 * xent(score[0], target) #tensor(6.4356, device='cuda:0', grad_fn=<AddBackward0>)  这里意思似乎是  0.25 后面成的是是4个part的平均的损失， 0.75后面是gloabl featue
        else:
                ID_LOSS = xent(score, target)

        if isinstance(feat, list): # 三元损失
                TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS) # tensor(20.4300, device='cuda:0', grad_fn=<DivBackward0>)
                TRI_LOSS = 0.25 * TRI_LOSS + 0.75 * triplet(feat[0], target)[0] # tensor(7.6633, device='cuda:0', grad_fn=<AddBackward0>) # 和上面的centre loss相同  也是 0.25 0.75比例

                center=center_criterion(feat[0], target)
                centr2 = [center_criterion2(feats, target) for feats in feat[1:]]
                centr2 = sum(centr2) / len(centr2)
                center=0.25 *centr2 +  0.75 *  center     # 2400+  同样 中心也选择的是 0.25 0.75比例
        else:
                TRI_LOSS = triplet(feat, target)[0]

        return   ID_LOSS+ TRI_LOSS, center # 三元损失+交叉熵损失，中心损失

    return  loss_func,center_criterion
                
    

