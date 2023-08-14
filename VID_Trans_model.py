
import torch
import torch.nn as nn
import copy
from vit_ID import TransReID,Block
from functools import partial
from torch.nn import functional as F

#明天重点看这个，是怎么混合特征的
def TCSS(features, shift, b,t):
    #aggregate features at patch level
    features=features.view(b,features.size(1),t*features.size(2)) # [128,129,768] -->[32,129,3072]
    token = features[:, 0:1] # [32,1,3072]

    batchsize = features.size(0) # 32
    dim = features.size(-1) # 3072
    
    
    #shift the patches with amount=shift
    features= torch.cat([features[:, shift:], features[:, 1:shift]], dim=1) # [32,129,3072] --> [32,128,3072]
    
    # Patch Shuffling by 2 part
    try:
        features = features.view(batchsize, 2, -1, dim) # [32,128,3072] --> [32,2,64,3072]
    except:
        features = torch.cat([features, features[:, -2:-1, :]], dim=1)
        features = features.view(batchsize, 2, -1, dim)
    
    features = torch.transpose(features, 1, 2).contiguous() # [32,2,64,3072] --> [32,64,2,3072] contiguous 方法返回一个在内存中连续存储的版本，这对某些操作是必要的
    features = features.view(batchsize, -1, dim) # [32,64,2,3072] --> [32,128,3072]
    
    return features,token    

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)




class VID_Trans(nn.Module):
    def __init__(self, num_classes, camera_num,pretrainpath):
        super(VID_Trans, self).__init__()
        self.in_planes = 768
        self.num_classes = num_classes
        
        
        self.base =TransReID(
        img_size=[256, 128], patch_size=16, stride_size=[16, 16], embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,\
        camera=camera_num,  drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0,norm_layer=partial(nn.LayerNorm, eps=1e-6),  cam_lambda=3.0)
        
          
        state_dict = torch.load(pretrainpath, map_location='cpu') # jx 这个是pretained vit，state_dict是一个字典，里面包括模型的各层的一些参数,map是加载的位置
        self.base.load_param(state_dict,load=True) #给模型加载这些参数
        
       
        #global stream
        block= self.base.blocks[-1] # 拿blocks的最后一个block
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        
        self.bottleneck = nn.BatchNorm1d(self.in_planes) #BatchNorm1d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottleneck.bias.requires_grad_(False) #在这个例子中，self.bottleneck 是一个 nn.BatchNorm1d 层，它包括一个偏置（bias）参数。通过 self.bottleneck.bias.requires_grad_(False)，作者将这个偏置参数的梯度计算设置为不需要进行梯度更新,但是weigth是需要更新的
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False) # Linear(in_features=768, out_features=625, bias=False)
        self.classifier.apply(weights_init_classifier)
       
        #-----------------------------------------------
        #-----------------------------------------------
        #假设你有一个全连接层的输出形状为 (batch_size, num_features)，其中 batch_size=8 和 num_features=512。对于 LayerNorm，你会对每个样本的 512 个特征进行标准化，而不是像 BatchNorm 那样对每个特征的 8 个样本值进行标准化。

        # building local video stream
        dpr = [x.item() for x in torch.linspace(0, 0, 12)]  # stochastic depth decay rule
        
        self.block1 = Block(
                dim=3072, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0, attn_drop=0, drop_path=dpr[11], norm_layer=partial(nn.LayerNorm, eps=1e-6)) #partial是偏函数，就是把nn.LayerNorm的参数eps=1e-6固定下来，这样就不用每次都写了
       
        self.b2 = nn.Sequential(
            self.block1,
            nn.LayerNorm(3072)#copy.deepcopy(layer_norm)
        )
        
        
        self.bottleneck_1 = nn.BatchNorm1d(3072) # BatchNorm1d(3072, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(3072)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(3072)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(3072)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)


        self.classifier_1 = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)


        #-------------------video attention-------------
        self.middle_dim = 256 # middle layer dimension
        self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim, [1,1]) # 7,4 cooresponds to 224, 112 input image size  Conv2d(768, 256, kernel_size=[1, 1], stride=(1, 1))
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        self.attention_conv.apply(weights_init_kaiming) 
        self.attention_tconv.apply(weights_init_kaiming) 
        #------------------------------------------
        
        self.shift_num = 5
        self.part = 4
        self.rearrange=True 
        



    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        b=x.size(0) # batch size 32
        t=x.size(1) # seq 4
        # 32 4 3 256 128  ---->  128 3 256 128
        x=x.view(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4))
        features = self.base(x, cam_label=cam_label) # 128 129 768
        
        
        # global branch
        b1_feat = self.b1(features) # [128, 129, 768]
        global_feat = b1_feat[:, 0] # [128, 768] 取第一个token
        
        global_feat=global_feat.unsqueeze(dim=2) # [128, 768, 1] tensor
        global_feat=global_feat.unsqueeze(dim=3) # [128, 768, 1, 1] tensor
        a = F.relu(self.attention_conv(global_feat)) # [128, 256, 1, 1] tensor
        a = a.view(b, t, self.middle_dim) # [32, 4, 256] tensor
        a = a.permute(0,2,1) # [32, 256, 4] tensor
        a = F.relu(self.attention_tconv(a)) # [32, 256, 4] ---> [32, 1, 4] tensor
        a = a.view(b, t) # [32, 1, 4] ---> [32, 4]
        a_vals = a 
        
        a = F.softmax(a, dim=1)
        x = global_feat.view(b, t, -1) # [128,3,256,128] ---> [32, 4, 768]
        a = torch.unsqueeze(a, -1) # [32, 4] ---> [32, 4, 1]
        a = a.expand(b, t, self.in_planes) # [32, 4, 1] ---> [32, 4, 768] 几层括号就是几维的tensor
        att_x = torch.mul(x,a) # [32, 4, 768]  element-wise multiplication
        att_x = torch.sum(att_x,1) # [32, 4, 768] ---> [32, 768]  沿着时间维度进行求和
        
        global_feat = att_x.view(b,self.in_planes) # [32, 768]
        feat = self.bottleneck(global_feat) #  [32, 768] BatchNorm1d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        



        #-------------------------------------------------
        #-------------------------------------------------


        # video patch patr features

        feature_length = features.size(1) - 1 # 129-1=128
        patch_length = feature_length // 4 # 128/4=32
        
        #Temporal clip shift and shuffled，混合完了  分出去4个头 然后各自有各自的loss
        x ,token=TCSS(features, self.shift_num, b,t) # [32,4,768] --> [32,128,3072]   [32,1,3072]
        
           
        # part1
        part1 = x[:, :patch_length] # [32, 32, 3072]
        part1 = self.b2(torch.cat((token, part1), dim=1))# [32, 32, 3072] ---> [32, 33, 3072]
        part1_f = part1[:, 0] # [32, 3072] 取第一个token

        # part2
        part2 = x[:, patch_length:patch_length*2]
        part2 = self.b2(torch.cat((token, part2), dim=1))
        part2_f = part2[:, 0]

        # part3
        part3 = x[:, patch_length*2:patch_length*3]
        part3 = self.b2(torch.cat((token, part3), dim=1))
        part3_f = part3[:, 0]

        # part4
        part4 = x[:, patch_length*3:patch_length*4]
        part4 = self.b2(torch.cat((token, part4), dim=1))
        part4_f = part4[:, 0]
       
        
        #4个头
        part1_bn = self.bottleneck_1(part1_f) # [32, 3072]
        part2_bn = self.bottleneck_2(part2_f)
        part3_bn = self.bottleneck_3(part3_f)
        part4_bn = self.bottleneck_4(part4_f)
        
        if self.training: # 要搞清楚 到底是一个batch 32意思是32个图片还是 32 个常委4的小视频
            
            Global_ID = self.classifier(feat) # [32, 768] ---> [32, 625]
            Local_ID1 = self.classifier_1(part1_bn)
            Local_ID2 = self.classifier_2(part2_bn)
            Local_ID3 = self.classifier_3(part3_bn)
            Local_ID4 = self.classifier_4(part4_bn)
                
            return [Global_ID, Local_ID1, Local_ID2, Local_ID3, Local_ID4 ], [global_feat, part1_f, part2_f, part3_f,part4_f],  a_vals #[global_feat, part1_f, part2_f, part3_f,part4_f],  a_vals 
        
        else:
              return torch.cat([feat, part1_bn/4 , part2_bn/4 , part3_bn /4, part4_bn/4 ], dim=1)
            


    def load_param(self, trained_path,load=False):
        if not load:
            param_dict = torch.load(trained_path)
            for i in param_dict:
               self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
               print('Loading pretrained model from {}'.format(trained_path))
        else:
            param_dict=trained_path
            for i in param_dict:
             #print(i)   
             if i not in self.state_dict() or 'classifier' in i or 'sie_embed' in i:
                continue
             self.state_dict()[i].copy_(param_dict[i])
           
            
            
    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))



           
           
