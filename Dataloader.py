import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import numpy as np
import math

from timm.data.random_erasing import RandomErasing
from utility import RandomIdentitySampler,RandomErasing3
from Datasets.MARS_dataset import Mars
from Datasets.iLDSVID import iLIDSVID
from Datasets.PRID_dataset import PRID

import os

__factory = {
    'Mars':Mars,
    'iLIDSVID':iLIDSVID,
    'PRID':PRID
}

def train_collate_fn(batch): #batch 是一个64的list，每个元素是 是一个含有4个元素的tuple，比如 0 是 一个 4 3 256 128 的tensor 1是78 2 是一个 list [4,4,4,4],3 是一个tensor tensor([1, 1, 1, 1])
    
    imgs, pids, camids,a= zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, torch.stack(a, dim=0) # imgs tuple64  tensor 4,3,256,128   torch.stack(imgs, dim=0): tensor 64,4,3,256,128

# def val_collate_fn(batch):
#
#     imgs, pids, camids, img_paths = zip(*batch)
#     viewids = torch.tensor(viewids, dtype=torch.int64)
#     camids_batch = torch.tensor(camids, dtype=torch.int64)
#     return torch.stack(imgs, dim=0), pids, camids_batch,  img_paths

def dataloader(Dataset_name):
    train_transforms = T.Compose([
            T.Resize([256, 128], interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            
            
        ]) #义了一个数据增强的转换序列 train_transforms，用于在训练数据上进行预处理操作。这些操作有助于提高模型的泛化能力和性能，同时也可以增加数据的多样性。
    #T.Compose 是一个转化，不用管里面什么结构 就是已定义了这个转化 然后用的时候
    # image = PIL.Image.open('image.jpg')
    # transformed_image = transform(image)

    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    

    dataset = __factory[Dataset_name]() # 这里指向mars数据集，看mars类中是怎么构造出这个数据集的 再这里,获得了datasets类的实例，这个实例就是mars数据集，里面有train，query，gallery等等
    train_set = VideoDataset_inderase(dataset.train, seq_len=4, sample='intelligent',transform=train_transforms) #可以认为是把mars数据集中的train部分拿出来，然后用VideoDataset类进行处理，这里是RandomErasing3， 得到train_set
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

   # 这里该bs
    train_loader = DataLoader(train_set, batch_size=32,sampler=RandomIdentitySampler(dataset.train, 32,4),num_workers=4, collate_fn=train_collate_fn) #这里定义了bs 这段代码使用了 PyTorch 中的 DataLoader 类，用于构建一个用于训练的数据加载器。DataLoader 提供了一种简便的方式来加载和处理训练数据，它可以在训练过程中自动进行批量化、随机化等操作。
  
    q_val_set = VideoDataset(dataset.query, seq_len=4, sample='dense', transform=val_transforms)
    g_val_set = VideoDataset(dataset.gallery, seq_len=4, sample='dense', transform=val_transforms)
    
    
    return train_loader, len(dataset.query), num_classes, cam_num, view_num,q_val_set,g_val_set



# def read_image(img_path):
#     """Keep reading image until succeed.
#     This can avoid IOError incurred by heavy IO process."""
#     got_img = False
#     while not got_img:
#         try:
#             img = Image.open(img_path).convert('RGB')
#             got_img = True
#         except IOError:
#             print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
#             pass
#     return img
def read_image(img_path):
    got_img = False
    while not got_img:
        try:
            real_img_path = os.path.realpath(img_path)
            img = Image.open(real_img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            return None
    return img

class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)



        # if self.sample == 'restricted_random':
        #     frame_indices = range(num)
        #     chunks = 
        #     rand_end = max(0, len(frame_indices) - self.seq_len - 1)
        #     begin_index = random.randint(0, rand_end)


        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            # print(begin_index, end_index, indices)
            if len(indices) < self.seq_len:
                indices=np.array(indices)
                indices = np.append(indices , [indices[-1] for i in range(self.seq_len - len(indices))])
            else:
                indices=np.array(indices)
            imgs = []
            targt_cam=[]
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                targt_cam.append(camid)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, targt_cam

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            # import pdb
            # pdb.set_trace()
        
            cur_index=0
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            # print(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            

            indices_list.append(last_seq)
            imgs_list=[]
            targt_cam=[]
            # print(indices_list , num , img_paths  )
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                    targt_cam.append(camid)
                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, targt_cam,img_paths
            #return imgs_array, pid, int(camid),trackid


        elif self.sample == 'dense_subset':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.max_length - 1)
            begin_index = random.randint(0, rand_end)
            

            cur_index=begin_index
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            # print(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            

            indices_list.append(last_seq)
            imgs_list=[]
            # print(indices_list , num , img_paths  )
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid
        
        elif self.sample == 'intelligent_random':
            # frame_indices = range(num)
            indices = []
            each = max(num//self.seq_len,1)
            for  i in range(self.seq_len):
                if i != self.seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            print(len(indices))
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

            

        
class VideoDataset_inderase(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length
        self.erase = RandomErasing3(probability=0.5, mean=[0.485, 0.456, 0.406])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index): #__getitem__ 方法通过使对象支持索引操作，使得我们可以按照序列的方式访问自定义类的元素,__是双下划线（__）开头和结尾的函数在Python中被称为"魔法方法"（magic methods）或"特殊方法"（special methods）。这些方法是Python类中的特殊函数，用于定义类的行为，例如初始化、比较、运算符重载等。这些魔法方法会在特定的上下文中被自动调用，而不需要直接调用它们。
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        if self.sample != "intelligent":
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices1 = frame_indices[begin_index:end_index]
            indices = []
            for index in indices1:
                if len(indices1) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
        else:
            # frame_indices = range(num)
            indices = []
            each = max(num//self.seq_len,1)
            for  i in range(self.seq_len):
                if i != self.seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            # print(len(indices), indices, num )
        imgs = []
        labels = []
        targt_cam=[]
        
        for index in indices:
            index=int(index)
            img_path = img_paths[index]
            
            img = read_image(img_path)

            if img is None:
                # Skip this image and continue with the next one
                continue


            if self.transform is not None:
                img = self.transform(img)
            img , temp  = self.erase(img)
            labels.append(temp)
            img = img.unsqueeze(0)
            imgs.append(img)
            targt_cam.append(camid)
        labels = torch.tensor(labels)
        imgs = torch.cat(imgs, dim=0)
        
        return imgs, pid, targt_cam ,labels
        

