from torch.utils.data import Dataset
import numpy as np
import torch
import os

class CMCCDataLoaderForFed(Dataset):
    def __init__(self,mode='train',semi=False,rank=1,world_size=3,window_size=20) -> None:
        super().__init__()
        x=np.load("./data/cmcc_winSize_{}/data_{}.npy".format(window_size,rank-1))

        if semi:
            y=np.load("./data/cmcc_winSize_{}/semi_label_{}.npy".format(window_size,rank-1))
        else:
            y=np.load("./data/cmcc_winSize_{}/label_{}.npy".format(window_size,rank-1))

        _len = len(y)
        if mode == 'train':
            self.x = x[:int(_len*0.8)]
            self.y = y[:int(_len*0.8)]
        else:
            self.x = x[int(_len*0.8):]
            self.y = y[int(_len*0.8):]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index],self.y[index]

class AliyunDataLoaderForFed(Dataset):
    def __init__(self,mode='train',semi=False,rank=1,world_size=3,window_size=20) -> None:
        super().__init__()
        x=np.load("/home/chenzhimin/czm/FedSplitLog/FedSplit/data/aliyun_winSize{}/data_{}.npy".format(window_size,rank-1))
        if semi:
            y=np.load("/home/chenzhimin/czm/FedSplitLog/FedSplit/data/aliyun_winSize{}/semi_label_{}.npy".format(window_size,rank-1))
        else:
            y=np.load("/home/chenzhimin/czm/FedSplitLog/FedSplit/data/aliyun_winSize{}/label_{}.npy".format(window_size,rank-1))

        _len = len(y)
        if mode == 'train':
            self.x = x[:int(_len*0.8)]
            self.y = y[:int(_len*0.8)]
        else:
            self.x = x[int(_len*0.8):]
            self.y = y[int(_len*0.8):]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index],self.y[index]
