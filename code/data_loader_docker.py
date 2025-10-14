from torch.utils.data import Dataset
import numpy as np
import torch
import os



class CMCCDataLoaderForFed(Dataset):
    def __init__(self,mode='train',semi=False,rank=1,world_size=3) -> None:
        super().__init__()
        x=np.load("./data/cmcc_0930/data_{}.npy".format(rank-1))
        if semi:
            y=np.load("./data/cmcc_0930/semi_label_{}.npy".format(rank-1))
        else:
            y=np.load("./data/cmcc_0930/label_{}.npy".format(rank-1))

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
    def __init__(self,mode='train',semi=False,rank=1,world_size=3) -> None:
        super().__init__()
        # data_dir = "/home/zhangshenglin/chezeyu/log/aliyun0823/result_0823"
        # file_list = os.listdir(data_dir)
        # num_of_file = len(file_list)
        # seq = num_of_file // (world_size-1)
        # file_list = file_list[(rank-1)*seq:rank*seq]
        # data_list = []
        # label_list = []
        # for file in file_list:
        #     data = np.load(os.path.join(data_dir,file,"data.npy"))
        #     label = np.load(os.path.join(data_dir,file,"label.npy"))
        #     if len(label)==0:
        #         continue
        #     data_list.append(data)
        #     label_list.append(label)
        # x = np.concatenate(data_list)
        # y = np.concatenate(label_list)
        x=np.load("./data/aliyun_0930/data_{}.npy".format(rank-1))
        if semi:
            y=np.load("./data/aliyun_0930/semi_label_{}.npy".format(rank-1))
        else:
            y=np.load("./data/aliyun_0930/label_{}.npy".format(rank-1))

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
