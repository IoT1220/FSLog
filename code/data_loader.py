from torch.utils.data import Dataset
import numpy as np
import torch
import os

# class CMCCDataLoader(Dataset):
#     def __init__(self,mode='train',semi=False) -> None:
#         super().__init__()
#         x = np.load("0704data/data.npy",allow_pickle=True)
#         y = np.load("0704data/label.npy",allow_pickle=True)
#         if semi:
#             y = np.load("0704data/semi-label-4x250.npy",allow_pickle=True)
#         x = torch.from_numpy(x).float()
#         y = torch.from_numpy(y)
#         _len = len(y)
#         if mode == 'train':
#             self.x = x[:int(_len*0.8)]
#             self.y = y[:int(_len*0.8)]
#         else:
#             self.x = x[int(_len*0.8):]
#             self.y = y[int(_len*0.8):]

#     def __len__(self):
#         return len(self.y)
    
#     def __getitem__(self, index):
#         return self.x[index],self.y[index]

class CMCCDataLoaderForFed(Dataset):
    def __init__(self,mode='train',semi=False,rank=1,world_size=3) -> None:
        super().__init__()
        x=np.load("/home/zhangshenglin/chezeyu/log/cmcc_0929/data/data_{}.npy".format(rank-1))
        if semi:
            y=np.load("/home/zhangshenglin/chezeyu/log/cmcc_0929/data/semi_label_{}.npy".format(rank-1))
        else:
            y=np.load("/home/zhangshenglin/chezeyu/log/cmcc_0929/data/label_{}.npy".format(rank-1))

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

        
# class CMCCDataLoaderForFed(Dataset):
#     def __init__(self,mode='train',semi=False,rank=1,world_size=3) -> None:
#         super().__init__()
#         x = np.load("0704data/data.npy",allow_pickle=True)
#         y = np.load("0704data/label.npy",allow_pickle=True)
#         if semi:
#             y = np.load("0704data/semi-label-4x250.npy",allow_pickle=True)
#         x = torch.from_numpy(x).float()
#         y = torch.from_numpy(y)
#         _len = len(y)
#         if mode == 'train':
#             x = x[:int(_len*0.8)]
#             y = y[:int(_len*0.8)]
#         else:
#             x = x[int(_len*0.8):]
#             y = y[int(_len*0.8):]
#         seq = len(y)//(world_size-1)
#         self.x = x[(rank-1)*seq:rank*seq]
#         self.y = y[(rank-1)*seq:rank*seq]

#     def __len__(self):
#         return len(self.y)
    
#     def __getitem__(self, index):
#         return self.x[index],self.y[index]
    
# class AliyunDataLoaderForFed(Dataset):
#     def __init__(self,mode='train',semi=False,rank=1,world_size=3) -> None:
#         super().__init__()
#         x = np.load("data/aliyun/data.npy",allow_pickle=True)
#         y = np.load("data/aliyun/label.npy",allow_pickle=True)
#         if semi:
#             y = np.load("data/aliyun/semi-label-4x250.npy",allow_pickle=True)
#         x = torch.from_numpy(x).float()
#         y = torch.from_numpy(y)
#         _len = len(y)
#         if mode == 'train':
#             x = x[:int(_len*0.8)]
#             y = y[:int(_len*0.8)]
#         else:
#             x = x[int(_len*0.8):]
#             y = y[int(_len*0.8):]
#         seq = len(y)//(world_size-1)
#         self.x = x[(rank-1)*seq:rank*seq]
#         self.y = y[(rank-1)*seq:rank*seq]

#     def __len__(self):
#         return len(self.y)
    
#     def __getitem__(self, index):
#         return self.x[index],self.y[index]
    



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
        x=np.load("/home/zhangshenglin/chezeyu/log/aliyun0823/data/data_{}.npy".format(rank-1))
        if semi:
            y=np.load("/home/zhangshenglin/chezeyu/log/aliyun0823/data/semi_label_{}.npy".format(rank-1))
        else:
            y=np.load("/home/zhangshenglin/chezeyu/log/aliyun0823/data/label_{}.npy".format(rank-1))

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
