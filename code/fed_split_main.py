from data_loader import CMCCDataLoaderForFed,AliyunDataLoaderForFed
from transformer_fed_bert import Embedding, TransformerEncoding, Head
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import copy
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
import argparse
from thop import profile 
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--classes', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--global_epochs', type=int, default=10)
parser.add_argument('--local_epochs', type=int, default=10)
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--embed_dim', type=int, default=768)
parser.add_argument('--world_size', type=int, default=6)
parser.add_argument('--out_dir', type=str, default='0827')
parser.add_argument('--datatype', type=str, default='aliyun')
parser.add_argument('--is_train', type=bool, default=True)
parser.add_argument('--is_semi', type=int, default=1)
parser.add_argument('--is_ema', type=int, default=1)

args = parser.parse_args()

seed = args.seed
classes = args.classes
batch_size = args.batch_size
learning_rate = args.learning_rate
global_epochs = args.global_epochs
local_epochs = args.local_epochs
device = torch.device('cuda:{}'.format(args.gpu_id))
embed_dim = args.embed_dim
world_size = args.world_size
out_path = args.out_dir
datatype = args.datatype
is_train = args.is_train
is_semi = True if args.is_semi==1 else False
is_ema = True if args.is_ema==1 else False

out_dir = f"Output/{datatype}_logs_{out_path}_{global_epochs}e_{local_epochs}locEpoch_{learning_rate}lr_{batch_size}bs_{world_size}ws_{is_semi}semi_{is_ema}EAM"
save_dir = os.path.join(out_dir,"model_save")


def torch_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def load_datasets_cmcc(rank,world_size):
    train_db = CMCCDataLoaderForFed(mode='train',semi=is_semi,rank=rank,world_size=world_size)
    test_db = CMCCDataLoaderForFed(mode='test',semi=is_semi,rank=rank,world_size=world_size)
    train_loader = DataLoader(train_db,batch_size=batch_size,shuffle=True,drop_last=True) 
    test_loader = DataLoader(test_db,batch_size=batch_size)
    val_loader = test_loader
    return train_loader,val_loader,test_loader

def load_datasets_aliyun(rank,world_size):
    train_db = AliyunDataLoaderForFed(mode='train',semi=is_semi,rank=rank,world_size=world_size)
    test_db = AliyunDataLoaderForFed(mode='test',semi=is_semi,rank=rank,world_size=world_size)
    print(rank,len(train_db),len(test_db))
    train_loader = DataLoader(train_db,batch_size=batch_size,shuffle=True,drop_last=True) 
    test_loader = DataLoader(test_db,batch_size=batch_size)
    val_loader = test_loader
    return train_loader,val_loader,test_loader

def unlabeled_weight(epoch,T1=2*2,T2=4*2,af=0.3):
    alpha = 0.0
    if epoch > T1:
        alpha = (epoch-T1) / (T2-T1)*af
        if epoch > T2:
            alpha = af
    return alpha

# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def pu(orgin_model,new_model):
    similarities = []
    for a,b in zip(orgin_model.items(),new_model.items()):
       a,b=a[1],b[1]
        # v = torch.var(torch.vstack([a,b]),axis=0)
       similarities.append(torch.sqrt(torch.sum((a-b)**2)))
    similarities = np.array(similarities)
    norm_similarities = (similarities-similarities.min())/(similarities.max()-similarities.min())
    u = np.sum(norm_similarities)/len(norm_similarities)
    model = copy.deepcopy(orgin_model)
    for key in model.keys():
        model[key] = u*orgin_model[key]+(1-u)*new_model[key]
    return model
    
def cal_f1(label_list, pred_list,fw=None):
    label_arr = np.array(label_list)
    pred_arr = np.array(pred_list)
    # 异常检测
    ad_label = np.where(label_arr>0,1,0)
    ad_pred = np.where(pred_arr>0,1,0)
    print("异常检测结果:")
    print(classification_report(ad_label,ad_pred))
    if fw:
        fw.write("异常检测结果:\n")
        fw.write(classification_report(ad_label,ad_pred))
        fw.write('\n\n')
    fault_index = (label_arr!=0)&(pred_arr!=0)
    fd_label = label_arr[fault_index]
    fd_pred = pred_arr[fault_index]
    print("故障诊断结果:")
    print(classification_report(fd_label,fd_pred))
    if fw:
        fw.write("故障诊断结果:\n")
        fw.write(classification_report(fd_label,fd_pred))
        fw.write('\n\n')

if __name__=='__main__':
    torch_seed(seed)
    if datatype == 'aliyun':
        loader_list = [load_datasets_aliyun(i,world_size) for i in range(1,world_size)]
    else:
        loader_list = [load_datasets_cmcc(i,world_size) for i in range(1,world_size)]

    embedding_model = Embedding(1024,embed_dim=embed_dim,device=device).to(device)
    client_list = [Head(embed_dim=embed_dim,classes=classes).to(device) for i in range(1,world_size)]
    server_list = [TransformerEncoding(2,embed_dim=embed_dim,num_heads=12,ff_dim=1024).to(device) for i in range(1,world_size)]
    # optimizer_client_list = [optim.Adam(client_list[i].parameters(), lr=learning_rate) for i in range(world_size-1)]
    # optimizer_server_list = [optim.Adam(server_list[i].parameters(), lr = learning_rate) for i in range(world_size-1)]
    best_score_list = [0 for _ in range(world_size-1)]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fw_list = [open("./{}/log{}.txt".format(out_dir, i),'w') for i in range(0,world_size-1)]
    criteon = nn.CrossEntropyLoss().to(device)

    if is_train:
        for i,model in enumerate(client_list):
            torch.save(model.state_dict(),os.path.join(save_dir,f"head_param{i}.pkl"))
        for i,model in enumerate(server_list):
            torch.save(model.state_dict(),os.path.join(save_dir,f"server_param{i}.pkl"))

        
        for it in range(global_epochs):
            print('*'*5,it,'*'*5)
            #train
            for i in range(world_size-1):
                flops, params = {}, {}
                client_model = client_list[i]
                server_model = server_list[i]
                optimizer_client = optim.Adam(client_list[i].parameters(), lr=learning_rate)
                optimizer_server = optim.Adam(server_list[i].parameters(), lr = learning_rate)
                client_model.train()
                server_model.train()
                train_loader = loader_list[i][0]
                for e in range(local_epochs):
                    print('server',i ,'train epoch:', e)
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        # print(data.dtype,inputs.dtype)
                        # logits = model(data)

                        out_embedding = embedding_model(data)
                        out_tfencoding = server_model(out_embedding)
                        out_tfencoding = torch.from_numpy(out_tfencoding.cpu().detach().numpy())
                        out_tfencoding_fx = out_tfencoding.requires_grad_(True)
                        logits = client_model(out_tfencoding.to(device))

                        # semi learning
                        labeled_targets = target[target!=-1]
                        unlabeled_targets = logits.max(1)[1][target==-1]
                        labeled_logits = logits[target!=-1]
                        unlabeled_logits = logits[target==-1]

                        labeled_loss = criteon(labeled_logits,labeled_targets)
                        unlabeled_loss = criteon(unlabeled_logits,unlabeled_targets)
                        
                        loss = torch.tensor(0).float().to(device)
                        if not labeled_loss.isnan():
                            loss += labeled_loss
                        if not unlabeled_loss.isnan():
                            loss += unlabeled_weight(e)*unlabeled_loss
                        # loss = labeled_loss + unlabeled_weight(e,T1=9,T2=14)*unlabeled_loss
                        # loss = criteon(logits, target)
                        optimizer_client.zero_grad()
                        optimizer_server.zero_grad()
                        loss.backward()
                        dfx_server = out_tfencoding.grad.clone().detach()
                        optimizer_server.step()
                        out_tfencoding.backward(dfx_server)
                        optimizer_client.step()

                        if len(list(flops.keys())) == 0:
                            fl, pa = profile(copy.deepcopy(embedding_model), inputs=(data, ))
                            flops['embedding'] = fl
                            params['embedding'] = pa
                            fl, pa = profile(copy.deepcopy(server_model), inputs=(out_embedding, ))
                            flops['encoder'] = fl
                            params['encoder'] = pa
                            fl, pa = profile(copy.deepcopy(client_model), inputs=(out_tfencoding.to(device),))
                            flops['head'] = fl
                            params['head'] = pa
                            params['out_embedding'] = torch.numel(out_embedding)
                            params['out_encoding'] = torch.numel(out_tfencoding)

            client_list = [client.cpu() for client in client_list]
            server_list = [server.cpu() for server in server_list]
            client_model = FedAvg(list(map(lambda x:x.state_dict(),client_list)))
            server_model = FedAvg(list(map(lambda x:x.state_dict(),server_list)))
            if is_ema:
                for i in range(world_size-1):
                    client_list[i].load_state_dict(pu(client_list[i].state_dict(),client_model))
                    server_list[i].load_state_dict(pu(server_list[i].state_dict(),server_model))
            else:
                for i in range(world_size-1):
                    client_list[i].load_state_dict(client_list[i].state_dict())
                    server_list[i].load_state_dict(server_list[i].state_dict())
            client_list = [client.to(device) for client in client_list]
            server_list = [server.to(device) for server in server_list]
                   
            for i in range(world_size-1):
                client_model = client_list[i]
                server_model = server_list[i]
                client_model.eval()
                server_model.eval()
                val_loader = loader_list[i][1]
                #valid
                test_loss = 0
                y_true = []
                y_pred = []
                for data, target in val_loader:
                    y_true.extend(target)
                    data, target = data.to(device), target.to(device)
                    out_embedding = embedding_model(data)
                    out_tfencoding = server_model(out_embedding)
                    logits = client_model(out_tfencoding)
                    # logits = model(data)
                    test_loss += criteon(logits, target).item()

                    pred = logits.data.topk(1)[1].flatten().cpu()
                    y_pred.extend(pred)

                test_loss /= len(val_loader.dataset)

                # F1_Score = f1_score(y_true, y_pred)
                acc = accuracy_score(y_true, y_pred)

                print('\n{},VALID set: Average loss: {:.4f},score:{}\n'.format(
                    i,test_loss,round(acc.item(),4)))
                
                print('{}, Client Counting flops: {:.4f}GFlops, Server Counting flops: {:.4f}GFlops, Transmitted params: {:.4f}G\n'.format(
                    it, (flops['embedding'] + flops['head']) * 3 * local_epochs * batch_idx * (it + 1) / 10e9, flops['encoder'] * 3 * local_epochs * batch_idx * (it + 1) / 10e9, \
                        (params['embedding'] + params['out_embedding'] * local_epochs * batch_idx * batch_size + \
                            params['out_encoding'] * local_epochs * batch_idx * batch_size) * (it + 1) / 10e9))
                
                fw_list[i].write('\n{},VALID set: Average loss: {:.4f},score:{}\n'.format(
                    it,test_loss,round(acc.item(),4)))
                fw_list[i].write('{}, Client Counting flops: {:.4f}GFlops, Server Counting flops: {:.4f}GFlops, Transmitted params: {:.4f}G\n'.format(
                    it, (flops['embedding'] + flops['head']) * 3 * local_epochs * batch_idx * (it + 1) / 10e9, flops['encoder'] * 3 * local_epochs * batch_idx * (it + 1) / 10e9, \
                        (params['embedding'] + params['out_embedding'] * local_epochs * batch_idx * batch_size + \
                            params['out_encoding'] * local_epochs * batch_idx * batch_size) * (it + 1) / 10e9))
              
                if acc>best_score_list[i]:
                    best_score_list[i] = acc
                    torch.save(client_model.state_dict(),os.path.join(save_dir,f"head_param{i}.pkl"))
                    torch.save(server_model.state_dict(),os.path.join(save_dir,f"server_param{i}.pkl"))
    for i in range(world_size-1):
        print("local server",i)
        client_model.load_state_dict(torch.load(os.path.join(save_dir,f"head_param{i}.pkl")))
        server_model.load_state_dict(torch.load(os.path.join(save_dir,f"server_param{i}.pkl")))
        client_model.eval()
        server_model.eval()
        test_loader = loader_list[i][2]
        pred_list = []
        label_list = []

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out_embedding = embedding_model(data)
            out_tfencoding = server_model(out_embedding)
            logits = client_model(out_tfencoding)
            pred = logits.data.topk(1)[1].flatten()
            pred_list.extend(list(pred.cpu()))
            label_list.extend(list(target.cpu()))
        cal_f1(label_list, pred_list,fw_list[i])

    for f in fw_list:
        f.close()
