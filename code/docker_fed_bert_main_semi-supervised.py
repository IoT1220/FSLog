from data_loader_docker import CMCCDataLoaderForFed,AliyunDataLoaderForFed
from transformer_fed_bert import Embedding, TransformerEncoding, Head
import torch
import  torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import copy
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
import argparse

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

out_dir = f"Output/docker_{datatype}_logs_{out_path}_{global_epochs}e_{local_epochs}locEpoch_{learning_rate}lr_{batch_size}bs_{world_size}ws_{is_semi}semi_{is_ema}EAM"
save_dir = os.path.join(out_dir,"model_save")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(os.path.join(save_dir,"client")):
    os.mkdir(os.path.join(save_dir,"client"))



global_epochs=2
local_epochs=2
max_len=20


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
    
def torch_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def unlabeled_weight(epoch,T1=2,T2=4,af=0.3):
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
def cal_f1(label_list, pred_list):
    label_arr = np.array(label_list)
    pred_arr = np.array(pred_list)
    # 异常检测
    ad_label = np.where(label_arr>0,1,0)
    ad_pred = np.where(pred_arr>0,1,0)
    print("异常检测结果:")
    print(classification_report(ad_label,ad_pred))
    fault_index = (label_arr!=0)&(pred_arr!=0)
    fd_label = label_arr[fault_index]
    fd_pred = pred_arr[fault_index]
    print("故障诊断结果:")
    print(classification_report(fd_label,fd_pred))

def main():
    # 初始化分布式训练环境
    dist.init_process_group(backend='gloo', init_method='env://')
    
    # 获取当前进程的rank和总进程数
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch_seed(2020+rank)
    print(world_size,rank,type(world_size),type(rank))
    log_f = open("./{}/log{}.txt".format(out_dir, rank),'w')
    if rank!=0:
        if datatype == 'aliyun':
            train_loader,val_loader,test_loader = load_datasets_aliyun(rank,world_size)
        else:
            train_loader,val_loader,test_loader = load_datasets_cmcc(rank,world_size)
    device = torch.device('cuda:0')
    cpu = torch.device('cpu')
    device = torch.device('cpu')

    # return
    if rank !=0 :
        embedding_model = Embedding(1024,embed_dim=embed_dim,device=device).to(device)
        head_model = Head(embed_dim=embed_dim,classes=classes).to(device)
    else:
        head_model = Head(embed_dim=embed_dim,classes=classes).to(device)
        net_glob_server = TransformerEncoding(2,embed_dim=embed_dim,num_heads=12,ff_dim=1024).to(device)
        w_glob_server = net_glob_server.state_dict()
        net_model_server = [net_glob_server for i in range(world_size)]
        net_server = copy.deepcopy(net_model_server[0]).to(device)

    criterion = nn.CrossEntropyLoss()

    best_acc = -1

    for iter in range(global_epochs):  #交互
        print('*'*5,iter,'*'*5)
        if rank !=0:
            # client train
            optimizer_client = torch.optim.Adam(head_model.parameters(), lr = learning_rate) 
            head_model.train()
            for epoch in range(local_epochs):
                print('train epoch:',epoch)
                for bacth_idx, (x,y) in enumerate(train_loader):
                    # if bacth_idx % 10 == 0:
                    #     print(bacth_idx,'/',len(train_loader))
                    # if bacth_idx == len(train_loader)-1:
                    #     break
                    x,y = x.to(device),y.to(device)
                    # x = x.to(device)
                    # x=x.unsqueeze(-1)
                    
                    #embedding
                    out_embedding = embedding_model(x)
                    # print("out_embedding.shape",out_embedding.shape)
                    dist.send(tensor=out_embedding.to(cpu),dst=0) #docker
                    
                    out_tfencoding=torch.ones_like(out_embedding).to(cpu)
                    dist.recv(out_tfencoding,0)
                    # out_tfencoding=out_tfencoding.to(device)
                    feature_vec = out_tfencoding.numpy().reshape(-1,max_len*embed_dim)
                    out_tfencoding.requires_grad_(True)
                    output_pred = head_model(out_tfencoding.to(device))
                    # print(feature_vec.shape)
                    # y=model.fit_predict(feature_vec)
                    # y=torch.from_numpy(y)

                    # semi learning
                    labeled_targets =  y[y!=-1]
                    unlabeled_targets = output_pred.max(1)[1][y==-1]
                    labeled_logits = output_pred[y!=-1]
                    unlabeled_logits = output_pred[y==-1]

                    labeled_loss = criterion(labeled_logits,labeled_targets)
                    unlabeled_loss = criterion(unlabeled_logits,unlabeled_targets)

                    loss = labeled_loss + unlabeled_weight(epoch)*unlabeled_loss
                    # loss = criterion(output_pred, y)

                    #backward
                    optimizer_client.zero_grad()
                    loss.backward()
                    dfx = out_tfencoding.grad.clone().detach().to(device)
                    optimizer_client.step()

                    dist.send(dfx.to(cpu),0)

                    flag = torch.tensor([epoch,local_epochs-1,bacth_idx,len(train_loader)-1]).to(device)
                    dist.send(flag.to(cpu),0)

                    print(bacth_idx,'/',len(train_loader),'loss:',loss)
                log_f.write(f"[iter:{iter}]/[epoch:{epoch}]-loss:{loss}\n")

        else:
            finished_ranks = []

            while len(finished_ranks) != world_size-1:

                out_embedding = torch.ones(batch_size,max_len,embed_dim).float().to(cpu)
                dfx = torch.ones(batch_size,max_len,embed_dim).float().to(cpu)
                flag = torch.ones(4).long().to(cpu)
                # print("out_embedding.shape",out_embedding.shape)
                # print("dfx.shape",dfx.shape)
                src = dist.recv(out_embedding)
                print(src)
                net_server = net_model_server[src].to(device)
                net_server.train()
                optimizer_server = torch.optim.Adam(net_server.parameters(), lr = learning_rate)
                optimizer_server.zero_grad()
                out_embedding=out_embedding.to(device)
                out_tfencoding = net_server(out_embedding)
                dist.send(out_tfencoding.to(cpu),src)
                dist.recv(dfx,src)
                dist.recv(flag,src)
                out_tfencoding.backward(dfx.to(device))
                optimizer_server.step()
                if flag[0]==flag[1] and flag[2]==flag[3]:
                    finished_ranks.append(src)

        dist.barrier()

        w_client_gather = [None]*world_size
        dist.gather_object(head_model.state_dict(),w_client_gather if rank==0 else None, dst=0)
        if rank==0:
            w_server = []
            for net in net_model_server[1:]:
                w_server.append(net.state_dict())
            w_glob_server = FedAvg(w_server) 
            for i in range(1,world_size):
                 net_model_server[i].load_state_dict(pu(w_server[i-1],w_glob_server))
            net_glob_server.load_state_dict(w_glob_server)
            # net_model_server = [net_glob_server for _ in range(world_size)]
            w_client = []
            for net in w_client_gather[1:]:
                w_client.append(net)
            w_glob_client = FedAvg(w_client)
            
            head_model.load_state_dict(w_glob_client)
            model_list = [head_model for _ in range(world_size)]
            for i in range(1,world_size):
                model_list[i].load_state_dict(pu(w_client[i-1],w_glob_client))
        else:
            model_list=[None]*world_size

        output_list=[None]
        dist.scatter_object_list(output_list,model_list,src=0)  
        if rank!=0:
            head_model = output_list[0]
        
        if rank !=0 :
            model_list=[None]*world_size
        else:
            model_list=copy.deepcopy(net_model_server)
        output_list=[None]
        dist.scatter_object_list(output_list,model_list,src=0)  
        if rank!=0:
            y_true = []
            y_pred = []
            transformer_encoding = output_list[0]
            #val
            head_model.eval()
            transformer_encoding.eval()
            with torch.no_grad():
                for bacth_idx, (x,y) in enumerate(val_loader):
                    y_true.extend(y)
                    x,y = x.to(device),y.to(device)
                    out_embedding = embedding_model(x)
                    out_tfencoding = transformer_encoding(out_embedding)
                    output_pred = head_model(out_tfencoding)

                    pred = output_pred.data.topk(1)[1].flatten().cpu()
                    y_pred.extend(pred)

            val_acc = accuracy_score(y_true, y_pred)
            print('val acc:',val_acc)
            if best_acc < val_acc:
                best_acc = val_acc
                torch.save(transformer_encoding.state_dict(),f"{save_dir}/client/transformer_encoding_{rank}.pkl")
                
                torch.save(head_model.state_dict(),f"{save_dir}/client/head_model_{rank}.pkl") 


    # 评估
    if rank!=0:
        transformer_encoding.load_state_dict(torch.load(f"{save_dir}/client/transformer_encoding_{rank}.pkl"))
        head_model.load_state_dict(torch.load(f"{save_dir}/client/head_model_{rank}.pkl"))
        head_model.eval()
        transformer_encoding.eval()

        y_test,y_predict=[],[]
        with torch.no_grad():
            for bacth_idx, (x,y) in enumerate(test_loader):

                x,y = x.to(device),y.to(device)
                # x=x.unsqueeze(-1)
                # x = x.to(device)

                out_embedding = embedding_model(x)
                out_tfencoding = transformer_encoding(out_embedding)
                output_pred = head_model(out_tfencoding)

                out_tfencoding = out_tfencoding.numpy().reshape(-1,max_len*embed_dim)
                # print(out_tfencoding.shape)
                # y=model.fit_predict(out_tfencoding)
                # y=torch.from_numpy(y)
                # print(output_pred.shape,y.shape)
                y_test.extend(y.tolist())
                y_predict.extend(output_pred.max(1, keepdim=True)[1].squeeze(axis=-1).tolist())
        
        # print(classification_report(y_test, y_predict))
        cal_f1(y_test, y_predict)
        
    print('ok!')

if __name__=='__main__':
    main()
