from data_loader import CMCCDataLoaderForFed,AliyunDataLoaderForFed
from transformer_fed_bert import Embedding, TransformerEncoding, Head
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import copy
import numpy as np
from sklearn.metrics import classification_report,accuracy_score,f1_score
import argparse
from thop import profile 
import time

from utils.login import *

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--classes', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--global_epochs', type=int, default=10)
parser.add_argument('--local_epochs', type=int, default=10)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--embed_dim', type=int, default=768)
parser.add_argument('--world_size', type=int, default=6)
parser.add_argument('--out_dir', type=str, default='0421')
parser.add_argument('--datatype', type=str, default='aliyun')
parser.add_argument('--is_train', type=bool, default=True)
parser.add_argument('--is_semi', type=int, default=1)
parser.add_argument('--is_ema', type=int, default=1)

parser.add_argument('--ema_delta_server', '--delta1', dest='ema_delta_server', type=float, default=0.9)
parser.add_argument('--ema_delta_client', '--delta2', dest='ema_delta_client', type=float, default=0.9)
parser.add_argument('--dim_num', type=int, default=1024)
parser.add_argument('--layer_num', type=int, default=2)
parser.add_argument('--split_k', type=int, default=0)
parser.add_argument('--window_size', type=int, default=20)
parser.add_argument('--enable_privacy_chain', type=int, default=1)
parser.add_argument('--privacy_dim', type=int, default=128)


args = parser.parse_args()

seed = args.seed
classes = args.classes
batch_size = args.batch_size
learning_rate = args.learning_rate
global_epochs = args.global_epochs
local_epochs = args.local_epochs
device = torch.device('cuda:{}'.format(args.gpu_id)) # device = torch.device("cpu")
embed_dim = args.embed_dim
world_size = args.world_size
out_path = args.out_dir
datatype = args.datatype
is_train = args.is_train
is_semi = True if args.is_semi==1 else False
is_ema = True if args.is_ema==1 else False
ema_delta_server = args.ema_delta_server
ema_delta_client = args.ema_delta_client
assert 0.0 <= ema_delta_server <= 1.0, f"ema_delta_server/delta1 must be in [0, 1], got {ema_delta_server}"
assert 0.0 <= ema_delta_client <= 1.0, f"ema_delta_client/delta2 must be in [0, 1], got {ema_delta_client}"
dim_num = args.dim_num
layer_num = args.layer_num
window_size = args.window_size
split_k = args.split_k
enable_privacy_chain = True if args.enable_privacy_chain == 1 else False
privacy_dim = args.privacy_dim

out_dir = f"Output/{out_path}/seed{seed}_{datatype}_logs_{out_path}_{global_epochs}e_{local_epochs}locEpoch_{learning_rate}lr_{batch_size}bs_{world_size}ws_{is_semi}semi_{is_ema}EMA_paper_dT{ema_delta_server}_dH{ema_delta_client}_{dim_num}dim_{layer_num}layer_{split_k}splitK_{window_size}window_privacy{int(enable_privacy_chain)}_pdim{privacy_dim}"
save_dir = os.path.join(out_dir,"model_save")
print(f"model save: {save_dir}")
print(f"privacy chain: enabled={enable_privacy_chain}, dim={privacy_dim}")

time_stats = {
        'train': [[] for _ in range(world_size-1)],  # 各客户端每个epoch的训练时间
        'val': [[] for _ in range(world_size-1)],    # 各客户端每个epoch的验证时间
        'test': [0.0 for _ in range(world_size-1)]   # 各客户端的最终测试时间
    }


def torch_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PrivacyChain(nn.Module):
    def __init__(self, embed_dim=768, bottleneck_dim=128):
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError(f"privacy bottleneck_dim must be positive, got {bottleneck_dim}")

        self.down = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.up = nn.Linear(bottleneck_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        z = self.down(x)
        z = self.act(z)
        z = self.dropout(z)
        z = self.up(z)
        z = self.norm(z)
        return z


def apply_transformer_layers(transformer_model, x, start_layer, end_layer):
    for layer_idx in range(start_layer, end_layer):
        x = transformer_model.layers[layer_idx](x)
    return x


class ClientFlopsModel(nn.Module):
    def __init__(self, embedding_model, transformer_model, head_model, privacy_chain, k, l):
        super().__init__()
        assert 0 <= k <= l
        self.embedding = embedding_model
        self.local_layers = nn.ModuleList([transformer_model.layers[i] for i in range(k)])
        self.privacy_chain = privacy_chain if privacy_chain is not None else nn.Identity()
        self.head = head_model

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.local_layers:
            x = layer(x)
        x = self.privacy_chain(x)
        # The H-layer FLOPs are independent of whether it is executed
        # before or after the VS-side Transformer because the tensor shape
        # remains [batch, seq_len, embed_dim].
        x = self.head(x)
        return x


class ServerFlopsModel(nn.Module):
    def __init__(self, transformer_model, k, l):
        super().__init__()
        assert 0 <= k <= l
        self.server_layers = nn.ModuleList([transformer_model.layers[i] for i in range(k, l)])

    def forward(self, x):
        for layer in self.server_layers:
            x = layer(x)
        return x


def profile_split_flops(embedding_model, transformer_model, head_model, privacy_chain, dummy_data, k, l):
    assert 0 <= k <= l, f"split_k={k} must be in [0, layer_num={l}]"

    client_profile_model = copy.deepcopy(
        ClientFlopsModel(embedding_model, transformer_model, head_model, privacy_chain, k, l)
    ).to(dummy_data.device)
    client_profile_model.eval()
    client_forward_flops, _ = profile(client_profile_model, inputs=(dummy_data,), verbose=False)

    if k == l:
        server_forward_flops = 0
    else:
        with torch.no_grad():
            cut_activation = embedding_model(dummy_data)
            cut_activation = apply_transformer_layers(transformer_model, cut_activation, 0, k)
            if privacy_chain is not None:
                cut_activation = privacy_chain(cut_activation)
        server_profile_model = copy.deepcopy(ServerFlopsModel(transformer_model, k, l)).to(dummy_data.device)
        server_profile_model.eval()
        server_forward_flops, _ = profile(server_profile_model, inputs=(cut_activation,), verbose=False)

    return client_forward_flops, server_forward_flops


def split_forward_train(data, embedding_model, transformer_model, head_model, privacy_chain, k, l):
    client_cut = embedding_model(data)
    client_cut = apply_transformer_layers(transformer_model, client_cut, 0, k)

    # CS-side privacy chain before uploading activation to VS.
    if privacy_chain is not None:
        client_cut = privacy_chain(client_cut)

    # CS -> VS protected cut activation.
    server_input = client_cut.detach().requires_grad_(True)
    server_output = apply_transformer_layers(transformer_model, server_input, k, l)

    # VS -> CS activation for local H-layer classification.
    head_input = server_output.detach().requires_grad_(True)
    logits = head_model(head_input)

    return logits, client_cut, server_input, server_output, head_input


def backward_split(loss, client_cut, server_input, server_output, head_input):
    loss.backward()

    if head_input.grad is not None:
        grad_to_server_output = head_input.grad.detach()
        server_output.backward(grad_to_server_output)

    if server_input.grad is not None and client_cut.requires_grad:
        grad_to_client_cut = server_input.grad.detach()
        client_cut.backward(grad_to_client_cut)


def split_forward_eval(data, embedding_model, transformer_model, head_model, privacy_chain, k, l):
    x = embedding_model(data)
    x = apply_transformer_layers(transformer_model, x, 0, k)
    if privacy_chain is not None:
        x = privacy_chain(x)
    x = apply_transformer_layers(transformer_model, x, k, l)
    logits = head_model(x)
    return logits


def load_datasets_cmcc(rank,world_size):
    train_db = CMCCDataLoaderForFed(mode='train',semi=is_semi,rank=rank,world_size=world_size,window_size=window_size)
    test_db = CMCCDataLoaderForFed(mode='test',semi=is_semi,rank=rank,world_size=world_size,window_size=window_size)
    train_loader = DataLoader(train_db,batch_size=batch_size,shuffle=True,drop_last=True) 
    test_loader = DataLoader(test_db,batch_size=batch_size)
    val_loader = test_loader
    return train_loader,val_loader,test_loader

def load_datasets_aliyun(rank,world_size):
    train_db = AliyunDataLoaderForFed(mode='train',semi=is_semi,rank=rank,world_size=world_size,window_size=window_size)
    test_db = AliyunDataLoaderForFed(mode='test',semi=is_semi,rank=rank,world_size=world_size,window_size=window_size)
    print(rank,len(train_db),len(test_db))
    train_loader = DataLoader(train_db,batch_size=batch_size,shuffle=True,drop_last=False) 
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

def pu(orgin_model, new_model, delta):
    if not 0.0 <= delta <= 1.0:
        raise ValueError(f"EMA delta must be in [0, 1], got {delta}")

    model = copy.deepcopy(orgin_model)
    for key in model.keys():
        if torch.is_floating_point(orgin_model[key]):
            model[key] = delta * orgin_model[key] + (1.0 - delta) * new_model[key]
        else:
            # Non-floating buffers cannot be interpolated. Keep the previous
            # state when delta=1 so server-side delta1=1 remains exactly FedAvg;
            # otherwise use the current state.
            model[key] = orgin_model[key] if delta == 1.0 else new_model[key]
    return model


def server_sequential_ema(server_local_states, server_fedavg_state, delta):
    ema_state = copy.deepcopy(server_fedavg_state)
    for local_state in server_local_states:
        ema_state = pu(ema_state, local_state, delta)
    return ema_state

def cal_f1(label_list, pred_list,fw=None):
    label_arr = np.array(label_list)
    pred_arr = np.array(pred_list)
    # 异常检测
    ad_label = np.where(label_arr>0,1,0)
    ad_pred = np.where(pred_arr>0,1,0)
    _acc = accuracy_score(ad_label, ad_pred)
    _f1 = f1_score(ad_label, ad_pred)
    print("异常检测结果:")
    print(classification_report(ad_label,ad_pred, zero_division=0))
    if fw:
        fw.write("异常检测结果:\n")
        fw.write(f"Accuracy:{round(_acc,4)}, F1-Score:{round(_f1,4)}\n")
        fw.write(classification_report(ad_label,ad_pred, zero_division=0))
        fw.write('\n\n')
    fault_index = (label_arr!=0)&(pred_arr!=0)
    fd_label = label_arr[fault_index]
    fd_pred = pred_arr[fault_index]
    _macrof1 = f1_score(fd_label,fd_pred,average='macro')
    print("故障诊断结果:")
    print(classification_report(fd_label,fd_pred, zero_division=0))
    # print(classification_report(label_arr,pred_arr, zero_division=0))
    if fw:
        fw.write("故障诊断结果:\n")
        fw.write(f"Macro-F1 Score:{round(_macrof1,4)}\n")
        fw.write(classification_report(fd_label,fd_pred, zero_division=0))
        # fw.write(classification_report(label_arr,pred_arr, zero_division=0))
        fw.write('\n\n')



if __name__=='__main__':
    torch_seed(seed)
    if datatype == 'aliyun':
        loader_list = [load_datasets_aliyun(i,world_size) for i in range(1,world_size)]
    else:
        loader_list = [load_datasets_cmcc(i,world_size) for i in range(1,world_size)]

    embedding_model = Embedding(1024,embed_dim=embed_dim,device=device).to(device)
    client_list = [Head(embed_dim=embed_dim,classes=classes).to(device) for i in range(1,world_size)]
    server_list = [TransformerEncoding(layer_num,embed_dim=embed_dim,num_heads=12,ff_dim=dim_num).to(device) for i in range(1,world_size)]

    if enable_privacy_chain:
        privacy_chain_list = [
            PrivacyChain(
                embed_dim=embed_dim,
                bottleneck_dim=privacy_dim
            ).to(device)
            for _ in range(world_size - 1)
        ]
    else:
        privacy_chain_list = [None for _ in range(world_size - 1)]
    head_ema_state_list = [
        {k: v.detach().cpu().clone() for k, v in client_list[i].state_dict().items()}
        for i in range(world_size-1)
    ]

    assert 0 <= split_k <= layer_num, \
        f"split_k={split_k} must be in [0, layer_num={layer_num}]"

    best_score_list = [0 for _ in range(world_size-1)]

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    fw_list = [open("./{}/log{}.txt".format(out_dir, i),'w') for i in range(0,world_size-1)]
    criteon = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)

    if is_train:
        for i,model in enumerate(client_list):
            torch.save(model.state_dict(),os.path.join(save_dir,f"head_param{i}.pkl"))
        for i,model in enumerate(server_list):
            torch.save(model.state_dict(),os.path.join(save_dir,f"server_param{i}.pkl"))
        for i,model in enumerate(privacy_chain_list):
            if model is not None:
                torch.save(model.state_dict(), os.path.join(save_dir, f"privacy_chain_param{i}.pkl"))

        
        for it in range(global_epochs):
            print('*'*5,it,'*'*5)
            flops_list = [{} for _ in range(world_size-1)]
            params_list = [{} for _ in range(world_size-1)]
            batch_idx_list = [0 for _ in range(world_size-1)]

            # global_epoch_start = time.time()

            #train
            for i in range(world_size-1):
                client_train_start = time.time()

                flops, params = {}, {}
                client_model = client_list[i]
                server_model = server_list[i]
                privacy_chain = privacy_chain_list[i]

                client_params = list(client_model.parameters())
                if privacy_chain is not None:
                    client_params += list(privacy_chain.parameters())

                optimizer_client = optim.Adam(client_params, lr=learning_rate)
                optimizer_server = optim.Adam(server_model.parameters(), lr = learning_rate)
                client_model.train()
                server_model.train()
                if privacy_chain is not None:
                    privacy_chain.train()
                train_loader = loader_list[i][0]
                for e in range(local_epochs):
                    print('server',i ,'train epoch:', e)
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        # print(data.dtype,inputs.dtype)
                        # logits = model(data)

                        logits, client_cut, server_input, server_output, head_input = split_forward_train(
                            data, embedding_model, server_model, client_model, privacy_chain, split_k, layer_num
                        )

                        # semi learning with confidence-based pseudo-label filtering
                        loss_terms = []

                        labeled_mask = target != -1
                        unlabeled_mask = target == -1

                        # 1. labeled samples
                        if labeled_mask.any():
                            labeled_logits = logits[labeled_mask]
                            labeled_targets = target[labeled_mask].long()
                            labeled_loss = criteon(labeled_logits, labeled_targets)
                            loss_terms.append(labeled_loss)

                        # 2. unlabeled samples with confidence filtering
                        if unlabeled_mask.any():
                            unlabeled_logits = logits[unlabeled_mask]

                            with torch.no_grad():
                                probs = torch.softmax(unlabeled_logits, dim=1)
                                conf, pseudo_targets = probs.max(dim=1)

                            keep_mask = conf >= 0.9

                            if keep_mask.any():
                                unlabeled_loss = criteon(
                                    unlabeled_logits[keep_mask],
                                    pseudo_targets[keep_mask].long()
                                )
                                loss_terms.append(unlabeled_weight(e) * unlabeled_loss)

                        # 3. if this batch has no valid training signal, skip it
                        if len(loss_terms) == 0:
                            continue

                        loss = sum(loss_terms)
                        # loss = labeled_loss + unlabeled_weight(e,T1=9,T2=14)*unlabeled_loss
                        # loss = criteon(logits, target)
                        optimizer_client.zero_grad()
                        optimizer_server.zero_grad()
                        backward_split(loss, client_cut, server_input, server_output, head_input)
                        optimizer_server.step()
                        optimizer_client.step()

                        if len(list(flops.keys())) == 0:
                            client_fl, server_fl = profile_split_flops(
                                embedding_model, server_model, client_model, privacy_chain, data, split_k, layer_num
                            )
                            flops['client'] = client_fl
                            flops['server'] = server_fl
                            params['cut_activation'] = torch.numel(server_input)
                            params['server_output'] = torch.numel(head_input)
                
                flops_list[i] = flops
                params_list[i] = params
                batch_idx_list[i] = batch_idx

                train_time = time.time() - client_train_start
                time_stats['train'][i].append(train_time)
                fw_list[i].write(f"[Epoch {it}] Local Train Time: {train_time:.2f}s\n")

            client_list = [client.cpu() for client in client_list]
            server_list = [server.cpu() for server in server_list]

            # 1) Only Transformer/server-side parameters participate in FedAvg.
            # 2) Client Heads remain local and are smoothed only by temporal EMA.
            server_states = list(map(lambda x:x.state_dict(),server_list))
            server_model = FedAvg(server_states)

            if is_ema:
                server_ema_model = server_sequential_ema(
                    server_local_states=server_states,
                    server_fedavg_state=server_model,
                    delta=ema_delta_server
                )
                for i in range(world_size-1):
                    server_list[i].load_state_dict(server_ema_model)

                    head_ema_state_list[i] = pu(
                        head_ema_state_list[i],
                        client_list[i].state_dict(),
                        ema_delta_client
                    )
                    client_list[i].load_state_dict(head_ema_state_list[i])
            else:
                for i in range(world_size-1):
                    server_list[i].load_state_dict(server_model)
                    head_ema_state_list[i] = {
                        k: v.detach().cpu().clone()
                        for k, v in client_list[i].state_dict().items()
                    }
            client_list = [client.to(device) for client in client_list]
            server_list = [server.to(device) for server in server_list]
                   
            for i in range(world_size-1):
                val_start = time.time()
                client_model = client_list[i]
                server_model = server_list[i]
                privacy_chain = privacy_chain_list[i]
                flops = flops_list[i]
                params = params_list[i]
                batch_idx_for_flops = batch_idx_list[i]
                client_model.eval()
                server_model.eval()
                if privacy_chain is not None:
                    privacy_chain.eval()
                val_loader = loader_list[i][1]
                #valid
                test_loss = 0
                y_true = []
                y_pred = []
                for data, target in val_loader:
                    y_true.extend(target)
                    data, target = data.to(device), target.to(device)
                    logits = split_forward_eval(data, embedding_model, server_model, client_model, privacy_chain, split_k, layer_num)
                    test_loss += criteon(logits, target).item()

                    pred = logits.data.topk(1)[1].flatten().cpu()
                    y_pred.extend(pred)

                test_loss /= len(val_loader.dataset)
                

                
                acc = accuracy_score(y_true, y_pred)
                # macro_f1 = f1_score(y_true, y_pred, average='macro')
                label_arr = np.array(y_true)
                pred_arr = np.array(y_pred)
                ad_label = np.where(label_arr>0,1,0)
                ad_pred = np.where(pred_arr>0,1,0)
                _acc = accuracy_score(ad_label, ad_pred)
                _f1 = f1_score(ad_label, ad_pred)
                label_arr = np.array(y_true)
                pred_arr = np.array(y_pred)
                fault_index = (label_arr!=0)&(pred_arr!=0)
                fd_label = label_arr[fault_index]
                fd_pred = pred_arr[fault_index]
                # _acc = accuracy_score(fd_label, fd_pred)
                _macrof1 = f1_score(fd_label,fd_pred,average='macro')
                _macrof1_new = f1_score(label_arr,pred_arr,average='macro')

                print('\n{},VALID set: Average loss: {:.4f}, Fault detection Accuracy:{} F1-Score:{}, Fault classification Macro-F1-score:{} or {}\n'.format(
                    i,test_loss,round(_acc,4),round(_f1,4),round(_macrof1,4),round(_macrof1_new,4)))
                # print(f'classification_report: {class_report}')
                
                print('{}, Client Counting flops: {:.4f}GFlops, Server Counting flops: {:.4f}GFlops, Transmitted params: {:.4f}G\n'.format(
                    it, flops['client'] * 3 * local_epochs * batch_idx_for_flops * (it + 1) / 10e9, flops['server'] * 3 * local_epochs * batch_idx_for_flops * (it + 1) / 10e9, \
                        (params['cut_activation'] + params['server_output']) * local_epochs * batch_idx_for_flops * (it + 1) / 10e9))
                
                fw_list[i].write('\n{},VALID set: Average loss: {:.4f}, Fault detection Accuracy:{} F1-Score:{}, Fault classification Macro-F1-score:{} or {}\n'.format(
                    i,test_loss,round(_acc,4),round(_f1,4),round(_macrof1,4),round(_macrof1_new,4)))
                
                # fw_list[i].write(f'classification_report: {class_report}\n')
                # cal_f1(y_true, y_pred,fw_list[i])
                
                fw_list[i].write('{}, Client Counting flops: {:.4f}GFlops, Server Counting flops: {:.4f}GFlops, Transmitted params: {:.4f}G\n'.format(
                    it, flops['client'] * 3 * local_epochs * batch_idx_for_flops * (it + 1) / 10e9, flops['server'] * 3 * local_epochs * batch_idx_for_flops * (it + 1) / 10e9, \
                        (params['cut_activation'] + params['server_output']) * local_epochs * batch_idx_for_flops * (it + 1) / 10e9))
                
                select_score =  acc

                if select_score > best_score_list[i]:
                    best_score_list[i] = select_score
                    torch.save(client_model.state_dict(), os.path.join(save_dir, f"head_param{i}.pkl"))
                    torch.save(server_model.state_dict(), os.path.join(save_dir, f"server_param{i}.pkl"))
                    if privacy_chain is not None:
                        torch.save(
                            privacy_chain.state_dict(),
                            os.path.join(save_dir, f"privacy_chain_param{i}.pkl")
                        )

                # 记录验证耗时
                val_time = time.time() - val_start
                time_stats['val'][i].append(val_time)
                fw_list[i].write(f"[Epoch {it}] Validation Time: {val_time:.2f}s\n")
                fw_list[i].write('==========================\n')

    for i in range(world_size-1):
        test_start = time.time()
        print("local server",i)
        client_model = client_list[i]
        server_model = server_list[i]
        privacy_chain = privacy_chain_list[i]
        client_model.load_state_dict(torch.load(os.path.join(save_dir,f"head_param{i}.pkl")))
        server_model.load_state_dict(torch.load(os.path.join(save_dir,f"server_param{i}.pkl")))
        if privacy_chain is not None:
            privacy_chain.load_state_dict(
                torch.load(os.path.join(save_dir, f"privacy_chain_param{i}.pkl"))
            )
        client_model.eval()
        server_model.eval()
        if privacy_chain is not None:
            privacy_chain.eval()
        test_loader = loader_list[i][2]
        pred_list = []
        label_list = []

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = split_forward_eval(data, embedding_model, server_model, client_model, privacy_chain, split_k, layer_num)
            pred = logits.data.topk(1)[1].flatten()
            pred_list.extend(list(pred.cpu()))
            label_list.extend(list(target.cpu()))
        cal_f1(label_list, pred_list,fw_list[i])
        test_time = time.time() - test_start
        time_stats['test'][i] = test_time
        fw_list[i].write(f"Final Test Time: {test_time:.2f}s\n")
    
    # 汇总统计到日志
    for i in range(world_size-1):
        fw = fw_list[i]
        fw.write("\n========== Time Summary ==========\n")
        fw.write(f"Client {i} Time Statistics:\n")
        fw.write(f"- Total Training Time: {sum(time_stats['train'][i]):.2f}s\n")
        fw.write(f"- Average Training Time per Epoch: {np.mean(time_stats['train'][i]):.2f}s\n")
        fw.write(f"- Total Validation Time: {sum(time_stats['val'][i]):.2f}s\n")
        fw.write(f"- Final Test Time: {time_stats['test'][i]:.2f}s\n")

    for f in fw_list:
        f.close()
