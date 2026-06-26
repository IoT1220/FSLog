import torch.nn as nn
import torch.optim as optim
import torch
# embed_dim = 768  # Embedding size for each token
# num_heads = 12  # Number of attention heads
# ff_dim = 2048  # Hidden layer size in feed forward network inside transformer
# max_len = 20

def get_angles(pos, i, d_model):
    angle_rates = 1 / torch.pow(10000, (2 * i) / torch.tensor(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(torch.unsqueeze(torch.arange(position), 1),
                            torch.unsqueeze(torch.arange(d_model), 0),
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

    pos_encoding = torch.unsqueeze(angle_rads, 0)

    return pos_encoding.float()

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim, device):
        super(PositionEmbedding, self).__init__()
        self.pos_encoding = positional_encoding(max_len,
                                                embed_dim).to(device)

    def forward(self, x):
        seq_len = x.shape[1]
        x =x+  self.pos_encoding[:, :seq_len, :].clone()
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()

        self.att = nn.MultiheadAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs):
        attn_output = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output[0])
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

# class PositionEmbedding(nn.Module):
#     def __init__(self, max_len, embed_dim, device):
#         super(PositionEmbedding, self).__init__()
#         self.pos_encoding = nn.Parameter(torch.randn(1, max_len,embed_dim)).to(device)

#     def forward(self, x):
#         x = x + self.pos_embedding.data[:, :x.shape[1], :]
#         return x

class Embedding(nn.Module):
    def __init__(self, max_len, embed_dim, device):
        super(Embedding, self).__init__()
        self.pos_encoding = positional_encoding(max_len,embed_dim).to(device)

    def forward(self, x):
        seq_len = x.shape[1]
        # x =x+  self.pos_encoding[:, :seq_len, :x.shape[2]].clone()
        x =x+  self.pos_encoding[:, :seq_len, :].clone()
        return x

# 新版本，修复了层数统计错误，实际层数与 layer_num 参数一致
class TransformerEncoding(nn.Module):
    def __init__(self, num_lay,embed_dim, num_heads, ff_dim, rate=0.1) -> None:
        super().__init__()
        self.num_lay = num_lay
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, rate)
            for _ in range(num_lay)
        ])

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

# 之前的错误版本，layer_num 没有生效，导致实际层数始终为 2
# class TransformerEncoding(nn.Module):
#     def __init__(self, num_lay,embed_dim, num_heads, ff_dim, rate=0.1) -> None:
#         super().__init__()
#         self.model = nn.Sequential()
#         for i in range(num_lay):
#             self.model.add_module('i',TransformerBlock(embed_dim, num_heads, ff_dim, rate))
#     def forward(self,x):
#         return self.model(x)
    
class Head(nn.Module):
    def __init__(self, embed_dim=768, dropout=0.1, classes=2) -> None:
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(embed_dim, 32)
        self.final = nn.Linear(32, classes)
    def forward(self,x):
        x = torch.transpose(x, 1, 2)
        x = self.pooling(x)
        x = self.dropout(torch.flatten(x, 1))
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.final(x)
        return x

class FedSplitClassifier(nn.Module):
    def __init__(self,embed_dim=768,classes=4,device='cpu') -> None:
        super().__init__()
        self.embedding = Embedding(1024,embed_dim=embed_dim,device=device)
        self.head = Head(embed_dim=embed_dim,classes=classes)
        self.transformer = TransformerEncoding(2,embed_dim=embed_dim,num_heads=2,ff_dim=1024)
    def forward(self,x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.head(x)
        return x


class Transformer_Classifier(nn.Module):
    def __init__(self, embed_dim=768, ff_dim=1024, num_heads=12, dropout=0.1, device='cpu', classes=2):
    # def __init__(self, embed_dim=768, ff_dim=1024, num_heads=2, dropout=0.1, device='cpu', classes=2):
        super(Transformer_Classifier, self).__init__()
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.embedding_layer = PositionEmbedding(1024, embed_dim, device)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(embed_dim, 32)
        self.final = nn.Linear(32, classes)

    def forward(self, inputs, device=None):
        if type(inputs) == list:
            inputs = inputs[0]
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = torch.transpose(x, 1, 2)
        x = self.pooling(x)
        x = self.dropout(torch.flatten(x, 1))
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.final(x)
        return x

def test():
    model = Transformer_Classifier(embed_dim=768, ff_dim=2048, num_heads=1, dropout=0.1, classes=6)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for _ in range(2):
        inputs = torch.zeros((1, 20, 768))
        
        label = torch.randint(0, 6, (1,))

        output = model(inputs)
        loss = criterion(output, label)

        print("Output shape:", output.shape)
        print("Label:", label)
        print("Loss:", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test2():
    criteon = nn.MSELoss()

    embedding = Embedding(1024,embed_dim=768,device='cpu')
    transformer_encoding = TransformerEncoding(2,embed_dim=768,num_heads=12,ff_dim=1024)
    head = Head(embed_dim=768,classes=5)

    inputs = torch.zeros((5,20,768))
    out_embedding = embedding(inputs)
    print(out_embedding.shape)
    out_tfencoding = transformer_encoding(out_embedding)
    print(out_tfencoding.shape)
    output = head(out_tfencoding)
    print(output.shape)
    
    label = torch.ones_like(output)
    loss=criteon(label,output)
    print(loss)

def test3():
    
    criteon = nn.CrossEntropyLoss()

    embedding = Embedding(1024,embed_dim=1,device='cpu')
    transformer_encoding = TransformerEncoding(2,embed_dim=1,num_heads=1,ff_dim=2048)
    head = Head(embed_dim=1,classes=2)
    learning_rate=1e-4
    optimizer_client = torch.optim.Adam(head.parameters(), lr = learning_rate) 
    optimizer_server = torch.optim.Adam(transformer_encoding.parameters(), lr = learning_rate) 
    input = torch.zeros((1,11,1))


    optimizer_client.zero_grad()
    optimizer_server.zero_grad()

    out_embedding = embedding(input)
    print('out_embedding shape',out_embedding.shape)
    
    out_tfencoding = transformer_encoding(out_embedding)
    print('out_tfencoding shape',out_tfencoding.shape)


    # out_tfencoding=torch.zeros((1,11,1))
    out_tfencoding = torch.from_numpy(out_tfencoding.detach().numpy())
    out_tfencoding_fx = out_tfencoding.requires_grad_(True)
    output_pred = head(out_tfencoding)
    print('output_pred shape', output_pred.shape)
    
    label = torch.ones_like(output_pred)
    loss=criteon(output_pred,label)
    print(loss)

    loss.backward()
    dfx_server = out_tfencoding.grad.clone().detach()
    optimizer_server.step()
    print(dfx_server.shape)
    out_tfencoding.backward(dfx_server)
    optimizer_client.step()


def test4():
    import numpy as np
    import copy
    a = torch.Tensor([1,2,3])
    b = torch.Tensor([2,3,4])
    print(a)
    print(b)
    v = torch.var(torch.vstack([a,b]),axis=0)
    print(v)
    print(torch.sqrt(torch.sum((a-b)**2)))
    transformer_encoding1 = TransformerEncoding(2,embed_dim=1,num_heads=1,ff_dim=2048)
    transformer_encoding2 = TransformerEncoding(2,embed_dim=1,num_heads=1,ff_dim=2048)
    def pu(orgin_model,new_model):
        similarities = []
        for a,b in zip(orgin_model.items(),new_model.items()):
            # v = torch.var(torch.vstack([a,b]),axis=0)
            a,b=a[1],b[1]
            similarities.append(torch.sqrt(torch.sum((a-b)**2)))
        similarities = np.array(similarities)
        norm_similarities = (similarities-similarities.min())/(similarities.max()-similarities.min())
        u = np.sum(norm_similarities)/len(norm_similarities)
        model = copy.deepcopy(orgin_model)
        for key in model.keys():
            model[key] = u*orgin_model[key]+(1-u)*new_model[key]
        return model
    pu(transformer_encoding1.state_dict(),transformer_encoding2.state_dict())


if __name__ == '__main__':
    test()
    
    
    
