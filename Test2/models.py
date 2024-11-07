import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Attention, ModeProduct


class CP(nn.Module):#单个CP
    def __init__(self, num_dim, latent_dim):
        super(CP, self).__init__()
        num_user, num_item, num_time = tuple(num_dim)
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_user, latent_dim)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_item, latent_dim)))
        self.time_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_time, latent_dim)))

    def forward(self, i_input, j_input, k_input ):
        i_embeds = self.user_embeddings[i_input]
        j_embeds = self.item_embeddings[j_input]
        k_embeds = self.time_embeddings[k_input]

        xijk = torch.einsum('nd, nd, nd -> n', i_embeds, j_embeds, k_embeds)
        # xijk = i_embeds* j_embeds * k_embeds # [128, 50]
        # xijk = torch.sum(i_embeds* j_embeds * k_embeds, dim = -1) # [bs, dim]

        return xijk

class AggregationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)  

    def forward(self, output1, output2, output3):
        
        combined_output = torch.stack((output1, output2, output3), dim=-1)  # shape: (batch_size, 3)
        
        return self.linear(combined_output).squeeze(-1)
    
class MergL(nn.Module):
    def __init__(self, num_dim, latent_dim):
        super(MergL, self).__init__()
        self.cp1 = CP(num_dim, latent_dim)
        self.cp2 = CP(num_dim, latent_dim)
        self.cp3 = CP(num_dim, latent_dim)
        self.cp4 = CP(num_dim, latent_dim)

    def forward(self, i, j, k, labels):
        # 创建一个空的张量来存储最终的预测结果
        predictions = torch.zeros(i.size(0), device=i.device)
        
        # 对每个标签进行分配
        for label in range(4):
            # 获取当前批次中，属于该标签的数据
            selected_idx = labels == label

            if selected_idx.sum() > 0:  # 如果该标签的数据非空
                if label == 0:
                    predictions[selected_idx] = self.cp1(i[selected_idx], j[selected_idx], k[selected_idx])
                elif label == 1:
                    predictions[selected_idx] = self.cp2(i[selected_idx], j[selected_idx], k[selected_idx])
                elif label == 2:
                    predictions[selected_idx] = self.cp3(i[selected_idx], j[selected_idx], k[selected_idx])
                elif label == 3:
                    predictions[selected_idx] = self.cp4(i[selected_idx], j[selected_idx], k[selected_idx])

        return predictions


class Tucker(nn.Module):
    def __init__(self, num_dim, latent_dim):
        super(Tucker, self).__init__()
        num_user, num_item, num_time = tuple(num_dim)
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_user, latent_dim[0])))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_item, latent_dim[1])))
        self.time_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_time, latent_dim[2])))
        self.core_tensor = nn.Parameter(nn.init.xavier_uniform_(torch.empty(latent_dim[0], latent_dim[1], latent_dim[2])))

    def forward(self, i_input, j_input, k_input, ks_input=None):
        i_embeds = self.user_embeddings[i_input]
        j_embeds = self.item_embeddings[j_input]
        k_embeds = self.time_embeddings[k_input]

        xijk = torch.einsum('np,nq,nr->npqr', i_embeds, j_embeds, k_embeds)
        xijk = torch.einsum('npqr,pqr->n', xijk, self.core_tensor)
        return xijk


class LTP(nn.Module):
    def __init__(self, num_dim, latent_dim, period):
        super(LTP, self).__init__()
        num_user, num_item, num_time = tuple(num_dim)
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_user, latent_dim)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_item, latent_dim)))
        self.time_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_time+period, latent_dim)))

        self.lstm = nn.LSTM(latent_dim, latent_dim, 1, batch_first=True)       # (input_size/feature/rank, hidden_size, num_layers)
        self.atten = Attention(latent_dim)

        self.w1 = nn.Parameter(torch.eye(latent_dim))
        self.b1 = nn.Parameter(torch.zeros(latent_dim))
        self.w2 = nn.Parameter(torch.eye(latent_dim))
        self.b2 = nn.Parameter(torch.zeros(latent_dim))
        self.w3 = nn.Parameter(torch.eye(latent_dim))
        self.b3 = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, i_input, j_input, k_input, ks_input=None):
        i_embeds = self.user_embeddings[i_input]
        j_embeds = self.item_embeddings[j_input]
        # k_embeds = self.time_embeddings[k_input]
        k_embeds = self.time_embeddings[ks_input]
        k_embeds, (_, _) = self.lstm(k_embeds)
        k_embeds = self.atten(k_embeds)

        i_embeds = torch.mm(i_embeds, self.w1) + self.b1
        j_embeds = torch.mm(j_embeds, self.w2) + self.b2
        k_embeds = torch.mm(k_embeds, self.w3) + self.b3
        xijk = torch.sum(torch.mul(torch.mul(i_embeds, j_embeds), k_embeds), 1)

        # xijk = torch.einsum('ni,nj,nk->nijk', i_embeds, j_embeds, k_embeds)
        # xijk = torch.sum(torch.mul(self.U, xijk), dim=(1, 2, 3))
        return xijk


class NTC(nn.Module):
    def __init__(self, num_dim, latent_dim, channels):
        super(NTC, self).__init__()
        num_user, num_item, num_time = tuple(num_dim)
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_user, latent_dim)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_item, latent_dim)))
        self.time_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_time, latent_dim)))

        self.Conv = nn.Sequential(  # input: [batch, channel, depth, height, width]
            nn.Conv3d(1, channels, 2, 2),  # in_channels, out_channels, kernel_size, stride, padding=0
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2),
            nn.Conv3d(channels, channels, 2, 2),     # , kernel_size=2, stride=2
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2),
            nn.Conv3d(channels, channels, 2, 2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2),
            # nn.Conv3d(channels, channels, 2, 2),
            # nn.ReLU(inplace=True),
            # nn.Dropout3d(0.2),
        )
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels*((latent_dim // (2 ** 3)) ** 3), 1)
            # nn.Linear(channels, 1)
        )

    def forward(self, i_input, j_input, k_input, ks_input=None):
        i_embeds = self.user_embeddings[i_input]
        j_embeds = self.item_embeddings[j_input]
        k_embeds = self.time_embeddings[k_input]
        
        xijk = torch.einsum('ni,nj,nk->nijk', i_embeds, j_embeds, k_embeds)
        xijk = xijk.unsqueeze(1)
        xijk = self.Conv(xijk)
        xijk = self.FC(xijk)
        xijk = torch.sigmoid(xijk)
        return torch.squeeze(xijk)


class NTM(nn.Module):
    def __init__(self, num_dim, latent_dim, k=100, c=5):
        super(NTM, self).__init__()
        num_user, num_item, num_time = tuple(num_dim)
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_user, latent_dim)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_item, latent_dim)))
        self.time_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_time, latent_dim)))

        self.w = nn.Parameter(torch.Tensor(latent_dim, k))
        self.mode = ModeProduct(latent_dim, latent_dim, latent_dim, c, c, c)
        self.flat = nn.Flatten()
        self.FC = nn.Linear(in_features=c ** 3 + k, out_features=1)
        self.c = c

    def forward(self, i_input, j_input, k_input, ks_input=None):
        i_embeds = self.user_embeddings[i_input]
        j_embeds = self.item_embeddings[j_input]
        k_embeds = self.time_embeddings[k_input]

        # GCP
        gcp = F.relu(torch.mm(torch.mul(torch.mul(i_embeds, j_embeds), k_embeds), self.w))
        # output product
        x = torch.einsum('ni,nj,nk->nijk', i_embeds, j_embeds, k_embeds)
        # mode product
        x = self.mode(x)
        x = self.flat(x)
        x = torch.cat((gcp, x), 1)
        x = self.FC(x)
        return torch.squeeze(x)

