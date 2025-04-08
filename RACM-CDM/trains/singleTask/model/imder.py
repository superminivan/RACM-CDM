# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from ...subNets import BertTextEncoder
# from ...subNets.transformers_encoder.transformer import TransformerEncoder
# from .scoremodel import ScoreNet, loss_fn, Euler_Maruyama_sampler
# import functools
# from .rcan import Group
# from random import sample

# __all__ = ['IMDER']



# class MSE(nn.Module): #均方误差
#     def __init__(self):
#         super(MSE, self).__init__()

#     def forward(self, pred, real):
#         diffs = torch.add(real, -pred)
#         n = torch.numel(diffs.data)
#         mse = torch.sum(diffs.pow(2)) / n

#         return mse

# # Set up the SDE (SDE is used to define Diffusion Process)
# device = 'cuda'
# def marginal_prob_std(t, sigma): #边际概率标准差
#     """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

#     Args:
#       t: A vector of time steps.
#       sigma: The $\sigma$ in our SDE.

#     Returns:
#       The standard deviation.
#     """
#     t = torch.as_tensor(t, device=device) #将t转换为张量
#     return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma)) #返回标准差

# def diffusion_coeff(t, sigma): #扩散系数
#     """Compute the diffusion coefficient of our SDE.

#     Args:
#       t: A vector of time steps.
#       sigma: The $\sigma$ in our SDE.

#     Returns:
#       The vector of diffusion coefficients.
#     """
#     return torch.as_tensor(sigma ** t, device=device)

# # Set up IMDer
# class IMDER(nn.Module):
#     def __init__(self, args):
#         super(IMDER, self).__init__()
#         if args.use_bert: #使用bert
#             self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
#                                               pretrained=args.pretrained)
#         self.use_bert = args.use_bert
#         dst_feature_dims, nheads = args.dst_feature_dim_nheads #目标特征维度和头数
#         self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims #原始特征维度
#         self.d_l = self.d_a = self.d_v = dst_feature_dims #目标特征维度
#         self.num_heads = nheads #头数
#         self.layers = args.nlevels #层数
#         self.attn_dropout = args.attn_dropout #注意力dropout
#         self.attn_dropout_a = args.attn_dropout_a #注意力dropout
#         self.attn_dropout_v = args.attn_dropout_v #注意力dropout
#         self.relu_dropout = args.relu_dropout #relu dropout
#         self.embed_dropout = args.embed_dropout #嵌入dropout
#         self.res_dropout = args.res_dropout #残差dropout
#         self.output_dropout = args.output_dropout #输出dropout
#         self.text_dropout = args.text_dropout #文本dropout
#         self.attn_mask = args.attn_mask #注意力掩码
#         self.MSE = MSE() #均方误差

#         combined_dim = 2 * (self.d_l + self.d_a + self.d_v) #组合维度

#         output_dim = args.num_classes if args.train_mode == "classification" else 1 #输出维度

#         sigma = 25.0 #sigma，SDE中的参数
#         self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
#         self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)  # used for sample
#         self.score_l = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn) #得分网络
#         self.score_v = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn)
#         self.score_a = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn)

#         self.cat_lv = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)   #将d_l*2维度的特征转换为d_l维度
#         self.cat_la = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)
#         self.cat_va = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)

#         self.rec_l = nn.Sequential(     #重建网络
#             nn.Conv1d(self.d_l, self.d_l*2, 1),     #卷积层
#             Group(num_channels=self.d_l*2, num_blocks=20, reduction=16),    #组
#             nn.Conv1d(self.d_l*2, self.d_l, 1)  #卷积层
#         )

#         self.rec_v = nn.Sequential(
#             nn.Conv1d(self.d_v, self.d_v*2, 1),
#             Group(num_channels=self.d_v*2, num_blocks=20, reduction=16),
#             nn.Conv1d(self.d_v*2, self.d_v, 1)
#         )

#         self.rec_a = nn.Sequential(
#             nn.Conv1d(self.d_a, self.d_a*2, 1),
#             Group(num_channels=self.d_a*2, num_blocks=20, reduction=16),
#             nn.Conv1d(self.d_a*2, self.d_a, 1)
#         )

#         # 1. Temporal convolutional layers
#         self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
#         self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
#         self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

#         # 2. Crossmodal Attentions
#         self.trans_l_with_a = self.get_network(self_type='la')
#         self.trans_l_with_v = self.get_network(self_type='lv')

#         self.trans_a_with_l = self.get_network(self_type='al')
#         self.trans_a_with_v = self.get_network(self_type='av')

#         self.trans_v_with_l = self.get_network(self_type='vl')
#         self.trans_v_with_a = self.get_network(self_type='va')

#         # 3. Self Attentions
#         self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
#         self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
#         self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

#         # Projection layers
#         self.proj1 = nn.Linear(combined_dim, combined_dim)
#         self.proj2 = nn.Linear(combined_dim, combined_dim)
#         self.out_layer = nn.Linear(combined_dim, output_dim)

#     def get_network(self, self_type='l', layers=-1): #获取网络
#         if self_type in ['l', 'al', 'vl']:
#             embed_dim, attn_dropout = self.d_l, self.attn_dropout
#         elif self_type in ['a', 'la', 'va']:
#             embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
#         elif self_type in ['v', 'lv', 'av']:
#             embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
#         elif self_type == 'l_mem':
#             embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
#         elif self_type == 'a_mem':
#             embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
#         elif self_type == 'v_mem':
#             embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
#         else:
#             raise ValueError("Unknown network type")

#         # TODO: Replace with nn.TransformerEncoder
#         return TransformerEncoder(embed_dim=embed_dim,
#                                   num_heads=self.num_heads,
#                                   layers=max(self.layers, layers),
#                                   attn_dropout=attn_dropout,
#                                   relu_dropout=self.relu_dropout,
#                                   res_dropout=self.res_dropout,
#                                   embed_dropout=self.embed_dropout,
#                                   attn_mask=self.attn_mask)

    
    
#     def forward(self, text, audio, video, num_modal=None): #前向传播
#         #文本处理：如果使用BERT，将文本输入 text_model 进行编码。应用dropout到文本特征。
#         with torch.no_grad():
#             if self.use_bert:
#                 text = self.text_model(text)
#         x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
#         x_a = audio.transpose(1, 2)
#         x_v = video.transpose(1, 2)
#         # Project the textual/visual/audio features
#         #将文本/视觉/音频特征投影到更低维度的空间
#         with torch.no_grad():
#             proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
#             proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
#             proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
#             gt_l, gt_v, gt_a = proj_x_l, proj_x_v, proj_x_a
            

#         #  random select modality
#         #随机选择模态
#         modal_idx = [0, 1, 2]  # (0:text, 1:vision, 2:audio)
#         ava_modal_idx = sample(modal_idx, num_modal)  # sample available modality
#         if num_modal == 1:  # one modality is available
#             if ava_modal_idx[0] == 0:  # has text
#                 conditions = proj_x_l
#                 loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=conditions)
#                 loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=conditions)
#                 loss_score_l = torch.tensor(0)
#                 # Generate samples from score-based models with the Euler_Maruyama_sampler
#                 proj_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
#                                                   device='cuda', condition=conditions)
#                 proj_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
#                                                   device='cuda', condition=conditions)
                

#                 #  refine modality
#                 proj_x_a = self.rec_a(proj_x_a)
#                 proj_x_v = self.rec_v(proj_x_v)
#                 loss_rec = self.MSE(proj_x_a, gt_a) + self.MSE(proj_x_v, gt_v)
#             elif ava_modal_idx[0] == 1:  # has video
#                 conditions = proj_x_v
#                 loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=conditions)
#                 loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=conditions)
#                 loss_score_v = torch.tensor(0)
#                 # Generate samples from score-based models with the Euler_Maruyama_sampler
#                 proj_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
#                                                   device='cuda', condition=conditions)
#                 proj_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
#                                                   device='cuda', condition=conditions)
                
#                 #  refine modality
#                 proj_x_l = self.rec_l(proj_x_l)
#                 proj_x_a = self.rec_a(proj_x_a)
#                 loss_rec = self.MSE(proj_x_l, gt_l) + self.MSE(proj_x_a, gt_a)
#             else:  # has audio
#                 conditions = proj_x_a
#                 loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=conditions)
#                 loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=conditions)
#                 loss_score_a = torch.tensor(0)
#                 # Generate samples from score-based models with the Euler_Maruyama_sampler
#                 proj_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
#                                                   device='cuda', condition=conditions)
#                 proj_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
#                                                   device='cuda', condition=conditions)
#                 #  refine modality
#                 proj_x_l = self.rec_l(proj_x_l)
#                 proj_x_v = self.rec_v(proj_x_v)
#                 loss_rec = self.MSE(proj_x_l, gt_l) + self.MSE(proj_x_v, gt_v)
#         if num_modal == 2:  # two modalities are available
#             if set(modal_idx) - set(ava_modal_idx) == {0}:  # L is missing (V,A available)
#                 conditions = self.cat_va(torch.cat([proj_x_v, proj_x_a], dim=1))  # cat two avail modalities as conditions
#                 loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=conditions)
#                 loss_score_v, loss_score_a = torch.tensor(0), torch.tensor(0)
#                 # Generate samples from score-based models with the Euler_Maruyama_sampler
#                 proj_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
#                                                   device='cuda', condition=conditions)
#                 #  refine modality
#                 proj_x_l = self.rec_l(proj_x_l)
#                 loss_rec = self.MSE(proj_x_l, gt_l)
#             if set(modal_idx) - set(ava_modal_idx) == {1}:  # V is missing (L,A available)
#                 conditions = self.cat_la(torch.cat([proj_x_l, proj_x_a], dim=1))  # cat two avail modalities as conditions
#                 loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=conditions)
#                 loss_score_l, loss_score_a = torch.tensor(0), torch.tensor(0)
#                 # Generate samples from score-based models with the Euler_Maruyama_sampler
#                 proj_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
#                                                   device='cuda', condition=conditions)
#                 #  refine modality
#                 proj_x_v = self.rec_v(proj_x_v)
#                 loss_rec = self.MSE(proj_x_v, gt_v)
#             if set(modal_idx) - set(ava_modal_idx) == {2}:  # A is missing (L,V available)
#                 conditions = self.cat_lv(torch.cat([proj_x_l, proj_x_v], dim=1))  # cat two avail modalities as conditions
#                 loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=conditions)
#                 loss_score_l, loss_score_v = torch.tensor(0), torch.tensor(0)
#                 # Generate samples from score-based models with the Euler_Maruyama_sampler
#                 proj_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
#                                                   device='cuda', condition=conditions)
#                 #  refine modality
#                 proj_x_a = self.rec_a(proj_x_a)
#                 loss_rec = self.MSE(proj_x_a, gt_a)
#         if num_modal == 3:  # no missing
#             loss_score_l, loss_score_v, loss_score_a = torch.tensor(0), torch.tensor(0), torch.tensor(0)
#             loss_rec = torch.tensor(0)

#         proj_x_a = proj_x_a.permute(2, 0, 1)
#         proj_x_v = proj_x_v.permute(2, 0, 1)
#         proj_x_l = proj_x_l.permute(2, 0, 1)

#         # (V,A) --> L
#         h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
#         h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
#         h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
#         h_ls = self.trans_l_mem(h_ls)
#         if type(h_ls) == tuple:
#             h_ls = h_ls[0]
#         last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

#         # (L,V) --> A
#         h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
#         h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
#         h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
#         h_as = self.trans_a_mem(h_as)
#         if type(h_as) == tuple:
#             h_as = h_as[0]
#         last_h_a = last_hs = h_as[-1]

#         # (L,A) --> V
#         h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
#         h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
#         h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
#         h_vs = self.trans_v_mem(h_vs)
#         if type(h_vs) == tuple:
#             h_vs = h_vs[0]
#         last_h_v = last_hs = h_vs[-1]

#         last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
#         # A residual block
#         last_hs_proj = self.proj2(
#             F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
#         last_hs_proj += last_hs

#         output = self.out_layer(last_hs_proj)

#         res = {
#             'Feature_t': last_h_l,
#             'Feature_a': last_h_a,
#             'Feature_v': last_h_v,
#             'Feature_f': last_hs,
#             'loss_score_l': loss_score_l,
#             'loss_score_v': loss_score_v,
#             'loss_score_a': loss_score_a,
#             'loss_rec': loss_rec,
#             'ava_modal_idx': ava_modal_idx,
#             'M': output
#         }
#         return res
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
from ...subNets.transformers_encoder.RJC_transformer import MultimodalTransformerEncoder
from .scoremodel import ScoreNet, loss_fn, Euler_Maruyama_sampler
import functools
from .rcan import Group
from random import sample

__all__ = ['IMDER']


class MSE(nn.Module): #均方误差
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred) # 计算预测与真实值之差
        n = torch.numel(diffs.data) # 获取差异数据的元素数量
        mse = torch.sum(diffs.pow(2)) / n # 计算均方误差

        return mse

# Set up the SDE (SDE is used to define Diffusion Process)
device = 'cuda'
def marginal_prob_std(t, sigma): # 定义边际概率的标准差计算，用于SDE（扩散过程中的参数）
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    t = torch.as_tensor(t, device=device) # 将时间步 t 转换为张量
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma)) # 返回标准差

# 定义扩散系数的计算，用于SDE
def diffusion_coeff(t, sigma): #扩散系数
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    return torch.as_tensor(sigma ** t, device=device)

# Set up IMDer
class IMDER(nn.Module):
    def __init__(self, args):
        super(IMDER, self).__init__()
        if args.use_bert: # 使用BERT进行文本编码
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads #目标特征维度和头数
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims #原始特征维度
        self.d_l = self.d_a = self.d_v = dst_feature_dims #目标特征维度
        self.num_heads = nheads # 多头注意力的头数
        self.layers = args.nlevels  # Transformer的层数
        # 定义模型的各种dropout参数
        self.attn_dropout = args.attn_dropout #注意力dropout
        self.attn_dropout_a = args.attn_dropout_a #注意力dropout
        self.attn_dropout_v = args.attn_dropout_v #注意力dropout
        self.relu_dropout = args.relu_dropout #relu dropout
        self.embed_dropout = args.embed_dropout #嵌入dropout
        self.res_dropout = args.res_dropout #残差dropout
        self.output_dropout = args.output_dropout #输出dropout
        self.text_dropout = args.text_dropout #文本dropout
        self.attn_mask = args.attn_mask #注意力掩码

        self.MSE = MSE() #均方误差

        # 组合特征维度的计算
        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)

        # 定义输出维度：分类模式下为类别数，否则为1
        output_dim = args.num_classes if args.train_mode == "classification" else 1

        # 设置扩散模型参数
        sigma = 25.0
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)  # used for sample
        # 初始化得分网络，用于推断缺失模态的特征
        self.score_l = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn)
        self.score_v = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn)
        self.score_a = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn)

        # 定义卷积层用于特征连接和降维处理
        self.cat_lv = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)
        self.cat_la = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)
        self.cat_va = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)

        # 定义用于特征重建的网络
        self.rec_l = nn.Sequential(     #重建网络
            nn.Conv1d(self.d_l, self.d_l*2, 1),     #卷积层
            Group(num_channels=self.d_l*2, num_blocks=20, reduction=16),    #组
            nn.Conv1d(self.d_l*2, self.d_l, 1)  #卷积层
        )

        self.rec_v = nn.Sequential(
            nn.Conv1d(self.d_v, self.d_v*2, 1),
            Group(num_channels=self.d_v*2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_v*2, self.d_v, 1)
        )

        self.rec_a = nn.Sequential(
            nn.Conv1d(self.d_a, self.d_a*2, 1),
            Group(num_channels=self.d_a*2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_a*2, self.d_a, 1)
        )

        # 1. Temporal convolutional layers
        # 定义时间卷积层，用于处理各模态特征
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        self.rjcmala = MultimodalTransformerEncoder(modalities=['l', 'a'],
                                                input_dim={ 'l': self.d_l, 'a': self.d_a},
                                                modal_dim=self.d_l,
                                                num_heads=self.num_heads,
                                                dropout=0)
        self.rjcmalv = MultimodalTransformerEncoder(modalities=['l', 'v'],
                                                input_dim={ 'l': self.d_l, 'v': self.d_v},
                                                modal_dim=self.d_l,
                                                num_heads=self.num_heads,
                                                dropout=0)
        self.rjcmava = MultimodalTransformerEncoder(modalities=['v', 'a'],
                                                        input_dim={ 'v': self.d_v, 'a': self.d_a},
                                                        modal_dim=self.d_v,
                                                        num_heads=self.num_heads,
                                                        dropout=0)

        # 2. Crossmodal Attentions
        # 定义交叉模态注意力机制
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')

        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')

        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions
        # 定义自注意力机制用于各模态的记忆
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # Projection layers
        # 定义投影层，用于最终的输出
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
        ##
        self.cond_adjust1 = nn.Conv1d(64, 48, kernel_size=1, padding=0)
    def get_network(self, self_type='l', layers=-1):
        # 根据指定的类型和层数，获取Transformer编码器
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        # TODO: Replace with nn.TransformerEncoder
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
        

    def forward(self, text, audio, video, num_modal=None): #前向传播
        #文本处理：如果使用BERT，将文本输入 text_model 进行编码。应用dropout到文本特征。
        with torch.no_grad():
            if self.use_bert:
                text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)
        # Project the textual/visual/audio features
        #将文本/视觉/音频特征投影到更低维度的空间                                                                                                         
        with torch.no_grad():
            proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
            proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
            proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
            gt_l, gt_v, gt_a = proj_x_l, proj_x_v, proj_x_a

        #  random select modality
        #随机选择模态
        modal_idx = [0, 1, 2]  # (0:text, 1:vision, 2:audio)
        ava_modal_idx = sample(modal_idx, num_modal)  # sample available modality
        if num_modal == 1:  # one modality is available 如果只有一种模态可用：
            if ava_modal_idx[0] == 0:  # has text
                conditions = proj_x_l# 将文本特征作为条件
                # 计算音频和视频的得分损失
                loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=conditions)
                loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=conditions)
                loss_score_l = torch.tensor(0)# 文本不缺失，损失为0
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                # 使用Euler_Maruyama采样器生成音频和视频特征
                proj_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                proj_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                # 通过重建网络对生成的特征进行细化
                proj_x_a = self.rec_a(proj_x_a)
                proj_x_v = self.rec_v(proj_x_v)
                loss_rec = self.MSE(proj_x_a, gt_a) + self.MSE(proj_x_v, gt_v)
            elif ava_modal_idx[0] == 1:  # has video
                conditions = proj_x_v
                loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=conditions)
                loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=conditions)
                loss_score_v = torch.tensor(0)
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                proj_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                proj_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                proj_x_l = self.rec_l(proj_x_l)
                proj_x_a = self.rec_a(proj_x_a)
                loss_rec = self.MSE(proj_x_l, gt_l) + self.MSE(proj_x_a, gt_a)
            else:  # has audio
                conditions = proj_x_a
                loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=conditions)
                loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=conditions)
                loss_score_a = torch.tensor(0)
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                proj_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                proj_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                proj_x_l = self.rec_l(proj_x_l)
                proj_x_v = self.rec_v(proj_x_v)
                loss_rec = self.MSE(proj_x_l, gt_l) + self.MSE(proj_x_v, gt_v)
        if num_modal == 2:  # two modalities are available
            if set(modal_idx) - set(ava_modal_idx) == {0}:  # L is missing (V,A available)
                # conditions = self.cat_va(torch.cat([proj_x_v, proj_x_a], dim=1))  # cat two avail modalities as conditions
                ##
                
                conditions = self.rjcmava({'a': proj_x_a.permute(0,2,1), 'v': proj_x_v.permute(0,2,1)})
                conditions = self.cat_va(conditions.permute(0,2,1))

                # conditions = self.cond_adjust1(conditions.permute(0,2,1)).permute(0,2,1)
                # print(torch.cat([proj_x_v, proj_x_a], dim=1).shape)
                # print(conditions.shape)
                loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=conditions)
                loss_score_v, loss_score_a = torch.tensor(0), torch.tensor(0)
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                proj_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                proj_x_l = self.rec_l(proj_x_l)
                loss_rec = self.MSE(proj_x_l, gt_l)
            if set(modal_idx) - set(ava_modal_idx) == {1}:  # V is missing (L,A available)
                # conditions = self.cat_la(torch.cat([proj_x_l, proj_x_a], dim=1))  # cat two avail modalities as conditions
                ##
                conditions = self.rjcmala({'l': proj_x_l.permute(0,2,1), 'a': proj_x_a.permute(0,2,1)})
                conditions = self.cat_va(conditions.permute(0,2,1))

                # conditions = self.cond_adjust1(conditions.permute(0,2,1)).permute(0,2,1)
                # print(torch.cat([proj_x_v, proj_x_a], dim=1).shape)
                # print(conditions.shape)
                loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=conditions)
                loss_score_l, loss_score_a = torch.tensor(0), torch.tensor(0)
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                proj_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                proj_x_v = self.rec_v(proj_x_v)
                loss_rec = self.MSE(proj_x_v, gt_v)
            if set(modal_idx) - set(ava_modal_idx) == {2}:  # A is missing (L,V available)
                # conditions = self.cat_lv(torch.cat([proj_x_l, proj_x_v], dim=1))  # cat two avail modalities as conditions
                ##
                conditions = self.rjcmalv({'l': proj_x_l.permute(0,2,1), 'v': proj_x_v.permute(0,2,1)})
                conditions = self.cat_va(conditions.permute(0,2,1))

                # conditions = self.cond_adjust1(conditions.permute(0,2,1)).permute(0,2,1)
                # print(torch.cat([proj_x_v, proj_x_a], dim=1).shape)
                # print(conditions.shape)
                loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=conditions)
                loss_score_l, loss_score_v = torch.tensor(0), torch.tensor(0)
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                proj_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                proj_x_a = self.rec_a(proj_x_a)
                loss_rec = self.MSE(proj_x_a, gt_a)
        if num_modal == 3:  # no missing
            loss_score_l, loss_score_v, loss_score_a = torch.tensor(0), torch.tensor(0), torch.tensor(0)
            loss_rec = torch.tensor(0)

        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]

        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        # A residual block
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)

        res = {
            'Feature_t': last_h_l,
            'Feature_a': last_h_a,
            'Feature_v': last_h_v,
            'Feature_f': last_hs,
            'loss_score_l': loss_score_l,
            'loss_score_v': loss_score_v,
            'loss_score_a': loss_score_a,
            'loss_rec': loss_rec,
            'ava_modal_idx': ava_modal_idx,
            'M': output
        }
        return res