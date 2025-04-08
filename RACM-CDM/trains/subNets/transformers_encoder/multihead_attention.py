import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

# 定义多头自注意力类，继承自PyTorch的nn.Module模块
class MultiheadAttention(nn.Module):
    # 初始化函数，设置多头自注意力的必要参数
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim # 嵌入的维度
        self.num_heads = num_heads  # 头的数量
        self.attn_dropout = attn_dropout  # 注意力权重的dropout概率
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        # 确保embed_dim可以被num_heads整除
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5# 缩放因子，用于平衡多头注意力

        # 权重参数，用于输入的线性变换
        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        # 如果需要bias，则初始化
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        # 输出的线性变换
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # 如果add_bias_kv为True，则添加键和值的bias
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn # 是否添加零注意力

        # 初始化参数
        self.reset_parameters()

    # 参数初始化函数
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)  # 使用Xavier均匀分布初始化权重
        nn.init.xavier_uniform_(self.out_proj.weight)  # 使用Xavier均匀分布初始化输出层权重
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)  # 初始化偏置为0
            nn.init.constant_(self.out_proj.bias, 0.)  # 初始化输出层偏置为0
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)  # 使用Xavier正态分布初始化键的偏置
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)  # 使用Xavier正态分布初始化值的偏置

     # 前向传播函数
    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        # 检查query, key, value是否指向同一内存地址，以确定是否为自注意力
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        # 如果query, key, value相同，则为自注意力，否则为编码器-解码器注意力
        if qkv_same:
            # 自注意力，将query, key, value合并后进行线性变换
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # 编码器-解码器注意力，分别对query和key, value进行线性变换
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            # 分别对query, key, value进行线性变换
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling  # 应用缩放因子

        # 如果存在键和值的偏置，则添加偏置
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        # 变换query, key, value的形状以适应多头自注意力计算
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        # 如果需要添加零注意力，则扩展key和value
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
        
        # 计算query和key的点积，得到注意力权重
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # 如果存在注意力mask，则添加到注意力权重中
        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False
                
        # 将权重进行softmax归一化，并应用dropout
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        # 计算注意力输出
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # 将注意力输出重新变形为原始形状，并经过输出层
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # 返回注意力输出和权重
        return attn, attn_weights

    # 以下为辅助函数，用于不同情况下的输入投影
    def in_proj_qkv(self, query):
        # 将query投影到三个子空间（query, key, value）
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        # 将key投影到两个子空间（key, value），从embed_dim开始
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        # 将query投影到其子空间，直到embed_dim结束
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        # 将key投影到其子空间，从embed_dim开始，直到2 * embed_dim结束
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        # 将value投影到其子空间，从2 * embed_dim开始
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        # 内部线性投影函数，处理输入的线性变换
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)