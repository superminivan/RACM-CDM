import math  # 导入数学库，用于数学计算
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块

def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(1)  # 计算最大位置数
    device = tensor.get_device()  # 获取张量所在的设备（CPU或GPU）
    buf_name = f'range_buf_{device}'  # 为设备创建一个缓冲区名称
    if not hasattr(make_positions, buf_name):
        # 如果make_positions函数没有该缓冲区属性，创建它
        setattr(make_positions, buf_name, tensor.new())
    setattr(make_positions, buf_name, getattr(make_positions, buf_name).type_as(tensor))
    # 将缓冲区的类型设置为与输入张量相同
    if getattr(make_positions, buf_name).numel() < max_pos:
        # 如果缓冲区的元素数量小于最大位置数，则重新生成位置范围
        torch.arange(padding_idx + 1, max_pos, out=getattr(make_positions, buf_name))
    
    mask = tensor.ne(padding_idx)  # 创建一个掩码，标记非填充符号的位置
    positions = getattr(make_positions, buf_name)[:tensor.size(1)].expand_as(tensor)
    # 获取位置索引并扩展到与输入张量相同的形状
    if left_pad:
        # 如果是左填充，则调整位置索引
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    
    new_tensor = tensor.clone()  # 克隆输入张量以避免修改原始张量
    return new_tensor.masked_scatter_(mask, positions[mask]).long()
    # 用位置索引替换非填充位置的值，并返回新的张量

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()  # 调用父类的构造函数
        self.embedding_dim = embedding_dim  # 存储嵌入维度
        self.padding_idx = padding_idx  # 存储填充索引
        self.left_pad = left_pad  # 存储填充方向（左或右）
        self.weights = dict()   # device --> actual weight; due to nn.DataParallel :-(
        # 存储不同设备的权重，支持数据并行
        self.register_buffer('_float_tensor', torch.FloatTensor(1))  # 注册一个浮点张量缓冲区

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2  # 计算一半的维度
        emb = math.log(10000) / (half_dim - 1)  # 计算缩放因子
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)  # 计算衰减因子
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        # 生成位置向量
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # 将正弦和余弦值连接起来
        if embedding_dim % 2 == 1:
            # 如果嵌入维度是奇数，添加零填充
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0  # 如果有填充索引，则将对应位置的嵌入设置为零
        return emb  # 返回生成的嵌入

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()  # 获取输入的批次大小和序列长度
        max_pos = self.padding_idx + 1 + seq_len  # 计算最大位置数
        device = input.get_device()  # 获取输入张量所在的设备
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # 如果设备不在权重字典中，或者最大位置数大于当前权重的大小，则重新计算嵌入
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights[device] = self.weights[device].type_as(self._float_tensor).to(input.device)
        # 将权重转换为与输入张量相同的类型并移动到相同的设备
        positions = make_positions(input, self.padding_idx, self.left_pad)  # 计算位置
        return self.weights[device].index_select(0, positions.contiguous().view(-1)).view(bsz, seq_len, -1).detach()
        # 根据位置索引选择嵌入，调整形状并返回

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number，返回支持的最大位置数