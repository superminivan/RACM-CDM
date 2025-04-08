import math
import torch
import torch.nn.functional as F
from torch import nn
from .multihead_attention import MultiheadAttention
from .position_embedding import SinusoidalPositionalEmbedding

class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        # 调用父类的构造方法
        super().__init__()
        
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout   # Attention dropout
        self.embed_dim = embed_dim         # 嵌入维度
        self.embed_scale = math.sqrt(embed_dim)  # 嵌入缩放因子
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)  # 正弦位置嵌入
        
        self.attn_mask = attn_mask  # 是否在注意力权重上应用掩码

        self.layers = nn.ModuleList([])  # 存储多个编码器层
        for layer in range(layers):
            # 创建新的编码器层并添加到层列表中
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))  # 注册一个版本张量
        self.normalize = True  # 是否进行层归一化
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)  # 创建层归一化实例

    def forward(self, x_in, x_in_k = None, x_in_v = None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """

        # 嵌入输入和位置
        x = self.embed_scale * x_in  # 缩放嵌入输入
        if self.embed_positions is not None:
            # 如果有位置嵌入，将其加到输入上
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)  # 应用嵌入丢弃

        if x_in_k is not None and x_in_v is not None:
            # 嵌入键和值
            x_k = self.embed_scale * x_in_k  # 缩放键输入
            x_v = self.embed_scale * x_in_v  # 缩放值输入
            if self.embed_positions is not None:
                # 如果有位置嵌入，将其加到键和值上
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)  # 应用键丢弃
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)  # 应用值丢弃
        
        # 编码器层
        intermediates = [x]  # 存储中间结果
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                # 如果提供了键和值，使用它们
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)  # 否则只使用输入x
            intermediates.append(x)  # 记录中间结果

        if self.normalize:
            x = self.layer_norm(x)  # 应用层归一化

        return x  # 返回编码器输出

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions  # 如果没有位置嵌入，返回最大源位置
        return min(self.max_source_positions, self.embed_positions.max_positions())  # 返回最大位置

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        # 调用父类的构造方法
        super().__init__()
        self.embed_dim = embed_dim  # 嵌入维度
        self.num_heads = num_heads    # 注意力头数
        
        # 初始化多头注意力机制
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask  # 是否应用注意力掩码

        self.relu_dropout = relu_dropout  # ReLU激活后的丢弃率
        self.res_dropout = res_dropout      # 残差连接后的丢弃率
        self.normalize_before = True  # 是否在层之前进行归一化

        # 前馈网络的线性层
        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # 线性变换
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)   # 线性变换
        # 创建两个层归一化实例
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x  # 保存输入用于残差连接
        x = self.maybe_layer_norm(0, x, before=True)  # 在前处理阶段应用层归一化
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None  # 生成未来掩码
        if x_k is None and x_v is None:
            # 如果没有提供键和值，使用自注意力
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            # 如果提供了键和值，将它们归一化
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True) 
            # 使用提供的键和值进行注意力计算
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)  # 应用残差丢弃
        x = residual + x  # 残差连接
        x = self.maybe_layer_norm(0, x, after=True)  # 在后处理阶段应用层归一化

        residual = x  # 保存当前输出用于下一个残差连接
        x = self.maybe_layer_norm(1, x, before=True)  # 在前处理阶段应用层归一化
        x = F.relu(self.fc1(x))  # 通过第一个线性层并应用ReLU
        x = F.dropout(x, p=self.relu_dropout, training=self.training)  # 应用ReLU后的丢弃
        x = self.fc2(x)  # 通过第二个线性层
        x = F.dropout(x, p=self.res_dropout, training=self.training)  # 应用残差丢弃
        x = residual + x  # 残差连接
        x = self.maybe_layer_norm(1, x, after=True)  # 在后处理阶段应用层归一化
        return x  # 返回编码后的输出

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after  # 确保只能在前或后处理而不是两者
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)  # 应用相应的层归一化
        else:
            return x  # 不应用归一化

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)  # 填充负无穷大

def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)  # 获取第一个张量的维度
    if tensor2 is not None:
        dim2 = tensor2.size(0)  # 如果有第二个张量，获取其维度
    # 创建一个上三角矩阵，填充负无穷大，表示未来掩码
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.to(tensor.device)  # 将掩码移动到相同的设备
    return future_mask[:dim1, :dim2]  # 返回适当大小的掩码

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)  # 创建线性层
    nn.init.xavier_uniform_(m.weight)  # 用Xavier均匀分布初始化权重
    if bias:
        nn.init.constant_(m.bias, 0.)  # 将偏置初始化为0
    return m  # 返回线性层

def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)  # 创建层归一化层
    return m  # 返回层归一化层

if __name__ == '__main__':
    encoder = TransformerEncoder(300, 4, 2)  # 创建一个Transformer编码器实例
    x = torch.tensor(torch.rand(20, 2, 300))  # 生成随机输入张量
    print(encoder(x).shape)  # 打印编码器输出的形状