
import torch.nn as nn
"""
CALayer 是一个注意力机制模块，它通过自适应平均池化 (AdaptiveAvgPool1d) 将输入 x 压缩到一个单一的向量，然后通过两个一维卷积层 (Conv1d) 来生成注意力权重。
第一个卷积层将通道数从 num_channels 减少到 num_channels//reduction，然后通过 ReLU 激活函数，再通过第二个卷积层将通道数恢复到 num_channels。
最后，通过 Sigmoid 激活函数生成权重 y，这个权重用来加权原始输入 x，实现通道注意力。
"""
class CALayer(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_du = nn.Sequential(
            nn.Conv1d(num_channels, num_channels//reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels//reduction, num_channels, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

"""
RCAB 是一个包含注意力机制的残差块，它由两个一维卷积层和一个 CALayer 组成。
残差连接允许网络直接将输入 x 加到经过 body（包含卷积和注意力层）处理后的输出上，增强了网络的学习能力。
res_scale 参数用于缩放残差连接的权重，以帮助稳定训练。
"""
class RCAB(nn.Module):
    def __init__(self, num_channels, reduction, res_scale):
        super().__init__()

        body = [
            nn.Conv1d(num_channels, num_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels, num_channels, 3, 1, 1),
        ]
        body.append(CALayer(num_channels, reduction))

        self.body = nn.Sequential(*body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

"""
Group 类似于 RCAB，但是它包含了多个 RCAB 块，这些块被堆叠在一起，以形成更深的网络结构。
除了 RCAB 块外，Group 还包含一个额外的一维卷积层，用于在所有 RCAB 块之后进一步处理特征。
同样地，Group 也使用残差连接将输入 x 加到 body 的输出上。
"""
class Group(nn.Module):
    def __init__(self, num_channels, num_blocks, reduction, res_scale=1.0):
        super().__init__()

        body = list()
        for _ in range(num_blocks):
            body += [RCAB(num_channels, reduction, res_scale)]
        body += [nn.Conv1d(num_channels, num_channels, 3, 1, 1)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res