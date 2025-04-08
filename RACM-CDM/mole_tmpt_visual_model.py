import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout
from transformers import AutoConfig, AutoModel

class MoEExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lora_rank=8, init_lora_weights=True):
        super(MoEExpert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.lora_A = nn.Parameter(torch.randn(hidden_dim, lora_rank))
        self.lora_B = nn.Parameter(torch.randn(lora_rank, output_dim))
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        
        # 初始化 LoRA 参数
        self.reset_parameters(init_lora_weights)

    def reset_parameters(self, init_lora_weights=True):
        """
        初始化 LoRA 参数
        """
        if init_lora_weights:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = self.fc2(x)
        return x

'''
class MoEAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=4, lora_rank=8, init_lora_weights=True):
        super(MoEAdapter, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            MoEExpert(input_dim, hidden_dim, output_dim, lora_rank, init_lora_weights) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        self.threshold_fn = nn.Linear(input_dim, 1, bias=False)
        self.max_threshold = 1.0 / num_experts

    def forward(self, x):
        gate_logits = F.softmax(self.gate(x), dim=-1)
        thresholds = F.sigmoid(self.threshold_fn(x)) * self.max_threshold
        adapted_gate_logits = gate_logits - thresholds
        selected_experts = (adapted_gate_logits > 0).float()
        weights = adapted_gate_logits * selected_experts
        weight_sums = weights.sum(dim=-1, keepdim=True)
        weights = weights / (weight_sums + 1e-6)
        
        # 初始化 results，确保维度与 expert(x) 一致
        results = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            batch_idx = (selected_experts[:, i] > 0).nonzero(as_tuple=True)[0]
            if len(batch_idx) > 0:
                expert_output = expert(x[batch_idx])
                print(f"weights: {weights[batch_idx, i].unsqueeze(-1).shape}")
                print(f"expert_output: {expert_output.shape}")
                print(f"results: {results[batch_idx].shape}")
                results[batch_idx] += weights[batch_idx, i].unsqueeze(-1) * expert_output
        return results
'''

class MoEAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=4, lora_rank=8, init_lora_weights=True):
        super(MoEAdapter, self).__init__()
        # experts 模块列表，每个 expert 输出维度为 output_dim
        self.experts = nn.ModuleList([
            MoEExpert(input_dim, hidden_dim, output_dim, lora_rank, init_lora_weights) 
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        self.threshold_fn = nn.Linear(input_dim, 1, bias=False)
        self.max_threshold = 1.0 / num_experts
        self.layer_loss = None  # 用于记录当前层的负载均衡损失

    def get_layer_loss(self, gate_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """
        计算负载均衡损失，参考 Switch Transformer 的方法
        """
        num_inputs = gate_logits.shape[0]
        num_experts = len(self.experts)
        expert_counts = torch.sum(selected_experts, dim=0)
        expert_fractions = expert_counts / num_inputs
        expert_probs = torch.sum(gate_logits, dim=0) / num_inputs
        layer_loss = num_experts * torch.sum(expert_fractions * expert_probs)
        return layer_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape [B, seq_len, hidden]
        """
        B, T, H = x.shape
        # 将输入展平为二维张量：[B*T, H]
        flattened_inputs = x.view(-1, H)
        
        # 计算门控得分
        gate_logits = torch.softmax(self.gate(flattened_inputs), dim=-1)  # [B*T, num_experts]
        thresholds = torch.sigmoid(self.threshold_fn(flattened_inputs)) * self.max_threshold  # [B*T, 1]
        adapted_gate_logits = gate_logits - thresholds
        selected_experts = (adapted_gate_logits >= 0).to(flattened_inputs.dtype)  # [B*T, num_experts]
        
        weights = adapted_gate_logits * selected_experts  # [B*T, num_experts]
        weight_sums = torch.sum(weights, dim=-1, keepdim=True)
        # 避免除0
        weight_sums = torch.where(weight_sums == 0, torch.ones_like(weight_sums), weight_sums)
        weights = weights / weight_sums

        # 初始化 results，形状与 expert(flattened_inputs) 输出一致，假设输出维度为 output_dim
        expert_output_dim = self.experts[0](flattened_inputs).shape[-1]
        results = torch.zeros(flattened_inputs.shape[0], expert_output_dim, 
                              dtype=flattened_inputs.dtype, device=flattened_inputs.device)

        # 针对每个 expert 分支
        for i, expert in enumerate(self.experts):
            # 找出对应 expert 被选中的位置，返回一维索引（注意：这里展平了 batch 和 seq_len）
            idx = torch.where(selected_experts[:, i] > 0)[0]
            if idx.numel() > 0:
                expert_output = expert(flattened_inputs[idx])
                # weights[idx, i] 的形状为 [N]，扩展为 [N, 1]后与 expert_output 相乘
                results[idx] += weights[idx, i].unsqueeze(-1) * expert_output

        # 将结果还原为原始形状：[B, T, output_dim]
        results = results.view(B, T, expert_output_dim)

        # 记录 layer_loss（只有在反向传播时才计算）
        if flattened_inputs.requires_grad:
            self.layer_loss = self.get_layer_loss(gate_logits=adapted_gate_logits, selected_experts=selected_experts)
        return results

class MoETransformerLayer(nn.Module):
    def __init__(self, original_layer, input_dim, hidden_dim, output_dim, num_experts=4, lora_rank=8, init_lora_weights=True):
        super(MoETransformerLayer, self).__init__()
        self.original_layer = original_layer  # 原始的 Transformer 层
        self.moe_adapter = MoEAdapter(input_dim, hidden_dim, output_dim, num_experts, lora_rank, init_lora_weights)
        
        for param in original_layer.parameters():
            param.requires_grad = False

    def forward(self, x, layer_head_mask=None, output_attentions=False):
        # 经过原始 Transformer 层
        outputs = self.original_layer(x, layer_head_mask, output_attentions)
        original_output = outputs[0]  # 原输出
        
        # MoEAdapter 也使用原始输入
        moe_output = self.moe_adapter(x)

        # 求和
        combined_output = original_output + moe_output

        # 返回格式保持一致
        outputs = (combined_output,) + outputs[1:]
        return outputs

class MoleTMPTVisualModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.visual_transformer_name)
        self.visual_plm = AutoModel.from_pretrained(args.visual_transformer_name, self.config)
        self.visual_soft_tokens = args.visual_soft_tokens
        self.hidden_size = self.config.hidden_size

        self.num_experts = getattr(args, 'num_experts', 4)
        self.lora_rank = getattr(args, 'lora_rank', 8)
        self.init_lora_weights = getattr(args, 'init_lora_weights', True)

        self.soft_prompt_dropout = Dropout(args.visual_soft_prompt_dropout)

        val = math.sqrt(6. / float(self.hidden_size * 2))
        self.soft_prompt_embeds = nn.Parameter(torch.zeros(1, self.visual_soft_tokens, self.hidden_size))
        nn.init.uniform_(self.soft_prompt_embeds.data, -val, val)

        # 修改每个 Transformer 层，添加 MoE 适配器
        self.visual_plm.encoder.layer = nn.ModuleList([
            MoETransformerLayer(
                layer,  # 原始 Transformer 层
                self.hidden_size,  # 输入维度
                self.hidden_size,  # 隐藏层维度
                self.hidden_size,  # 输出维度
                self.num_experts,  # 专家数量
                self.lora_rank,    # LoRA 秩
                self.init_lora_weights  # 是否初始化 LoRA 权重
            )
            for layer in self.visual_plm.encoder.layer
        ])

    def incorporate_prompt(self, pixel_values):
        batch_size = pixel_values.shape[0]
        x = self.visual_plm.embeddings(pixel_values)
        x = torch.cat((
                x[:, :1, :],
                self.soft_prompt_dropout(self.soft_prompt_embeds.expand(batch_size, -1, -1)),
                x[:, 1:, :]
            ), dim=1)
        return x

    def train(self, mode=True):
        if mode:
            self.visual_plm.encoder.eval()
            self.visual_plm.embeddings.eval()
            self.visual_plm.layernorm.eval()
            self.visual_plm.pooler.eval()
            # 遍历每个 transformer 层，解冻其中的 moe_adapter 参数
            for layer in self.visual_plm.encoder.layer:
                for param in layer.moe_adapter.parameters():
                    param.requires_grad = True
                # 可选：将 moe_adapter 设为训练模式
                # layer.moe_adapter.train()
            self.soft_prompt_dropout.train()
        else:
            for module in self.children():
                module.train(mode)

    def forward(self, input_data):
        embedding_output = self.incorporate_prompt(**{k: v for k, v in input_data.items() if k in inspect.signature(self.incorporate_prompt).parameters})
        
        encoder_outputs = self.visual_plm.encoder(
            embedding_output,
            output_attentions=input_data.get('output_attentions', None),
            output_hidden_states=input_data.get('output_hidden_states', None),
            return_dict=True
        )
        
        last_hidden_state = encoder_outputs['last_hidden_state']
        pooled_output = self.visual_plm.pooler(last_hidden_state)
        soft_hidden_state = self.visual_plm.layernorm(last_hidden_state)[:, 1:1+self.visual_soft_tokens, :]
        soft_hidden_state = torch.avg_pool1d(soft_hidden_state.transpose(1, 2), kernel_size=self.visual_soft_tokens).squeeze(-1)
        return {
            'last_hidden_state': soft_hidden_state,
            'pooler_output': pooled_output
        }

if __name__ == '__main__':
    class Args():
        # visual_transformer_name = 'google/vit-base-patch16-224'
        visual_transformer_name = '/root/autodl-tmp/model/vit-base-patch16-224'
        visual_soft_tokens = 5
        visual_soft_prompt_dropout = 0.2
        num_experts = 4
        lora_rank = 8

    args = Args()
    model = MoleTMPTVisualModel(args)
    # print(model)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    image_tensor = torch.randn(size=[16, 3, 224, 224])
    input_data = {'pixel_values': image_tensor}
    logits = model(input_data)
    print(f'logits.shape: {logits["last_hidden_state"].shape}')
    print(f'pooler_output.shape: {logits["pooler_output"].shape}')