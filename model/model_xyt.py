import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import torch.nn.init as init
from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        # 输入维度大于训练维度
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            # 计算 频率b 对应的序列号
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # 把输入维度分为低频、中频、高频三部分
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin):
    """应用旋转位置编码"""
    def rotate_half(x):
        # [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复KV heads以匹配query heads数量"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)

class Attention(nn.Module):
    """简化的多头注意力机制（带KV缓存）"""
    def __init__(self, dim: int, n_heads: int, kv_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = kv_heads
        self.n_rep = n_heads // kv_heads
        self.head_dim = dim // n_heads
        
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x, cos_sin, past_kv=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        
        # 投影得到Q, K, V
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        
        # 应用旋转位置编码
        cos, sin = cos_sin
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # KV缓存
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=1)
            v = torch.cat([past_kv[1], v], dim=1)
        present_kv = (k, v) if use_cache else None
        
        # 调整形状并重复KV
        q = q.transpose(1, 2)
        k = repeat_kv(k, self.n_rep).transpose(1, 2)
        v = repeat_kv(v, self.n_rep).transpose(1, 2)
        
        # 注意力计算
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 因果掩码
        if seq_len > 1:
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
            scores[:, :, :, -seq_len:] += mask
        
        # 应用注意力掩码
        if attention_mask is not None:
            scores = scores + attention_mask
            
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        output = scores @ v
        
        # 输出投影
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(output), present_kv

class FeedForward(nn.Module):
    """标准的前馈网络"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        # 每个专家有一个选择权重
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim))) # [n, h]
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # moe时，只关心token值，不关心其位置，可将bsz和seq_len合并
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h) # [bsz*seq_len, h]
        # 计算各个专家的得分
        logits = F.linear(hidden_states, self.weight, None) # [bsz*seq_len, n]
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False) # [bsz*seq_len, topk]

        # 权重归一化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 计算辅助损失
        if self.training and self.alpha > 0.0: # alpha：对辅助损失的看重程度
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1) # [bsz, seq_len*topk]

            # 方法1：序列级辅助损失（更加精细）
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1) # [bsz, seq_len, n]
                # 计算一个句子中各个专家的使用概率
                # sum(ce[0]) == seq_len * topk
                # 如果每个专家选择次数一样，那么ce中每个数等于1
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                # 每个专家：sum(频率 * 平均分数) == 损失
                # 分数越高，损失越多
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            # 方法2：批次级辅助损失（直接计算一整个batch的辅助损失）
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        # 每个专家相当于一个普通的FFNN
        # 专门专家层：需要选择topk个
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)

        if config.n_shared_experts > 0:
            # 共享专家层：每个token都要经过
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1]) # [bsz*seq_len, h]
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0) # x 复制topk遍
            y = torch.empty_like(x, dtype=x.dtype)
            # 遍历所有专家
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i]) # 每个专家各自的输出（而不是按token分）
                if expert_out.shape[0] > 0: 
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else: # 该专家没被选上（没产生梯度）
                    # 强制加入计算图
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            # 合并topk个专家的输出（每个输出*权重在求和）
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    # 将每个专家要处理的token集中到一起，不复制输入
    # 不计算梯度
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0) # 每个专家要处理的token数
        # 在flat_expert_indices中，一个token占topk个数。整除topk才是token真正的序列号
        token_idxs = idxs // self.config.num_experts_per_tok # 整除topk
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache

class TransformerBlock(nn.Module):
    """单个Transformer块"""
    def __init__(self, dim: int, n_heads: int, kv_heads: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attention = Attention(dim, n_heads, kv_heads, dropout)
        self.ffn = FeedForward(dim, hidden_dim, dropout)
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        
    def forward(self, x, cos_sin, past_kv=None, use_cache=False, attention_mask=None):
        # 自注意力
        residual = x
        attn_out, present_kv = self.attention(
            self.attn_norm(x), cos_sin, past_kv, use_cache, attention_mask
        )
        x = residual + attn_out
        
        # 前馈网络
        residual = x
        ffn_out = self.ffn(self.ffn_norm(x))
        x = residual + ffn_out
        
        return x, present_kv

class TransformerLM(nn.Module):
    """完整的Transformer语言模型"""
    def __init__(self, vocab_size: int, dim: int, n_layers: int, n_heads: int, 
                 kv_heads: int, hidden_dim: int, max_seq_len: int = 32768, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        
        # 词嵌入
        self.token_embeddings = nn.Embedding(vocab_size, dim)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, kv_heads, hidden_dim, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # 共享嵌入权重
        self.lm_head.weight = self.token_embeddings.weight
        
        # 预计算旋转位置编码
        cos, sin = precompute_freqs_cis(dim // n_heads, max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
    def forward(self, input_ids, past_key_values=None, use_cache=False, attention_mask=None):
        bsz, seq_len = input_ids.shape
        
        # 嵌入查找
        x = self.token_embeddings(input_ids)
        
        # 位置编码
        start_pos = 0
        if past_key_values is not None and past_key_values[0] is not None:
            start_pos = past_key_values[0][0].shape[1]
        
        cos_slice = self.cos[start_pos:start_pos + seq_len]
        sin_slice = self.sin[start_pos:start_pos + seq_len]
        cos_sin = (cos_slice, sin_slice)
        
        # 初始化KV缓存
        if past_key_values is None:
            past_key_values = [None] * self.n_layers
            
        # 前向传播
        present_key_values = []
        for i, layer in enumerate(self.layers):
            x, present_kv = layer(
                x, cos_sin, past_key_values[i], use_cache, attention_mask
            )
            if use_cache:
                present_key_values.append(present_kv)
        
        # 输出投影
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return {
            "logits": logits,
            "past_key_values": present_key_values if use_cache else None
        }

# 使用示例
if __name__ == "__main__":
    # 创建模型
    config = {
        "vocab_size": 6400,
        "dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "kv_heads": 2,
        "hidden_dim": 1360,  # 大约 dim * 8/3
        "max_seq_len": 32768,
        "dropout": 0.0
    }
    
    model = TransformerLM(**config)
    
    # 推理示例
    input_ids = torch.randint(0, config["vocab_size"], (2, 10))  # 批量大小2，序列长度10
    output = model(input_ids)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"输入形状: {input_ids.shape}")
    print(f"输出logits形状: {output['logits'].shape}")