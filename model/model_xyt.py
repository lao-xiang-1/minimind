import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class RMSNorm(nn.Module):
    """简化的RMS归一化层"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def precompute_freqs_cis(dim: int, end: int, rope_base: float = 1e6):
    """预计算RoPE旋转位置编码的频率"""
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
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