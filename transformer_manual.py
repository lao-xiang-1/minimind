import argparse
import math
import os
import time

import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from contextlib import nullcontext

from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import Logger


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0)) / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn = torch.softmax(scores, dim=-1)

    if dropout:
        attn = dropout(attn)
    
    return attn @ value, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        def transform(x, linear):
            x = linear(x)
            return x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        query = transform(query, self.linear_q)
        key = transform(key, self.linear_k)
        value = transform(value, self.linear_v)

        x, _ = attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.linear_out(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
class AddNorm(nn.Module):
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = nn.ModuleList(
            [
                AddNorm(d_model, dropout),
                AddNorm(d_model, dropout)
            ]
        )

    def forward(self, x, mask=None):
        out1 = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        out2 = self.sublayers[1](out1, self.feed_forward)
        
        return out2

class DecoderLayers(nn.Module):
    def __init__(self, d_model, self_attn, cross_attn, feed_forward, dropout):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayers = nn.ModuleList(
            [
                AddNorm(d_model, dropout),
                AddNorm(d_model, dropout),
                AddNorm(d_model, dropout)
            ]
        )

    def forward(self, x, memory, src_mask, tag_mask):
        out1 = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tag_mask))
        out2 = self.sublayers[1](out1, lambda x: self.cross_attn(out1, memory, memory, src_mask))
        out3 = self.sublayers[2](out2, self.feed_forward)
        
        return out3

class Transformer(nn.Module):
    def __init__(self, src_vocab, tag_vocab, dropout=0.1, d_model=512, N=6, h=8, d_ff=2048):
        super().__init__()
        self.src_embed = nn.Sequential(
            Embeddings(src_vocab, d_model),
            PositionalEncoding(d_model)
        )

        self.tag_embed = nn.Sequential(
            Embeddings(tag_vocab, d_model),
            PositionalEncoding(d_model)
        )

        attn = lambda: MultiHeadAttention(h, d_model, dropout)
        ff = lambda: FeedForward(d_model, d_ff, dropout)

        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, attn(), ff(), dropout) for _ in range(N)
        ])

        self.decoder = nn.ModuleList([
            DecoderLayers(d_model, attn(), attn(), ff(), dropout) for _ in range(N)
        ])
        
        self.out = nn.Linear(d_model, tag_vocab)

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x
    
    def decode(self, tag, memory, src_mask, tag_mask):
        x = self.tag_embed(tag)
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tag_mask)
        return x

    def forward(self, src, tag, src_mask=None, tag_mask=None):
        memory = self.encode(src, src_mask)
        out = self.decode(tag, memory, src_mask, tag_mask)

        return self.out(out)

def causal_mask(seq_len: int, device: str):
    mask = torch.ones(1, seq_len, seq_len, device=device, dtype=torch.bool)
    return torch.tril(mask)


def train_epoch(epoch, model, optimizer, loss_fct, loader, autocast_ctx, scaler, args, device="cuda"):
    model.train()
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(loader, start=1):
        X = X.to(device)
        Y = Y.to(device)
        loss_mask = loss_mask.to(device)

        src_mask = loss_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
        tgt_mask = src_mask & causal_mask(Y.size(1), device)  # (B,1,L,L)

        with autocast_ctx:
            logits = model(X, X, src_mask=src_mask, tag_mask=tgt_mask)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            logits_loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = logits_loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == len(loader):
            spent = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            lr = optimizer.param_groups[0]['lr']
            eta_min = spent / step * len(loader) / 60 - spent / 60
            Logger(f"Epoch {epoch + 1}/{args.epochs} | Step {step}/{len(loader)} | loss {current_loss:.4f} | lr {lr:.6f} | eta {eta_min:.2f} min")

        if args.save_interval and (step % args.save_interval == 0 or step == len(loader)):
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_dir, f"{args.save_weight}_epoch{epoch+1}_step{step}.pth")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "step": step,
            }, ckpt_path)

        del X, Y, loss_mask, logits, loss

    ckpt_path = os.path.join(args.save_dir, f"{args.save_weight}_epoch{epoch+1}_step{step}.pth")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
    }, ckpt_path)


def test():
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 10
    vocab_size = 500
    model = Transformer(src_vocab=vocab_size, tag_vocab=vocab_size)

    src_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    src_pad_mask = (src_input != 0).unsqueeze(1).unsqueeze(2)
    tgt_pad_mask = (tgt_input != 0).unsqueeze(1).unsqueeze(2)
    seq_mask = torch.ones(1, seq_len, seq_len).tril(diagonal=0).bool()
    tgt_mask = tgt_pad_mask & seq_mask

    with torch.no_grad():
        outputs = model(src_input, tgt_input, src_mask=src_pad_mask, tag_mask=tgt_mask)
    print(f"输出形状: {outputs.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Transformer LM Training")
    parser.add_argument("--data_path", type=str, default="./dataset/s_pretrain_hq.jsonl", help="训练数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="./model", help="分词器路径")
    parser.add_argument("--save_dir", type=str, default="./out", help="模型保存目录")
    parser.add_argument("--save_weight", type=str, default="manual_transformer", help="模型保存前缀")
    parser.add_argument("--max_seq_len", type=int, default=340, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="dataloader线程")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="学习率")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--log_interval", type=int, default=50, help="日志间隔step")
    parser.add_argument("--save_interval", type=int, default=0, help="保存间隔step，为0则不保存")
    parser.add_argument("--d_model", type=int, default=512, help="模型维度")
    parser.add_argument("--n_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--n_layers", type=int, default=6, help="编码/解码层数")
    parser.add_argument("--d_ff", type=int, default=2048, help="前馈层维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout比例")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float16", help="混合精度类型")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    vocab_size = tokenizer.vocab_size

    model = Transformer(
        src_vocab=vocab_size,
        tag_vocab=vocab_size,
        d_model=args.d_model,
        N=args.n_layers,
        h=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    autocast_ctx = nullcontext() if device == "cpu" else torch.cuda.amp.autocast(dtype=torch_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    for epoch in range(args.epochs):
        train_epoch(epoch, model, optimizer, loss_fct, loader, autocast_ctx, scaler, args, device)

