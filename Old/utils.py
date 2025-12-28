import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """固定所有随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_log_probs(model, input_ids, attention_mask, labels, chunk_size: int = 128, batch_chunk_size: int = 1):
    """
    返回 [B, T] 的 token log-prob（第 0 位补 0）。
    关键：不再 permute logits 到 [B, V, T]，避免 cross_entropy 内部触发大规模拷贝；
         采用时间维分块，降低峰值显存。
    新增：batch_chunk_size，支持对 batch 维度进行切分，防止 OOM。
    """
    B, T = input_ids.shape
    if B > batch_chunk_size:
        log_probs_list = []
        for i in range(0, B, batch_chunk_size):
            end = min(i + batch_chunk_size, B)
            batch_input_ids = input_ids[i:end]
            batch_attention_mask = attention_mask[i:end]
            batch_labels = labels[i:end]
            
            # 递归调用处理子 batch
            batch_log_probs = compute_log_probs(
                model, batch_input_ids, batch_attention_mask, batch_labels, 
                chunk_size=chunk_size, batch_chunk_size=batch_chunk_size
            )
            log_probs_list.append(batch_log_probs)
        
        return torch.cat(log_probs_list, dim=0)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits  # [B, T, V]

    shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()      # [B, T-1]

    B, Tm1, V = shift_logits.shape
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    nll_chunks = []
    for s in range(0, Tm1, chunk_size):
        e = min(s + chunk_size, Tm1)
        sl = shift_logits[:, s:e, :].contiguous().view(-1, V)  # [(B*(e-s)), V]
        tl = shift_labels[:, s:e].contiguous().view(-1)        # [(B*(e-s))]
        nll = loss_fct(sl, tl).view(B, e - s)                  # [B, (e-s)]
        nll_chunks.append(nll)

    nll = torch.cat(nll_chunks, dim=1)  # [B, T-1]
    zeros = torch.zeros(B, 1, device=nll.device, dtype=nll.dtype)
    token_log_probs = torch.cat([zeros, -nll], dim=1)          # [B, T]
    return token_log_probs