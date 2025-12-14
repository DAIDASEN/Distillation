
import re
import torch

def extract_answer(text: str) -> str:
    """从文本中提取最终数字答案"""
    if "####" not in text:
        return ""
    # Take the last segment after ####
    last_segment = text.split("####")[-1]
    # Find the first number in this segment
    # Allow for some text before the number, e.g. "The answer is 72"
    # Also handle comma in numbers like 1,000 (remove comma first?)
    # GSM8K usually has clean numbers, but let's be safe.
    # Simple regex for float/int
    match = re.search(r"(-?\d+(\.\d+)?)", last_segment)
    if match:
        return match.group(1).strip()
    return ""

def compute_reward_batch(model_outputs, ref_answers):
    """
    比较模型输出和参考答案。
    Returns: Tensor [B], 1.0 (正确) or 0.0 (错误)
    """
    rewards = []
    for out_str, ref_str in zip(model_outputs, ref_answers):
        # Handle reference answer: if it has ####, extract it; otherwise assume it's clean
        if "####" in ref_str:
            ref_ans = extract_answer(ref_str)
        else:
            ref_ans = ref_str.strip()
            
        pred_ans = extract_answer(out_str)   # 从模型生成结果提取
        
        # 简单全匹配 (String Exact Match)
        # Remove commas for comparison (e.g. 1,000 vs 1000)
        ref_clean = ref_ans.replace(",", "")
        pred_clean = pred_ans.replace(",", "")
        
        if ref_clean != "" and pred_clean != "" and ref_clean == pred_clean:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return torch.tensor(rewards, dtype=torch.float32)
