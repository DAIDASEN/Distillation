
import os
import argparse
import json
import signal
import sys
from typing import List, Tuple

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig

from utils import set_seed, get_device, compute_log_probs
from data_loader import load_gsm8k, iter_gsm8k_batches
from reward_math import compute_reward_batch

# --- Helper Functions ---

def apply_chat_template_no_thinking(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

class AnswerStopper(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len, window=128):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len  # Track where the prompt ends
        self.window = window
        self.stop_strings = ["####", "\\boxed{"]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check only the GENERATED tokens (after prompt), not the prompt itself
        batch_size = input_ids.shape[0]
        finished = [False] * batch_size
        
        for i in range(batch_size):
            # Only check tokens AFTER the prompt
            generated_tokens = input_ids[i, self.prompt_len:]
            if generated_tokens.numel() == 0:
                # No tokens generated yet
                continue
            # Get last window tokens of generated part only
            check_tokens = generated_tokens[-self.window:] if generated_tokens.numel() > self.window else generated_tokens
            decoded = self.tokenizer.decode(check_tokens, skip_special_tokens=True)
            for s in self.stop_strings:
                if s in decoded:
                    finished[i] = True
                    break
        
        return all(finished)

# --- Main Logic ---

def build_prompts(questions):
    messages_batch = []
    for q in questions:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{q}\n\nPlease reason step by step, and put your final answer at the end in the format: #### <answer>"},
        ]
        messages_batch.append(messages)
    return messages_batch

def make_generation_inputs(tokenizer, messages_batch, k_samples):
    texts = []
    group_indices = []
    for i, msg in enumerate(messages_batch):
        # Use the new helper function
        text = apply_chat_template_no_thinking(tokenizer, msg)
        for _ in range(k_samples):
            texts.append(text)
            group_indices.append(i)
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    return enc["input_ids"], enc["attention_mask"], torch.tensor(group_indices)

def group_normalize_rewards(rewards, group_indices):
    norm_rewards = torch.zeros_like(rewards)
    for g in torch.unique(group_indices):
        idx = (group_indices == g)
        if idx.sum() > 1:
            r_group = rewards[idx]
            mean, std = r_group.mean(), r_group.std() + 1e-8
            norm_rewards[idx] = (r_group - mean) / std
        else:
            norm_rewards[idx] = rewards[idx]
    return norm_rewards

def generate_teacher_outputs(model, tokenizer, input_ids, attention_mask, max_new_tokens, device):
    """教师模型生成（每个问题只生成1个样本，用于对比）"""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_len = input_ids.shape[1]
    
    stopper = AnswerStopper(tokenizer, prompt_len=prompt_len, window=128)
    stopping_criteria = StoppingCriteriaList([stopper])
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for teacher
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria
        )
    
    decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_texts

def generate_student_rollouts(model, tokenizer, input_ids, attention_mask, max_new_tokens, device, args):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_len = input_ids.shape[1]
    
    # Setup stopping criteria - pass prompt_len so it only checks generated tokens
    stopper = AnswerStopper(tokenizer, prompt_len=prompt_len, window=128)
    stopping_criteria = StoppingCriteriaList([stopper])
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria
        )
    
    # Calculate metrics
    # 1. Truncation rate
    gen_ids = outputs[:, prompt_len:]
    if tokenizer.pad_token_id is not None:
        token_counts = (gen_ids != tokenizer.pad_token_id).sum(dim=1)
        truncation_mask = (token_counts == max_new_tokens)
        truncation_rate = truncation_mask.float().mean().item()
    else:
        # Fallback if no pad token
        gen_len = outputs.shape[1] - prompt_len
        truncation_rate = 1.0 if gen_len == max_new_tokens else 0.0
    
    # Decode for other metrics
    decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # 2. Answer found rate & 4. Avg new tokens to answer
    answer_found_count = 0
    tokens_to_answer_list = []
    
    # We need to find where the answer starts in the decoded text to estimate tokens
    # This is an approximation because mapping char index to token index is not 1:1
    # But we can re-tokenize the prefix.
    
    # Get prompt texts to subtract them? 
    # Actually decoded_texts includes prompt if we didn't slice outputs?
    # batch_decode(outputs) includes prompt.
    # But we want to check the *generated* part.
    # Let's decode only the generated part for easier analysis.
    generated_ids = outputs[:, prompt_len:]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    for i, text in enumerate(generated_texts):
        found = False
        stop_idx = -1
        
        # Check for ####
        if "####" in text:
            found = True
            stop_idx = text.find("####")
        elif "\\boxed{" in text:
            found = True
            stop_idx = text.find("\\boxed{")
            
        if found:
            answer_found_count += 1
            # Estimate tokens to answer
            # Take text up to stop_idx
            prefix_text = text[:stop_idx]
            # Tokenize prefix
            prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
            tokens_to_answer_list.append(len(prefix_ids))
            
    answer_found_rate = answer_found_count / len(generated_texts)
    avg_new_tokens_to_answer = sum(tokens_to_answer_list) / len(tokens_to_answer_list) if tokens_to_answer_list else 0.0
    
    # 3. Avg new tokens
    # Count non-pad tokens in generated part
    # Note: generated_ids might contain pad tokens if some finished early
    # But wait, model.generate with padding_side=left (tokenizer) usually pads on left?
    # No, generation output usually has padding on the right if batching?
    # Actually, if we use `pad_token_id`, generate might output pad tokens after EOS.
    # Let's count non-pad tokens.
    if tokenizer.pad_token_id is not None:
        new_token_counts = (generated_ids != tokenizer.pad_token_id).sum(dim=1)
    else:
        new_token_counts = torch.tensor([gen_len] * len(generated_texts))
        
    avg_new_tokens = new_token_counts.float().mean().item()
    
    metrics = {
        "truncation_rate": truncation_rate,
        "answer_found_rate": answer_found_rate,
        "avg_new_tokens": avg_new_tokens,
        "avg_new_tokens_to_answer": avg_new_tokens_to_answer
    }

    labels = outputs.clone()
    labels[:, :prompt_len] = -100
    # Note: We return the full decoded texts (including prompt) for reward computation if needed, 
    # or just generated? 
    # The original code used `tokenizer.batch_decode(outputs, ...)` which includes prompt.
    # Let's stick to that.
    full_decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return outputs, (outputs != tokenizer.pad_token_id).long(), labels, full_decoded_texts, metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--k_samples", type=int, default=8)
    # C.1 Change default max_new_tokens
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lambda_env", type=float, default=1.0)
    parser.add_argument("--lambda_kl", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="runs/qwen2.5_gsm8k_distill")
    
    # F. Generation parameters
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log file definition
    log_file_path = os.path.join(args.output_dir, "training_log.jsonl")
    print(f"Logging metrics to {log_file_path}")

    set_seed(42)
    device = get_device()

    # 1. Tokenizer (CRITICAL FIX: padding_side='left')
    print("Loading tokenizer with padding_side='left'...")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Models
    print("Loading models...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        dtype=torch.bfloat16,
    ).to(device)
    student.gradient_checkpointing_enable()
    student.config.use_cache = False

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        quantization_config=bnb_cfg,
        device_map="cuda",
    )
    teacher.eval()
    teacher.config.use_cache = False
    for p in teacher.parameters():
        p.requires_grad_(False)
    
    optimizer = AdamW(student.parameters(), lr=args.lr)
    
    # Load GSM8K dataset
    ds = load_gsm8k()
    iter_batches = iter_gsm8k_batches
    
    student.train()
    global_step = 0

    # --- Signal Handler for Graceful Shutdown ---
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}. Saving checkpoint at step {global_step}...")
        save_path = os.path.join(args.output_dir, f"ckpt_interrupted_{global_step}")
        student.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Checkpoint saved to {save_path}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    pbar = tqdm(total=args.max_steps)

    while global_step < args.max_steps:
        for questions, answers in iter_batches(ds, args.batch_size):
            if global_step >= args.max_steps: break
            
            # --- Sampling ---
            msgs = build_prompts(questions)
            in_ids, att_mask, g_idx = make_generation_inputs(tokenizer, msgs, args.k_samples)
            
            full_ids, full_mask, labels, texts, gen_metrics = generate_student_rollouts(
                student, tokenizer, in_ids, att_mask, args.max_new_tokens, device, args
            )
            
            # --- Reward ---
            ref_ans_expanded = [answers[i] for i in g_idx.tolist()]
            raw_rewards = compute_reward_batch(texts, ref_ans_expanded).to(device)
            norm_rewards = group_normalize_rewards(raw_rewards, g_idx.to(device))
            
            # --- Teacher Evaluation (每10步评估一次教师，减少开销) ---
            teacher_acc = 0.0
            if global_step % 10 == 0:
                # 教师只对原始问题生成（不扩展k_samples）
                teacher_in_ids, teacher_att_mask, _ = make_generation_inputs(tokenizer, msgs, k_samples=1)
                teacher_texts = generate_teacher_outputs(
                    teacher, tokenizer, teacher_in_ids, teacher_att_mask, args.max_new_tokens, device
                )
                teacher_rewards = compute_reward_batch(teacher_texts, answers)
                teacher_acc = teacher_rewards.mean().item()

                # Save teacher samples
                teacher_samples_file = os.path.join(args.output_dir, "teacher_samples.jsonl")
                for i, (q, pred, ref, reward) in enumerate(zip(questions, teacher_texts, answers, teacher_rewards.tolist())):
                    sample_entry = {
                        "step": global_step,
                        "sample_idx": i,
                        "question": q,
                        "prediction": pred,
                        "reference": ref,
                        "correct": reward == 1.0,
                        "model": "teacher"
                    }
                    with open(teacher_samples_file, "a") as f:
                        f.write(json.dumps(sample_entry, ensure_ascii=False) + "\n")
            
            # --- Distillation ---
            with torch.inference_mode():
                # Teacher inference (chunked internally by compute_log_probs if needed)
                logp_teacher = compute_log_probs(teacher, full_ids, full_mask, labels, batch_chunk_size=4)
            
            # Gradient Accumulation Loop
            optimizer.zero_grad()
            
            valid_mask_all = (labels != -100).float()
            total_valid_tokens = valid_mask_all.sum()
            
            # Use small mini-batch size to save memory during backward
            mini_batch_size = 2 
            total_loss = 0.0
            
            B_total = full_ids.shape[0]
            for i in range(0, B_total, mini_batch_size):
                end = min(i + mini_batch_size, B_total)
                
                # Slice batch
                mb_ids = full_ids[i:end]
                mb_mask = full_mask[i:end]
                mb_labels = labels[i:end]
                mb_logp_teacher = logp_teacher[i:end]
                mb_norm_rewards = norm_rewards[i:end]
                
                # Student Forward
                mb_logp_student = compute_log_probs(student, mb_ids, mb_mask, mb_labels)
                
                # Advantage
                mb_r_kl = mb_logp_teacher - mb_logp_student.detach()
                mb_r_env = mb_norm_rewards.unsqueeze(-1).expand_as(mb_logp_student)
                mb_advantage = args.lambda_env * mb_r_env + args.lambda_kl * mb_r_kl
                
                # Loss
                mb_valid_mask = (mb_labels != -100).float()
                # Normalize by global token count to match batch loss definition
                mb_loss = - (mb_advantage * mb_logp_student * mb_valid_mask).sum() / (total_valid_tokens + 1e-8)
                
                # Backward
                mb_loss.backward()
                
                total_loss += mb_loss.item()
            
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            
            # --- Logging & Monitoring ---
            avg_acc = raw_rewards.mean().item()
            curr_loss = total_loss
            
            global_step += 1
            pbar.update(1)
            
            # Update postfix with new metrics (显示教师acc仅在评估步)
            postfix = {
                "loss": f"{curr_loss:.4f}",
                "stu": f"{avg_acc:.2f}",
                "tea": f"{teacher_acc:.2f}" if global_step % 10 == 0 else "-",
                "ans": f"{gen_metrics['answer_found_rate']:.2f}",
                "len": f"{gen_metrics['avg_new_tokens']:.0f}"
            }
            pbar.set_postfix(postfix)
            
            # Write to JSONL for plotting
            log_entry = {
                "step": global_step,
                "loss": curr_loss,
                "student_acc": avg_acc,
                "teacher_acc": teacher_acc if global_step % 10 == 0 else None,
                **gen_metrics
            }
            with open(log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            # Save samples periodically (every 10 steps) or when answer rate is high
            if global_step % 10 == 0 or gen_metrics['answer_found_rate'] >= 0.5:
                samples_file = os.path.join(args.output_dir, "samples.jsonl")
                # Get questions for this batch
                batch_questions = [questions[i] for i in g_idx.tolist()]
                for i, (q, pred, ref, reward) in enumerate(zip(batch_questions, texts, ref_ans_expanded, raw_rewards.tolist())):
                    sample_entry = {
                        "step": global_step,
                        "sample_idx": i,
                        "question": q,
                        "prediction": pred,
                        "reference": ref,
                        "correct": reward == 1.0,
                        "answer_found_rate": gen_metrics['answer_found_rate']
                    }
                    with open(samples_file, "a") as f:
                        f.write(json.dumps(sample_entry, ensure_ascii=False) + "\n")

            if global_step % 200 == 0:
                 student.save_pretrained(os.path.join(args.output_dir, f"ckpt_{global_step}"))
                 tokenizer.save_pretrained(os.path.join(args.output_dir, f"ckpt_{global_step}"))

    student.save_pretrained(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

if __name__ == "__main__":
    main()
