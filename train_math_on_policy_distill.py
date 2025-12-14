
import os
import argparse
import json
from typing import List, Tuple

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig

from utils import set_seed, get_device, compute_log_probs
from data_loader import load_deepscaler, iter_deepscaler_batches
from reward_math import compute_reward_batch

# --- Helper Functions ---

def apply_chat_template_no_thinking(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

class AnswerStopper(StoppingCriteria):
    def __init__(self, tokenizer, window=128):
        self.tokenizer = tokenizer
        self.window = window
        self.stop_strings = ["####", "\\boxed{"]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check only the last 'window' tokens for efficiency
        # We check if ALL sequences in the batch have encountered the stop string.
        
        batch_size = input_ids.shape[0]
        finished = [False] * batch_size
        
        for i in range(batch_size):
            # Get last window tokens
            last_tokens = input_ids[i, -self.window:]
            decoded = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
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
            {"role": "system", "content": "You are a math problem solving assistant.\n\nRequirements:\n1) Keep your reasoning concise (no more than 20 lines).\n2) You MUST output the final answer on a separate line in the exact format: \"#### <final_number>\".\n3) After printing the \"####\" line, STOP immediately and output nothing else."},
            {"role": "user", "content": q},
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

def generate_student_rollouts(model, tokenizer, input_ids, attention_mask, max_new_tokens, device, args):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_len = input_ids.shape[1]
    
    # Setup stopping criteria
    stopper = AnswerStopper(tokenizer, window=128)
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
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen3-1.7B")
    # A.1 Change default teacher model
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--k_samples", type=int, default=4)
    # C.1 Change default max_new_tokens
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lambda_env", type=float, default=1.0)
    parser.add_argument("--lambda_kl", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="runs/qwen3_distill")
    
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
    
    # Load DeepScaleR dataset
    ds = load_deepscaler()
    iter_batches = iter_deepscaler_batches
    
    student.train()
    global_step = 0
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
            
            # --- Distillation ---
            with torch.inference_mode():
                logp_teacher = compute_log_probs(teacher, full_ids, full_mask, labels)
            
            logp_student = compute_log_probs(student, full_ids, full_mask, labels)
            
            # Advantage = Env + KL
            r_kl = logp_teacher - logp_student.detach()
            r_env = norm_rewards.unsqueeze(-1).expand_as(logp_student)
            advantage = args.lambda_env * r_env + args.lambda_kl * r_kl
            
            valid_mask = (labels != -100).float()
            loss = - (advantage * logp_student * valid_mask).sum() / valid_mask.sum()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            
            # --- Logging & Monitoring ---
            avg_acc = raw_rewards.mean().item()
            curr_loss = loss.item()
            
            global_step += 1
            pbar.update(1)
            
            # Update postfix with new metrics
            postfix = {
                "loss": f"{curr_loss:.4f}",
                "acc": f"{avg_acc:.2f}",
                "trunc": f"{gen_metrics['truncation_rate']:.2f}",
                "ans_rate": f"{gen_metrics['answer_found_rate']:.2f}",
                "len": f"{gen_metrics['avg_new_tokens']:.1f}"
            }
            pbar.set_postfix(postfix)
            
            # Write to JSONL for plotting
            log_entry = {
                "step": global_step,
                "loss": curr_loss,
                "accuracy": avg_acc,
                **gen_metrics
            }
            with open(log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if global_step % 200 == 0:
                 student.save_pretrained(os.path.join(args.output_dir, f"ckpt_{global_step}"))
                 tokenizer.save_pretrained(os.path.join(args.output_dir, f"ckpt_{global_step}"))

    student.save_pretrained(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

if __name__ == "__main__":
    main()
