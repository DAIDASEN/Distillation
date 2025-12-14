
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_loader import load_gsm8k
from reward_math import compute_reward_batch

def evaluate_model(model_name, dataset, num_samples=20, batch_size=4, device="cuda"):
    print(f"\nEvaluating model: {model_name}")
    print(f"Loading tokenizer and model...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    
    # Select subset
    subset = dataset.select(range(min(num_samples, len(dataset))))
    questions = [ex["question"] for ex in subset]  # GSM8K uses 'question' field
    answers = [ex["answer"] for ex in subset]
    
    all_rewards = []
    token_lengths = []
    
    print(f"Generating responses for {len(questions)} examples...")
    
    for i in tqdm(range(0, len(questions), batch_size)):
        batch_q = questions[i:i+batch_size]
        batch_a = answers[i:i+batch_size]
        
        messages_batch = []
        for q in batch_q:
            messages = [
                {"role": "system", "content": "Solve the math problem step by step. Show your reasoning, then give the final answer as: #### <number>"},
                {"role": "user", "content": q},
            ]
            messages_batch.append(messages)
            
        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
        
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=3072, 
                do_sample=False, # Greedy decoding for evaluation
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Calculate new tokens generated
        new_tokens = outputs[:, input_ids.shape[1]:]
        lengths = (new_tokens != tokenizer.pad_token_id).sum(dim=1).cpu().tolist()
        token_lengths.extend(lengths)
        
        decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        rewards = compute_reward_batch(decoded_texts, batch_a)
        all_rewards.extend(rewards.tolist())
        
    accuracy = np.mean(all_rewards)
    avg_len = np.mean(token_lengths)
    max_len = np.max(token_lengths)
    p95_len = np.percentile(token_lengths, 95)
    
    print(f"Results for {model_name}:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Avg Token Length: {avg_len:.1f}")
    print(f"  Max Token Length: {max_len}")
    print(f"  95th Percentile Length: {p95_len:.1f}")
    
    # Clean up to save memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return {
        "accuracy": accuracy,
        "avg_len": avg_len,
        "max_len": max_len,
        "p95_len": p95_len
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    ds = load_gsm8k(split="test")  # Use test split for evaluation
    
    student_model = "Qwen/Qwen2.5-0.5B"
    teacher_model = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Evaluate Student
    student_stats = evaluate_model(student_model, ds, num_samples=20, device=device)
    
    # Evaluate Teacher
    teacher_stats = evaluate_model(teacher_model, ds, num_samples=20, device=device)
    
    print("\n" + "="*40)
    print("FINAL COMPARISON (GSM8K)")
    print("="*40)
    print(f"{'Metric':<20} | {'Student':<20} | {'Teacher':<20}")
    print("-" * 66)
    print(f"{'Accuracy':<20} | {student_stats['accuracy']:.2%}             | {teacher_stats['accuracy']:.2%}")
    print(f"{'Avg Length':<20} | {student_stats['avg_len']:.1f}                 | {teacher_stats['avg_len']:.1f}")
    print(f"{'Max Length':<20} | {student_stats['max_len']:<20} | {teacher_stats['max_len']}")
    print(f"{'95% Length':<20} | {student_stats['p95_len']:.1f}                 | {teacher_stats['p95_len']:.1f}")
    print("="*40)

if __name__ == "__main__":
    main()
