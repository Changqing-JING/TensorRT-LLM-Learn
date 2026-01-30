import time
import torch
from transformers import AutoTokenizer

import tensorrt_llm
from tensorrt_llm.runtime import ModelRunnerCpp

if __name__ == '__main__':
    engine_dir = "./qwen2.5_1.5b_engine"
    tokenizer_dir = "./Qwen2.5-1.5B-Instruct"

    print(f"Loading TensorRT engine from: {engine_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    end_id = tokenizer.eos_token_id
    
    # Load the TensorRT engine using ModelRunnerCpp
    batch_size = 1
    runner = ModelRunnerCpp.from_dir(
        engine_dir=engine_dir,
        rank=0,
        max_output_len=128,
    )
    
    # Prepare input with chat template
    prompt = "What's new in C++ 20"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    batch_input_ids = [torch.tensor(input_ids, dtype=torch.int32)]
    
    print(f"Input: {prompt}")
    print(f"Formatted prompt:\n{formatted_prompt}")
    print(f"Input token count: {len(input_ids)}")
    
    # Warmup
    print("\nWarmup...")
    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=batch_input_ids,
            max_new_tokens=128,
            end_id=end_id,
            pad_id=pad_id,
            temperature=0.8,
            top_p=0.95,
            return_dict=True,
            output_sequence_lengths=True,
        )
    
    # Benchmark
    print("\nRunning benchmark...")
    num_runs = 5
    total_tokens = 0
    start_time = time.time()

    for i in range(num_runs):
        with torch.no_grad():
            outputs = runner.generate(
                batch_input_ids=batch_input_ids,
                max_new_tokens=128,
                end_id=end_id,
                pad_id=pad_id,
                temperature=0.8,
                top_p=0.95,
                return_dict=True,
                output_sequence_lengths=True,
            )
        seq_len = outputs['sequence_lengths'][0][0].item()
        total_tokens += seq_len - len(input_ids)

    elapsed = time.time() - start_time
    throughput = total_tokens / elapsed

    print(f"\n=== TENSORRT ENGINE RESULTS (INT4 AWQ, Batch Size 1) ===")
    print(f"Total runs: {num_runs}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} tokens/sec")

    # Print sample output
    output_ids = outputs['output_ids'][0][0].tolist()
    output_text = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True)
    print(f"\nSample output:")
    print(output_text)
