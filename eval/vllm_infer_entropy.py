import json
import argparse
import time
import os
from vllm import LLM, SamplingParams
import torch.distributed as dist
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="vLLM Inference with Top-X% Average Entropy calculation")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)

    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--block-size", type=int, default=128)

    parser.add_argument("--num-responses", "-n", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)

    args = parser.parse_args()

    is_master = not dist.is_initialized() or dist.get_rank() == 0

    if is_master:
        print(f">>> Loading model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)

        original_data = []
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    original_data.append(json.loads(line))

        prompts_to_process = []
        for item in original_data:
            messages = []
            if "system" in item and item["system"]:
                messages.append({"role": "system", "content": item["system"]})

            convs = item.get("conversations", item.get("messages", []))
            messages.extend(convs)

            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts_to_process.append(prompt)
            except Exception as e:
                print(f"Warning: Template error for item {item.get('id', 'unknown')}: {e}")
                prompts_to_process.append("")
    else:
        prompts_to_process = []

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        block_size=args.block_size
    )

    sampling_params = SamplingParams(
        n=args.num_responses,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        logprobs=1  # 必须开启以获取概率
    )

    start_time = time.time()
    outputs = llm.generate(prompts_to_process, sampling_params)

    if is_master:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output_file, 'w', encoding='utf-8') as f_out:
            for i, output in enumerate(outputs):
                if i >= len(original_data): break

                item_copy = original_data[i].copy()
                res_list = []

                avg_ent_list = []
                top01_avg_list = []
                top02_avg_list = []
                top03_avg_list = []
                top04_avg_list = []
                top05_avg_list = []
                top10_avg_list = []
                top20_avg_list = []
                top30_avg_list = []
                top40_avg_list = []
                top50_avg_list = []

                for res in output.outputs:
                    res_list.append(res.text)

                    token_entropies = []
                    if res.logprobs:
                        for tid, lp_dict in zip(res.token_ids, res.logprobs):
                            if lp_dict and tid in lp_dict:
                                token_entropies.append(-lp_dict[tid].logprob)

                    if token_entropies:
                        avg_ent = sum(token_entropies) / len(token_entropies)

                        sorted_ent_desc = sorted(token_entropies, reverse=True)
                        seq_len = len(sorted_ent_desc)

                        def get_top_p_avg(p):
                            k = int(seq_len * p)
                            if k == 0:
                                k = 1
                            top_k_elements = sorted_ent_desc[:k]
                            return sum(top_k_elements) / len(top_k_elements)

                        top01_avg = get_top_p_avg(0.01)
                        top02_avg = get_top_p_avg(0.02)
                        top03_avg = get_top_p_avg(0.03)
                        top04_avg = get_top_p_avg(0.04)
                        top05_avg = get_top_p_avg(0.05)
                        top10_avg = get_top_p_avg(0.1)
                        top20_avg = get_top_p_avg(0.2)
                        top30_avg = get_top_p_avg(0.3)
                        top40_avg = get_top_p_avg(0.4)
                        top50_avg = get_top_p_avg(0.5)

                    else:
                        avg_ent = 0.0
                        top01_avg = 0.0
                        top02_avg = 0.0
                        top03_avg = 0.0
                        top04_avg = 0.0
                        top05_avg = 0.0
                        top10_avg = 0.0
                        top20_avg = 0.0
                        top30_avg = 0.0
                        top40_avg = 0.0
                        top50_avg = 0.0

                    avg_ent_list.append(avg_ent)
                    top01_avg_list.append(top01_avg)
                    top02_avg_list.append(top02_avg)
                    top03_avg_list.append(top03_avg)
                    top04_avg_list.append(top04_avg)
                    top05_avg_list.append(top05_avg)
                    top10_avg_list.append(top10_avg)
                    top20_avg_list.append(top20_avg)
                    top30_avg_list.append(top30_avg)
                    top40_avg_list.append(top40_avg)
                    top50_avg_list.append(top50_avg)

                item_copy["responses"] = res_list
                item_copy["avg_native_entropies"] = avg_ent_list
                item_copy["top01_avg_entropies"] = top01_avg_list
                item_copy["top02_avg_entropies"] = top02_avg_list
                item_copy["top03_avg_entropies"] = top03_avg_list
                item_copy["top04_avg_entropies"] = top04_avg_list
                item_copy["top05_avg_entropies"] = top05_avg_list
                item_copy["top10_avg_entropies"] = top10_avg_list
                item_copy["top20_avg_entropies"] = top20_avg_list
                item_copy["top30_avg_entropies"] = top30_avg_list
                item_copy["top40_avg_entropies"] = top40_avg_list
                item_copy["top50_avg_entropies"] = top50_avg_list

                f_out.write(json.dumps(item_copy, ensure_ascii=False) + "\n")

        print(f">>> Finished. Saved to {args.output_file}")
        print(f">>> Cost time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()