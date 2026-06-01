import json
import argparse
import time
import os
import re
import uuid
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from sandbox import GLOBAL_SANDBOX_MANAGER

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "A tool for executing code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to execute."
                    }
                },
                "required": ["code"]
            }
        }
    }
]

QWEN_AGENT_SYSTEM_PROMPT = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_json}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
\nThe answer format must be: \\boxed{{'The final answer goes here.'}}
"""

REGEX_TOOL_XML = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)


def extract_tool_call(text: str) -> Optional[Dict]:
    if not text:
        return None

    matches = REGEX_TOOL_XML.findall(text)
    if not matches:
        return None

    json_str = matches[-1]

    try:
        data = json.loads(json_str, strict=False)
        return _parse_tool_payload(data)
    except Exception as e:
        try:
            import ast
            data = ast.literal_eval(json_str)
            return _parse_tool_payload(data)
        except:
            return None


def _parse_tool_payload(data: Any) -> Optional[Dict]:
    if not isinstance(data, dict):
        return None

    if data.get("name") == "code_interpreter":
        args = data.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except:
                pass

        if isinstance(args, dict) and "code" in args:
            return {"code": args["code"]}

    return None


def execute_request_code(req_data: Dict, code: str, timeout: float) -> Dict:
    req_id = req_data['original_item']['id']

    result = GLOBAL_SANDBOX_MANAGER.run_code(req_id, code, timeout=timeout)

    max_obs_len = 2000
    if len(result) > max_obs_len:
        result = result[:max_obs_len] + f"\n... (truncated, total {len(result)} chars)"

    tool_msg = {
        "role": "user",
        "content": f"<tool_response>\n{result}\n</tool_response>"
    }

    req_data['messages'].append(tool_msg)
    req_data['turn_count'] += 1
    return req_data


def main():
    parser = argparse.ArgumentParser(description="Schema-based Tool Inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, help="Tokenizer path")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)

    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)

    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument("--max-tool-turns", type=int, default=16)
    parser.add_argument("--tool-timeout", type=float, default=10.0)
    parser.add_argument("--max-concurrent-tools", type=int, default=32)
    parser.add_argument("--num-responses", "-n", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=50)

    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f">>> Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

    stop_tokens = ["</tool_call>", "<|im_end|>"]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=stop_tokens,
        logprobs=1
    )

    tools_list_str = "\n".join([json.dumps(t, ensure_ascii=False) for t in TOOLS_SCHEMA])
    system_prompt_content = QWEN_AGENT_SYSTEM_PROMPT.format(tools_json=tools_list_str)

    active_requests = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            if "id" not in item: item["id"] = str(uuid.uuid4())

            msgs = []
            has_system = False
            if "conversations" in item:
                for m in item["conversations"]:
                    if m['role'] == 'system':
                        m['content'] = system_prompt_content
                        has_system = True
                        break

            if not has_system:
                msgs.append({"role": "system", "content": system_prompt_content})

            if "conversations" in item:
                for m in item["conversations"]:
                    if m['role'] != 'system':
                        msgs.append(m)

            for m in msgs:
                if "token_count" not in m:
                    m["token_count"] = len(tokenizer.encode(m["content"]))

            active_requests.append({
                "original_item": item,
                "messages": msgs,
                "turn_count": 0,
                "final_response": None
            })

    completed_requests = []
    print(f">>> Loaded {len(active_requests)} requests.")

    current_turn = 0

    while active_requests and current_turn < args.max_tool_turns:
        iter_start = time.time()
        print(f"\n=== Turn {current_turn + 1}/{args.max_tool_turns} | Active: {len(active_requests)} ===")

        prompts = []
        valid_indices = []

        for i, req in enumerate(active_requests):
            try:
                p = tokenizer.apply_chat_template(req["messages"], tokenize=False, add_generation_prompt=True)
                prompts.append(p)
                valid_indices.append(i)
            except Exception as e:
                print(f"[Error] Template failed for ID {req['original_item']['id']}: {e}")
                req["error"] = str(e)
                completed_requests.append(req)

        if not prompts:
            break

        outputs = llm.generate(prompts, sampling_params)

        execution_tasks = []
        next_cycle_candidates = []
        finished_in_this_round = []

        for i, vllm_out in enumerate(outputs):
            idx = valid_indices[i]
            req = active_requests[idx]

            generated_text = vllm_out.outputs[0].text
            finish_reason = vllm_out.outputs[0].finish_reason
            token_ids = vllm_out.outputs[0].token_ids
            logprobs = vllm_out.outputs[0].logprobs

            if finish_reason == "stop" and not generated_text.strip().endswith("</tool_call>"):
                if "<tool_call>" in generated_text:
                    generated_text += "</tool_call>"

            entropies = []
            if logprobs:
                for tid, lp_dict in zip(token_ids, logprobs):
                    if lp_dict and tid in lp_dict:
                        # Logprob通常为负数，熵值为负的对数概率
                        entropies.append(-lp_dict[tid].logprob)

            avg_native_entropy = sum(entropies) / len(entropies) if entropies else 0.0
            token_count = len(token_ids)

            req["messages"].append({
                "role": "assistant",
                "content": generated_text,
                "token_count": token_count,
                "avg_native_entropy": avg_native_entropy
            })

            tool_payload = extract_tool_call(generated_text)

            if tool_payload and "code" in tool_payload:
                execution_tasks.append((req, tool_payload["code"]))
            else:
                req["final_response"] = generated_text
                req["finish_reason"] = "stop"
                finished_in_this_round.append(req)

        completed_requests.extend(finished_in_this_round)

        if execution_tasks:
            print(f">>> Executing tool calls for {len(execution_tasks)} requests...")

            with ThreadPoolExecutor(max_workers=args.max_concurrent_tools) as executor:
                future_to_req = {
                    executor.submit(execute_request_code, req, code, args.tool_timeout): req
                    for req, code in execution_tasks
                }

                for future in as_completed(future_to_req):
                    req = future.result()

                    # === [修改] 工具执行完毕后，为刚才追加进去的工具消息计算 token_count ===
                    last_msg = req["messages"][-1]
                    if last_msg["role"] == "user" and "token_count" not in last_msg:
                        last_msg["token_count"] = len(tokenizer.encode(last_msg["content"]))

                    if req["turn_count"] < args.max_tool_turns:
                        next_cycle_candidates.append(req)
                    else:
                        req["finish_reason"] = "max_turns"
                        req["final_response"] = req["messages"][-1]["content"]
                        completed_requests.append(req)

        active_requests = next_cycle_candidates
        print(f">>> Turn cost: {time.time() - iter_start:.2f}s. Total completed: {len(completed_requests)}")
        current_turn += 1

    GLOBAL_SANDBOX_MANAGER.close_all()

    # 5. 保存
    output_data = []
    for req in completed_requests:
        item = {
            "id": req["original_item"]["id"],
            "history": req["messages"]
        }

        for k, v in req["original_item"].items():
            if k not in item and k != "conversations":
                item[k] = v

        output_data.append(item)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f">>> Finished. Saved to {args.output_file}")


if __name__ == "__main__":
    main()