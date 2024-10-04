import argparse
import json
import time
import asyncio
from typing import List, Dict, Any

import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import DEVICE_OPTIONS
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.utils import FlexibleArgumentParser

def read_json_input(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_metrics(input_requests: List[Dict[str, Any]], 
                      outputs: List[Dict[str, Any]], 
                      total_time: float,
                      baseline_latency_per_token: float = 0.1) -> Dict[str, float]:
    total_requests = len(input_requests)
    slo_attained = 0
    slo_output_tokens = 0
    total_output_tokens = 0

    for request, output in zip(input_requests, outputs):
        input_tokens = request['input_length']
        output_tokens = len(output['generated_text'].split())
        total_output_tokens += output_tokens
        
        decode_time = output['latency'] - (input_tokens * baseline_latency_per_token)
        slo_target = request['slo_ratio'] * baseline_latency_per_token * output_tokens * 1000
        
        if decode_time * 1000 <= slo_target:
            slo_attained += 1
            slo_output_tokens += input_tokens + output_tokens

    slo_attainment = slo_attained / total_requests
    goodput = slo_output_tokens / total_time

    return {
        "slo_attainment": slo_attainment,
        "goodput": goodput,
        "total_output_tokens": total_output_tokens,
        "total_time": total_time
    }

async def process_batch(llm, batch, start_time):
    prompts = [request['prompt'] for request in batch]
    max_tokens = max(request['output_length'] for request in batch)
    
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=max_tokens,
    )
    
    st = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end = time.perf_counter()
    
    return [{
        "generated_text": output.outputs[0].text,
        "latency": end - st
    } for output in outputs]

async def main(args: argparse.Namespace):
    print(args)

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        device=args.device,
    )

    input_requests = read_json_input(args.input_file)
    input_requests.sort(key=lambda x: x['emission_time_ms'])  # Sort by emission time

    # Warm-up phase
    print("Warming up...")
    dummy_prompt = "This is a dummy prompt for warm-up."
    dummy_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=10)
    for _ in range(5):
        llm.generate([dummy_prompt], dummy_params)

    start_time = time.perf_counter()
    outputs = []
    current_batch = []
    last_emission_time = 0

    for request in tqdm(input_requests, desc="Processing requests"):
        # Wait until the emission time
        wait_time = (request['emission_time_ms'] - last_emission_time) / 1000
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        current_batch.append(request)
        
        # Process the batch if it's full or if it's the last request
        if len(current_batch) == args.batch_size or request == input_requests[-1]:
            batch_outputs = await process_batch(llm, current_batch, start_time)
            outputs.extend(batch_outputs)
            current_batch = []
        
        last_emission_time = request['emission_time_ms']

    end_time = time.perf_counter()
    total_time = end_time - start_time

    metrics = calculate_metrics(input_requests, outputs, total_time, args.baseline_latency_per_token)

    print(f"SLO Attainment: {metrics['slo_attainment']:.4f}")
    print(f"Goodput: {metrics['goodput']:.2f} tokens/second")
    print(f"Total output tokens: {metrics['total_output_tokens']}")
    print(f"Total time: {metrics['total_time']:.2f} seconds")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    parser = FlexibleArgumentParser(description='Benchmark using emission.json input file')
    parser.add_argument('--input-file', type=str, required=True, help='Path to emission.json input file')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization', '-q', choices=[*QUANTIZATION_METHODS, None], default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--max-model-len', type=int, default=None)
    parser.add_argument('--dtype', type=str, default='auto', 
                        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'])
    parser.add_argument('--enforce-eager', action='store_true')
    parser.add_argument('--device', type=str, default="cuda", choices=DEVICE_OPTIONS)
    parser.add_argument('--output-json', type=str, default='vllm_results.json', 
                        help='Path to save the results in JSON format')
    parser.add_argument('--baseline-latency-per-token', type=float, default=0.1,
                        help='Baseline latency per token in seconds')
    parser.add_argument('--batch-size', type=int, default=8, help='Maximum batch size for processing requests')
    
    args = parser.parse_args()
    asyncio.run(main(args))