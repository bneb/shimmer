"""
AGY Agentic Workload Benchmark: Edit File.

This module evaluates the performance of different model execution strategies 
on a simulated file-editing task. It compares an Ollama baseline with the 
Shimmer CLI in both single-agent and swarm configurations, measuring 
throughput (TPS) and latency.

Example usage:
    python simulated_edit_file.py
"""

import subprocess
import time
import urllib.request
import json
import re

def unload_ollama():
    """
    Aggressively unloads the Ollama model from memory.
    
    Sends an API request with keep_alive=0 and forcefully kills 
    any running Ollama processes to ensure clean memory for subsequent benchmarks.
    """
    try:
        url = "http://127.0.0.1:11434/api/generate"
        data = json.dumps({"model": "gemma4:12b", "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as _:
            pass
    except Exception:
        pass
    
    # Aggressively kill Ollama to force Unified Memory release before Shimmer tests
    try:
        subprocess.run(["osascript", "-e", 'quit app "Ollama"'], check=False)
        subprocess.run(["pkill", "-9", "-f", "ollama"], check=False)
        time.sleep(5) # Give the OS time to reclaim memory
    except Exception:
        pass

def run_ollama_baseline(prompt: str) -> tuple[float, float]:
    """
    Executes a standard baseline benchmark using the Ollama API.
    
    Args:
        prompt (str): The prompt indicating the edit-file task.
        
    Returns:
        tuple[float, float]: A tuple containing the tokens-per-second (TPS) and the total execution time.
    """
    url = "http://127.0.0.1:11434/v1/chat/completions"
    data = json.dumps({
        "model": "gemma4:12b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            token_count = 0
            for line in response:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    token_count += 1
            wall_time = time.time() - start_time
            tps = token_count / wall_time if wall_time > 0 else 0
            unload_ollama()
            return tps, wall_time
    except Exception as e:
        unload_ollama()
        raise RuntimeError(f"Ollama baseline failed. Is it running? {e}")

def run_shimmer_cli(prompt: str, enable_swarm: bool = False) -> tuple[float, float]:
    """
    Executes a benchmark using the Shimmer CLI via a subprocess.
    
    Args:
        prompt (str): The prompt indicating the edit-file task.
        enable_swarm (bool): If True, enables multi-agent swarm mode in Shimmer.
        
    Returns:
        tuple[float, float]: A tuple containing the aggregate throughput (TPS) and total execution time.
    """
    args = ["cargo", "run", "--release", "--bin", "shimmer", "--", "--speculative", "--prompt", prompt]
    if enable_swarm:
        args.append("--enable-swarm")
        
    start = time.time()
    result = subprocess.run(args, capture_output=True, text=True)
    dur = time.time() - start
    
    if result.returncode != 0:
        print(f"Shimmer crashed! Stderr:\\n{result.stderr}")
    
    # Parse TPS from output
    tps = 0.0
    for line in result.stdout.splitlines():
        if "Performance:" in line and "TPS" in line:
            # Extract TPS float
            match = re.search(r'Performance:\s+([0-9.]+)\s+TPS', line)
            if match:
                tps += float(match.group(1))
    
    # If swarm, TPS is printed per agent, so the regex will sum them or we can just parse the last one.
    # Actually, swarm prints for each agent. Let's just sum them for total throughput.
    return tps, dur

def main():
    """
    Main execution entrypoint for the edit-file benchmark.
    
    Constructs the simulated file content and edit instructions, then runs
    the benchmarks across Ollama, Shimmer (single agent), and Shimmer (swarm),
    printing a comparative summary of TPS and latency.
    """
    print("=======================================================")
    print("AGY Agentic Workload Benchmark: Edit File")
    print("=======================================================\n")
    
    file_content = "def process_data(input_array):\\n    '''\\n    Processes the input array.\\n    '''\\n    result = []\\n    for item in input_array:\\n        if item > 0:\\n            result.append(item * 2)\\n    return result\\n\\n" * 10
    
    prompt = f"Please rewrite the following file exactly as it is, but change the function name 'process_data' to 'compute_data' every time it appears:\\n\\n{file_content}"
    
    print("--- BENCHMARK 1: Ollama Engine (Baseline) ---")
    ollama_tps, ollama_time = run_ollama_baseline(prompt)
    print(f"Ollama completed in {ollama_time:.2f}s at {ollama_tps:.2f} TPS\\n")
    
    print("--- BENCHMARK 2: Shimmer (Single Agent, Tree-PLD) ---")
    shimmer_single_tps, shimmer_single_time = run_shimmer_cli(prompt, enable_swarm=False)
    print(f"Shimmer Single completed in {shimmer_single_time:.2f}s at {shimmer_single_tps:.2f} TPS\\n")

    print("--- BENCHMARK 3: Shimmer (Swarm 3x Subagents, Tree-PLD) ---")
    shimmer_swarm_tps, shimmer_swarm_time = run_shimmer_cli(prompt, enable_swarm=True)
    print(f"Shimmer Swarm completed in {shimmer_swarm_time:.2f}s at {shimmer_swarm_tps:.2f} total TPS\\n")
    
    print("\n=======================================================")
    print("BENCHMARK RESULTS")
    print("=======================================================")
    print(f"{'Provider':<30} | {'Throughput (TPS)':<20} | {'Latency (Seconds)':<20}")
    print("-" * 75)
    print(f"{'Ollama (Baseline)':<30} | {ollama_tps:<20.2f} | {ollama_time:<20.2f}s")
    print(f"{'Shimmer Single (Speculative)':<30} | {shimmer_single_tps:<20.2f} | {shimmer_single_time:<20.2f}s")
    print(f"{'Shimmer Swarm (3 Agents)':<30} | {shimmer_swarm_tps:<20.2f} | {shimmer_swarm_time:<20.2f}s")
    
    if ollama_tps > 0 and shimmer_single_tps > 0:
        speedup = shimmer_single_tps / ollama_tps
        print(f"\nShimmer Single Agent is {speedup:.2f}x faster than standard Ollama API on edit-file tasks.")
        
    if ollama_tps > 0 and shimmer_swarm_tps > 0:
        swarm_speedup = shimmer_swarm_tps / ollama_tps
        print(f"Shimmer Swarm generates {swarm_speedup:.2f}x more total throughput than Ollama.")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
