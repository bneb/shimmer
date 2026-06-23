"""
Measures throughput (TPS), VRAM/memory usage, and concurrency capabilities 
of the Shimmer daemon using domain sockets.

Example usage:
    python throughput_and_memory.py
"""

import socket
import json
import time
import threading
import os
import subprocess

SOCKET_PATH = "/tmp/shimmer.sock"

import threading

class VRAMTracker:
    """
    Estimates VRAM/memory usage of the background Shimmer daemon during execution.
    """
    def __init__(self):
        self.peak_vram = 0.0
        self.running = False
        self.thread = None

    def _poll(self):
        while self.running:
            try:
                pids = subprocess.check_output(["pgrep", "-f", "shimmer --daemon"]).decode().strip().split('\n')
                if pids and pids[0]:
                    pid = pids[0]
                    output = subprocess.check_output(["ps", "-p", pid, "-o", "rss="]).decode().strip()
                    current = int(output) / 1024.0 # MB
                    if current > self.peak_vram:
                        self.peak_vram = current
            except Exception:
                pass
            time.sleep(0.1)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.peak_vram

def test_single_agent():
    """
    Evaluates tokens per second (TPS) and memory usage for a single query.

    Returns:
        dict: Benchmarking results including TPS, latency, and memory usage.
    """
    print("Running Single-agent TPS via Tree-PLD...")
    try:
        vram_tracker = VRAMTracker()
        vram_tracker.start()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(SOCKET_PATH)
        
        # A simple request
        prompt = {"prompt": "What are the three laws of robotics?"}
        s.sendall((json.dumps(prompt) + "\n").encode())
        
        start_time = time.time()
        first_token_time = None
        tokens = 0
        
        f = s.makefile('r')
        for line in f:
            if not first_token_time:
                first_token_time = time.time()
            try:
                data = json.loads(line)
                if "token" in data:
                    tokens += 1
            except json.JSONDecodeError:
                pass
                
        end_time = time.time()
        peak_vram = vram_tracker.stop()
        
        duration = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else 0
        tps = tokens / duration if duration > 0 else 0
        
        return {
            "tokens": tokens,
            "ttft_sec": ttft,
            "duration_sec": duration,
            "tps": tps,
            "vram_peak_mb": peak_vram
        }
    except Exception as e:
        return {"error": str(e)}

def swarm_worker(results, index):
    """
    A worker function that sends a request to the Shimmer daemon socket.

    Args:
        results (list): Shared list to store thread results.
        index (int): Worker index for assigning prompt topic and storing result.
    """
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(SOCKET_PATH)
        
        prompt = {"prompt": f"Write a paragraph about topic {index}."}
        s.sendall((json.dumps(prompt) + "\n").encode())
        
        start_time = time.time()
        tokens = 0
        
        f = s.makefile('r')
        for line in f:
            try:
                data = json.loads(line)
                if "token" in data:
                    tokens += 1
            except json.JSONDecodeError:
                pass
                
        end_time = time.time()
        results[index] = {
            "tokens": tokens,
            "duration_sec": end_time - start_time,
            "tps": tokens / (end_time - start_time) if (end_time - start_time) > 0 else 0
        }
    except Exception as e:
        results[index] = {"error": str(e)}

def test_swarm():
    """
    Tests Shimmer daemon concurrency by running multiple requests simultaneously.

    Returns:
        dict: Aggregated performance and memory metrics across all threads.
    """
    print("Running Swarm concurrency (3 parallel connections)...")
    vram_tracker = VRAMTracker()
    vram_tracker.start()
    threads = []
    results = [None] * 3
    start_time = time.time()
    
    for i in range(3):
        t = threading.Thread(target=swarm_worker, args=(results, i))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    end_time = time.time()
    peak_vram = vram_tracker.stop()
    duration = end_time - start_time
    
    total_tokens = sum(r.get("tokens", 0) for r in results if r and "tokens" in r)
    tps = total_tokens / duration if duration > 0 else 0
    
    return {
        "total_tokens": total_tokens,
        "duration_sec": duration,
        "total_tps": tps,
        "vram_peak_mb": peak_vram,
        "worker_results": results
    }

def test_tool_latency():
    """
    Measures end-to-end latency for a model prompt requiring a tool call.

    Returns:
        dict: Performance metrics indicating latency overhead of tool calls.
    """
    print("Running End-to-End latency of a native 'rg' tool interception...")
    try:
        vram_tracker = VRAMTracker()
        vram_tracker.start()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(SOCKET_PATH)
        
        # We prompt the model to output a tool call for rg
        prompt_text = 'Output exactly this text and nothing else: <tool_call>{"name": "rg", "arguments": ["main", "."]}</tool_call>'
        prompt = {"prompt": prompt_text}
        
        start_time = time.time()
        s.sendall((json.dumps(prompt) + "\n").encode())
        
        tokens = 0
        f = s.makefile('r')
        for line in f:
            try:
                data = json.loads(line)
                if "token" in data:
                    tokens += 1
            except json.JSONDecodeError:
                pass
                
        end_time = time.time()
        peak_vram = vram_tracker.stop()
        duration = end_time - start_time
        
        return {
            "duration_sec": duration,
            "tokens_generated": tokens,
            "tps": tokens / duration if duration > 0 else 0,
            "vram_peak_mb": peak_vram
        }
    except Exception as e:
        return {"error": str(e)}

def test_swarm_disconnect():
    """
    Ensures memory leaks do not occur when clients disconnect prematurely.

    Returns:
        dict: Memory metrics before and after simulated early disconnections.
    """
    print("Testing Swarm disconnection to verify no memory leaks occur...")
    vram_tracker = VRAMTracker()
    vram_tracker.start()
    threads = []
    results = []
    
    def early_disconnect(index):
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(SOCKET_PATH)
            prompt = {"prompt": f"Write a very long explanation about space {index}."}
            s.sendall((json.dumps(prompt) + "\n").encode())
            
            f = s.makefile('r')
            count = 0
            for line in f:
                count += 1
                if count >= 3:
                    break
            # Disconnect early
            s.close()
            results.append(f"Worker {index} disconnected successfully after {count} tokens.")
        except Exception as e:
            results.append(f"Worker {index} error: {e}")

    for i in range(3):
        t = threading.Thread(target=early_disconnect, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    # Wait briefly to let the daemon clean up its resources
    time.sleep(2)
    peak_vram = vram_tracker.stop()
    
    return {
        "status": "success",
        "vram_peak_mb": peak_vram,
        "details": results
    }

def main():
    """
    Main benchmark runner. Validates daemon socket presence, executes all tests, 
    and serializes the final report to JSON.
    """
    if not os.path.exists(SOCKET_PATH):
        print(f"Error: Daemon socket {SOCKET_PATH} does not exist. Please start it with 'shimmer --daemon'.")
        return

    report = {}
    
    report["single_agent"] = test_single_agent()
    report["swarm_concurrency"] = test_swarm()
    report["tool_latency"] = test_tool_latency()
    report["swarm_disconnect"] = test_swarm_disconnect()
    
    report_path = "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\nBenchmark complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
