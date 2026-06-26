"""Shimmer inference runner.  Single function: run_shimmer."""

import os, subprocess, threading, time

INSTANCE_TIMEOUT = 900  # 15 minutes per instance
SHIMMER_BIN = None  # Override for testing: set to a fake shimmer script path

GPU_NOISE_PATTERNS = [
    r'ggml_metal', r'llama_', r'load_tensor', r'print_info',
    r'sched_reserve', r'create_tensor', r'done_getting',
    r'load_tensors', r'set_abort', r'init_tokenizer',
    r'load: \\d', r'load: control', r'load: printing',
    r'load: special', r'\\.{10,}', r'graph_reserve',
    r'~llama_context', r'ggml_metal_free',
]


def _reader_thread(process, response_chunks, done):
    """Read lines from process stdout; run in a daemon thread."""
    for line in process.stdout:
        print(line, end="", flush=True)
        response_chunks.append(line)
    done.set()


def _kill_process(process, timeout_sec):
    """Try SIGKILL with fallback for stuck GPU kernels."""
    process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        import signal
        os.kill(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"\n[TIMEOUT after {timeout_sec}s — process unkillable (GPU hang)]",
                  flush=True)


def run_shimmer(prompt, cwd, model_path, sample_config, no_tools=False):
    """Run shimmer inference with wall-clock timeout. Returns output text."""
    shimmer_bin = SHIMMER_BIN or f"{cwd}/target/release/shimmer"
    cmd = [
        shimmer_bin,
        "--main-model", model_path,
        "--prompt", prompt,
        "--no-blind-edit-blocker",
        "--sample", sample_config,
    ]
    if no_tools:
        cmd.append("--no-tools")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, cwd=cwd)
    response_chunks = []
    done = threading.Event()
    t = threading.Thread(target=_reader_thread,
                         args=(process, response_chunks, done), daemon=True)
    t.start()
    t.join(timeout=INSTANCE_TIMEOUT)

    if not done.is_set():
        _kill_process(process, INSTANCE_TIMEOUT)
        print(f"\n[TIMEOUT after {INSTANCE_TIMEOUT}s]", flush=True)
        response_chunks.append("\n[TIMEOUT]\n")

    return "".join(response_chunks)
