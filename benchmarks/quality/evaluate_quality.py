"""
Evaluates output quality of Shimmer against a standard Ollama baseline.
Runs multiple predefined prompts to ensure optimizations do not degrade output.

Example usage:
    python evaluate_quality.py
"""

import os
import re
import subprocess

PROMPTS = [
    {
        "name": "Prompt 1 (Code/Structure)",
        "text": "Write a Rust function that implements a recursive binary search tree insertion. Provide only the code, no explanation."
    },
    {
        "name": "Prompt 2 (Repetitive Text)",
        "text": "Write a short poem with heavy repetition about a ticking clock. The word 'tick' must appear many times."
    },
    {
        "name": "Prompt 3 (Creative/Prose)",
        "text": "Write a highly descriptive paragraph about a futuristic neon city in the rain. Try to avoid repeating any adjectives."
    }
]

CONFIGS = [
    {"name": "Ollama (Baseline)", "is_shimmer": False},
    {"name": "Shimmer (N=3, M=24)", "is_shimmer": True, "n": 3, "m": 24},
    {"name": "Shimmer (N=4, M=24)", "is_shimmer": True, "n": 4, "m": 24},
    {"name": "Shimmer (N=3, M=27)", "is_shimmer": True, "n": 3, "m": 27},
    {"name": "Shimmer (N=4, M=27)", "is_shimmer": True, "n": 4, "m": 27},
]

SPECULATIVE_RS_PATH = "src/speculative.rs"
ARTIFACT_PATH = "quality_evaluation.md"

def modify_constants(ngram_size, draft_tokens):
    """
    Temporarily modifies speculative.rs to set n-gram size and max draft tokens.

    Args:
        ngram_size (int): New n-gram size.
        draft_tokens (int): New maximum draft tokens limit.
    """
    with open(SPECULATIVE_RS_PATH, "r") as f:
        content = f.read()
    
    content = re.sub(r"let ngram_size = \d+;", f"let ngram_size = {ngram_size};", content)
    content = re.sub(r"pub const MAX_DRAFT_TOKENS: usize = \d+;", f"pub const MAX_DRAFT_TOKENS: usize = {draft_tokens};", content)
    
    with open(SPECULATIVE_RS_PATH, "w") as f:
        f.write(content)

import urllib.request
import json

def run_ollama(prompt):
    """
    Queries the local Ollama instance as a baseline.

    Args:
        prompt (str): The prompt text to generate against.

    Returns:
        str: The generated response or an error message.
    """
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "gemma4:12b",
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0
    }
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode())
            return result.get("response", "").strip()
    except Exception as e:
        return f"Error: {e}"

def run_shimmer(prompt_text):
    """
    Runs the Shimmer speculative decoder locally to generate text.

    Args:
        prompt_text (str): The prompt text to evaluate.

    Returns:
        str: Extracted assistant generation text or an error message.
    """
    env = os.environ.copy()
    env["SHIMMER_PROMPT"] = prompt_text
    try:
        res = subprocess.run(["cargo", "run", "--release", "--", "--speculative"], capture_output=True, text=True, env=env, timeout=90)
        out = res.stdout
    except subprocess.TimeoutExpired:
        return "Error: Timeout expired during generation."
        
    # Extract everything between [Generation Result] and the next =====================
    match = re.search(r"\[Generation Result\]\n(.*?)\n=====================", out, re.DOTALL)
    if match:
        text = match.group(1).strip()
        assistant_idx = text.find("<|im_start|>assistant")
        if assistant_idx != -1:
            text = text[assistant_idx + len("<|im_start|>assistant"):].strip()
        text = text.replace("<|channel>thought", "").replace("<channel|>", "").strip()
        return text
    return "Error: Could not parse Shimmer output."

def main():
    """
    Entrypoint that builds Shimmer, iterates through configured model variants 
    and prompts, and writes the output quality results to a markdown artifact.
    """
    subprocess.run(["cargo", "build", "--release"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    markdown_content = "# Output Quality Evaluation\n\nThis artifact compares the output of standard Ollama with several Shimmer Speculative Decoding configurations to ensure our optimizations haven't degraded output quality or caused repeating loops.\n\n"
    
    for prompt in PROMPTS:
        print(f"Running {prompt['name']}...", flush=True)
        markdown_content += f"## {prompt['name']}\n**Prompt:** `{prompt['text']}`\n\n"
        
        for config in CONFIGS:
            print(f"  -> {config['name']}", flush=True)
            markdown_content += f"### {config['name']}\n"
            if not config["is_shimmer"]:
                output = run_ollama(prompt['text'])
            else:
                modify_constants(config["n"], config["m"])
                # Rebuild is fast and incremental
                subprocess.run(["cargo", "build", "--release"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                output = run_shimmer(prompt['text'])
            
            markdown_content += f"```text\n{output}\n```\n\n"
    
    with open(ARTIFACT_PATH, "w") as f:
        f.write(markdown_content)
        
    # Restore defaults
    modify_constants(3, 24)
    print(f"Done. Saved to {ARTIFACT_PATH}")

if __name__ == "__main__":
    main()
