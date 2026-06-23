#!/usr/bin/env python3
"""
Generates a synthetic monorepo and server logs for an agentic stress test environment.
Injects a specific vulnerability and related crash logs.

Example usage:
    python generate_stress_test.py
"""

import os
import sys
import random
import datetime
import subprocess

NUM_SERVICES = 50
FILES_PER_SERVICE = 20
NUM_COMMITS = 500
BUGGY_COMMIT_OFFSET = 100 # How many commits back from HEAD the bug was introduced
LOG_SIZE_MB = 10

def run_cmd(cmd, cwd=None):
    """
    Executes a shell command silently.

    Args:
        cmd (str): Command to execute.
        cwd (str, optional): Working directory for the command.
    """
    subprocess.run(cmd, shell=True, check=True, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def generate_codebase(root_dir):
    """
    Generates a synthetic codebase with multiple services and commits, 
    injecting a NULL_POINTER_DEREFERENCE bug into a specific commit.

    Args:
        root_dir (str): Root directory for the synthetic codebase.

    Returns:
        tuple: (bug_service (str), bug_func (str)) identifying where the bug was injected.
    """
    print(f"Generating synthetic codebase in {root_dir}...")
    os.makedirs(root_dir, exist_ok=True)
    
    # Initialize Git
    run_cmd("git init", cwd=root_dir)
    run_cmd('git config user.name "Test Bot"', cwd=root_dir)
    run_cmd('git config user.email "bot@example.com"', cwd=root_dir)

    # Bug definition
    bug_service = f"service_{random.randint(1, NUM_SERVICES):03d}"
    bug_file = f"src/auth_handler.py"
    bug_func = "verify_token_integrity"
    
    # Generate initial files
    files_created = []
    for s in range(1, NUM_SERVICES + 1):
        service_name = f"service_{s:03d}"
        service_dir = os.path.join(root_dir, service_name, "src")
        os.makedirs(service_dir, exist_ok=True)
        
        for f in range(1, FILES_PER_SERVICE + 1):
            file_path = os.path.join(service_dir, f"module_{f:03d}.py")
            with open(file_path, "w") as f_out:
                f_out.write(f"# Auto-generated module for {service_name}\n")
                f_out.write("def do_work():\n    pass\n")
            files_created.append(file_path)

    run_cmd("git add .", cwd=root_dir)
    run_cmd('git commit -m "Initial commit of monorepo framework"', cwd=root_dir)

    print(f"Generating {NUM_COMMITS} commits...")
    bug_commit_hash = None
    
    for i in range(1, NUM_COMMITS + 1):
        target_file = random.choice(files_created)
        with open(target_file, "a") as f:
            f.write(f"\n# Commit {i} modification\n")
            
            # Inject the bug at the specified offset
            if i == NUM_COMMITS - BUGGY_COMMIT_OFFSET:
                bug_target = os.path.join(root_dir, bug_service, bug_file)
                os.makedirs(os.path.dirname(bug_target), exist_ok=True)
                with open(bug_target, "w") as bf:
                    bf.write(f"""
def {bug_func}(token):
    # CRITICAL VULNERABILITY INJECTED HERE
    if token is None:
        raise RuntimeError("CRITICAL: NULL_POINTER_DEREFERENCE in {bug_func}")
    return True
""")
                run_cmd(f"git add {bug_service}/{bug_file}", cwd=root_dir)
                run_cmd('git commit -m "Refactor auth middleware for performance"', cwd=root_dir)
                bug_commit_hash = subprocess.check_output("git rev-parse HEAD", shell=True, cwd=root_dir).decode('utf-8').strip()
                continue
                
        run_cmd(f"git add {os.path.relpath(target_file, root_dir)}", cwd=root_dir)
        run_cmd(f'git commit -m "chore: update {os.path.basename(target_file)}"', cwd=root_dir)
        
    print(f"Codebase generated. Bug injected in {bug_service}/{bug_file}")
    print(f"Bug commit hash: {bug_commit_hash}")
    return bug_service, bug_func

def generate_logs(root_dir, bug_func):
    """
    Generates synthetic production logs and injects the target stack trace exactly once.

    Args:
        root_dir (str): Root directory containing the logs directory.
        bug_func (str): Function name to inject into the stack trace.
    """
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "production.log")
    
    print(f"Generating {LOG_SIZE_MB}MB of synthetic logs...")
    
    target_bytes = LOG_SIZE_MB * 1024 * 1024
    written = 0
    
    start_time = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    
    bug_timestamp = "14:32:01"
    
    with open(log_file, "w") as f:
        while written < target_bytes:
            current_time = start_time + datetime.timedelta(seconds=written / 1000)
            time_str = current_time.strftime("%H:%M:%S")
            
            # Inject the specific bug exactly once around the middle of the log
            if written > target_bytes / 2 and bug_timestamp not in "injected":
                bug_log = f"""
[INFO] {bug_timestamp} - User request started
[DEBUG] {bug_timestamp} - Authenticating payload
[ERROR] {bug_timestamp} - 500 INTERNAL SERVER ERROR
Traceback (most recent call last):
  File "server.py", line 42, in handle_request
  File "auth_handler.py", line 4, in {bug_func}
RuntimeError: CRITICAL: NULL_POINTER_DEREFERENCE in {bug_func}
[ERROR] Affected User IDs: usr_198x7, usr_229a1, usr_000b4, usr_918c2, usr_777z9
[INFO] {bug_timestamp} - Request terminated
"""
                f.write(bug_log)
                written += len(bug_log.encode('utf-8'))
                bug_timestamp = "injected"
                
            line = f"[INFO] {time_str} - Processed request from IP 192.168.1.{random.randint(1, 255)} - Status 200 OK - Latency {random.randint(10, 500)}ms\n"
            f.write(line)
            written += len(line.encode('utf-8'))

    print("Logs generated.")

def main():
    """
    Main execution flow. Checks for existing directory, generates the codebase 
    and logs, and prints the instructions for the agent's stress test goal.
    """
    root_dir = os.path.join(os.getcwd(), "stress_test_env")
    if os.path.exists(root_dir):
        print(f"Directory {root_dir} already exists. Please remove it to regenerate.")
        sys.exit(1)
        
    bug_service, bug_func = generate_codebase(root_dir)
    generate_logs(root_dir, bug_func)
    
    print("\n--- Benchmark Environment Ready ---")
    print(f"Location: {root_dir}")
    print("Goal for Agent:")
    print("1. Find the 500 error at 14:32:01 in logs/production.log")
    print("2. Identify the failing function name and the affected User IDs")
    print("3. Find which microservice contains this function in the codebase")
    print("4. Find the git commit hash that introduced this function")

if __name__ == "__main__":
    main()
