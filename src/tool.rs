//! Shell command execution and output formatting for agent tools.

use std::process::{Child, Command, Output, Stdio};
use std::time::Duration;

const MAX_OUTPUT_CHARS: usize = 12000;
const TOOL_TIMEOUT_SECS: u64 = 30;

/// Determines if a shell command and its arguments represent an idempotent, read-only operation.
pub fn is_idempotent(tool_name: &str, args: &[String]) -> Result<bool, String> {
    match tool_name {
        "rg" | "grep" | "fd" | "find" | "cat" | "read_file" | "ls" => Ok(true),
        "git" => {
            if let Some(subcmd) = args.first() {
                match subcmd.as_str() {
                    "status" | "diff" => Ok(true),
                    "commit" | "add" | "push" | "pull" | "checkout" | "branch" => Ok(false),
                    _ => Ok(false), // default to false for unknown git commands
                }
            } else {
                Ok(true) // Just "git" is idempotent (prints help)
            }
        }
        "sed" => {
            let is_destructive = args
                .iter()
                .any(|arg| arg == "-i" || arg.starts_with("--in-place"));
            if is_destructive {
                Err("[SYSTEM ERROR: 'sed -i' is forbidden. Use XML edit blocks to modify files.]".to_string())
            } else {
                Ok(true)
            }
        }
        "run_test" => Ok(false),
        "bash" | "python" | "cargo" => Ok(false),
        _ => Ok(false),
    }
}

/// Spawns a child process for the specified tool and arguments, returning a handle to the executing process.
pub fn spawn_tool(tool_name: &str, args: &[String]) -> std::io::Result<Child> {
    let cmd_name = if tool_name == "read_file" { "cat" } else { tool_name };
    let mut cmd = Command::new(cmd_name);

    // Hardcode default args for ripgrep
    if tool_name == "rg" {
        cmd.arg("--color=never")
            .arg("--no-heading")
            .arg("--line-number");
    }

    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
}

/// Waits for a spawned tool child process to finish, processing and wrapping
/// its output into an observation string. Applies a timeout to prevent hangs.
pub fn process_tool_child(mut child: Child) -> String {
    let content = wait_with_timeout(&mut child, TOOL_TIMEOUT_SECS);
    wrap_observation(&content)
}

/// Waits for a child process with a timeout, returning partial output on timeout.
fn wait_with_timeout(child: &mut Child, timeout_secs: u64) -> String {
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(timeout_secs);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = child.stdout.take().map(|mut p| {
                    let mut buf = Vec::new();
                    use std::io::Read;
                    let _ = p.read_to_end(&mut buf);
                    buf
                }).unwrap_or_default();
                let stderr = child.stderr.take().map(|mut p| {
                    let mut buf = Vec::new();
                    use std::io::Read;
                    let _ = p.read_to_end(&mut buf);
                    buf
                }).unwrap_or_default();
                return process_tool_output(Output { status, stdout, stderr });
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return format!(
                        "ERROR: Tool timed out after {}s. Try a more targeted command.",
                        timeout_secs
                    );
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => return format!("ERROR: Failed to wait on command: {}", e),
        }
    }
}

/// Executes a tool command synchronously, waits for it to complete, and formats its output.
pub fn execute_tool(tool_name: &str, args: &[String]) -> String {
    if tool_name == "cat" || tool_name == "read_file" {
        let file_path = args.first().map(|s| s.as_str()).unwrap_or("");
        if let Ok(content) = std::fs::read_to_string(file_path) {
            let mut start_line = 0;
            let mut end_line = usize::MAX;
            if args.len() > 1
                && let Ok(s) = args[1].parse::<usize>() { start_line = s.saturating_sub(1); }
            if args.len() > 2
                && let Ok(e) = args[2].parse::<usize>() { end_line = e; }
            let lines: Vec<&str> = content.lines().collect();
            let sliced_lines = &lines[start_line.min(lines.len())..end_line.min(lines.len())];

            let mut numbered = String::new();
            for (i, line) in sliced_lines.iter().enumerate() {
                numbered.push_str(&format!("{}: {}\n", start_line + i + 1, line));
            }

            let raw_content: String = sliced_lines
                .iter()
                .fold(String::new(), |mut acc, l| {
                    acc.push_str(l);
                    acc.push('\n');
                    acc
                });

            let budget = MAX_OUTPUT_CHARS / 2;
            let numbered_trunc: String = numbered.chars().take(budget).collect();
            let raw_trunc: String = raw_content.chars().take(budget).collect();

            let mut combined = numbered_trunc;
            if !raw_trunc.is_empty() {
                combined.push_str("\n[RAW CONTENT FOR SEARCH BLOCK — copy from here]\n");
                combined.push_str(&raw_trunc);
                combined.push_str("[END RAW CONTENT]\n");
            }

            let total_chars = numbered.chars().count() + raw_content.chars().count();
            if total_chars > MAX_OUTPUT_CHARS {
                combined.push_str(&format!(
                    "\n[...TRUNCATED: output too long. Use `cat {} <start> <end>` for specific sections.]",
                    file_path
                ));
            }
            return wrap_observation(&combined);
        }
    }

    if tool_name == "run_test" {
        let test_cmd = if args.is_empty() {
            "python -m pytest".to_string()
        } else if args.len() == 1 {
            args[0].clone()
        } else {
            let (first, rest) = args.split_at(1);
            let rest_quoted: Vec<String> = rest.iter().map(|a| {
                shlex::try_quote(a)
                    .unwrap_or_else(|_| std::borrow::Cow::Owned(a.clone()))
                    .into_owned()
            }).collect();
            format!("{} {}", first[0], rest_quoted.join(" "))
        };
        let mut cmd = Command::new("bash");

        let output_result = cmd
            .arg("-c")
            .arg(&test_cmd)
            .stdin(Stdio::null())
            .output();

        let content = match output_result {
            Ok(out) => process_tool_output(out),
            Err(e) => format!("ERROR: Failed to execute test command: {}", e),
        };
        return wrap_observation(&content);
    }

    let cmd_name = if tool_name == "read_file" { "cat" } else { tool_name };
    let mut cmd = Command::new(cmd_name);

    if tool_name == "rg" {
        cmd.arg("--color=never")
            .arg("--no-heading")
            .arg("--line-number");
    }

    let output_result = cmd
        .args(args)
        .stdin(Stdio::null())
        .output();

    let content = match output_result {
        Ok(out) => process_tool_output(out),
        Err(e) => format!("ERROR: Failed to execute command: {}", e),
    };

    wrap_observation(&content)
}

/// Formats the raw output of a command execution, truncating large payloads and handling error codes.
fn process_tool_output(out: Output) -> String {
    let code = out.status.code().unwrap_or(2);

    if code == 1 {
        let stdout_str = String::from_utf8_lossy(&out.stdout).into_owned();
        let stderr_str = String::from_utf8_lossy(&out.stderr).into_owned();
        if !stdout_str.trim().is_empty() || !stderr_str.trim().is_empty() {
            return format!("{}\n{}", stdout_str, stderr_str);
        }
        return "Command completed with exit code 1 (often means no matches or minor failure)."
            .to_string();
    }

    if code > 1 {
        let err_msg = String::from_utf8_lossy(&out.stderr);
        let stdout_str = String::from_utf8_lossy(&out.stdout).into_owned();
        if !stdout_str.trim().is_empty() {
            return format!("Partial output (command failed with code {}):\n{}\n\nStderr: {}",
                code, stdout_str, err_msg.trim());
        }
        return format!("ERROR - Command failed: {}", err_msg.trim());
    }

    let lossy_str = String::from_utf8_lossy(&out.stdout).into_owned();
    
    // SMART CHUNKING
    let lines: Vec<&str> = lossy_str.lines().collect();
    if lines.len() > 150 || lossy_str.chars().count() > MAX_OUTPUT_CHARS {
        let mut anchors = Vec::new();
        let anchor_keywords = ["Traceback", "FAIL:", "ERROR:", "Exception:", "AssertionError", "Err", "warning:"];
        
        for (i, line) in lines.iter().enumerate() {
            if anchor_keywords.iter().any(|&k| line.contains(k)) {
                anchors.push(i);
            }
        }
        
        if !anchors.is_empty() {
            let mut include_lines = std::collections::HashSet::new();
            for a in anchors {
                let start = a.saturating_sub(30);
                let end = (a + 30).min(lines.len() - 1);
                for i in start..=end {
                    include_lines.insert(i);
                }
            }
            
            let mut sorted_includes: Vec<usize> = include_lines.into_iter().collect();
            sorted_includes.sort_unstable();
            
            let mut smart_output = String::new();
            let mut last_idx: i32 = -1;
            
            for &i in &sorted_includes {
                if last_idx != -1 && i as i32 > last_idx + 1 {
                    let skipped = i as i32 - last_idx - 1;
                    smart_output.push_str(&format!("\n... [Truncated {} lines] ...\n", skipped));
                }
                smart_output.push_str(lines[i]);
                smart_output.push('\n');
                last_idx = i as i32;
            }
            
            smart_output.push_str("\n[Output was smart-chunked around errors. Use 'grep -C 50 <error_string>' if you need more context]");
            
            if smart_output.chars().count() <= MAX_OUTPUT_CHARS {
                return smart_output;
            }
        }

        // Fallback to naive truncation if no anchors found or smart chunking is still too large
        let truncated: String = lossy_str.chars().take(MAX_OUTPUT_CHARS).collect();
        return format!("{}\n\n[...TRUNCATED]", truncated);
    }

    if lossy_str.is_empty() {
        return "[Command executed successfully with no output]".to_string();
    }

    lossy_str
}

/// Wraps formatted tool output text into a standardized system observation block.
fn wrap_observation(content: &str) -> String {
    format!(
        "\n[SYSTEM OBSERVED OUTPUT START]\n{}\n[SYSTEM OBSERVED OUTPUT END]\n",
        content
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_idempotent_read_tools() {
        for tool in &["rg", "grep", "fd", "find", "cat", "read_file", "ls"] {
            assert!(is_idempotent(tool, &[]).unwrap_or(false));
        }
    }

    #[test]
    fn test_is_idempotent_git_status() {
        assert!(is_idempotent("git", &["status".into()]).unwrap_or(false));
    }

    #[test]
    fn test_is_idempotent_git_commit_blocked() {
        assert!(!is_idempotent("git", &["commit".into()]).unwrap_or(true));
    }

    #[test]
    fn test_is_idempotent_sed_blocked() {
        assert!(is_idempotent("sed", &["-i".into(), "file".into()]).is_err());
    }

    #[test]
    fn test_is_idempotent_sed_allowed() {
        assert!(is_idempotent("sed", &["-n".into(), "p".into()]).unwrap_or(false));
    }

    #[test]
    fn test_is_idempotent_run_test() {
        assert!(!is_idempotent("run_test", &[]).unwrap_or(true));
    }

    #[test]
    fn test_is_idempotent_unknown_tool() {
        assert!(!is_idempotent("nonexistent_tool_xyz", &[]).unwrap_or(true));
    }

    #[test]
    fn test_execute_tool_echo_output() {
        let result = execute_tool("echo", &["hello_test".into()]);
        assert!(result.contains("hello_test"));
        assert!(result.contains("[SYSTEM OBSERVED OUTPUT START]"));
    }

    #[test]
    fn test_execute_tool_cat_nonexistent() {
        let result = execute_tool("cat", &["nonexistent_file_xyz_12345".into()]);
        assert!(result.contains("[SYSTEM OBSERVED OUTPUT"));
    }

    #[test]
    fn test_spawn_tool_read_file_aliases_to_cat() {
        let child = spawn_tool("read_file", &["Cargo.toml".into()]);
        assert!(child.is_ok());
    }
}
