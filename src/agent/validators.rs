//! Validators for the agent generation loop.
//!
//! Each check returns `None` if the check passes (no intervention needed)
//! or `Some(nudge_message)` if the model needs to be redirected. The caller
//! injects the nudge text into the generation stream and returns early.

use crate::agent::state::{AgentConfig, EngineState, HARD_TOOL_LIMIT, STUBBORN_ABORT_LIMIT};
use crate::interceptor::ToolInterceptor;
use crate::models::CanonicalModel;

/// Blind edit blocker: fires when `</edit>` is seen without any prior tool calls.
///
/// Catches both complete edits and incomplete ones (missing `<replace>`).
/// Checking on the closing tag prevents infinite loops from mid-thought interruption.
pub(crate) fn check_blind_edit(
    state: &EngineState,
    config: &AgentConfig,
    interceptor: &mut ToolInterceptor,
    model_type: &CanonicalModel,
) -> Option<String> {
    if !interceptor.edit_tag_closed {
        return None;
    }
    interceptor.edit_tag_closed = false;
    if config.enable_blind_edit_blocker && state.tool_calls == 0 {
        interceptor.detected_full_edit.take();
        let msg = crate::models::format_system_nudge(
            model_type,
            "You produced an `<edit>` block without using any tools to investigate the codebase \
             first. Please use tools like `rg`, `ls`, or `cat` to verify the file path and \
             content before making any edits.",
        );
        interceptor.reset_edit_state();
        Some(msg)
    } else {
        None
    }
}

/// TDD enforcement: fires on complete edit blocks when tests haven't been run.
///
/// Requires a complete `<replace>` + `</edit>` block to evaluate. Incomplete
/// edits are caught by the blind edit blocker instead.
pub(crate) fn check_tdd_enforcement(
    state: &EngineState,
    config: &AgentConfig,
    interceptor: &mut ToolInterceptor,
    model_type: &CanonicalModel,
) -> Option<String> {
    if interceptor.detected_full_edit.is_some()
        && config.enable_tdd_enforcement
        && state.tests_since_last_edit == 0
    {
        interceptor.detected_full_edit.take();
        let msg = crate::models::format_system_nudge(
            model_type,
            "You produced an `<edit>` block but haven't run tests yet. Please use the 'run_test' \
             tool first.",
        );
        interceptor.reset_edit_state();
        Some(msg)
    } else {
        None
    }
}

/// Path blocker: rejects edits targeting non-existent files.
///
/// When the edit path does not exist, the check resets interceptor state and
/// returns a nudge suggesting the model use `ls`, `find`, or `run_command`
/// to create the file. When the path exists, `path_validated` is set so the
/// check only runs once per edit path.
pub(crate) fn check_path_blocker(
    interceptor: &mut ToolInterceptor,
    model_type: &CanonicalModel,
) -> Option<String> {
    if interceptor.path_validated {
        return None;
    }
    let path = interceptor.detected_edit_path.as_ref()?;
    if !std::path::Path::new(path).exists() {
        let err_msg = format!(
            "</edit>\n{}",
            crate::models::format_system_nudge(
                model_type,
                &format!(
                    "File '{}' does not exist. If you meant to create it, use the `run_command` \
                     tool to create it. Otherwise, verify the path using `ls` or `find`.",
                    path
                ),
            )
        );
        interceptor.buffer.clear();
        interceptor.detected_edit_path = None;
        Some(err_msg)
    } else {
        interceptor.path_validated = true;
        None
    }
}

/// Search verifier: validates that `<search>` blocks match exactly once.
///
/// Zero matches means the search string does not appear in the file; the
/// model is told to use `cat` to read the actual content. Multiple matches
/// means the search is ambiguous; the model is told to include more context.
/// Dedup tracking via `edit_history` prevents infinite retry loops.
pub(crate) fn check_search_verifier(
    state: &mut EngineState,
    interceptor: &mut ToolInterceptor,
    model_type: &CanonicalModel,
) -> Option<String> {
    let (path, search_content) = interceptor.detected_search_block.take()?;
    let normalized_search = search_content.replace("\r\n", "\n");
    let file_content = std::fs::read_to_string(&path).unwrap_or_default();
    let matches = file_content.matches(&normalized_search).count();

    if matches != 1 {
        // Dedup: if the model retries the same wrong search, escalate
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        path.hash(&mut hasher);
        normalized_search.hash(&mut hasher);
        let search_hash = hasher.finish();

        let stale_retry = !state.edit_history.insert(search_hash);
        let hint = format!(
            "Use `cat {}` to read the actual file content, then copy the exact lines into your \
             `<search>` block.",
            path
        );
        let stale_note = if stale_retry {
            " You already tried this exact search — it will not work. Read the file instead of \
             guessing. "
        } else {
            ""
        };

        let err_msg = if matches == 0 {
            format!(
                "\n</edit>\n{}",
                crate::models::format_system_nudge(
                    model_type,
                    &format!(
                        "The code in your `<search>` block was not found in {path}. \
                         {stale_note}{hint}",
                        path = path,
                        stale_note = stale_note,
                        hint = hint
                    ),
                )
            )
        } else {
            format!(
                "\n</edit>\n{}",
                crate::models::format_system_nudge(
                    model_type,
                    &format!(
                        "Your `<search>` block matched {matches} times in {path}. Include more \
                         surrounding context to make it unique. {hint}",
                        matches = matches,
                        path = path,
                        hint = hint
                    ),
                )
            )
        };

        interceptor.buffer.clear();
        interceptor.detected_edit_path = None;
        interceptor.path_validated = false;
        Some(err_msg)
    } else {
        None
    }
}

/// Syntax checker: AST-compiles patched Python files and rejects errors.
///
/// Writes the patched content to a temp file, runs `python3 -c "compile(...)"`,
/// and removes the temp file. On AST failure, returns a nudge with the
/// compiler error. On success, resets `tests_since_last_edit` to require
/// tests before the next edit.
pub(crate) fn check_syntax_checker(
    state: &mut EngineState,
    interceptor: &mut ToolInterceptor,
    model_type: &CanonicalModel,
) -> Option<String> {
    let (path, search, replace) = interceptor.detected_full_edit.take()?;
    if !path.ends_with(".py") {
        return None;
    }

    let file_content = std::fs::read_to_string(&path).unwrap_or_default();
    let normalized_search = search.replace("\r\n", "\n");
    let normalized_replace = replace.replace("\r\n", "\n");

    if !file_content.contains(&normalized_search) {
        return None;
    }

    let new_content = file_content.replace(&normalized_search, &normalized_replace);
    let temp_path = format!("{}.syntax_check_tmp_{}", path, std::process::id());

    if std::fs::write(&temp_path, new_content).is_err() {
        return None;
    }

    let mut cmd = std::process::Command::new("python3");
    cmd.arg("-c")
        .arg("import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec')")
        .arg(&temp_path);

    let result = cmd.output();
    let _ = std::fs::remove_file(&temp_path);

    match result {
        Ok(output) if !output.status.success() => {
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            let cleaned_stderr = stderr.replace(&temp_path, &path);

            let err_msg = crate::models::format_system_nudge(
                model_type,
                &format!(
                    "The patch you provided has syntax errors and failed AST compilation. Please \
                     fix the errors:\n\n{}",
                    cleaned_stderr
                ),
            );
            interceptor.buffer.clear();
            interceptor.detected_edit_path = None;
            interceptor.path_validated = false;
            interceptor.detected_search_block = None;
            Some(err_msg)
        }
        Ok(_) => {
            // Edit passed syntax check — require tests before next edit
            state.tests_since_last_edit = 0;
            None
        }
        Err(_) => None,
    }
}

/// Insanity detector: catches repeated tool calls (same tool + same args)
/// and escalates from nudge to hard rejection.
///
/// - First duplicate: injects a nudge suggesting a different approach.
/// - At `HARD_TOOL_LIMIT` (8) duplicates: injects a hard tool rejection.
/// - At `STUBBORN_ABORT_LIMIT` (11) duplicates: returns `Err` to abort.
///
/// On success (no violation) returns `Ok(None)`. The caller is responsible
/// for handling the mutating-tool history reset and setting `pending_tool`.
pub(crate) fn check_insanity_detector(
    state: &mut EngineState,
    config: &AgentConfig,
    name: &str,
    args: &serde_json::Value,
    model_type: &CanonicalModel,
) -> Result<Option<String>, anyhow::Error> {
    let call_key = format!("{}:{:?}", name, args);
    if config.enable_insanity_detector && !state.tool_history.insert(call_key) && name != "run_test"
    {
        if state.tool_calls >= STUBBORN_ABORT_LIMIT && name != "run_test" {
            return Err(anyhow::anyhow!(
                "Model is stuck in a loop and refusing to yield a patch. Aborting."
            ));
        }
        let nudge_msg = if state.tool_calls >= HARD_TOOL_LIMIT {
            crate::models::format_tool_rejection(model_type)
        } else {
            crate::models::format_tool_nudge(
                model_type,
                name,
                &format!(".shimmer_tool_{}.txt", state.tool_calls + 1),
            )
        };
        state.tool_calls += 1;
        state.last_output_token_count = 0;
        Ok(Some(nudge_msg))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_blocker_rejects_nonexistent_path() {
        let mut interceptor = ToolInterceptor::new(false, true);
        interceptor.feed_token("<edit file=\"nonexistent/path/for/testing/file.py\">");
        assert!(interceptor.detected_edit_path.is_some());
        assert!(!interceptor.path_validated);

        let result = check_path_blocker(&mut interceptor, &CanonicalModel::Gemma4_12B);
        assert!(result.is_some());
        assert!(interceptor.detected_edit_path.is_none());
    }

    #[test]
    fn path_blocker_passes_existing_file() {
        let mut interceptor = ToolInterceptor::new(false, true);
        interceptor.feed_token("<edit file=\"src/main.rs\">");

        let result = check_path_blocker(&mut interceptor, &CanonicalModel::Gemma4_12B);
        assert!(result.is_none());
        assert!(interceptor.path_validated);
    }

    #[test]
    fn path_blocker_skips_when_already_validated() {
        let mut interceptor = ToolInterceptor::new(false, true);
        interceptor.path_validated = true;
        interceptor.detected_edit_path = Some("nonexistent/file.py".into());

        let result = check_path_blocker(&mut interceptor, &CanonicalModel::Gemma4_12B);
        assert!(result.is_none());
    }
}
