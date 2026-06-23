//! Asynchronous Tool Interceptor
//!
//! Evaluates incoming token streams for tool trigger sequences without blocking
//! the primary Metal generation graph.

use crate::tool;
use serde::Deserialize;
use std::process::Child;

/// Upper bound byte length for the interceptor buffer.
const MAX_BUFFER_LEN: usize = 4096;
const TOOL_MARKER_START: &str = "```json\n";
const TOOL_MARKER_END: &str = "```";

/// A deserialized payload representing a specific tool invocation.
#[derive(Debug, Deserialize)]
pub struct ToolCallPayload {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// A process task executed speculatively before a tool call finishes generation.
pub struct SpeculativeTask {
    pub child: Child,
    pub tool_name: String,
    pub args: serde_json::Value,
}

/// Monitors a generation stream for tool invocations and handles speculative execution.
pub struct ToolInterceptor {
    pub buffer: String,
    pub detected_call: Option<(String, serde_json::Value, Option<Child>)>,
    pub spec_task: Option<SpeculativeTask>,
    pub enable_time_travel: bool,
    /// When false, JSON tool-call markers (```json) are treated as plain text.
    /// Edit tag parsing always runs regardless.
    pub detect_json_tools: bool,
    pub detected_edit_path: Option<String>,
    pub path_validated: bool,
    pub detected_search_block: Option<(String, String)>,
    pub detected_full_edit: Option<(String, String, String)>,
    /// Set when </edit> is seen with a known edit path — signals ANY edit
    /// block closure, even incomplete ones missing <replace>.
    pub edit_tag_closed: bool,
}

impl ToolInterceptor {
    pub fn new(enable_time_travel: bool, detect_json_tools: bool) -> Self {
        Self {
            buffer: String::with_capacity(MAX_BUFFER_LEN),
            detected_call: None,
            spec_task: None,
            enable_time_travel,
            detect_json_tools,
            detected_edit_path: None,
            path_validated: false,
            detected_search_block: None,
            detected_full_edit: None,
            edit_tag_closed: false,
        }
    }

    /// Clears edit-related parser state after a validator rejects an edit,
    /// preventing stale state from affecting the next attempt.
    pub fn reset_edit_state(&mut self) {
        self.buffer.clear();
        self.detected_edit_path = None;
        self.path_validated = false;
        self.detected_search_block = None;
    }

    pub fn feed_token(&mut self, token_str: &str) -> bool {
        self.buffer.push_str(token_str);

        if self.detected_edit_path.is_none()
            && let Some(start) = self.buffer.find("<edit file=\"") {
                let rest = &self.buffer[start + 12..];
                if let Some(end) = rest.find("\">") {
                    self.detected_edit_path = Some(rest[..end].to_string());
                }
            }

        if self.detected_search_block.is_none()
            && let Some(ref path) = self.detected_edit_path
                && let Some(s_start) = self.buffer.find("<search>") {
                    let s_rest = &self.buffer[s_start + 8..];
                    if let Some(s_end) = s_rest.find("</search>") {
                        self.detected_search_block = Some((path.clone(), s_rest[..s_end].to_string()));
                    }
                }

        if self.detected_full_edit.is_none()
            && let Some((ref path, ref search)) = self.detected_search_block
                && let Some(r_start) = self.buffer.find("<replace>") {
                    let r_rest = &self.buffer[r_start + 9..];
                    if let Some(r_end) = r_rest.find("</replace>")
                        && self.buffer[r_start + 9 + r_end..].contains("</edit>") {
                            self.detected_full_edit = Some((path.clone(), search.clone(), r_rest[..r_end].to_string()));
                        }
                }

        // Detect ANY edit block closure (with or without <replace>).
        // This catches incomplete edits that the full_edit detector misses.
        if !self.edit_tag_closed
            && self.detected_edit_path.is_some()
            && self.buffer.contains("</edit>") {
                self.edit_tag_closed = true;
            }

        if self.detect_json_tools
            && let Some(start) = self.buffer.find(TOOL_MARKER_START) {
            let rest = &self.buffer[start + TOOL_MARKER_START.len()..];
            if rest.contains(TOOL_MARKER_END) {
                return self.process_full_call();
            }
        }

        self.process_speculative_call();
        self.handle_buffer_overflow();

        false
    }

    pub fn extract_json_payload(line: &str) -> Option<ToolCallPayload> {
        let start = line.find(TOOL_MARKER_START)?;
        let rest = &line[start + TOOL_MARKER_START.len()..];
        let end_idx = rest.find(TOOL_MARKER_END).unwrap_or(rest.len());
        let payload_str = &rest[..end_idx];
        match serde_json::from_str::<ToolCallPayload>(payload_str) {
            Ok(payload) => Some(payload),
            Err(e) => {
                tracing::debug!("JSON parse error: {}. Payload: {:?}", e, payload_str);
                None
            }
        }
    }

    fn process_full_call(&mut self) -> bool {
        if let Some(payload) = Self::extract_json_payload(&self.buffer) {
            let child_opt = self.take_matching_task(&payload);
            self.detected_call = Some((payload.name, payload.arguments, child_opt));
            self.buffer.clear();
            true
        } else {
            self.kill_speculative_task();
            self.buffer.clear();
            false
        }
    }

    fn take_matching_task(&mut self, payload: &ToolCallPayload) -> Option<Child> {
        let mut task = self.spec_task.take()?;
        if task.tool_name == payload.name && task.args == payload.arguments {
            Some(task.child)
        } else {
            let _ = task.child.kill();
            None
        }
    }

    fn kill_speculative_task(&mut self) {
        if let Some(mut task) = self.spec_task.take() {
            let _ = task.child.kill();
        }
    }

    fn process_speculative_call(&mut self) {
        if !self.enable_time_travel {
            return;
        }

        if self.spec_task.is_none() {
            if self.buffer.contains(TOOL_MARKER_START)
                && let Some(payload) = Self::extract_json_payload(&self.buffer) {
                    self.try_spawn_speculative(payload);
                }
        } else if Self::has_diverged(&self.buffer) {
            self.kill_speculative_task();
            self.buffer.clear();
        }
    }

    fn try_spawn_speculative(&mut self, payload: ToolCallPayload) {
        let args_vec: Vec<String> = match &payload.arguments {
            serde_json::Value::Array(arr) => arr.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
            serde_json::Value::Object(obj) => obj.values().filter_map(|v| v.as_str().map(String::from)).collect(),
            _ => Vec::new(),
        };
        if tool::is_idempotent(&payload.name, &args_vec).unwrap_or(false)
            && let Ok(child) = tool::spawn_tool(&payload.name, &args_vec) {
                self.spec_task = Some(SpeculativeTask {
                    child,
                    tool_name: payload.name,
                    args: payload.arguments,
                });
            }
    }

    fn handle_buffer_overflow(&mut self) {
        if self.buffer.len() <= MAX_BUFFER_LEN {
            return;
        }

        let is_in_tool = self.buffer.contains(TOOL_MARKER_START);
        let is_in_edit = self.buffer.contains("<edit file=");

        if is_in_tool || is_in_edit {
            if self.buffer.len() > MAX_BUFFER_LEN * 16 {
                self.kill_speculative_task();
                // We keep the last chunk so we don't totally lose state, but for <edit> this might break.
                // However, <search> blocks should not exceed 64KB.
                self.buffer.clear();
            }
            return;
        }

        self.kill_speculative_task();
        let target_len = 100;
        let start = self.buffer.len().saturating_sub(target_len);
        let valid_start = (start..self.buffer.len())
            .find(|&i| self.buffer.is_char_boundary(i))
            .unwrap_or(self.buffer.len());
        self.buffer.drain(..valid_start);
    }

    fn has_diverged(line: &str) -> bool {
        if let Some(idx) = line.find(TOOL_MARKER_START) {
            let rest = &line[idx..];
            let chars_after = rest
                .chars()
                .filter(|c| !c.is_whitespace() && *c != '"' && *c != '{' && *c != '}')
                .count();
            if chars_after > 500 {
                return true;
            }
        }
        false
    }
}

impl Default for ToolInterceptor {
    fn default() -> Self {
        Self::new(false, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_tool_parse() {
        let mut interceptor = ToolInterceptor::new(false, true);
        let token1 = "```json\n";
        let token2 = r#"{"name": "rg", "arguments": ["pattern", "dir"]}"#;
        let token3 = "```";

        assert!(!interceptor.feed_token(token1));
        assert!(!interceptor.feed_token(token2));
        assert!(interceptor.feed_token(token3));

        let call = interceptor.detected_call.take().unwrap();
        assert_eq!(call.0, "rg");
        assert_eq!(call.1, serde_json::json!(["pattern", "dir"]));
    }

    #[test]
    fn test_multiline_json_tool_payloads_never_panic() {
        let mut interceptor = ToolInterceptor::new(false, true);
        interceptor.feed_token("```json\n");
        let chunk =
            "{\n  \"name\": \"test\",\n  \"arguments\": [\n    \"arg1\",\n    \"arg2\"\n  ]\n}\n";
        for line in chunk.lines() {
            interceptor.feed_token(line);
            interceptor.feed_token("\n");
        }
        interceptor.feed_token("```");
        let call = interceptor.detected_call.take().unwrap();
        assert_eq!(call.0, "test");
        assert_eq!(call.1, serde_json::json!(["arg1", "arg2"]));
    }

    #[test]
    fn test_handle_buffer_overflow_with_multibyte() {
        let mut interceptor = ToolInterceptor::new(false, true);
        let huge = "a".repeat(4090);
        interceptor.feed_token(&huge);
        interceptor.feed_token("Hello 🌍"); // forces overflow
        assert!(interceptor.buffer.len() <= 104); // 100 target + emoji
        assert!(interceptor.buffer.ends_with("🌍"));
    }

    #[test]
    fn test_handle_buffer_overflow() {
        let mut interceptor = ToolInterceptor::new(false, true);
        let token = "a".repeat(5000);
        interceptor.feed_token(&token);
        assert_eq!(interceptor.buffer.len(), 100);
    }

    #[test]
    fn test_interceptor_xml_parsing() {
        let mut interceptor = ToolInterceptor::new(false, true);
        interceptor.feed_token("<edit file=\"src/main.rs\">\n");
        assert_eq!(interceptor.detected_edit_path.as_deref(), Some("src/main.rs"));

        interceptor.feed_token("<search>\nfn main() {\n</search>\n");
        let search_block = interceptor.detected_search_block.as_ref().unwrap();
        assert_eq!(search_block.0, "src/main.rs");
        assert_eq!(search_block.1, "\nfn main() {\n");

        interceptor.feed_token("<replace>\nfn main() -> Result<()> {\n</replace>\n</edit>");
        let full_edit = interceptor.detected_full_edit.as_ref().unwrap();
        assert_eq!(full_edit.0, "src/main.rs");
        assert_eq!(full_edit.1, "\nfn main() {\n");
        assert_eq!(full_edit.2, "\nfn main() -> Result<()> {\n");
    }

    #[test]
    fn test_process_speculative_call() {
        let mut interceptor = ToolInterceptor::new(true, true);
        let token1 = "```json\n";
        let token2 = r#"{"name": "rg", "arguments": ["pattern", "dir"]}"#;

        assert!(!interceptor.feed_token(token1));
        assert!(!interceptor.feed_token(token2));
        assert!(interceptor.feed_token("```"));
    }

    #[test]
    fn test_tdd_enforcement_buffer_matching() {
        let mut interceptor = ToolInterceptor::new(false, true);
        interceptor.feed_token("<edi");
        interceptor.feed_token("t file=\"foo.py\">");
        assert!(interceptor.buffer.contains("<edit "));
        
        let mut interceptor2 = ToolInterceptor::new(false, true);
        interceptor2.feed_token("<ed");
        interceptor2.feed_token("it>");
        assert!(interceptor2.buffer.contains("<edit>"));
        
        // Ensure it doesn't accidentally trigger on non-related <edit tags without space
        let mut interceptor3 = ToolInterceptor::new(false, true);
        interceptor3.feed_token("<editor>");
        assert!(!interceptor3.buffer.contains("<edit "));
        assert!(!interceptor3.buffer.contains("<edit>"));
    }

    #[test]
    fn test_extract_json_payload() {
        let text = "```json\n{\"name\": \"cat\", \"arguments\": [\"django/db/utils.py\"]}\n```The fix";
        let payload = ToolInterceptor::extract_json_payload(text);
        assert!(payload.is_some(), "Failed to parse: {:?}", text);
    }

    #[test]
    fn test_no_tools_mode_ignores_json_but_parses_edits() {
        // detect_json_tools=false: JSON tool calls become plain text
        let mut interceptor = ToolInterceptor::new(false, false);

        // Feed a JSON tool call — should NOT be detected
        assert!(!interceptor.feed_token("```json\n"));
        assert!(!interceptor.feed_token(
            r#"{"name": "rg", "arguments": ["pattern", "dir"]}"#
        ));
        assert!(!interceptor.feed_token("```"));
        assert!(interceptor.detected_call.is_none());

        // Feed an edit tag — SHOULD still be detected
        assert!(!interceptor.feed_token("<edit file=\"src/main.rs\">\n"));
        assert_eq!(interceptor.detected_edit_path.as_deref(), Some("src/main.rs"));

        // Feed search block
        interceptor.feed_token("<search>\nfn main() {\n</search>\n");
        assert!(interceptor.detected_search_block.is_some());
        let (path, search) = interceptor.detected_search_block.as_ref().unwrap();
        assert_eq!(path, "src/main.rs");
        assert_eq!(search, "\nfn main() {\n");

        // Feed replace + close — should detect full edit
        interceptor.feed_token("<replace>\nfn main() {}\n</replace>\n</edit>");
        assert!(interceptor.detected_full_edit.is_some());
        let (path, _search, replace) = interceptor.detected_full_edit.as_ref().unwrap();
        assert_eq!(path, "src/main.rs");
        assert_eq!(replace, "\nfn main() {}\n");
    }

    #[test]
    fn test_no_tools_edit_tag_closed_detected() {
        // Edit tag closure should still fire when detect_json_tools=false
        let mut interceptor = ToolInterceptor::new(false, false);
        interceptor.feed_token("<edit file=\"lib.py\">\n");
        interceptor.feed_token("some content\n");
        interceptor.feed_token("</edit>\n");
        assert!(interceptor.edit_tag_closed);
        assert!(interceptor.detected_call.is_none()); // no JSON detected
    }
}
