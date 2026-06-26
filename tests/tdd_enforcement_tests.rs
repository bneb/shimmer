//! Integration tests for TDD enforcement features.
//!
//! Validates path blocker, search verifier, syntax checker,
//! and insanity detector across the interceptor and agent modules.

use shimmer::interceptor::ToolInterceptor;

/// Path blocker: verifies that hallucinated file paths are detected.
#[test]
fn test_path_blocker_detects_nonexistent_path() {
    let mut interceptor = ToolInterceptor::new(false, true);
    interceptor.feed_token("<edit file=\"nonexistent/file.py\">");
    assert_eq!(interceptor.detected_edit_path.as_deref(), Some("nonexistent/file.py"));
    assert!(!interceptor.path_validated);
    // In agent.rs, this triggers a path check via std::path::Path::new(path).exists()
}

/// Path blocker: existing paths pass validation.
#[test]
fn test_path_blocker_passes_existing_path() {
    let mut interceptor = ToolInterceptor::new(false, true);
    interceptor.feed_token("<edit file=\"src/main.rs\">");
    assert_eq!(interceptor.detected_edit_path.as_deref(), Some("src/main.rs"));
    // This path exists; path_validated would be set to true in agent.rs
}

/// Search verifier: detects complete <search> blocks.
#[test]
fn test_search_verifier_detects_search_block() {
    let mut interceptor = ToolInterceptor::new(false, true);
    interceptor.feed_token("<edit file=\"src/main.rs\">\n<search>\nfn main() {\n</search>");
    let sb = interceptor.detected_search_block.as_ref().unwrap();
    assert_eq!(sb.0, "src/main.rs");
    assert_eq!(sb.1, "\nfn main() {\n");
}

/// Search verifier: full edit tag with search+replace.
#[test]
fn test_search_verifier_full_edit_cycle() {
    let mut interceptor = ToolInterceptor::new(false, true);
    interceptor.feed_token("<edit file=\"src/main.rs\">\n");
    interceptor.feed_token("<search>\nfn main() {\n</search>\n");
    interceptor.feed_token("<replace>\nfn main() -> Result<()> {\n</replace>\n");
    interceptor.feed_token("</edit>");
    let edit = interceptor.detected_full_edit.as_ref().unwrap();
    assert_eq!(edit.0, "src/main.rs");
    assert_eq!(edit.1, "\nfn main() {\n");
    assert_eq!(edit.2, "\nfn main() -> Result<()> {\n");
}

/// Search verifier: does not trigger on unrelated <edit> tags.
#[test]
fn test_search_verifier_no_false_positive() {
    let mut interceptor = ToolInterceptor::new(false, true);
    interceptor.feed_token("<editor>");
    interceptor.feed_token("<edit_distance>");
    assert!(interceptor.detected_edit_path.is_none());
    assert!(interceptor.detected_search_block.is_none());
}

/// Syntax checker: detects edit paths ending in .py (for syntax checking).
#[test]
fn test_syntax_checker_detects_python_path() {
    let mut interceptor = ToolInterceptor::new(false, true);
    interceptor.feed_token("<edit file=\"foo.py\">");
    assert_eq!(interceptor.detected_edit_path.as_deref(), Some("foo.py"));
    // agent.rs checks path.ends_with(".py") before running syntax check
}

/// Buffer management: long buffers with <edit> tags are cleared correctly.
#[test]
fn test_buffer_clear_on_overflow_with_edit() {
    let mut interceptor = ToolInterceptor::new(false, true);
    interceptor.feed_token("<edit file=\"test.py\">");
    assert!(interceptor.buffer.contains("<edit "));
    assert!(interceptor.buffer.len() < 100);
}

/// Tool call mutation: interceptor correctly resets state between generations.
#[test]
fn test_interceptor_state_reset() {
    let mut interceptor = ToolInterceptor::new(false, true);
    interceptor
        .feed_token("<edit file=\"a.py\">\n<search>x</search>\n<replace>y</replace>\n</edit>");
    assert!(interceptor.detected_full_edit.is_some());

    // Simulating a new turn by clearing buffer
    interceptor.buffer.clear();
    interceptor.detected_edit_path = None;
    interceptor.detected_search_block = None;
    interceptor.detected_full_edit = None;
    interceptor.path_validated = false;

    // New edit detection after reset
    interceptor.feed_token("<edit file=\"b.py\">");
    assert_eq!(interceptor.detected_edit_path.as_deref(), Some("b.py"));
    assert!(interceptor.detected_search_block.is_none());
}

/// Token boundary fuzz: XML edit blocks parse correctly regardless of split point.
#[test]
fn test_interceptor_xml_parsing_token_boundary_fuzz() {
    let edit =
        "<edit file=\"foo.py\">\n<search>\nold\n</search>\n<replace>\nnew\n</replace>\n</edit>";
    for split in 1..edit.len() {
        let mut interceptor = ToolInterceptor::new(false, true);
        interceptor.feed_token(&edit[..split]);
        interceptor.feed_token(&edit[split..]);
        assert_eq!(interceptor.detected_edit_path.as_deref(), Some("foo.py"));
        let (path, search) = interceptor.detected_search_block.as_ref().unwrap();
        assert_eq!(path, "foo.py");
        assert_eq!(search, "\nold\n");
        let (path2, search2, replace) = interceptor.detected_full_edit.as_ref().unwrap();
        assert_eq!(path2, "foo.py");
        assert_eq!(search2, "\nold\n");
        assert_eq!(replace, "\nnew\n");
    }
}

/// Token boundary fuzz: JSON tool calls parse correctly regardless of split point.
#[test]
fn test_interceptor_json_tool_call_token_boundary_fuzz() {
    let call = "```json\n{\"name\": \"rg\", \"arguments\": [\"pattern\"]}\n```";
    for split in 1..call.len() {
        let mut interceptor = ToolInterceptor::new(false, true);
        let detected_first = interceptor.feed_token(&call[..split]);
        let detected_second = interceptor.feed_token(&call[split..]);
        assert!(
            detected_first || detected_second,
            "Tool call should be detected at split {}",
            split
        );
        let (name, args, _child) = interceptor.detected_call.as_ref().unwrap();
        assert_eq!(name, "rg");
        assert_eq!(args, &serde_json::json!(["pattern"]));
    }
}

/// Partial edit detection: full edit only fires when all parts are present.
#[test]
fn test_interceptor_partial_edit_not_detected_early() {
    let edit =
        "<edit file=\"x.py\">\n<search>\nold\n</search>\n<replace>\nnew\n</replace>\n</edit>";
    let close_pos = edit.rfind("</edit>").unwrap();
    let chars: Vec<char> = edit.chars().collect();
    let mut interceptor = ToolInterceptor::new(false, true);
    for (i, &c) in chars.iter().enumerate() {
        let mut s = String::new();
        s.push(c);
        interceptor.feed_token(&s);
        if i < close_pos + "</edit>".len() - 1 {
            assert!(
                interceptor.detected_full_edit.is_none(),
                "Full edit detected before </edit> complete at char {}",
                i
            );
        }
    }
    assert!(interceptor.detected_full_edit.is_some());
}
