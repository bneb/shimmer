//! KV cache compaction and entropy-based summarization.
//!
//! Manages context window limits by evicting low-value token ranges
//! and summarizing tool output pages to preserve key information.

use super::state::EngineState;
use crate::compaction::ZoneType;
use anyhow::Result;
use std::collections::HashSet;

impl super::Agent {
    /// Enforces context window limits by triggering compaction when necessary.
    /// Uses entropy-based summarization: for tool output pages, extracts
    /// high-value lines (errors, paths, key findings) before discarding.
    pub(crate) fn enforce_compaction(
        &self,
        state: &mut EngineState,
        draft_size: usize,
    ) -> Result<()> {
        if !state.compactor.needs_compaction(draft_size) {
            return Ok(());
        }
        self.summarize_if_tool_output(state);

        let _original_history = state.history.clone();
        if let Some((start, end)) = state.compactor.compact(&mut state.history, 0) {
            let seq_id = 0_u32;
            let shift_amount = end - start;

            state
                .ctx
                .clear_kv_cache_seq(Some(seq_id), Some(start as u32), Some(end as u32))
                .map_err(|e| anyhow::anyhow!("Compaction KV Clear error: {:?}", e))?;

            state
                .ctx
                .kv_cache_seq_add(
                    seq_id as i32,
                    Some(end as u32),
                    None,
                    -(shift_amount as i32),
                )
                .map_err(|e| anyhow::anyhow!("Compaction KV Shift error: {:?}", e))?;

            state.n_cur -= shift_amount as i32;
        }
        Ok(())
    }

    /// If the next compaction target is a ToolOutput page, replace its content
    /// with a high-entropy summary before compaction discards the rest.
    fn summarize_if_tool_output(&self, state: &mut EngineState) {
        let target_idx = match state.compactor.find_compaction_target() {
            Some(i) if state.compactor.pages[i].zone == ZoneType::ToolOutput => i,
            _ => return,
        };
        let page_start = state.compactor.pages[target_idx].start;
        let page_end = state.compactor.pages[target_idx].end;
        let tokens: Vec<_> = state.history[page_start..page_end].to_vec();
        let text: String = tokens.iter().filter_map(|t| {
            self.main_model.token_to_piece_bytes(*t, 256, true, None).ok()
        }).map(|b| String::from_utf8_lossy(&b).into_owned()).collect();
        if text.is_empty() { return; }

        let summary = Self::extract_high_entropy_summary(&text);
        if summary.len() >= text.len() / 2 { return; }

        let summary_tokens = self.main_model.str_to_token(
            &summary, llama_cpp_2::model::AddBos::Never
        ).unwrap_or_default();
        if summary_tokens.len() >= tokens.len() { return; }

        let new_len = summary_tokens.len();
        let diff = page_end - page_start - new_len;
        state.history.splice(page_start..page_end, summary_tokens);
        state.compactor.pages[target_idx].end = page_start + new_len;
        for p in state.compactor.pages.iter_mut().skip(target_idx + 1) {
            p.start = p.start.saturating_sub(diff);
            p.end = p.end.saturating_sub(diff);
        }
        if diff > 0 {
            state.n_cur -= diff as i32;
            let _ = state.ctx.clear_kv_cache_seq(
                Some(0_u32),
                Some((page_start + new_len) as u32),
                Some(page_end as u32),
            );
            let _ = state.ctx.kv_cache_seq_add(
                0_i32,
                Some(page_end as u32),
                None,
                -(diff as i32),
            );
        }
    }

    /// Extracts high-entropy lines from tool output for compaction summarization.
    /// Keeps error messages, file paths, line-numbered matches, and boundary context.
    pub fn extract_high_entropy_summary(text: &str) -> String {
        let keywords = ["Error", "Traceback", "FAIL", "Exception", "assert", "warning",
                        "error", "fail", "panic", "abort", "timeout", "killed"];
        let mut kept: HashSet<usize> = HashSet::new();
        let lines: Vec<&str> = text.lines().collect();
        let n = lines.len();
        for (i, line) in lines.iter().enumerate() {
            let s = line.trim();
            if keywords.iter().any(|&k| s.contains(k)) {
                for j in i.saturating_sub(3)..=(i + 3).min(n.saturating_sub(1)) { kept.insert(j); }
            }
            if s.contains('/') && (s.contains(".py") || s.contains(".rs") || s.contains(".js")) {
                kept.insert(i);
            }
            if let Some((path, rest)) = s.split_once(':')
                && let Some((num, _)) = rest.split_once(':')
                    && num.trim().parse::<usize>().is_ok() && path.contains('.') { kept.insert(i); }
        }
        for i in 0..2.min(n) { kept.insert(i); }
        for i in n.saturating_sub(2)..n { kept.insert(i); }
        if kept.is_empty() { return text.chars().take(300).collect(); }
        let mut sorted: Vec<usize> = kept.into_iter().collect();
        sorted.sort_unstable();
        let mut result = String::from("[summary]\n");
        let mut last: i32 = -2;
        for &i in &sorted {
            if i as i32 > last + 1 { result.push_str("...\n"); }
            result.push_str(lines[i].trim());
            result.push('\n');
            last = i as i32;
        }
        if result.len() > 500 { result.truncate(500); result.push_str("..."); }
        result
    }
}
