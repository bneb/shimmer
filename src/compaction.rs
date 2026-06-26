//! RoPE-Safe KV Compaction Module
//!
//! This module implements Shimmer's sliding window memory architecture.
//! It tracks token zones (System, History, ToolOutput) and automatically
//! ejects the oldest reasoning blocks when the context window reaches capacity.

use llama_cpp_2::token::LlamaToken;

/// Represents the type of content contained within a contiguous block of tokens.
#[derive(Debug, Clone, PartialEq)]
pub enum ZoneType {
    /// The system instructions, which are preserved during compaction.
    System,
    /// History tokens from user or assistant interactions.
    History,
    /// Output resulting from tool executions.
    ToolOutput,
}

/// A contiguous block of tokens with an associated semantic zone.
#[derive(Debug, Clone)]
pub struct Page {
    pub start: usize,
    pub end: usize,
    pub zone: ZoneType,
}

/// The size of each contiguous token page.
pub const PAGE_SIZE: usize = 32;

/// Manages token blocks and orchestrates KV cache compaction.
pub struct ContextManager {
    /// Active pages tracking token zones.
    pub pages: Vec<Page>,
    /// The maximum token capacity of the context window.
    pub max_capacity: usize,
    /// The fill threshold at which compaction is triggered.
    pub threshold_pct: f64,
    /// Number of most-recent ToolOutput pages to preserve during compaction.
    pub recent_tool_outputs: usize,
}

/// Core implementation block for ContextManager operations.
impl ContextManager {
    /// Initializes a new ContextManager with maximum capacity and compaction threshold.
    pub fn new(max_capacity: usize, threshold_pct: f64) -> Self {
        Self { pages: Vec::new(), max_capacity, threshold_pct, recent_tool_outputs: 3 }
    }

    /// Pushes a new block of tokens into the context tracker.
    pub fn push_block(&mut self, zone: ZoneType, len: usize) {
        if len == 0 {
            return;
        }
        let start = self.pages.last().map(|b| b.end).unwrap_or(0);
        self.pages.push(Page { start, end: start + len, zone });
    }

    /// Extends the final block if it matches the zone type, otherwise pushes a new block.
    pub fn extend_last_block(&mut self, zone: ZoneType, mut len: usize) {
        if len == 0 {
            return;
        }
        if let Some(last) = self.pages.last_mut()
            && last.zone == zone
            && (last.end - last.start) < PAGE_SIZE
        {
            let capacity = PAGE_SIZE - (last.end - last.start);
            let chunk = len.min(capacity);
            last.end += chunk;
            len -= chunk;
        }
        self.push_block(zone, len);
    }

    /// Determines if generating `upcoming_len` tokens will breach the compaction threshold.
    pub fn needs_compaction(&self, upcoming_len: usize) -> bool {
        let current_size = self.pages.last().map(|b| b.end).unwrap_or(0);
        let max_allowed = (self.max_capacity as f64 * self.threshold_pct) as usize;
        current_size + upcoming_len > max_allowed
    }

    /// Identifies the best candidate block for eviction.
    /// Prioritizes oldest ToolOutput (skipping the most recent N), then oldest History.
    /// System prompts are never evicted.
    pub fn find_compaction_target(&self) -> Option<usize> {
        // Collect indices of non-empty ToolOutput pages
        let tool_indices: Vec<usize> = self
            .pages
            .iter()
            .enumerate()
            .filter(|(_, b)| b.zone == ZoneType::ToolOutput && b.end > b.start)
            .map(|(i, _)| i)
            .collect();

        // Skip the most recent N tool outputs; evict the oldest among the rest
        let skip = self.recent_tool_outputs;
        if tool_indices.len() > skip {
            return Some(tool_indices[0]);
        }
        // If all tool outputs are recent, fall through to evict oldest History

        for (i, block) in self.pages.iter().enumerate() {
            if block.zone == ZoneType::History && i > 0 && block.end > block.start {
                return Some(i);
            }
        }
        None
    }

    /// Removes a targeted block and shifts the history array to maintain contiguity.
    /// Returns the (start, end) indices of the removed block.
    pub fn compact(
        &mut self,
        token_history: &mut Vec<LlamaToken>,
        compressed_len: usize,
    ) -> Option<(usize, usize)> {
        let idx = self.find_compaction_target()?;

        let mut removed_page = self.pages.remove(idx);
        let original_start = removed_page.start;
        let original_end = removed_page.end;

        let page_len = original_end - original_start;
        let c_len = compressed_len.min(page_len);
        let shift_amount = page_len - c_len;

        // Shift token history, leaving c_len dummy tokens at the start
        token_history.drain(original_start..original_start + shift_amount);

        // Re-insert only if there is remaining content
        if c_len > 0 {
            removed_page.end = original_start + c_len;
            self.pages.insert(idx, removed_page);
            for block in self.pages.iter_mut().skip(idx + 1) {
                block.start -= shift_amount;
                block.end -= shift_amount;
            }
        } else {
            for block in self.pages.iter_mut().skip(idx) {
                block.start -= shift_amount;
                block.end -= shift_amount;
            }
        }

        Some((original_start, original_end))
    }

    /// Reverts the context manager state to a specific token length, used during tool interception rollback.
    pub fn truncate_history(&mut self, new_len: usize) {
        while let Some(last) = self.pages.last_mut() {
            if last.start >= new_len {
                self.pages.pop();
            } else if last.end > new_len {
                last.end = new_len;
                if last.end == last.start {
                    self.pages.pop();
                }
                break;
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
/// Test suite for compaction logic.
mod tests {
    use llama_cpp_2::token::LlamaToken;

    use super::*;

    #[test]
    /// Tests pushing blocks and correctly compacting them.
    fn test_push_and_compact() {
        let mut cm = ContextManager::new(100, 0.9);
        cm.recent_tool_outputs = 0; // Allow evicting all tool outputs
        let mut hist = vec![LlamaToken::new(0); 30];

        cm.push_block(ZoneType::System, 10);
        cm.push_block(ZoneType::History, 10);
        cm.push_block(ZoneType::ToolOutput, 10);

        assert_eq!(cm.pages.len(), 3);
        assert_eq!(cm.pages[2].start, 20);
        assert_eq!(cm.pages[2].end, 30);

        assert!(cm.needs_compaction(70));

        let (start, end) = cm.compact(&mut hist, 64).unwrap();
        assert_eq!(start, 20); // Removed ToolOutput
        assert_eq!(end, 30);
        assert_eq!(hist.len(), 30);
        assert_eq!(cm.pages.len(), 3);
        assert_eq!(cm.pages[2].end, 30);
    }

    #[test]
    /// Tests extending a block properly when zone types match.
    fn test_extend_last_block() {
        let mut cm = ContextManager::new(100, 0.9);
        cm.extend_last_block(ZoneType::System, 10);
        cm.extend_last_block(ZoneType::System, 5);
        assert_eq!(cm.pages.len(), 1);
        assert_eq!(cm.pages[0].end, 15);

        cm.extend_last_block(ZoneType::History, 5);
        assert_eq!(cm.pages.len(), 2);
    }

    #[test]
    /// Tests truncating history back to a specific token length.
    fn test_truncate_history() {
        let mut cm = ContextManager::new(100, 0.9);
        cm.push_block(ZoneType::System, 10);
        cm.push_block(ZoneType::History, 10);
        cm.push_block(ZoneType::History, 5); // 3rd block ends at 25

        cm.truncate_history(15);
        assert_eq!(cm.pages.len(), 2);
        assert_eq!(cm.pages[1].end, 15);
    }

    #[test]
    /// Tests fallback targeting when no primary eviction candidate is present.
    fn test_find_compaction_fallback() {
        let mut cm = ContextManager::new(100, 0.9);
        cm.push_block(ZoneType::System, 10);
        cm.push_block(ZoneType::History, 10);
        cm.push_block(ZoneType::History, 10);

        // No ToolOutput, so it should target oldest History (index 1)
        assert_eq!(cm.find_compaction_target(), Some(1));
    }

    #[test]
    /// Ensures no compaction occurs when conditions are insufficient.
    fn test_find_compaction_none() {
        let mut cm = ContextManager::new(100, 0.9);
        cm.push_block(ZoneType::System, 10);
        assert_eq!(cm.find_compaction_target(), None);
    }

    #[test]
    /// Verifies that compacting a block shifts indices for all subsequent blocks.
    fn test_compact_shifts_subsequent_blocks() {
        let mut cm = ContextManager::new(100, 0.9);
        cm.recent_tool_outputs = 0; // Allow evicting all tool outputs
        let mut hist = vec![LlamaToken::new(0); 40];

        cm.push_block(ZoneType::System, 10);
        cm.push_block(ZoneType::History, 10);
        cm.push_block(ZoneType::ToolOutput, 10);
        cm.push_block(ZoneType::History, 10);

        let (start, end) = cm.compact(&mut hist, 64).unwrap();
        assert_eq!(start, 20);
        assert_eq!(end, 30);
        assert_eq!(cm.pages.len(), 4);
        assert_eq!(cm.pages[2].start, 20);
        assert_eq!(cm.pages[2].end, 30);
        assert_eq!(cm.pages[3].start, 30);
        assert_eq!(cm.pages[3].end, 40);
    }

    #[test]
    /// Tests that history truncation cleanly exits when target length exceeds blocks.
    fn test_truncate_history_early_exit() {
        let mut cm = ContextManager::new(100, 0.9);
        cm.push_block(ZoneType::System, 10);
        cm.push_block(ZoneType::History, 10);

        // Truncate to a length that is beyond the current blocks should hit `break`
        cm.truncate_history(30);
        assert_eq!(cm.pages.len(), 2);
    }

    #[test]
    /// Validates RoPE-safe behavior during history shift and truncation.
    fn test_rope_safe_kv_shifting() {
        let mut cm = ContextManager::new(100, 0.9);
        let mut hist = (0..50).map(LlamaToken::new).collect::<Vec<_>>();

        cm.push_block(ZoneType::System, 10);
        cm.push_block(ZoneType::History, 20);
        cm.push_block(ZoneType::History, 20);

        let (start, end) = cm.compact(&mut hist, 5).unwrap();
        assert_eq!(start, 10);
        assert_eq!(end, 30);
        assert_eq!(hist.len(), 35);
        assert_eq!(hist[0].0, 0);
        assert_eq!(hist[9].0, 9);
        assert_eq!(hist[10].0, 25);
        assert_eq!(hist[14].0, 29);
        assert_eq!(hist[15].0, 30);
    }

    #[test]
    /// Verifies that recent tool outputs are preserved during compaction.
    fn test_preserves_recent_tool_outputs() {
        let mut cm = ContextManager::new(100, 0.9);
        // Default recent_tool_outputs is 3
        let mut hist = vec![LlamaToken::new(0); 50];

        cm.push_block(ZoneType::System, 10);
        // 4 tool output pages — only the oldest (1st) should be evictable
        cm.push_block(ZoneType::ToolOutput, 5); // oldest — evictable
        cm.push_block(ZoneType::ToolOutput, 5); // 2nd oldest — evictable
        cm.push_block(ZoneType::ToolOutput, 5); // recent (3rd newest)
        cm.push_block(ZoneType::ToolOutput, 5); // most recent (4th)

        // 4 tool outputs, skip=3 → first one (index 1, start=10) gets evicted
        let (start, _) = cm.compact(&mut hist, 0).unwrap();
        assert_eq!(start, 10); // Oldest ToolOutput at index 1
    }

    #[test]
    /// Tests edge case behavior when pushing blocks of length 0.
    fn test_push_and_extend_zero_len() {
        let mut cm = ContextManager::new(100, 0.9);
        cm.push_block(ZoneType::System, 0);
        assert_eq!(cm.pages.len(), 0);

        cm.push_block(ZoneType::System, 10);
        cm.extend_last_block(ZoneType::System, 0);
        assert_eq!(cm.pages.len(), 1);
        assert_eq!(cm.pages[0].end, 10);
    }
}
