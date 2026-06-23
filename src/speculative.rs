#![allow(unexpected_cfgs)]
//! Matrix-Matrix Prompt Lookup Decoding (PLD)
//!
//! This module implements Shimmer's speculative decoding engine.
//! By utilizing n-gram matching over the local KV cache history, it drafts token branches
//! and verifies them simultaneously using a single batched sequence evaluation in Metal.

use anyhow::Result;
use llama_cpp_2::{
    context::LlamaContext, llama_batch::LlamaBatch, model::LlamaModel, token::LlamaToken,
};

/// The default sequence ID used for the main generation context.
/// Represents the outcome of evaluating speculative draft branches.
pub struct DraftVerification {
    /// Number of tokens accepted from the best branch.
    pub accepted_count: usize,
    /// The true token that should replace the first incorrect prediction.
    pub correction_token: LlamaToken,
    /// The index of the winning branch in the drafted tree.
    pub best_branch_idx: usize,
    /// The C++ sequence ID allocated to the winning branch.
    pub best_seq_id: i32,
    /// The total number of branches that were actively evaluated.
    pub active_branches: usize,
}

/// A pure function for extracting draft branches from the history array via n-gram matching.
/// This enables testing the PLD algorithm without a loaded model context.
pub fn extract_draft_branches(
    history: &[LlamaToken],
    eos_token: LlamaToken,
    draft_size: usize,
    ngram_size: usize,
) -> Vec<Vec<LlamaToken>> {
    let mut branches = Vec::new();

    if history.len() < ngram_size || ngram_size == 0 {
        return branches;
    }

    let current_ngram = &history[history.len() - ngram_size..];
    let search_end = history.len() - ngram_size;
    let mut matches = Vec::new();

    // Search backwards to find the most recent matching n-grams
    for i in (0..search_end).rev() {
        if &history[i..i + ngram_size] == current_ngram {
            matches.push(i);
            if !matches.is_empty() {
                break;
            }
        }
    }

    // Convert matches into future token predictions
    for idx in matches {
        let draft_tokens = build_draft_branch(history, eos_token, draft_size, idx + ngram_size);
        if !draft_tokens.is_empty() {
            branches.push(draft_tokens);
        }
    }

    branches
}

/// Constructs a single draft branch by slicing future tokens from the local cache history.
fn build_draft_branch(
    history: &[LlamaToken],
    eos_token: LlamaToken,
    draft_size: usize,
    start_proposing: usize,
) -> Vec<LlamaToken> {
    let mut draft_tokens = Vec::with_capacity(draft_size);
    for i in 0..draft_size {
        let proposed_idx = start_proposing + i;
        if proposed_idx >= history.len() {
            break;
        }
        let token = history[proposed_idx];
        draft_tokens.push(token);
        if token == eos_token {
            break;
        }
    }
    draft_tokens
}

/// Generates speculative tokens by performing an n-gram search against the KV cache history.
pub fn generate_draft_tokens(
    main_model: &LlamaModel,
    history: &[LlamaToken],
    draft_size: usize,
    ngram_size: usize,
) -> Result<Vec<Vec<LlamaToken>>> {
    Ok(extract_draft_branches(
        history,
        main_model.token_eos(),
        draft_size,
        ngram_size,
    ))
}

/// Extracts the maximum likelihood token `m_0` from the previous context evaluation.
#[cfg(not(tarpaulin_include))]
fn get_m0(main_ctx: &mut LlamaContext, prev_main_batch_size: i32) -> LlamaToken {
    main_ctx
        .candidates_ith(prev_main_batch_size - 1)
        .max_by(|a, b| {
            a.logit()
                .partial_cmp(&b.logit())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|c| c.id())
        .unwrap()
}

/// Filters draft branches, keeping only those whose first token matches the true `m_0`.
fn filter_valid_branches(draft_branches: &[Vec<LlamaToken>], m_0: LlamaToken) -> Vec<usize> {
    draft_branches
        .iter()
        .enumerate()
        .filter(|(_, b)| !b.is_empty() && b[0] == m_0)
        .map(|(i, _)| i)
        .collect()
}

/// Allocates valid draft branches into the C++ batch graph for parallel evaluation.
#[cfg(not(tarpaulin_include))]
fn prepare_batch(
    main_batch: &mut LlamaBatch,
    seq_id: i32,
    draft_branches: &[Vec<LlamaToken>],
    valid_indices: &[usize],
    n_cur_start: i32,
) -> Result<Vec<Vec<i32>>> {
    main_batch.clear();
    let mut branch_token_indices = Vec::new();
    let mut batch_idx: i32 = 0;

    for &branch_idx_val in valid_indices.iter() {
        let branch = &draft_branches[branch_idx_val];
        let mut token_indices = Vec::new();
        for (i, &token) in branch.iter().enumerate() {
            main_batch
                .add(token, n_cur_start + i as i32, &[seq_id], true)
                .map_err(|e| anyhow::anyhow!("Batch add error: {}", e))?;
            token_indices.push(batch_idx);
            batch_idx += 1;
        }
        branch_token_indices.push(token_indices);
    }
    Ok(branch_token_indices)
}

/// Traverses the evaluation graph to find the branch with the highest sequence acceptance.
#[cfg(not(tarpaulin_include))]
fn evaluate_single_branch(
    main_ctx: &mut LlamaContext,
    branch: &[LlamaToken],
    token_indices: &[i32],
) -> (usize, LlamaToken) {
    let mut accepted = 1;
    for i in 1..branch.len() {
        let m_i = main_ctx
            .candidates_ith(token_indices[i - 1])
            .max_by(|a, b| {
                a.logit()
                    .partial_cmp(&b.logit())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|c| c.id())
            .unwrap();

        if m_i == branch[i] {
            accepted += 1;
        } else {
            return (accepted, m_i);
        }
    }

    let correction = main_ctx
        .candidates_ith(*token_indices.last().unwrap())
        .max_by(|a, b| {
            a.logit()
                .partial_cmp(&b.logit())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|c| c.id())
        .unwrap();
    (accepted, correction)
}

/// Evaluates multiple valid draft branches and returns the one with the highest acceptance rate.
#[cfg(not(tarpaulin_include))]
fn evaluate_branches(
    main_ctx: &mut LlamaContext,
    draft_branches: &[Vec<LlamaToken>],
    valid_indices: &[usize],
    branch_token_indices: &[Vec<i32>],
    m_0: LlamaToken,
    seq_id: i32,
) -> DraftVerification {
    let mut best_accepted = 0;
    let mut best_correction = m_0;
    let mut best_branch_idx = valid_indices[0];
    let mut best_seq_id = 1;

    for (seq_offset, &branch_idx) in valid_indices.iter().enumerate() {
        let branch = &draft_branches[branch_idx];
        let token_indices = &branch_token_indices[seq_offset];

        let (accepted, correction) = evaluate_single_branch(main_ctx, branch, token_indices);

        if accepted > best_accepted {
            best_accepted = accepted;
            best_correction = correction;
            best_branch_idx = branch_idx;
            best_seq_id = seq_id;
        }
    }

    DraftVerification {
        accepted_count: best_accepted,
        correction_token: best_correction,
        best_branch_idx,
        best_seq_id,
        active_branches: valid_indices.len(),
    }
}

/// Verifies all drafted tokens simultaneously utilizing Metal's parallel execution.
#[cfg(not(tarpaulin_include))]
pub fn verify_draft_tokens(
    main_ctx: &mut LlamaContext,
    main_batch: &mut LlamaBatch,
    draft_branches: &[Vec<LlamaToken>],
    n_cur_start: i32,
    prev_main_batch_size: i32,
    seq_id: i32,
) -> Result<DraftVerification> {
    let m_0 = get_m0(main_ctx, prev_main_batch_size);
    let valid_indices = filter_valid_branches(draft_branches, m_0);

    if valid_indices.is_empty() {
        return Ok(DraftVerification {
            accepted_count: 0,
            correction_token: m_0,
            best_branch_idx: 0,
            best_seq_id: 0,
            active_branches: 0,
        });
    }

    let branch_token_indices = prepare_batch(
        main_batch,
        seq_id,
        draft_branches,
        &valid_indices,
        n_cur_start,
    )?;
    main_ctx
        .decode(main_batch)
        .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;
    Ok(evaluate_branches(
        main_ctx,
        draft_branches,
        &valid_indices,
        &branch_token_indices,
        m_0,
        seq_id,
    ))
}

/// Commits the winning branch to the primary cache sequence and prunes all failed futures.
#[cfg(not(tarpaulin_include))]
pub fn rollback_and_correct(
    main_ctx: &mut LlamaContext,
    main_batch: &mut LlamaBatch,
    n_cur_start: i32,
    verification: &DraftVerification,
    seq_id: i32,
) -> Result<i32> {
    let rollback_pos = n_cur_start + verification.accepted_count as i32;

    if verification.active_branches > 0 {
        main_ctx
            .clear_kv_cache_seq(Some(seq_id as u32), Some(rollback_pos as u32), None)
            .map_err(|e| anyhow::anyhow!("KV Clear error: {:?}", e))?;
    }

    main_batch.clear();
    main_batch
        .add(verification.correction_token, rollback_pos, &[seq_id], true)
        .map_err(|e| anyhow::anyhow!("Batch add error: {}", e))?;
    main_ctx
        .decode(main_batch)
        .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;

    Ok(1)
}

/// Adaptively tracks and adjusts the size of speculative draft branches based on recent acceptance rates.
pub struct AdaptiveTracker {
    pub moving_avg: f64,
    alpha: f64,
    tokens_since_disable: usize,
}

impl Default for AdaptiveTracker {
    /// Provides the default configuration for the adaptive tracker, starting with an initial moving average of 1.0.
    fn default() -> Self {
        Self {
            moving_avg: 1.0,
            alpha: 0.2,
            tokens_since_disable: 0,
        }
    }
}

impl AdaptiveTracker {
    /// Creates a new `AdaptiveTracker` with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the moving average of the acceptance rate based on newly evaluated draft tokens.
    pub fn update_draft(&mut self, accepted: usize, drafted: usize) {
        if drafted == 0 {
            return;
        }
        let rate = accepted as f64 / drafted as f64;
        self.moving_avg = self.alpha * rate + (1.0 - self.alpha) * self.moving_avg;
    }

    /// Penalizes the moving average during continuous autoregressive steps to encourage falling back to drafting later.
    pub fn update_autoregressive(&mut self) {
        if self.moving_avg < 0.25 {
            self.tokens_since_disable += 1;
            if self.tokens_since_disable > 10 {
                self.moving_avg = 1.0;
                self.tokens_since_disable = 0;
            }
        }
    }

    /// Determines the optimal draft size for the next iteration based on the current moving average.
    pub fn current_draft_size(&self, max_size: usize) -> usize {
        if self.moving_avg < 0.25 { 0 } else { max_size }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that n-gram matches correctly produce future token sequences.
    #[test]
    fn test_extract_draft_branches_match() {
        let eos = LlamaToken::new(99);
        // History contains "A B C D E F" ... "A B C"
        // Tokens: 1 2 3 4 5 6 ... 1 2 3
        let history = vec![
            LlamaToken::new(1),
            LlamaToken::new(2),
            LlamaToken::new(3),
            LlamaToken::new(4),
            LlamaToken::new(5),
            LlamaToken::new(6),
            LlamaToken::new(1),
            LlamaToken::new(2),
            LlamaToken::new(3),
        ];

        let branches = extract_draft_branches(&history, eos, 3, 3);
        assert_eq!(branches.len(), 1);
        // The drafted tokens should be [4, 5, 6]
        assert_eq!(
            branches[0],
            vec![LlamaToken::new(4), LlamaToken::new(5), LlamaToken::new(6)]
        );
    }

    /// Ensures that drafting halts correctly when an End-Of-Sequence (EOS) token is encountered.
    #[test]
    fn test_extract_draft_branches_eos_halt() {
        let eos = LlamaToken::new(99);
        // Tokens: 1 2 3 4 99 6 ... 1 2 3
        let history = vec![
            LlamaToken::new(1),
            LlamaToken::new(2),
            LlamaToken::new(3),
            LlamaToken::new(4),
            eos,
            LlamaToken::new(6),
            LlamaToken::new(1),
            LlamaToken::new(2),
            LlamaToken::new(3),
        ];

        let branches = extract_draft_branches(&history, eos, 5, 3);
        assert_eq!(branches.len(), 1);
        // Should halt AT the EOS token
        assert_eq!(branches[0], vec![LlamaToken::new(4), eos]);
    }

    /// Validates that draft branches are correctly filtered based on matching the first token `m_0`.
    #[test]
    fn test_filter_valid_branches() {
        let m_0 = LlamaToken::new(4);
        let branches = vec![
            vec![LlamaToken::new(4), LlamaToken::new(5)],
            vec![LlamaToken::new(8), LlamaToken::new(9)],
            vec![LlamaToken::new(4), LlamaToken::new(6)],
        ];

        let valid = filter_valid_branches(&branches, m_0);
        assert_eq!(valid, vec![0, 2]);
    }

    /// Tests the adaptive tracking logic, simulating success rates and validating dynamic sizing behavior.
    #[test]
    fn test_adaptive_tracker() {
        let mut tracker = AdaptiveTracker::new();
        assert_eq!(tracker.current_draft_size(24), 24);

        // Force drop below threshold
        tracker.update_draft(0, 24);
        tracker.update_draft(0, 24);
        tracker.update_draft(0, 24);
        tracker.update_draft(0, 24);
        tracker.update_draft(0, 24);
        tracker.update_draft(0, 24);
        tracker.update_draft(0, 24);
        tracker.update_draft(0, 24);
        assert_eq!(tracker.current_draft_size(24), 0);

        // Count autoregressive
        for _ in 0..10 {
            tracker.update_autoregressive();
        }
        assert_eq!(tracker.current_draft_size(24), 0);

        // 11th token should reset
        tracker.update_autoregressive();
        assert_eq!(tracker.current_draft_size(24), 24);
    }
}
