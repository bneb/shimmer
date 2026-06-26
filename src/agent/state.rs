//! Execution state types for the Shimmer agent engine.
//!
//! Contains configuration, sampling parameters, benchmark results,
//! and per-agent execution state used across all agent sub-modules.

use crate::compaction::ContextManager;
use crate::speculative;

pub const BATCH_SIZE: usize = 512;
pub(crate) const CTX_SIZE: u32 = 16384;
pub const MAX_TOKENS: usize = 8192;
pub(crate) const MAX_TOKEN_BYTES: usize = 128;
pub(crate) const COMPACT_THRESH: f64 = 0.9;
pub(crate) const BATCH_SEQ_ID: i32 = 0;

/// Tool call limit: at this count, further tool calls are rejected with a
/// nudge to produce an edit instead.
pub(crate) const HARD_TOOL_LIMIT: usize = 8;
/// Tool call limit: at this count, generation is aborted entirely.
pub(crate) const STUBBORN_ABORT_LIMIT: usize = 11;

/// Contains metrics for a completed generation cycle.
pub struct BenchmarkResult {
    pub generated_text: String,
    pub token_count: usize,
    pub duration_secs: f64,
    pub tps: f64,
}

/// Sampling hyperparameters for token selection.
#[derive(Clone, Debug)]
pub struct SampleConfig {
    /// Temperature for logit scaling. 0.0 = greedy argmax.
    pub temperature: f32,
    /// Top-k filtering. 0 = disabled (all candidates considered).
    pub top_k: usize,
    /// Repetition penalty applied to tokens in the recent window.
    pub repetition_penalty: f32,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self { temperature: 0.0, top_k: 0, repetition_penalty: 1.0 }
    }
}

/// Configuration for agent behavior and speculative decoding.
#[derive(Clone, Debug, Default)]
pub struct AgentConfig {
    pub use_speculative: bool,
    pub draft_size: usize,
    pub ngram_size: usize,
    pub enable_time_travel: bool,
    pub execute_tools_locally: bool,
    pub enable_tdd_enforcement: bool,
    pub enable_search_verifier: bool,
    pub enable_path_blocker: bool,
    pub enable_insanity_detector: bool,
    pub enable_syntax_checker: bool,
    pub enable_blind_edit_blocker: bool,
    /// When true, the JSON tool-call detector is disabled. The model generates
    /// straight through — edit tag parsing still runs, but ```json markers are
    /// treated as plain text. Used by agentless single-shot mode.
    pub disable_tool_interceptor: bool,
    pub sample_config: SampleConfig,
}

/// Holds execution state for a single-agent generation loop.
pub(crate) struct EngineState<'a> {
    pub ctx: llama_cpp_2::context::LlamaContext<'a>,
    pub batch: llama_cpp_2::llama_batch::LlamaBatch<'a>,
    pub history: Vec<llama_cpp_2::token::LlamaToken>,
    pub compactor: ContextManager,
    pub sampler: Option<llama_cpp_2::sampling::LlamaSampler>,
    pub sample_config: SampleConfig,
    pub n_cur: i32,
    pub prev_size: i32,
    pub total_generated: usize,
    pub adaptive_tracker: speculative::AdaptiveTracker,
    pub pending_utf8: Vec<u8>,
    pub tool_calls: usize,
    pub tests_since_last_edit: usize,
    pub tool_history: std::collections::HashSet<String>,
    pub edit_history: std::collections::HashSet<u64>,
    pub continuation_count: usize,
    pub last_output_token_count: usize,
    pub pending_tool: Option<(String, serde_json::Value, Option<std::process::Child>)>,
}

/// Tracks independent state for a single agent running inside a Swarm.
pub struct AgentState {
    pub history: Vec<llama_cpp_2::token::LlamaToken>,
    pub compactor: ContextManager,
    pub sampler: Option<llama_cpp_2::sampling::LlamaSampler>,
    pub n_cur: i32,
    pub prev_size: i32,
    pub total_generated: usize,
    pub seq_id: i32,
    pub active: bool,
    pub last_batch_idx: i32,
    pub lora_path: Option<String>,
    pub lora_adapter: Option<llama_cpp_2::model::LlamaLoraAdapter>,
    pub pending_token: Option<llama_cpp_2::token::LlamaToken>,
    pub adaptive_tracker: speculative::AdaptiveTracker,
    pub pending_utf8: Vec<u8>,
}

/// Holds the collective execution context and state for a multi-agent swarm.
pub struct SwarmState<'a> {
    pub ctx: llama_cpp_2::context::LlamaContext<'a>,
    pub batch: llama_cpp_2::llama_batch::LlamaBatch<'a>,
    pub agents: Vec<AgentState>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_have_expected_values() {
        assert_eq!(BATCH_SIZE, 512);
        assert_eq!(CTX_SIZE, 16384);
        assert_eq!(MAX_TOKENS, 8192);
        assert_eq!(MAX_TOKEN_BYTES, 128);
        assert!((COMPACT_THRESH - 0.9).abs() < 0.001);
        assert_eq!(BATCH_SEQ_ID, 0);
        assert_eq!(HARD_TOOL_LIMIT, 8);
        assert_eq!(STUBBORN_ABORT_LIMIT, 11);
    }

    #[test]
    fn test_sample_config_default_is_greedy() {
        let cfg = SampleConfig::default();
        assert_eq!(cfg.temperature, 0.0);
        assert_eq!(cfg.top_k, 0);
        assert_eq!(cfg.repetition_penalty, 1.0);
    }

    #[test]
    fn test_sample_config_clone() {
        let cfg = SampleConfig { temperature: 0.5, top_k: 40, repetition_penalty: 1.05 };
        let cfg2 = cfg.clone();
        assert_eq!(cfg2.temperature, 0.5);
        assert_eq!(cfg2.top_k, 40);
    }

    #[test]
    fn test_agent_config_default_disables_tools() {
        let cfg = AgentConfig::default();
        assert!(!cfg.enable_tdd_enforcement);
        assert!(!cfg.enable_search_verifier);
        assert!(!cfg.enable_path_blocker);
        assert!(!cfg.enable_insanity_detector);
        assert!(!cfg.enable_syntax_checker);
        assert!(!cfg.enable_blind_edit_blocker);
        assert!(!cfg.disable_tool_interceptor);
        assert!(!cfg.execute_tools_locally);
    }

    #[test]
    fn test_agent_config_with_tools_disabled() {
        let cfg = AgentConfig {
            disable_tool_interceptor: true,
            execute_tools_locally: false,
            ..Default::default()
        };
        assert!(cfg.disable_tool_interceptor);
        assert!(!cfg.execute_tools_locally);
    }

    #[test]
    fn test_benchmark_result_construction() {
        let r = BenchmarkResult {
            generated_text: "hello".to_string(),
            token_count: 100,
            duration_secs: 2.5,
            tps: 40.0,
        };
        assert_eq!(r.generated_text, "hello");
        assert_eq!(r.token_count, 100);
        assert!((r.tps - 40.0).abs() < 0.01);
    }
}
