//! Tool execution and output injection.
//!
//! Handles tool dispatch, output capture, and injecting results as user turns
//! without KV cache rollback (append semantics).

use anyhow::Result;
use llama_cpp_2::model::AddBos;

use super::state::{BATCH_SIZE, EngineState};
use crate::compaction::ZoneType;
use crate::tool;

impl super::Agent {
    /// Executes a pending tool and appends the result as a user turn.
    /// No KV cache rollback — tool call tokens stay in history.
    pub(crate) fn execute_pending_tool(
        &self,
        pat: String,
        tgt: serde_json::Value,
        child_opt: Option<std::process::Child>,
        state: &mut EngineState,
        tx_out: Option<&std::sync::mpsc::Sender<String>>,
    ) -> Result<()> {
        state.tool_calls += 1;
        state.last_output_token_count = 0;
        if pat == "run_test" {
            state.tests_since_last_edit += 1;
        }
        let raw_obs = if let Some(child) = child_opt {
            tool::process_tool_child(child)
        } else {
            let args_vec: Vec<String> = match &tgt {
                serde_json::Value::Array(arr) => arr
                    .iter()
                    .filter_map(|v| {
                        if let Some(s) = v.as_str() {
                            Some(String::from(s))
                        } else {
                            v.as_number().map(|n| n.to_string())
                        }
                    })
                    .collect(),
                serde_json::Value::Object(obj) => {
                    obj.values().filter_map(|v| v.as_str().map(String::from)).collect()
                }
                _ => Vec::new(),
            };
            tool::execute_tool(&pat, &args_vec)
        };
        let tool_file = format!(".shimmer_tool_{}.txt", state.tool_calls);
        let _ = std::fs::write(&tool_file, &raw_obs);
        let obs = crate::models::format_tool_result(&self.model_type, &pat, &raw_obs);
        let obs_tokens = self.main_model.str_to_token(&obs, AddBos::Never)?;
        self.ensure_capacity(state, obs_tokens.len())?;
        // Use history length as the ground-truth position (not n_cur which can drift)
        let pos = state.history.len() as i32;
        state.batch.clear();
        for (j, &t) in obs_tokens.iter().enumerate() {
            let is_last = j == obs_tokens.len() - 1;
            state.batch.add(t, pos + j as i32, &[0], is_last)?;
            state.history.push(t);
            let _ = self.emit_token_output(t, tx_out, &mut state.pending_utf8)?;
            if state.batch.n_tokens() == BATCH_SIZE as i32 || is_last {
                state.ctx.decode(&mut state.batch)?;
                if !is_last {
                    state.batch.clear();
                }
            }
        }
        if tx_out.is_none() {
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        state.compactor.extend_last_block(ZoneType::ToolOutput, obs_tokens.len());
        state.prev_size = state.batch.n_tokens();
        state.n_cur = state.history.len() as i32;
        Ok(())
    }

    /// Ensures there is capacity for `needed` tokens, compacting if necessary.
    pub(crate) fn ensure_capacity(&self, state: &mut EngineState, needed: usize) -> Result<()> {
        let mut evictions = 0;
        while state.compactor.needs_compaction(needed) && evictions < 100 {
            let old_len = state.compactor.pages.len();
            self.enforce_compaction(state, needed)?;
            if state.compactor.pages.len() >= old_len {
                break;
            }
            evictions += 1;
        }
        Ok(())
    }
}
