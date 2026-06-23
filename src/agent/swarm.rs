//! Parallel batch inference and daemon mode.
//!
//! Runs the same prompt concurrently across multiple contexts sharing a
//! single KV cache arena, with per-agent LoRA adapters and compaction.

use super::state::{AgentState, SwarmState, AgentConfig, BenchmarkResult};
use super::state::{BATCH_SIZE, MAX_TOKENS, CTX_SIZE, COMPACT_THRESH, MAX_TOKEN_BYTES};
use crate::compaction::{ContextManager, ZoneType};
use crate::interceptor::ToolInterceptor;
use crate::models;
use crate::speculative;
use crate::tool;
use anyhow::Result;
use llama_cpp_2::{
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::AddBos,
};

impl super::Agent {
    /// Processes multiple prompts concurrently within a single unified context swarm.
    pub fn process_swarm(
        &self,
        backend: &LlamaBackend,
        prompts: &[String],
        lora_paths: &[Option<String>],
        config: &AgentConfig,
        grammar_str: Option<&str>,
    ) -> Result<Vec<BenchmarkResult>> {
        let mut state = self.init_swarm_state(backend, prompts, lora_paths, grammar_str)?;
        let gen_start = std::time::Instant::now();
        self.swarm_generation_loop(&mut state, config)?;
        let dur = gen_start.elapsed().as_secs_f64();
        Ok(self.build_swarm_results(&state, dur))
    }

    /// Initializes the collective swarm state for a batch of prompts.
    fn init_swarm_state<'a>(
        &'a self,
        backend: &LlamaBackend,
        prompts: &[String],
        lora_paths: &[Option<String>],
        grammar_str: Option<&str>,
    ) -> Result<SwarmState<'a>> {
        let k = prompts.len();
        let mut ctx = models::create_context(backend, &self.main_model, CTX_SIZE, k as u32)?;
        let mut batch = LlamaBatch::new(BATCH_SIZE * k, k as i32);
        let mut agents = self.create_swarm_agents(k, lora_paths, grammar_str)?;
        self.init_swarm_decode(&mut ctx, &mut batch, &mut agents, prompts)?;
        Ok(SwarmState { ctx, batch, agents })
    }

    /// Allocates and configures internal states for all agents in the swarm.
    fn create_swarm_agents(
        &self, k: usize, lora_paths: &[Option<String>], grammar_str: Option<&str>,
    ) -> Result<Vec<AgentState>> {
        let mut agents = Vec::with_capacity(k);
        for i in 0..k {
            let lora_path = lora_paths.get(i).cloned().flatten();
            agents.push(self.create_agent_state(i as i32, lora_path, grammar_str)?);
        }
        Ok(agents)
    }

    /// Initializes an individual agent state, including LORA adapter and sampler.
    pub(crate) fn create_agent_state(
        &self, seq_id: i32, lora_path: Option<String>, grammar_str: Option<&str>,
    ) -> Result<AgentState> {
        let comp = ContextManager::new(CTX_SIZE as usize, COMPACT_THRESH);
        let mut lora_adapter = None;
        if let Some(ref path) = lora_path {
            lora_adapter = Some(
                self.main_model.lora_adapter_init(path)
                    .map_err(|e| anyhow::anyhow!("LoRA load error: {:?}", e))?,
            );
        }
        let sampler = if let Some(g) = grammar_str {
            Some(llama_cpp_2::sampling::LlamaSampler::grammar(&self.main_model, g, "root")?)
        } else { None };
        Ok(AgentState {
            history: Vec::new(), compactor: comp, sampler, n_cur: 0, prev_size: 0,
            total_generated: 0, seq_id, active: true, last_batch_idx: 0,
            lora_path, lora_adapter, pending_token: None,
            adaptive_tracker: speculative::AdaptiveTracker::new(), pending_utf8: Vec::new(),
        })
    }

    fn init_swarm_decode(
        &self, ctx: &mut llama_cpp_2::context::LlamaContext, batch: &mut LlamaBatch,
        agents: &mut [AgentState], prompts: &[String],
    ) -> Result<()> {
        let mut groups: std::collections::HashMap<Option<String>, Vec<usize>> = std::collections::HashMap::new();
        for (i, a) in agents.iter().enumerate() { groups.entry(a.lora_path.clone()).or_default().push(i); }
        for (lora_path, agent_indices) in groups {
            self.set_swarm_lora_adapter(ctx, agents, &lora_path, &agent_indices)?;
            for &idx in &agent_indices {
                let prompt = &prompts[idx];
                let tokens = self.main_model.str_to_token(prompt, AddBos::Always)?;
                let a = &mut agents[idx];
                batch.clear();
                for (j, &t) in tokens.iter().enumerate() {
                    let is_last = j == tokens.len() - 1;
                    batch.add(t, a.n_cur + j as i32, &[a.seq_id], is_last)?;
                    a.history.push(t);
                    if batch.n_tokens() == BATCH_SIZE as i32 || is_last {
                        ctx.decode(batch)?;
                        if is_last {
                            a.last_batch_idx = batch.n_tokens() - 1;
                            let m_i = self.sample_token_for_agent(ctx, a)?;
                            if let Some(sampler) = &mut a.sampler { sampler.accept(m_i); }
                            a.pending_token = Some(m_i);
                        }
                        if !is_last { batch.clear(); }
                    }
                }
                a.n_cur += tokens.len() as i32;
                a.prev_size = a.n_cur;
                a.compactor.push_block(ZoneType::System, a.n_cur as usize);
            }
        }
        Ok(())
    }

    fn set_swarm_lora_adapter(
        &self, ctx: &mut llama_cpp_2::context::LlamaContext, agents: &mut [AgentState],
        lora_path: &Option<String>, agent_indices: &[usize],
    ) -> Result<()> {
        if let Some(_lora) = lora_path
            && let Some(&idx) = agent_indices.first()
            && let Some(adapter) = agents[idx].lora_adapter.as_mut()
        {
            let adapter_ptr = adapter as *mut llama_cpp_2::model::LlamaLoraAdapter;
            // Safety: adapter is a live &mut reference from agents[idx].lora_adapter,
            // obtained above via as_mut(). The pointer is used immediately and
            // does not outlive the borrow on agents.
            let _ = ctx.lora_adapter_set(unsafe { &mut *adapter_ptr }, 1.0);
        } else {
            let mut dummy = None;
            for a in &mut *agents {
                if let Some(adapter) = a.lora_adapter.as_mut() {
                    dummy = Some(adapter as *mut llama_cpp_2::model::LlamaLoraAdapter);
                    break;
                }
            }
            // Safety: dummy is derived from a.lora_adapter.as_mut() above,
            // a live reference within the agents slice. Used immediately.
            if let Some(dummy_ptr) = dummy { let _ = ctx.lora_adapter_remove(unsafe { &mut *dummy_ptr }); }
        }
        Ok(())
    }

    pub(crate) fn sample_token_for_agent(
        &self, ctx: &mut llama_cpp_2::context::LlamaContext, a: &mut AgentState,
    ) -> Result<llama_cpp_2::token::LlamaToken> {
        let m_i = if let Some(sampler) = &mut a.sampler {
            let candidates: Vec<_> = ctx.candidates_ith(a.last_batch_idx).collect();
            let mut data_array = llama_cpp_2::token::data_array::LlamaTokenDataArray::new(candidates, false);
            sampler.apply(&mut data_array);
            data_array.data.iter().max_by(|a, b| a.logit().partial_cmp(&b.logit()).unwrap_or(std::cmp::Ordering::Equal))
                .map(|d| d.id()).unwrap_or(llama_cpp_2::token::LlamaToken::new(0))
        } else {
            ctx.candidates_ith(a.last_batch_idx).max_by(|a, b| a.logit().partial_cmp(&b.logit()).unwrap_or(std::cmp::Ordering::Equal))
                .map(|d| d.id()).unwrap_or(llama_cpp_2::token::LlamaToken::new(0))
        };
        Ok(m_i)
    }

    fn swarm_generation_loop(&self, state: &mut SwarmState, config: &AgentConfig) -> Result<()> {
        let mut interceptors: Vec<_> = (0..state.agents.len())
            .map(|_| ToolInterceptor::new(config.enable_time_travel, true)).collect();
        loop {
            if state.agents.iter().all(|a| !a.active) { break; }
            self.enforce_swarm_compaction(state)?;
            self.generate_swarm_step(state)?;
            self.process_swarm_tools(state, &mut interceptors)?;
            for a in &mut state.agents {
                if !a.active { continue; }
                if a.total_generated >= MAX_TOKENS || self.check_eog(&a.history) { a.active = false; }
            }
        }
        Ok(())
    }

    pub fn enforce_swarm_compaction(&self, state: &mut SwarmState) -> Result<()> {
        let global_used: usize = state.agents.iter().filter(|a| a.active).map(|a| a.n_cur as usize).sum();
        if global_used > (CTX_SIZE as f64 * COMPACT_THRESH) as usize {
            self.compact_largest_agent(state)?;
        }
        for a in &mut state.agents {
            if a.active && a.compactor.needs_compaction(1) {
                self.compact_agent(a, &mut state.ctx)?;
            }
        }
        Ok(())
    }

    fn compact_largest_agent(&self, state: &mut SwarmState) -> Result<()> {
        if let Some(largest) = state.agents.iter_mut().filter(|a| a.active).max_by_key(|a| a.n_cur) {
            self.compact_agent(largest, &mut state.ctx)?;
        }
        Ok(())
    }

    fn compact_agent(&self, a: &mut AgentState, ctx: &mut llama_cpp_2::context::LlamaContext) -> Result<()> {
        let _original_history = a.history.clone();
        if let Some((start, end)) = a.compactor.compact(&mut a.history, 0) {
            let shift_amount = end - start;
            ctx.clear_kv_cache_seq(Some(a.seq_id as u32), Some(start as u32), Some(end as u32))?;
            ctx.kv_cache_seq_add(a.seq_id, Some(end as u32), None, -(shift_amount as i32))?;
            a.n_cur -= shift_amount as i32;
        }
        Ok(())
    }

    pub fn generate_swarm_step(&self, state: &mut SwarmState) -> Result<()> {
        let mut groups: std::collections::HashMap<Option<String>, Vec<usize>> = std::collections::HashMap::new();
        for (i, a) in state.agents.iter().enumerate() {
            if a.active { groups.entry(a.lora_path.clone()).or_default().push(i); }
        }
        for (lora_path, agent_indices) in groups {
            self.set_swarm_lora_adapter(&mut state.ctx, &mut state.agents, &lora_path, &agent_indices)?;
            state.batch.clear();
            let batch_idx = self.prepare_swarm_step_batch(&mut state.batch, &mut state.agents, &agent_indices)?;
            if batch_idx > 0 {
                state.ctx.decode(&mut state.batch)?;
                for &idx in &agent_indices {
                    let a = &mut state.agents[idx];
                    let m_i = self.sample_token_for_agent(&mut state.ctx, a)?;
                    if let Some(sampler) = &mut a.sampler { sampler.accept(m_i); }
                    a.pending_token = Some(m_i);
                }
            }
        }
        Ok(())
    }

    fn prepare_swarm_step_batch(&self, batch: &mut LlamaBatch, agents: &mut [AgentState], agent_indices: &[usize]) -> Result<i32> {
        let mut batch_idx = 0;
        for &idx in agent_indices {
            let a = &mut agents[idx];
            let Some(t) = a.pending_token else { continue; };
            a.history.push(t);
            a.compactor.extend_last_block(ZoneType::History, 1);
            a.total_generated += 1;
            batch.add(t, a.n_cur, &[a.seq_id], true)?;
            a.n_cur += 1;
            a.last_batch_idx = batch_idx;
            batch_idx += 1;
        }
        Ok(batch_idx)
    }

    pub fn process_swarm_tools(&self, state: &mut SwarmState, interceptors: &mut [ToolInterceptor]) -> Result<()> {
        let mut tools_to_run = Vec::new();
        for (i, a) in state.agents.iter_mut().enumerate() {
            if !a.active { continue; }
            let Some(last_t) = a.history.last().copied() else { continue; };
            let t_bytes = self.main_model.token_to_piece_bytes(last_t, MAX_TOKEN_BYTES, true, None).unwrap_or_default();
            let token_str = String::from_utf8_lossy(&t_bytes).to_string();
            if interceptors[i].feed_token(&token_str) && let Some(call) = interceptors[i].detected_call.take() {
                tools_to_run.push((i, call, a.n_cur));
            }
        }
        for (i, call, pos) in tools_to_run { self.execute_swarm_tool(state, i, call.0, call.1, pos)?; }
        Ok(())
    }

    fn execute_swarm_tool(&self, state: &mut SwarmState, agent_idx: usize, pat: String, tgt: serde_json::Value, tool_pos: i32) -> Result<()> {
        let args_vec: Vec<String> = match &tgt {
            serde_json::Value::Array(arr) => arr.iter().filter_map(|v| {
                if let Some(s) = v.as_str() { Some(String::from(s)) } else { v.as_number().map(|n| n.to_string()) }
            }).collect(),
            serde_json::Value::Object(obj) => obj.values().filter_map(|v| v.as_str().map(String::from)).collect(),
            _ => Vec::new(),
        };
        let obs = tool::execute_tool(&pat, &args_vec);
        let a = &mut state.agents[agent_idx];
        a.history.truncate(tool_pos as usize);
        a.compactor.truncate_history(tool_pos as usize);
        state.ctx.clear_kv_cache_seq(Some(a.seq_id as u32), Some(tool_pos as u32), None)?;
        let obs_tokens = self.main_model.str_to_token(&obs, AddBos::Never)?;
        state.batch.clear();
        for (j, &t) in obs_tokens.iter().enumerate() {
            let is_last = j == obs_tokens.len() - 1;
            state.batch.add(t, tool_pos + j as i32, &[a.seq_id], is_last)?;
            a.history.push(t);
            if state.batch.n_tokens() == BATCH_SIZE as i32 || is_last {
                state.ctx.decode(&mut state.batch)?;
                if is_last { a.last_batch_idx = state.batch.n_tokens() - 1; }
                if !is_last { state.batch.clear(); }
            }
        }
        a.compactor.extend_last_block(ZoneType::ToolOutput, obs_tokens.len());
        a.prev_size = state.batch.n_tokens();
        a.n_cur = tool_pos + obs_tokens.len() as i32;
        Ok(())
    }

    fn build_swarm_results(&self, state: &SwarmState, duration: f64) -> Vec<BenchmarkResult> {
        state.agents.iter().map(|a| {
            let mut final_conv = String::new();
            for &t in &a.history {
                let t_bytes = self.main_model.token_to_piece_bytes(t, MAX_TOKEN_BYTES, true, None).unwrap_or_default();
                final_conv.push_str(&String::from_utf8_lossy(&t_bytes));
            }
            BenchmarkResult { generated_text: final_conv, token_count: a.total_generated, duration_secs: duration, tps: a.total_generated as f64 / duration }
        }).collect()
    }

    /// Returns a continuation prompt if the model should keep going (SWE-bench).
    /// Three-phase convergence: search → force edit → cap.
    pub fn should_continue_swe_bench(
        model: &crate::models::CanonicalModel,
        history_str: &str,
        continuation_count: usize,
        last_output_tokens: usize,
        tool_calls: usize,
    ) -> Option<String> {
        let edits_started = history_str.matches("<edit file=").count();
        let edits_ended = history_str.matches("</edit>").count();

        // Hard cap
        if continuation_count >= 10 { return None; }

        // Phase 3: already started a real edit (beyond system prompt example) — help finish it
        if edits_started > 1 && edits_ended < edits_started {
            return Some(crate::models::format_system_nudge(model, "You have not provided the closing </edit> tag. Please finish the edit block. Do not stop."));
        }

        // Phase 2: enough investigation — force the edit
        if tool_calls >= 6 && edits_started <= 1 {
            return Some(crate::models::format_system_nudge(model, "You have enough information. Provide your fix now. Use either XML or JSON format. Do not use any more tools."));
        }

        // Phase 1: early investigation — encourage search
        if edits_started <= 1 && history_str.contains("CRITICAL: If a search tool yields no output") {
            if last_output_tokens == 0 {
                return Some(crate::models::format_system_nudge(model, "Your last response was empty or incorrectly formatted. You MUST either execute a JSON tool call or provide an <edit> block."));
            } else {
                return Some(crate::models::format_system_nudge(model, "Investigate using tools (rg, cat, fd). Find the relevant code, then provide your fix (XML or JSON format)."));
            }
        }

        None
    }
}
