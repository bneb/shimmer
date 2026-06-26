//! Agent execution loop — generation, tool delegation, and context management.

pub mod compaction;
pub mod sampling;
pub mod state;
pub mod swarm;
pub mod tools;
pub mod validators;

use anyhow::Result;
use llama_cpp_2::{
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel},
};
pub use state::{AgentConfig, AgentState, BenchmarkResult, SampleConfig, SwarmState};
use state::{
    BATCH_SEQ_ID, COMPACT_THRESH, CTX_SIZE, EngineState, HARD_TOOL_LIMIT, MAX_TOKEN_BYTES,
    STUBBORN_ABORT_LIMIT,
};
pub use state::{BATCH_SIZE, MAX_TOKENS};

use crate::compaction::{ContextManager, ZoneType};
use crate::interceptor::ToolInterceptor;
use crate::models;
use crate::speculative;

/// The primary agent coordinator wrapping the main LLM.
pub struct Agent {
    pub(crate) main_model: LlamaModel,
    pub(crate) model_type: crate::models::CanonicalModel,
}

/// Core implementation of the Agent coordinator.
impl Agent {
    /// Starts a background swarm daemon that listens for incoming generation requests.
    pub fn run_daemon_swarm(
        &self,
        backend: &LlamaBackend,
        _config: &AgentConfig,
        rx: std::sync::mpsc::Receiver<crate::daemon::DaemonRequest>,
    ) -> Result<()> {
        let max_agents = 4;
        let ctx =
            crate::models::create_context(backend, &self.main_model, CTX_SIZE, max_agents as u32)?;
        let batch =
            llama_cpp_2::llama_batch::LlamaBatch::new((BATCH_SIZE) * max_agents, max_agents as i32);

        let mut state = SwarmState { ctx, batch, agents: Vec::new() };
        let mut streams: Vec<Option<std::os::unix::net::UnixStream>> = Vec::new();
        let mut interceptors: Vec<ToolInterceptor> = Vec::new();
        loop {
            let new_reqs = self.poll_daemon_requests(&state, &rx)?;
            self.process_daemon_requests(new_reqs, &mut state, &mut streams, &mut interceptors)?;

            self.enforce_swarm_compaction(&mut state)?;
            self.generate_swarm_step(&mut state)?;
            self.process_swarm_tools(&mut state, &mut interceptors)?;

            self.flush_daemon_streams(&mut state, &mut streams)?;
            self.cleanup_inactive_agents(&mut state, &mut streams, &mut interceptors)?;
        }
    }

    /// Polls for incoming daemon requests, blocking if no agents are currently active.
    fn poll_daemon_requests(
        &self,
        state: &SwarmState,
        rx: &std::sync::mpsc::Receiver<crate::daemon::DaemonRequest>,
    ) -> Result<Vec<crate::daemon::DaemonRequest>> {
        let mut new_reqs = Vec::new();
        let has_active = state.agents.iter().any(|a| a.active);
        if !has_active {
            match rx.recv() {
                Ok(req) => new_reqs.push(req),
                Err(_) => return Err(anyhow::anyhow!("Daemon channel closed")),
            }
        } else {
            while let Ok(req) = rx.try_recv() {
                new_reqs.push(req);
            }
        }
        Ok(new_reqs)
    }

    /// Initializes new agent states for newly received daemon requests.
    fn process_daemon_requests(
        &self,
        reqs: Vec<crate::daemon::DaemonRequest>,
        state: &mut SwarmState,
        streams: &mut Vec<Option<std::os::unix::net::UnixStream>>,
        interceptors: &mut Vec<ToolInterceptor>,
    ) -> Result<()> {
        for req in reqs {
            state.batch.clear();
            let mut seq_id = 0;
            while state.agents.iter().any(|a| a.seq_id == seq_id) {
                seq_id += 1;
            }
            if seq_id >= 4 {
                tracing::warn!("Daemon warning: Max agents (4) reached. Dropping request.");
                continue;
            }
            let mut a = self.create_agent_state(seq_id, None, None)?;

            let prompt = format!(
                "<start_of_turn>model\nYou are an \
                 AI.<end_of_turn>\n<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                req.prompt
            );
            let tokens =
                self.main_model.str_to_token(&prompt, llama_cpp_2::model::AddBos::Always)?;

            for (j, &t) in tokens.iter().enumerate() {
                let is_last = j == tokens.len() - 1;
                state.batch.add(t, a.n_cur + j as i32, &[a.seq_id], is_last)?;
                a.history.push(t);
                if is_last {
                    a.last_batch_idx = j as i32;
                }
            }

            a.n_cur += tokens.len() as i32;
            a.prev_size = a.n_cur;
            state.ctx.decode(&mut state.batch)?;
            a.pending_token = Some(self.sample_token_for_agent(&mut state.ctx, &mut a)?);

            state.agents.push(a);
            streams.push(Some(req.stream));
            interceptors.push(ToolInterceptor::new(false, true));
            state.batch.clear();
        }
        Ok(())
    }

    /// Flushes pending generated tokens out to the associated daemon streams.
    fn flush_daemon_streams(
        &self,
        state: &mut SwarmState,
        streams: &mut [Option<std::os::unix::net::UnixStream>],
    ) -> Result<()> {
        use std::io::Write;
        for (i, a) in state.agents.iter_mut().enumerate() {
            if !a.active {
                continue;
            }
            if let Some(t) = a.pending_token {
                let token_str = self.format_token_safe(t, &mut a.pending_utf8);
                if token_str.is_empty() {
                    continue;
                }
                let payload = serde_json::json!({ "token": token_str });
                if let Some(stream) = &mut streams[i]
                    && writeln!(stream, "{}", payload).is_err()
                {
                    a.active = false;
                }
            }
            if a.total_generated >= MAX_TOKENS || self.check_eog(&a.history) {
                a.active = false;
            }
        }
        Ok(())
    }

    /// Removes and cleans up KV cache for agents that have finished generation.
    fn cleanup_inactive_agents(
        &self,
        state: &mut SwarmState,
        streams: &mut Vec<Option<std::os::unix::net::UnixStream>>,
        interceptors: &mut Vec<ToolInterceptor>,
    ) -> Result<()> {
        let mut i = 0;
        while i < state.agents.len() {
            if !state.agents[i].active {
                let a = state.agents.remove(i);
                streams.remove(i);
                interceptors.remove(i);
                let _ = state.ctx.clear_kv_cache_seq(Some(a.seq_id as u32), None, None);
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    /// Creates a new Agent instance, loading the main model from the specified path.
    pub fn new(backend: &LlamaBackend, main_path: &str) -> Result<Self> {
        let model_type = crate::models::CanonicalModel::from_path(main_path);
        Ok(Self { main_model: models::load_model(backend, main_path)?, model_type })
    }

    /// Processes a single query to completion and returns the generated text and metrics.
    pub fn process_query(
        &self,
        backend: &LlamaBackend,
        prompt: &str,
        config: &AgentConfig,
        grammar_str: Option<&str>,
    ) -> Result<BenchmarkResult> {
        let mut state = self.init_state(backend, prompt, grammar_str, &config.sample_config)?;
        let gen_start_time = std::time::Instant::now();

        tracing::debug!("Processing query: {} chars", prompt.len());

        self.generation_loop(&mut state, config, None)?;

        let duration = gen_start_time.elapsed().as_secs_f64();
        tracing::debug!("Timings: {}", state.ctx.timings());
        tracing::debug!("Tool Calls: {}", state.tool_calls);
        tracing::debug!("Evictions: {}", state.compactor.pages.len());
        Ok(self.build_result(&state, duration))
    }

    /// Streams the generation output for a given prompt via an mpsc channel.
    pub fn process_stream(
        &self,
        backend: &LlamaBackend,
        prompt: &str,
        config: &AgentConfig,
        tx_out: std::sync::mpsc::Sender<String>,
    ) -> Result<()> {
        let mut state = self.init_state(backend, prompt, None, &config.sample_config)?;
        self.generation_loop(&mut state, config, Some(&tx_out))
    }

    /// Determines if the model has emitted an End-Of-Generation token.
    fn check_eog(&self, emitted: &[llama_cpp_2::token::LlamaToken]) -> bool {
        emitted.last().is_some_and(|&t| self.main_model.is_eog_token(t))
    }

    /// Initializes a new engine state, creating the context and evaluating the prompt.
    fn init_state<'a>(
        &'a self,
        backend: &LlamaBackend,
        prompt: &str,
        grammar_str: Option<&str>,
        sample_config: &SampleConfig,
    ) -> Result<EngineState<'a>> {
        let mut ctx = models::create_context(backend, &self.main_model, CTX_SIZE, 1)?;
        let mut batch = LlamaBatch::new(BATCH_SIZE, 1);

        let mut compactor = ContextManager::new(CTX_SIZE as usize, COMPACT_THRESH);
        let mut history = Vec::new();

        let n_cur = self.init_prompt(prompt, &mut batch, &mut ctx, &mut history)?;
        compactor.push_block(ZoneType::System, n_cur as usize);

        let sampler = if let Some(g) = grammar_str {
            Some(llama_cpp_2::sampling::LlamaSampler::grammar(&self.main_model, g, "root")?)
        } else {
            None
        };
        let prev_size = batch.n_tokens();
        Ok(EngineState {
            ctx,
            batch,
            history,
            compactor,
            sampler,
            sample_config: sample_config.clone(),
            n_cur,
            prev_size,
            total_generated: 0,
            adaptive_tracker: speculative::AdaptiveTracker::new(),
            pending_utf8: Vec::new(),
            tool_calls: 0,
            tests_since_last_edit: 0,
            tool_history: std::collections::HashSet::new(),
            edit_history: std::collections::HashSet::new(),
            continuation_count: 0,
            last_output_token_count: 0,
            pending_tool: None,
        })
    }

    /// Executes a single generation step, optionally utilizing speculative decoding.
    fn generate_step(
        &self,
        state: &mut EngineState,
        use_speculative: bool,
        draft_size: usize,
        ngram_size: usize,
    ) -> Result<Vec<llama_cpp_2::token::LlamaToken>> {
        let n_start = state.n_cur;
        let current_m = state.adaptive_tracker.current_draft_size(draft_size);

        let draft_branches = if use_speculative && state.sampler.is_none() && current_m > 0 {
            speculative::generate_draft_tokens(
                &self.main_model,
                &state.history,
                current_m,
                ngram_size,
            )?
        } else {
            Vec::new()
        };

        if draft_branches.is_empty() {
            state.adaptive_tracker.update_autoregressive();
            return self.handle_empty_draft(state, n_start);
        }
        self.verify_and_rollback_draft(state, &draft_branches, n_start)
    }

    /// Handles generation when speculative draft branches are empty.
    fn handle_empty_draft(
        &self,
        state: &mut EngineState,
        n_start: i32,
    ) -> Result<Vec<llama_cpp_2::token::LlamaToken>> {
        let next_token = self.sample_engine_token(state)?;
        let verification = speculative::DraftVerification {
            accepted_count: 0,
            correction_token: next_token,
            best_branch_idx: 0,
            best_seq_id: 0,
            active_branches: 0,
        };
        state.prev_size = speculative::rollback_and_correct(
            &mut state.ctx,
            &mut state.batch,
            n_start,
            &verification,
            0,
        )?;
        Ok(vec![next_token])
    }

    /// Verifies draft tokens and rolls back the context state to match the accepted branch.
    fn verify_and_rollback_draft(
        &self,
        state: &mut EngineState,
        draft_branches: &[Vec<llama_cpp_2::token::LlamaToken>],
        n_start: i32,
    ) -> Result<Vec<llama_cpp_2::token::LlamaToken>> {
        tracing::debug!("==> verify_draft_tokens");
        let std_out = std::io::stdout();
        let _ = {
            use std::io::Write;
            std_out.lock().flush()
        };
        let verification = speculative::verify_draft_tokens(
            &mut state.ctx,
            &mut state.batch,
            draft_branches,
            n_start,
            state.prev_size,
            0,
        )?;
        tracing::debug!("<== verify_draft_tokens");
        let _ = {
            use std::io::Write;
            std_out.lock().flush()
        };

        let draft_len = draft_branches[verification.best_branch_idx].len();
        state.adaptive_tracker.update_draft(verification.accepted_count, draft_len);

        tracing::debug!("==> rollback_and_correct");
        let _ = {
            use std::io::Write;
            std_out.lock().flush()
        };
        state.prev_size = speculative::rollback_and_correct(
            &mut state.ctx,
            &mut state.batch,
            n_start,
            &verification,
            0,
        )?;
        tracing::debug!("<== rollback_and_correct");
        let _ = {
            use std::io::Write;
            std_out.lock().flush()
        };

        // draft_branches[...][0] is m_0, which was already emitted in the previous step.
        // The accepted draft tokens are at indices 1 to accepted_count.
        let mut emitted = draft_branches[verification.best_branch_idx]
            .iter()
            .skip(1)
            .take(verification.accepted_count)
            .copied()
            .collect::<Vec<_>>();
        emitted.push(verification.correction_token);
        Ok(emitted)
    }

    /// Processes emitted tokens, updating history and checking for tool interrupts.
    fn process_emitted_tokens(
        &self,
        emitted: &[llama_cpp_2::token::LlamaToken],
        interceptor: &mut crate::interceptor::ToolInterceptor,
        state: &mut EngineState,
        config: &AgentConfig,
        tx_out: Option<&std::sync::mpsc::Sender<String>>,
    ) -> Result<bool> {
        let mut tool_detected = false;
        for &token in emitted.iter() {
            if self.process_single_token(token, interceptor, state, config, tx_out)? {
                tool_detected = true;
                if state.pending_tool.is_some() {
                    state.n_cur += 1;
                }
                break;
            } else {
                state.n_cur += 1;
            }
        }
        Ok(tool_detected)
    }

    fn process_single_token(
        &self,
        token: llama_cpp_2::token::LlamaToken,
        interceptor: &mut crate::interceptor::ToolInterceptor,
        state: &mut EngineState,
        config: &AgentConfig,
        tx_out: Option<&std::sync::mpsc::Sender<String>>,
    ) -> Result<bool> {
        if let Some(sampler) = &mut state.sampler {
            sampler.accept(token);
        }
        state.history.push(token);
        state.compactor.extend_last_block(ZoneType::History, 1);
        state.total_generated += 1;
        state.last_output_token_count += 1;

        let token_str = self.emit_token_output(token, tx_out, &mut state.pending_utf8)?;

        if interceptor.feed_token(&token_str)
            && let Some((name, args, child)) = interceptor.detected_call.take()
        {
            if config.execute_tools_locally {
                if let Some(nudge_msg) = validators::check_insanity_detector(
                    state,
                    config,
                    &name,
                    &args,
                    &self.model_type,
                )? {
                    let tokens = self.main_model.str_to_token(&nudge_msg, AddBos::Never)?;
                    if !tokens.is_empty() {
                        state.batch.clear();
                        let pos = state.history.len() as i32;
                        for (j, &t) in tokens.iter().enumerate() {
                            let is_last = j == tokens.len() - 1;
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
                        state.compactor.extend_last_block(ZoneType::System, tokens.len());
                        state.prev_size = state.batch.n_tokens();
                        state.n_cur = state.history.len() as i32;
                    }
                    return Ok(true);
                }
                // Only destructive tools reset the dedup set
                let is_mutating = name == "edit"
                    || name == "run_command"
                    || name == "replace_file_content"
                    || name == "write_to_file";
                if is_mutating {
                    state.tool_history.clear();
                }
                state.pending_tool = Some((name, args, child));
            }
            return Ok(true);
        }

        if let Some(nudge_msg) =
            validators::check_blind_edit(state, config, interceptor, &self.model_type)
        {
            self.inject_nudge(state, &nudge_msg, tx_out)?;
            return Ok(true);
        }

        if let Some(nudge_msg) =
            validators::check_tdd_enforcement(state, config, interceptor, &self.model_type)
        {
            self.inject_nudge(state, &nudge_msg, tx_out)?;
            return Ok(true);
        }

        if config.enable_path_blocker
            && let Some(nudge_msg) = validators::check_path_blocker(interceptor, &self.model_type)
        {
            self.inject_nudge(state, &nudge_msg, tx_out)?;
            return Ok(true);
        }

        if config.enable_search_verifier
            && let Some(nudge_msg) =
                validators::check_search_verifier(state, interceptor, &self.model_type)
        {
            self.inject_nudge(state, &nudge_msg, tx_out)?;
            return Ok(true);
        }

        if config.enable_syntax_checker
            && let Some(nudge_msg) =
                validators::check_syntax_checker(state, interceptor, &self.model_type)
        {
            self.inject_nudge(state, &nudge_msg, tx_out)?;
            return Ok(true);
        }

        Ok(false)
    }

    fn format_token_safe(
        &self,
        token: llama_cpp_2::token::LlamaToken,
        pending_utf8: &mut Vec<u8>,
    ) -> String {
        let t_bytes = self
            .main_model
            .token_to_piece_bytes(token, MAX_TOKEN_BYTES, true, None)
            .unwrap_or_default();
        pending_utf8.extend_from_slice(&t_bytes);

        let mut emit_str = String::new();
        loop {
            match std::str::from_utf8(pending_utf8) {
                Ok(s) => {
                    emit_str.push_str(s);
                    pending_utf8.clear();
                    break;
                }
                Err(e) => {
                    let valid_len = e.valid_up_to();
                    if valid_len > 0 {
                        emit_str.push_str(
                            std::str::from_utf8(&pending_utf8[..valid_len])
                                .expect("valid_up_to guarantees valid UTF-8 boundary"),
                        );
                        pending_utf8.drain(..valid_len);
                    } else if let Some(err_len) = e.error_len() {
                        emit_str.push('\u{FFFD}');
                        pending_utf8.drain(..err_len);
                    } else {
                        break;
                    }
                }
            }
        }
        emit_str
    }

    fn emit_token_output(
        &self,
        token: llama_cpp_2::token::LlamaToken,
        tx_out: Option<&std::sync::mpsc::Sender<String>>,
        pending_utf8: &mut Vec<u8>,
    ) -> Result<String> {
        let emit_str = self.format_token_safe(token, pending_utf8);

        if !emit_str.is_empty() {
            if let Some(tx) = tx_out {
                tx.send(emit_str.clone()).map_err(|_| anyhow::anyhow!("Client disconnected"))?;
            } else {
                print!("{}", emit_str);
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
        }
        Ok(emit_str)
    }

    /// Injects a nudge message into the generation stream as a synthetic user turn.
    ///
    /// Used by all validators to redirect the model when a check fails.
    fn inject_nudge(
        &self,
        state: &mut EngineState,
        nudge_msg: &str,
        tx_out: Option<&std::sync::mpsc::Sender<String>>,
    ) -> Result<()> {
        let tokens = self.main_model.str_to_token(nudge_msg, AddBos::Never).unwrap_or_default();
        state.batch.clear();
        let pos = state.history.len() as i32;
        for (j, &t) in tokens.iter().enumerate() {
            let is_last = j == tokens.len() - 1;
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
        state.compactor.extend_last_block(ZoneType::System, tokens.len());
        state.prev_size = state.batch.n_tokens();
        state.n_cur = state.history.len() as i32;
        Ok(())
    }

    /// Tokenizes the prompt and decodes it into the context batch.
    fn init_prompt(
        &self,
        prompt: &str,
        m_batch: &mut LlamaBatch,
        m_ctx: &mut llama_cpp_2::context::LlamaContext,
        history: &mut Vec<llama_cpp_2::token::LlamaToken>,
    ) -> Result<i32> {
        let tokens = self.main_model.str_to_token(prompt, AddBos::Always)?;
        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            m_batch.add(token, i as i32, &[BATCH_SEQ_ID], is_last)?;
            history.push(token);
            if m_batch.n_tokens() == BATCH_SIZE as i32 || is_last {
                m_ctx.decode(m_batch)?;
                if !is_last {
                    m_batch.clear();
                }
            }
        }
        Ok(tokens.len() as i32)
    }

    /// Constructs a BenchmarkResult containing generated text and performance metrics.
    fn build_result(&self, state: &EngineState, duration: f64) -> BenchmarkResult {
        let mut final_conv = String::new();
        for &t in &state.history {
            let t_bytes = self
                .main_model
                .token_to_piece_bytes(t, MAX_TOKEN_BYTES, true, None)
                .unwrap_or_default();
            final_conv.push_str(&String::from_utf8_lossy(&t_bytes));
        }
        BenchmarkResult {
            generated_text: final_conv,
            token_count: state.total_generated,
            duration_secs: duration,
            tps: state.total_generated as f64 / duration,
        }
    }

    /// Starts an interactive Read-Eval-Print Loop (REPL) for the agent.
    pub fn run_repl(
        &self,
        backend: &LlamaBackend,
        system_prompt: &str,
        config: &AgentConfig,
        grammar_str: Option<&str>,
    ) -> Result<()> {
        let mut state =
            self.init_state(backend, system_prompt, grammar_str, &SampleConfig::default())?;
        let mut rl = rustyline::DefaultEditor::new()
            .map_err(|e| anyhow::anyhow!("Rustyline error: {}", e))?;
        self.run_repl_loop(&mut rl, &mut state, config)
    }

    /// Core loop handling user input and dispatching execution for the REPL.
    fn run_repl_loop(
        &self,
        rl: &mut rustyline::DefaultEditor,
        state: &mut EngineState,
        config: &AgentConfig,
    ) -> Result<()> {
        loop {
            match rl.readline("\x1b[32mshimmer>\x1b[0m ") {
                Ok(line) => {
                    let input = line.trim();
                    if input == "exit" || input == "quit" {
                        break;
                    }
                    if input.is_empty() {
                        continue;
                    }
                    let _ = rl.add_history_entry(input);
                    if let Err(e) = self.execute_repl_turn(state, input, config) {
                        tracing::error!("Error: {}", e);
                    }
                }
                Err(rustyline::error::ReadlineError::Interrupted)
                | Err(rustyline::error::ReadlineError::Eof) => break,
                Err(err) => {
                    tracing::error!("Error: {:?}", err);
                    break;
                }
            }
        }
        Ok(())
    }

    /// Executes a single conversational turn based on REPL input.
    fn execute_repl_turn(
        &self,
        state: &mut EngineState,
        input: &str,
        config: &AgentConfig,
    ) -> Result<()> {
        let user_prompt = format!(
            "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n<|channel>thought\\
             n<channel|>",
            input
        );
        let tokens = self.main_model.str_to_token(&user_prompt, AddBos::Never)?;
        self.prepare_repl_turn(state, &tokens)?;

        self.generation_loop(state, config, None)
    }

    /// Prepares the batch and token history for the upcoming REPL turn.
    fn prepare_repl_turn(
        &self,
        state: &mut EngineState,
        tokens: &[llama_cpp_2::token::LlamaToken],
    ) -> Result<()> {
        state.batch.clear();
        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            state.batch.add(token, state.n_cur + i as i32, &[BATCH_SEQ_ID], is_last)?;
            state.history.push(token);
            if state.batch.n_tokens() == BATCH_SIZE as i32 || is_last {
                state.ctx.decode(&mut state.batch)?;
                if !is_last {
                    state.batch.clear();
                }
            }
        }
        state.compactor.extend_last_block(ZoneType::History, tokens.len());
        state.ctx.decode(&mut state.batch)?;
        state.prev_size = state.batch.n_tokens();
        state.n_cur += tokens.len() as i32;
        state.total_generated = 0;
        Ok(())
    }

    /// Core generation loop shared by query, stream, and REPL modes.
    fn generation_loop(
        &self,
        state: &mut EngineState,
        config: &AgentConfig,
        tx_out: Option<&std::sync::mpsc::Sender<String>>,
    ) -> Result<()> {
        let mut interceptor = crate::interceptor::ToolInterceptor::new(
            config.enable_time_travel,
            !config.disable_tool_interceptor,
        );
        while state.total_generated < MAX_TOKENS {
            self.enforce_compaction(state, config.draft_size)?;
            let emitted = self.generate_step(
                state,
                config.use_speculative,
                config.draft_size,
                config.ngram_size,
            )?;
            let _tool_invoked =
                self.process_emitted_tokens(&emitted, &mut interceptor, state, config, tx_out)?;

            if config.execute_tools_locally && state.pending_tool.is_some() {
                // Execute any pending tool immediately, appending result as a user turn
                if let Some((name, args, child)) = state.pending_tool.take() {
                    if state.tool_calls >= STUBBORN_ABORT_LIMIT && name != "run_test" {
                        return Err(anyhow::anyhow!(
                            "Model is stuck in a loop and refusing to yield a patch. Aborting."
                        ));
                    }
                    if state.tool_calls >= HARD_TOOL_LIMIT && name != "run_test" {
                        // Hard cut: reject further tools, force edit now
                        let force = crate::models::format_tool_rejection(&self.model_type);
                        let tokens = self.main_model.str_to_token(&force, AddBos::Never)?;
                        if tokens.is_empty() {
                            self.execute_pending_tool(name, args, child, state, tx_out)?;
                            continue;
                        }
                        let pos = state.history.len() as i32;
                        state.batch.clear();
                        for (j, &t) in tokens.iter().enumerate() {
                            let is_last = j == tokens.len() - 1;
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
                        state.compactor.extend_last_block(ZoneType::System, tokens.len());
                        state.prev_size = state.batch.n_tokens();
                        state.n_cur = state.history.len() as i32;
                        continue;
                    }
                    self.execute_pending_tool(name, args, child, state, tx_out)?;
                    continue;
                }
            }

            if self.check_eog(&emitted) {
                // No pending tool — check continuation
                state.continuation_count += 1;
                let mut history_str = String::new();
                for &t in &state.history {
                    if let Ok(bytes) = self.main_model.token_to_piece_bytes(t, 256, true, None) {
                        history_str.push_str(&String::from_utf8_lossy(&bytes));
                    }
                }
                if let Some(prompt) = Self::should_continue_swe_bench(
                    &self.model_type,
                    &history_str,
                    state.continuation_count,
                    state.last_output_token_count,
                    state.tool_calls,
                ) {
                    let tokens =
                        self.main_model.str_to_token(&prompt, llama_cpp_2::model::AddBos::Never)?;
                    state.batch.clear();
                    let pos = state.history.len() as i32;
                    for (j, &t) in tokens.iter().enumerate() {
                        let is_last = j == tokens.len() - 1;
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
                    state
                        .compactor
                        .extend_last_block(crate::compaction::ZoneType::History, tokens.len());
                    state.prev_size = state.batch.n_tokens();
                    state.n_cur = state.history.len() as i32;
                    continue;
                }
                break;
            }
        }
        Ok(())
    }
}

/// Swarm-specific implementation methods for the Agent.

#[cfg(test)]
mod tests {
    use super::*;

    fn sys_prompt() -> &'static str {
        "CRITICAL: If a search tool yields no output... 
<edit file=\"path/to/file.py\">
</edit>
"
    }

    #[test]
    fn test_extract_high_entropy_summary_preserves_errors() {
        let input = "line 1\n  File \"/src/foo.py\", line 42, in bar\n    raise \
                     ValueError('bad')\nValueError: bad\nline 5";
        let out = Agent::extract_high_entropy_summary(input);
        assert!(out.contains("ValueError"));
        assert!(out.contains("foo.py"));
    }

    #[test]
    fn test_extract_high_entropy_summary_short_circuits() {
        let short = "small output";
        let out = Agent::extract_high_entropy_summary(short);
        assert!(!out.is_empty());
    }

    #[test]
    fn test_extract_high_entropy_summary_caps_at_500() {
        let long = "line\n".repeat(200);
        let out = Agent::extract_high_entropy_summary(&long);
        assert!(out.len() <= 510);
    }

    #[test]
    fn test_should_continue_swe_bench_no_swe_bench() {
        let history = "Just a normal conversation with an <edit file=\"foo.txt\"> tag.";
        assert_eq!(
            Agent::should_continue_swe_bench(
                &crate::models::CanonicalModel::Qwen25Coder7B,
                history,
                0,
                10,
                0
            ),
            None
        );
    }

    #[test]
    fn test_should_continue_swe_bench_unclosed_tag() {
        let history = format!(
            "{}
<edit file=\"foo.txt\">
Some code...
Let me make another fix:
        <edit file=\"bar.txt\">",
            sys_prompt()
        );
        assert!(
            Agent::should_continue_swe_bench(
                &crate::models::CanonicalModel::Qwen25Coder7B,
                &history,
                0,
                10,
                0
            )
            .unwrap()
            .contains("closing </edit> tag")
        );
    }

    #[test]
    fn test_should_continue_swe_bench_missing_edit() {
        let history = format!(
            "{}
        I have found the problem. The issue is in bar.py.",
            sys_prompt()
        );
        // Phase 1: early investigation — encourages search
        let prompt = Agent::should_continue_swe_bench(
            &crate::models::CanonicalModel::Qwen25Coder7B,
            &history,
            0,
            10,
            0,
        )
        .unwrap();
        assert!(prompt.contains("Investigate using tools"));
    }

    /// Phase 2: after 6+ tool calls with no edit, force convergence.
    #[test]
    fn test_should_continue_swe_bench_force_convergence() {
        let history = format!(
            "{}
I have found the problem.",
            sys_prompt()
        );
        let prompt = Agent::should_continue_swe_bench(
            &crate::models::CanonicalModel::Qwen25Coder7B,
            &history,
            0,
            10,
            6,
        )
        .unwrap();
        assert!(prompt.contains("Provide your fix now"));
    }

    #[test]
    fn test_should_continue_swe_bench_completed_edit() {
        let history = format!(
            "{}
<edit file=\"foo.txt\">
        Some code...
</edit>
All done!",
            sys_prompt()
        );
        assert_eq!(
            Agent::should_continue_swe_bench(
                &crate::models::CanonicalModel::Qwen25Coder7B,
                &history,
                0,
                10,
                0
            ),
            None
        );
    }

    #[test]
    fn test_should_continue_swe_bench_empty_response() {
        let history = format!(
            "{}
<|im_start|>assistant
<|im_end|>",
            sys_prompt()
        );
        let prompt = Agent::should_continue_swe_bench(
            &crate::models::CanonicalModel::Qwen25Coder7B,
            &history,
            0,
            0,
            0,
        )
        .unwrap();
        assert!(prompt.contains("empty or incorrectly formatted"));
    }

    #[test]
    fn test_should_continue_swe_bench_anti_spiral_cap() {
        let history = format!(
            "{}
I have found the problem.",
            sys_prompt()
        );
        // At continuation count 10, should return None (hard stop)
        assert_eq!(
            Agent::should_continue_swe_bench(
                &crate::models::CanonicalModel::Qwen25Coder7B,
                &history,
                10,
                10,
                0
            ),
            None
        );
        // At continuation count 9, should still trigger
        assert!(
            Agent::should_continue_swe_bench(
                &crate::models::CanonicalModel::Qwen25Coder7B,
                &history,
                9,
                10,
                0
            )
            .is_some()
        );
    }
}
