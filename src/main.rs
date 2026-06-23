//! Shimmer CLI Entrypoint
//!
//! Initializes the logger, parses arguments, and bootstraps the inference engine.
//!
//! # Example
//! ```bash
//! cargo run -- --serve
//! ```

use anyhow::Result;
use clap::Parser;

use llama_cpp_2::llama_backend::LlamaBackend;
use shimmer::{agent, engine};

const SYSTEM_PROMPT: &str = r#"You are a coding assistant with tools. To use a tool, output exactly:
```json
{"name": "tool_name", "arguments": ["arg1", "arg2"]}
```
Wait for the system to inject the results before continuing."#;

/// Command-line arguments for configuring the Shimmer AI engine.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the main GGUF model
    #[arg(
        long,
        default_value = "models/gemma4-12b.gguf"
    )]
    main_model: String,

    /// Prompt to send to the model
    #[arg(short, long)]
    prompt: Option<String>,

    /// Enable Speculative Decoding (Prompt Lookup Decoding). Defaults to model-specific recommendation if omitted.
    #[arg(long, default_missing_value = "true", num_args = 0..=1)]
    speculative: Option<bool>,

    /// Maximum draft tokens (M). Scale this based on your memory bandwidth.
    #[arg(long, default_value = "24")]
    draft_size: usize,

    /// N-Gram match size (N).
    #[arg(long, default_value = "3")]
    ngram_size: usize,

    /// Warm start the Metal engine before benchmarking
    #[arg(long)]
    warmup: bool,

    /// Enable Batched Multi-Agent Swarm Concurrency
    #[arg(long)]
    enable_swarm: bool,

    #[arg(long)]
    lora_paths: Vec<String>,

    /// Enable Speculative Tool Execution
    #[arg(long)]
    enable_time_travel: bool,

    /// Start the OpenAI-compatible REST API server
    #[arg(long)]
    serve: bool,

    /// Start the Unix Domain Socket Daemon
    #[arg(long)]
    daemon: bool,

    /// Address to bind the REST API server (default: 127.0.0.1:8080)
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,

    /// Optional API key for authenticating requests. When set, requests
    /// must include an Authorization: Bearer <key> header.
    #[arg(long)]
    api_key: Option<String>,

    /// Optional grammar string or path to .gbnf file
    #[arg(long)]
    grammar: Option<String>,

    /// Enable 'run_test' tool for test-driven generation
    #[arg(long)]
    teston: bool,

    /// Disable heuristic retrieval preprocessor
    #[arg(long)]
    no_preprocessor: bool,

    /// Disable TDD enforcement
    #[arg(long)]
    no_tdd_enforcement: bool,

    /// Disable <search> exact-match verification
    #[arg(long)]
    no_search_verifier: bool,

    /// Disable hallucinated file path blocker
    #[arg(long)]
    no_path_blocker: bool,

    /// Disable insanity detector
    #[arg(long)]
    no_insanity_detector: bool,

    /// Disable pre-flight syntax checker
    #[arg(long)]
    no_syntax_checker: bool,

    /// Disable blind edit blocker
    #[arg(long)]
    no_blind_edit_blocker: bool,

    /// Disable all tool detection and execution. The model generates straight
    /// through — JSON tool markers are plain text. Edit tags still parse.
    #[arg(long)]
    no_tools: bool,

    /// Sampling configuration: temp,topk,repp (e.g. "temp=0.2,topk=40,repp=1.05")
    #[arg(long, default_value = "temp=0.0,topk=0,repp=1.0")]
    sample: String,
}

fn run_serve(
    agent: std::sync::Arc<agent::Agent>,
    backend: std::sync::Arc<LlamaBackend>,
    config: std::sync::Arc<shimmer::agent::AgentConfig>,
    bind_addr: &str,
    api_key: Option<String>,
) -> Result<()> {
    tracing::info!("Starting Shimmer API Server on http://{}", bind_addr);
    let state = shimmer::server::AppState { agent, backend, config, api_key };
    let app = shimmer::server::create_router(state);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(bind_addr).await?;
        axum::serve(listener, app).await?;
        Ok::<_, anyhow::Error>(())
    })?;
    Ok(())
}

fn do_warmup(
    agent: &agent::Agent,
    backend: &LlamaBackend,
    config: &shimmer::agent::AgentConfig,
) -> Result<()> {
    tracing::info!("Warming up Metal engine...");
    let warmup_prompt = "<|im_start|>assistant\nYou are an AI.<|im_end|>\n<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n".to_string();
    let _ = agent.process_query(backend, &warmup_prompt, config, None)?;
    tracing::info!("Warmup complete");
    Ok(())
}

fn run_swarm(
    agent: &agent::Agent,
    backend: &LlamaBackend,
    args: &Args,
    config: &shimmer::agent::AgentConfig,
    grammar_str: Option<&str>,
    prompt: &str,
) -> Result<()> {
    tracing::info!("Running Swarm Benchmark with 3 agents...");
    let prompts = vec![prompt.to_string(), prompt.to_string(), prompt.to_string()];
    let mut lora_paths_opt = Vec::new();
    for i in 0..3 {
        lora_paths_opt.push(args.lora_paths.get(i).cloned());
    }
    let results = agent.process_swarm(backend, &prompts, &lora_paths_opt, config, grammar_str)?;

    for (i, result) in results.iter().enumerate() {
        print_result(result, &format!("Agent {}", i));
    }
    Ok(())
}

fn run_single_query(
    agent: &agent::Agent,
    backend: &LlamaBackend,
    config: &shimmer::agent::AgentConfig,
    grammar_str: Option<&str>,
    prompt: &str,
) -> Result<()> {
    let result = agent.process_query(backend, prompt, config, grammar_str)?;
    print_result(&result, "Generation Result");
    Ok(())
}

fn print_result(result: &shimmer::agent::BenchmarkResult, title: &str) {
    tracing::info!(
        "[{}] {} tokens, {:.4}s, {:.2} TPS, {} chars",
        title, result.token_count, result.duration_secs, result.tps,
        result.generated_text.len()
    );
    tracing::debug!("{}", result.generated_text);
}

fn main() -> Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    tracing::info!("Shimmer AI Engine starting");

    let backend = std::sync::Arc::new(engine::init_backend()?);
    tracing::info!("Loading Model: {}", args.main_model);
    let agent = std::sync::Arc::new(agent::Agent::new(&backend, &args.main_model)?);

    let grammar_str = load_grammar(args.grammar.as_deref())?;

    let canonical_model = shimmer::models::CanonicalModel::from_path(&args.main_model);
    tracing::info!("Canonical Model Detected: {:?}", canonical_model);

    let use_speculative = if args.enable_swarm {
        false
    } else {
        args.speculative.unwrap_or_else(|| canonical_model.supports_speculative_decoding())
    };

    tracing::info!(
        "Speculative Decoding: {}",
        if use_speculative {
            "ENABLED"
        } else {
            "DISABLED"
        }
    );

    let config = std::sync::Arc::new(create_agent_config(&args, use_speculative));

    if args.serve {
        return run_serve(agent, backend, config, &args.bind, args.api_key.clone());
    }

    if args.daemon {
        return shimmer::daemon::start_daemon(agent, backend, config);
    }

    if let Some(ref prompt_text) = args.prompt {
        if args.warmup {
            do_warmup(&agent, &backend, &config)?;
        }

        tracing::debug!("Prompt: {} chars", prompt_text.len());
        let tools_list = if args.teston {
            "rg, grep, fd, find, cat, read_file, ls, git, sed, run_test"
        } else {
            "rg, grep, fd, find, cat, read_file, ls, git, sed"
        };

        let tdd_instruction = if args.teston {
            "\nUse JSON tools: {\"name\":\"rg\",\"arguments\":[\"pattern\",\"path\"]}. Tools: rg, cat, fd, ls, git, run_test.\n\
            When you have found the fix, write an edit tag with file= set to the real path, containing search and replace tags with the exact code."
        } else {
            ""
        };

        let full_system_prompt = format!("{}{}", SYSTEM_PROMPT, tdd_instruction);

        let preprocessor_hint = if !args.no_preprocessor {
            match std::env::current_dir() {
                Ok(cwd) => shimmer::preprocessor::write_context_file(prompt_text, cwd.as_path())
                    .ok()
                    .flatten()
                    .unwrap_or_default(),
                Err(_) => String::new(),
            }
        } else {
            String::new()
        };

        let augmented_prompt = format!("{}{}", preprocessor_hint, prompt_text);

        let prompt = shimmer::models::format_prompt(
            &canonical_model,
            &full_system_prompt,
            tools_list,
            &augmented_prompt,
        );

        if args.enable_swarm {
            run_swarm(
                &agent,
                &backend,
                &args,
                &config,
                grammar_str.as_deref(),
                &prompt,
            )?;
        } else {
            run_single_query(&agent, &backend, &config, grammar_str.as_deref(), &prompt)?;
        }
    } else {
        let system_prompt = format!("<|im_start|>assistant\n{}<|im_end|>\n", SYSTEM_PROMPT);
        agent.run_repl(&backend, &system_prompt, &config, grammar_str.as_deref())?;
    }

    Ok(())
}

/// Loads a grammar file from a path or passes through the raw grammar string.
fn load_grammar(grammar_arg: Option<&str>) -> Result<Option<String>> {
    if let Some(path_or_str) = grammar_arg {
        if std::path::Path::new(path_or_str).exists() {
            Ok(Some(std::fs::read_to_string(path_or_str)?))
        } else {
            Ok(Some(path_or_str.to_string()))
        }
    } else {
        Ok(None)
    }
}

/// Creates an agent configuration struct from the parsed CLI arguments.
fn create_agent_config(args: &Args, use_speculative: bool) -> shimmer::agent::AgentConfig {
    let sample_config = parse_sample_config(&args.sample);
    shimmer::agent::AgentConfig {
        use_speculative,
        draft_size: args.draft_size,
        ngram_size: args.ngram_size,
        enable_time_travel: args.enable_time_travel,
        execute_tools_locally: !args.serve && !args.daemon && !args.no_tools,
        enable_tdd_enforcement: !args.no_tdd_enforcement && args.teston && !args.no_tools,
        enable_search_verifier: !args.no_search_verifier && !args.no_tools,
        enable_path_blocker: !args.no_path_blocker && !args.no_tools,
        enable_insanity_detector: !args.no_insanity_detector && !args.no_tools,
        enable_syntax_checker: !args.no_syntax_checker && !args.no_tools,
        enable_blind_edit_blocker: !args.no_blind_edit_blocker && !args.no_tools,
        disable_tool_interceptor: args.no_tools,
        sample_config,
    }
}

/// Parses a "temp=X,topk=Y,repp=Z" sample config string.
fn parse_sample_config(raw: &str) -> shimmer::agent::SampleConfig {
    let mut sc = shimmer::agent::SampleConfig::default();
    for part in raw.split(',') {
        let part = part.trim();
        if let Some(val) = part.strip_prefix("temp=") {
            if let Ok(v) = val.parse::<f32>() { sc.temperature = v; }
        } else if let Some(val) = part.strip_prefix("topk=") {
            if let Ok(v) = val.parse::<usize>() { sc.top_k = v; }
        } else if let Some(val) = part.strip_prefix("repp=")
            && let Ok(v) = val.parse::<f32>() { sc.repetition_penalty = v; }
    }
    sc
}
