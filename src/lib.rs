//! Shimmer — local LLM execution engine for agentic coding on Apple Silicon.
//!
//! Builds on llama.cpp and Metal to run GGUF models with native tool
//! interception, edit validation, and speculative decoding.

/// Agent execution loop, parallel batch inference, and edit validators.
pub mod agent;
/// Token page management and context window compaction.
pub mod compaction;
/// Background service for asynchronous processing over Unix sockets.
pub mod daemon;
/// Metal-optimized configuration parameters for execution.
pub mod engine;
/// Token stream observation and speculative tool execution.
pub mod interceptor;
/// Definitions and implementations for ML model backends.
pub mod models;
/// Heuristic prompt context search.
pub mod preprocessor;
/// HTTP API server.
pub mod server;
/// Speculative decoding and drafting heuristics.
pub mod speculative;
/// Tool execution subsystem and environment abstractions.
pub mod tool;
