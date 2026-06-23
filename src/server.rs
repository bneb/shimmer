//! Shimmer REST API Server
//!
//! This module implements an OpenAI-compatible HTTP server using Axum, providing streaming and non-streaming
//! chat completion endpoints.

use axum::{
    Json, Router, body::Body,
    extract::State,
    http::StatusCode,
    middleware::{self, from_fn_with_state},
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;
use uuid::Uuid;

use crate::agent::Agent;
use llama_cpp_2::llama_backend::LlamaBackend;

/// Represents a single chat message containing a role and its corresponding content.
#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Incoming request payload for the chat completions endpoint, specifying messages, model, and tool usage.
#[derive(Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
}

/// Describes an OpenAI-compatible function that the model can call.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct OpenAIFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
}

/// Wraps an `OpenAIFunction` into a tool object.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: OpenAIFunction,
}

/// Represents a delta update for a streaming chat chunk, containing partial content or tool calls.
#[derive(Serialize)]
pub struct ChatChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

/// Represents a single choice within a streaming chat chunk response.
#[derive(Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Represents a Server-Sent Events (SSE) chunk emitted during a streaming chat completion.
#[derive(Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

/// Application state shared across all HTTP routes, holding references to the agent and backend.
#[derive(Clone)]
pub struct AppState {
    pub agent: Arc<Agent>,
    pub backend: Arc<LlamaBackend>,
    pub config: Arc<crate::agent::AgentConfig>,
    pub api_key: Option<String>,
}

/// OpenAI-compatible error response body.
#[derive(Serialize)]
struct OpenAIErrorResponse {
    error: OpenAIErrorDetail,
}

/// Detail inside an OpenAI error response.
#[derive(Serialize)]
struct OpenAIErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: String,
}

/// Middleware that enforces a 300-second timeout on all requests.
async fn timeout_handler(
    request: axum::http::Request<Body>,
    next: middleware::Next,
) -> impl IntoResponse {
    let Ok(response) = tokio::time::timeout(std::time::Duration::from_secs(300), next.run(request)).await
    else {
        return (StatusCode::REQUEST_TIMEOUT, "Request timed out").into_response();
    };
    response
}

/// Creates and configures the Axum router for the HTTP server, injecting the required application state.
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route_layer(from_fn_with_state(state.clone(), auth_middleware))
        .route("/health", get(health_check))
        .layer(CorsLayer::permissive())
        .layer(middleware::from_fn(timeout_handler))
        .layer(RequestBodyLimitLayer::new(1024 * 1024))
        .with_state(state)
}

/// Constant-time byte comparison for API key verification.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).fold(0, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Rejects requests missing a valid Bearer token when an API key is configured.
/// The /health endpoint is exempt (registered after the route_layer).
async fn auth_middleware(
    State(state): State<AppState>,
    request: axum::http::Request<Body>,
    next: middleware::Next,
) -> impl IntoResponse {
    if let Some(ref expected_key) = state.api_key {
        let provided = request
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .unwrap_or("");
        if !constant_time_eq(provided.as_bytes(), expected_key.as_bytes()) {
            return (StatusCode::UNAUTHORIZED, Json(serde_json::json!({
                "error": {
                    "message": "Invalid or missing API key. Use Authorization: Bearer <key>",
                    "type": "authentication_error",
                    "code": "401"
                }
            }))).into_response();
        }
    }
    next.run(request).await
}

/// Compiles an array of chat messages and optional tools into a single prompt string formatted for the model.
fn build_prompt(messages: &[ChatMessage], tools: &Option<Vec<OpenAITool>>) -> String {
    let mut prompt = String::new();

    // Inject system prompt for tools if provided
    if let Some(t_vec) = tools {
        let tools_json = serde_json::to_string_pretty(t_vec).unwrap_or_else(|_| "[]".into());
        let sys_msg = format!(
            "You have access to the following tools. To use a tool, output exactly this JSON syntax:\n```json\n{{\"name\": \"tool_name\", \"arguments\": {{\"arg_name\": \"arg_value\"}}}}\n```\nAvailable tools schema:\n{}",
            tools_json
        );
        prompt.push_str(&format!("<start_of_turn>model\n{}<end_of_turn>\n", sys_msg));
    }

    for msg in messages {
        let role = if msg.role == "assistant" {
            "model"
        } else {
            &msg.role
        };
        prompt.push_str(&format!(
            "<start_of_turn>{}\n{}<end_of_turn>\n",
            role, msg.content
        ));
    }
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

/// Represents a parsed segment of the model's output, either text content or a tool invocation.
pub enum ApiChunk {
    Content(String),
    ToolCall(String, serde_json::Value), // name, args
}

/// State machine responsible for intercepting and parsing tool calls from the raw text stream.
struct ApiInterceptor {
    buffer: String,
}

impl ApiInterceptor {
    /// Initializes a new, empty API interceptor.
    fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Checks if the current buffer ends with a partial tool tag prefix, preventing premature flushing.
    fn get_tool_prefix_len(buf: &str) -> usize {
        let tag = "```json\n";
        for i in (1..=tag.len()).rev() {
            if buf.ends_with(&tag[..i]) {
                return i;
            }
        }
        0
    }

    /// Appends a new token to the buffer and extracts complete content chunks and tool calls.
    fn push(&mut self, token: &str) -> Vec<ApiChunk> {
        self.buffer.push_str(token);
        let mut chunks = Vec::new();
        let tag_start = "```json\n";
        let tag_end = "```";

        loop {
            if let Some(start_idx) = self.buffer.find(tag_start) {
                let rest = &self.buffer[start_idx + tag_start.len()..];
                if let Some(end_offset) = rest.find(tag_end) {
                    if start_idx > 0 {
                        let content = self.buffer[..start_idx].to_string();
                        chunks.push(ApiChunk::Content(content));
                    }
                    
                    let end_idx = start_idx + tag_start.len() + end_offset + tag_end.len();
                    let payload = crate::interceptor::ToolInterceptor::extract_json_payload(&self.buffer[..end_idx]);
                    if let Some(p) = payload {
                        chunks.push(ApiChunk::ToolCall(p.name, p.arguments));
                    } else {
                        // Fallback if payload is malformed
                        let content = self.buffer[..end_idx].to_string();
                        chunks.push(ApiChunk::Content(content));
                    }
                    self.buffer = self.buffer[end_idx..].to_string();
                    continue;
                } else if self.buffer.len() > 8192 {
                    // Flush if missing end tag for too long
                    let content = self.buffer.clone();
                    chunks.push(ApiChunk::Content(content));
                    self.buffer.clear();
                    break;
                } else {
                    // Wait for the rest of the tool call
                    if start_idx > 0 {
                        let content = self.buffer[..start_idx].to_string();
                        chunks.push(ApiChunk::Content(content));
                        self.buffer = self.buffer[start_idx..].to_string();
                    }
                    break;
                }
            } else {
                let prefix_len = Self::get_tool_prefix_len(&self.buffer);
                if prefix_len > 0 {
                    let content_len = self.buffer.len() - prefix_len;
                    if content_len > 0 {
                        let content = self.buffer[..content_len].to_string();
                        chunks.push(ApiChunk::Content(content));
                        self.buffer = self.buffer[content_len..].to_string();
                    }
                } else if !self.buffer.is_empty() {
                    let content = self.buffer.clone();
                    chunks.push(ApiChunk::Content(content));
                    self.buffer.clear();
                }
                break;
            }
        }
        chunks
    }
}

/// Spawns a background thread to run the agent inference process and pipes output back to an async channel.
fn spawn_agent_thread(state: AppState, prompt: String, tx: mpsc::UnboundedSender<ApiChunk>) {
    tokio::task::spawn_blocking(move || {
        let (std_tx, std_rx): (std::sync::mpsc::Sender<String>, _) = std::sync::mpsc::channel();
        let tx_clone = tx.clone();

        std::thread::spawn(move || {
            let mut interceptor = ApiInterceptor::new();
            while let Ok(msg) = std_rx.recv() {
                let chunks = interceptor.push(&msg);
                for chunk in chunks {
                    if tx_clone.send(chunk).is_err() {
                        return;
                    }
                }
            }
        });

        let _ = state
            .agent
            .process_stream(&state.backend, &prompt, &state.config, std_tx);
    });
}

/// Represents a fully formed response message containing role and content.
#[derive(Serialize)]
pub struct ChatMessageResponse {
    pub role: String,
    pub content: String,
}

/// Represents a single choice in a complete, non-streaming chat response.
#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessageResponse,
}

/// Represents the final, complete response payload for a non-streaming chat completions request.
#[derive(Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
}

const OBJECT_CHAT_COMPLETION: &str = "chat.completion";
const OBJECT_CHAT_STREAM_CHUNK: &str = "chat.completion.chunk";
const ROLE_ASSISTANT: &str = "assistant";
const TOOL_CALL_TYPE: &str = "function";

/// Constructs an OpenAI-compatible error response.
fn error_response(status: StatusCode, error_type: &str, message: &str) -> (StatusCode, Json<OpenAIErrorResponse>) {
    (
        status,
        Json(OpenAIErrorResponse {
            error: OpenAIErrorDetail {
                message: message.to_string(),
                error_type: error_type.to_string(),
                code: status.as_u16().to_string(),
            },
        }),
    )
}

/// GET /health — returns a simple status indicator.
async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

/// Axum handler for processing POST requests to the `/v1/chat/completions` endpoint.
async fn chat_completions(
    State(state): State<AppState>,
    req: Result<Json<ChatRequest>, axum::extract::rejection::JsonRejection>,
) -> axum::response::Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(_) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "The request body is not valid JSON or does not match the expected schema.",
            ).into_response();
        }
    };

    let chat_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    tracing::info!("Received request for model {}", req.model);
    let (tx, rx) = mpsc::unbounded_channel();
    let prompt = build_prompt(&req.messages, &req.tools);
    let model_name = req.model.clone();

    spawn_agent_thread(state, prompt, tx);

    if req.stream {
        handle_streaming_response(rx, model_name, chat_id, created).into_response()
    } else {
        handle_sync_response(rx, model_name, chat_id, created).await.into_response()
    }
}

/// Formats a single ApiChunk into a streaming SSE Event.
fn format_sse_chunk(chunk: ApiChunk, model_name: &str, tool_id: &mut usize, chat_id: &str, created: u64) -> Event {
    let mut delta = ChatChunkDelta { content: None, tool_calls: None };

    match chunk {
        ApiChunk::Content(c) => delta.content = Some(c),
        ApiChunk::ToolCall(name, args) => {
            let id = format!("call_{}", tool_id);
            *tool_id += 1;
            let args_str = serde_json::to_string(&args).unwrap_or_else(|_| "[]".into());
            delta.tool_calls = Some(vec![serde_json::json!({
                "index": 0, "id": id, "type": TOOL_CALL_TYPE,
                "function": { "name": name, "arguments": args_str }
            })]);
        }
    }

    let sse_chunk = ChatChunk {
        id: chat_id.into(),
        object: OBJECT_CHAT_STREAM_CHUNK.into(),
        created,
        model: model_name.to_string(),
        choices: vec![ChatChunkChoice { index: 0, delta, finish_reason: None }],
    };
    Event::default().data(serde_json::to_string(&sse_chunk).unwrap_or_else(|_| "{}".into()))
}

/// Manages the streaming response lifecycle.
fn handle_streaming_response(
    rx: mpsc::UnboundedReceiver<ApiChunk>,
    model_name: String,
    chat_id: String,
    created: u64,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let mut tool_id = 0;
    let stream = UnboundedReceiverStream::new(rx).map(move |chunk| {
        Ok::<_, Infallible>(format_sse_chunk(chunk, &model_name, &mut tool_id, &chat_id, created))
    });

    let done_stream = tokio_stream::iter(vec![Ok::<_, Infallible>(Event::default().data("[DONE]"))]);
    Sse::new(stream.chain(done_stream))
}

/// Accumulates chunks for a non-streaming sync response.
async fn handle_sync_response(
    mut rx: mpsc::UnboundedReceiver<ApiChunk>,
    model_name: String,
    chat_id: String,
    created: u64,
) -> Json<serde_json::Value> {
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut tool_id = 0;

    while let Some(chunk) = rx.recv().await {
        match chunk {
            ApiChunk::Content(c) => content.push_str(&c),
            ApiChunk::ToolCall(name, args) => {
                let id = format!("call_{}", tool_id);
                tool_id += 1;
                let args_str = serde_json::to_string(&args).unwrap_or_else(|_| "[]".into());
                calls.push(serde_json::json!({
                    "id": id, "type": TOOL_CALL_TYPE,
                    "function": { "name": name, "arguments": args_str }
                }));
            }
        }
    }

    Json(build_sync_json_payload(model_name, content, calls, chat_id, created))
}

/// Constructs the final JSON block for a sync response.
fn build_sync_json_payload(model: String, content: String, calls: Vec<serde_json::Value>, chat_id: String, created: u64) -> serde_json::Value {
    let mut msg = serde_json::json!({ "role": ROLE_ASSISTANT });
    if !content.is_empty() { msg["content"] = serde_json::Value::String(content); }
    if !calls.is_empty() { msg["tool_calls"] = serde_json::Value::Array(calls); }

    serde_json::json!({
        "id": chat_id,
        "object": OBJECT_CHAT_COMPLETION,
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "message": msg, "finish_reason": "stop" }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests that pure content chunks are parsed and emitted correctly.
    #[test]
    fn test_api_interceptor_content() {
        let mut interceptor = ApiInterceptor::new();
        let chunks = interceptor.push("Hello");
        assert_eq!(chunks.len(), 1);
        if let ApiChunk::Content(c) = &chunks[0] {
            assert_eq!(c, "Hello");
        } else {
            panic!("Expected content");
        }
    }

    /// Tests that tool call XML blocks are successfully intercepted, buffered, and parsed into `ApiChunk::ToolCall`.
    #[test]
    fn test_api_interceptor_tool_call() {
        let mut interceptor = ApiInterceptor::new();
        let chunks = interceptor.push("```js");
        assert!(chunks.is_empty()); // Buffered, waiting

        let chunks =
            interceptor.push("on\n{\"name\": \"rg\", \"arguments\": [\"a\"]}\n```");
        assert_eq!(chunks.len(), 1);
        if let ApiChunk::ToolCall(name, args) = &chunks[0] {
            assert_eq!(name, "rg");
            assert_eq!(args, &serde_json::json!(["a"]));
        } else {
            panic!("Expected tool call");
        }
    }

    /// Tests the interceptor's ability to extract both text content and a trailing tool call from the same input stream.
    #[test]
    fn test_api_interceptor_mixed() {
        let mut interceptor = ApiInterceptor::new();
        let chunks = interceptor.push("Okay! Let me search. ```json\n{\"name\": \"rg\", \"arguments\": [\"a\"]}\n```");

        // Wait, because we sent it all at once, the loop should extract BOTH the content before and the tool call.
        assert_eq!(chunks.len(), 2);

        if let ApiChunk::Content(c) = &chunks[0] {
            assert_eq!(c, "Okay! Let me search. ");
        } else {
            panic!("Expected content");
        }

        if let ApiChunk::ToolCall(name, args) = &chunks[1] {
            assert_eq!(name, "rg");
            assert_eq!(args, &serde_json::json!(["a"]));
        } else {
            panic!("Expected tool call");
        }
    }

    #[test]
    fn test_constant_time_eq_matching() {
        assert!(constant_time_eq(b"key123", b"key123"));
    }

    #[test]
    fn test_constant_time_eq_mismatch() {
        assert!(!constant_time_eq(b"key123", b"key456"));
    }

    #[test]
    fn test_constant_time_eq_different_lengths() {
        assert!(!constant_time_eq(b"short", b"longer_key"));
    }

    #[test]
    fn test_build_prompt_formats_messages() {
        use crate::server::{ChatMessage, build_prompt};
        let msgs = vec![
            ChatMessage { role: "user".into(), content: "hello".into() },
        ];
        let prompt = build_prompt(&msgs, &None);
        assert!(prompt.contains("hello"));
        assert!(prompt.contains("user"));
    }

    #[test]
    fn test_build_prompt_includes_tools() {
        use crate::server::{ChatMessage, build_prompt};
        let msgs = vec![
            ChatMessage { role: "user".into(), content: "test".into() },
        ];
        let tools = Some(vec![crate::server::OpenAITool {
            tool_type: "function".into(),
            function: crate::server::OpenAIFunction {
                name: "rg".into(),
                description: Some("search".into()),
                parameters: None,
            },
        }]);
        let prompt = build_prompt(&msgs, &tools);
        assert!(prompt.contains("rg"));
    }

    #[test]
    fn test_error_response_returns_correct_status() {
        let (status, body) = error_response(
            axum::http::StatusCode::UNAUTHORIZED,
            "authentication_error",
            "Invalid key",
        );
        assert_eq!(status, axum::http::StatusCode::UNAUTHORIZED);
        assert!(body.error.message.contains("Invalid key"));
    }
}
