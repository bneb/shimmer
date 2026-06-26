//! Model detection, loading, and per-model configuration.
//!
//! Each supported model declares its chat template, end-of-generation
//! tokens, and default parameters through the CanonicalModel enum.
//! Callers match on the enum variant to select model-specific behavior
//! rather than hardcoding logic per model.
//!
//! Pipeline:
//!   Model path → detect CanonicalModel → get ChatTemplate
//!   → format prompt via template → model generates raw text
//!   → interceptor parses tool calls → agent executes → template formats results

use anyhow::{Context, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;

use crate::engine;

// Model adapter

/// A chat template with slots for system prompt, tools list, and user message.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    /// Full prompt format. Slots: {system}, {tools}, {user}
    pub prompt: &'static str,
    /// Wraps a raw tool result before injecting as a user turn.
    pub tool_result_wrapper: &'static str,
    /// Text the interceptor looks for to detect tool calls. Usually ```json.
    pub tool_call_marker: &'static str,
    /// EOG token text that signals end of generation.
    pub eog_text: &'static str,
}

/// Supported canonical models that Shimmer can intelligently configure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalModel {
    Gemma4_12B,
    Qwen25Coder7B,
    Unknown(String),
}

impl CanonicalModel {
    /// Detects the canonical model type from the file path.
    pub fn from_path(path: &str) -> Self {
        let lower = path.to_lowercase();
        if lower.contains("gemma") {
            Self::Gemma4_12B
        } else if lower.contains("qwen2") || (lower.contains("qwen") && lower.contains("2.5")) {
            Self::Qwen25Coder7B
        } else {
            Self::Unknown(path.to_string())
        }
    }

    /// Returns the chat template for this model family.
    pub fn chat_template(&self) -> ChatTemplate {
        match self {
            Self::Qwen25Coder7B => ChatTemplate {
                prompt: "<|im_start|>system\n{system}\nAvailable tools: \
                         {tools}\n<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\\
                         n<|im_start|>assistant\n",
                tool_result_wrapper: "<|im_end|>\n<|im_start|>user\n[Tool '{name}' \
                                      executed]\n{output}\n<|im_end|>\n<|im_start|>assistant\n",
                tool_call_marker: "```json",
                eog_text: "<|im_end|>",
            },
            Self::Gemma4_12B => ChatTemplate {
                // Separate system instructions from user problem to prevent echoing
                prompt: "<start_of_turn>user\n{system}\n\nAvailable tools: \
                         {tools}\n<end_of_turn>\n<start_of_turn>model\nI'll investigate and fix \
                         the issue.<end_of_turn>\n<start_of_turn>user\n{user}\n\nProvide your fix \
                         at the end.\n<end_of_turn>\n<start_of_turn>model\n<|channel>thought\\
                         n<channel|>",
                tool_result_wrapper: "<end_of_turn>\n<start_of_turn>user\n[Tool '{name}' \
                                      executed]\n{output}\n<end_of_turn>\n<start_of_turn>model\\
                                      n<|channel>thought\n<channel|>",
                tool_call_marker: "```json",
                eog_text: "<end_of_turn>",
            },
            Self::Unknown(_) => ChatTemplate {
                prompt: "<|im_start|>system\n{system}\nAvailable tools: \
                         {tools}\n<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\\
                         n<|im_start|>assistant\n",
                tool_result_wrapper: "<|im_end|>\n<|im_start|>user\n[Tool '{name}' \
                                      executed]\n{output}\n<|im_end|>\n<|im_start|>assistant\n",
                tool_call_marker: "```json",
                eog_text: "<|im_end|>",
            },
        }
    }

    /// Returns the recommended default for speculative decoding.
    ///
    /// Disabled by default: n-gram drafts from prompt text cause token
    /// corruption in structured output — missing braces and quotes in
    /// JSON tool calls, fused XML tags, dropped characters.
    /// Enable manually with --speculative for raw text generation only.
    pub fn supports_speculative_decoding(&self) -> bool {
        false
    }
}

/// Formats a prompt using the model's chat template.
pub fn format_prompt(model: &CanonicalModel, system: &str, tools: &str, user: &str) -> String {
    let t = model.chat_template();
    t.prompt.replace("{system}", system).replace("{tools}", tools).replace("{user}", user)
}

/// Formats a tool result for injection as a user turn.
pub fn format_tool_result(model: &CanonicalModel, name: &str, output: &str) -> String {
    model.chat_template().tool_result_wrapper.replace("{name}", name).replace("{output}", output)
}

/// Formats a tool nudge for injection as a user turn when the model loops.
pub fn format_tool_nudge(model: &CanonicalModel, name: &str, output: &str) -> String {
    let base = format_tool_result(model, name, output);
    let note = format!(
        "\n[Note: you already ran '{}' with the same arguments. The result is unchanged. Try a \
         different approach or produce your edit.]\n",
        name
    );
    base.replace(
        &format!("[Tool '{}' executed]\n", name),
        &format!("[Tool '{}' executed]\n{}", name, note),
    )
}

/// Formats a tool rejection message for injection as a user turn when the model exceeds the tool limit.
pub fn format_tool_rejection(model: &CanonicalModel) -> String {
    let base = format_tool_result(model, "REJECTED", "");
    base.replace(
        "[Tool 'REJECTED' executed]\n",
        "Tools disabled. You MUST provide the final patch using the <edit> tags immediately. Do \
         not use any more tools.\n",
    )
}

/// Formats a system nudge message for injection as a user turn.
pub fn format_system_nudge(model: &CanonicalModel, msg: &str) -> String {
    let base = format_tool_result(model, "NUDGE", "NUDGE");
    base.replace("[Tool 'NUDGE' executed]\nNUDGE", msg)
}

// Model loading

/// Loads a llama-cpp-2 model from the specified file path.
pub fn load_model(backend: &LlamaBackend, path: &str) -> Result<LlamaModel> {
    if !std::path::Path::new(path).exists() {
        anyhow::bail!("Model file not found: {}", path);
    }
    let params = engine::optimized_model_params();
    let model = LlamaModel::load_from_file(backend, path, &params)
        .with_context(|| format!("Failed to load model from {}", path))?;
    Ok(model)
}

/// Creates an optimized execution context with a given context window size.
pub fn create_context<'a>(
    backend: &LlamaBackend,
    model: &'a LlamaModel,
    ctx_size: u32,
    n_seq_max: u32,
) -> Result<LlamaContext<'a>> {
    let params = engine::optimized_ctx_params(ctx_size, n_seq_max);
    let ctx = model.new_context(backend, params).with_context(|| "Failed to create context")?;
    Ok(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_model_invalid_path_fails_gracefully() {
        let backend = engine::init_backend().unwrap();
        let result = load_model(&backend, "invalid_path.gguf");
        assert!(result.is_err());
    }

    #[test]
    fn test_canonical_model_detection() {
        assert_eq!(
            CanonicalModel::from_path("/path/to/gemma-12b.gguf"),
            CanonicalModel::Gemma4_12B
        );
        assert_eq!(
            CanonicalModel::from_path("/path/to/qwen2.5-coder-7b.gguf"),
            CanonicalModel::Qwen25Coder7B
        );
        assert_eq!(
            CanonicalModel::from_path("/path/to/llama-3-8b.gguf"),
            CanonicalModel::Unknown("/path/to/llama-3-8b.gguf".to_string())
        );
    }

    #[test]
    fn test_chat_template_qwen() {
        let t = CanonicalModel::Qwen25Coder7B.chat_template();
        assert!(t.prompt.contains("<|im_start|>"));
        assert!(t.eog_text.contains("<|im_end|>"));
    }

    #[test]
    fn test_chat_template_gemma() {
        let t = CanonicalModel::Gemma4_12B.chat_template();
        assert!(t.prompt.contains("<start_of_turn>"));
        assert!(t.eog_text.contains("<end_of_turn>"));
    }

    #[test]
    fn test_format_prompt() {
        let prompt = format_prompt(&CanonicalModel::Qwen25Coder7B, "sys", "tools", "user");
        assert!(prompt.contains("sys"));
        assert!(prompt.contains("tools"));
        assert!(prompt.contains("user"));
    }

    #[test]
    fn test_format_tool_result() {
        let result =
            format_tool_result(&CanonicalModel::Qwen25Coder7B, "rg", ".shimmer_tool_1.txt");
        assert!(result.contains("rg"));
        assert!(result.contains(".shimmer_tool_1.txt"));
    }
}
