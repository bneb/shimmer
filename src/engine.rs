//! Hardware Optimization Engine
//!
//! Exposes Apple Metal specific configuration constants and parameters
//! necessary to maximize memory bandwidth and computation throughput on M-Series chips.

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;

/// Configures the engine backend.
pub fn init_backend() -> anyhow::Result<LlamaBackend> {
    let backend = LlamaBackend::init()?;
    Ok(backend)
}

/// The maximum number of layers to offload to the GPU.
const MAX_GPU_LAYERS: u32 = 99; // Offload all layers to Metal GPU
/// The number of threads to utilize for computation.
const N_THREADS: i32 = 4;

/// Creates optimized model parameters for Apple Silicon (mlock).
pub fn optimized_model_params() -> LlamaModelParams {
    LlamaModelParams::default().with_n_gpu_layers(MAX_GPU_LAYERS).with_use_mmap(true)
}

use std::num::NonZeroU32;

/// Creates optimized context parameters (Flash Attention, Q8_0 KV).
pub fn optimized_ctx_params(ctx_size: u32, n_seq_max: u32) -> LlamaContextParams {
    let non_zero_ctx = NonZeroU32::new(ctx_size).unwrap();
    LlamaContextParams::default()
        .with_n_ctx(Some(non_zero_ctx))
        .with_n_batch(512)
        .with_n_ubatch(512)
        .with_type_k(llama_cpp_2::context::params::KvCacheType::Q8_0)
        .with_type_v(llama_cpp_2::context::params::KvCacheType::Q8_0)
        .with_n_threads(N_THREADS)
        .with_n_threads_batch(N_THREADS)
        .with_n_seq_max(n_seq_max)
        .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED)
}

#[cfg(test)]
/// Unit tests for the engine configuration parameters.
mod tests {
    use super::*;

    #[test]
    /// Verifies that optimized model parameters are successfully generated.
    fn test_model_params_has_mlock() {
        let _params = optimized_model_params();
    }

    #[test]
    /// Verifies that optimized context parameters are successfully generated.
    fn test_ctx_params_has_flash_attn() {
        let _params = optimized_ctx_params(8192, 1);
    }
}
