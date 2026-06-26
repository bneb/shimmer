//! Token sampling and logit manipulation for the agent engine.
//!
//! Applies temperature scaling, repetition penalty, and top-k filtering
//! to logit distributions during token generation. When temperature is 0,
//! selects the argmax token (greedy). When temperature > 0, samples from
//! the softmax distribution.

use super::state::EngineState;
use anyhow::Result;
use rand::Rng;

impl super::Agent {
    /// Samples the next token from the engine state. Greedy argmax when
    /// temperature is 0; multinomial sampling from the softmax distribution
    /// when temperature > 0.
    pub(crate) fn sample_engine_token(
        &self,
        state: &mut EngineState,
    ) -> Result<llama_cpp_2::token::LlamaToken> {
        let candidates: Vec<_> = state.ctx.candidates_ith(state.prev_size - 1).collect();
        let mut data_array =
            llama_cpp_2::token::data_array::LlamaTokenDataArray::new(candidates, false);
        if let Some(sampler) = &mut state.sampler {
            sampler.apply(&mut data_array);
        }

        let sc = &state.sample_config;
        let mut scored: Vec<(llama_cpp_2::token::LlamaToken, f32)> =
            data_array.data.iter().map(|d| (d.id(), d.logit())).collect();

        self.apply_temperature(&mut scored, sc.temperature);
        self.apply_repetition_penalty(&mut scored, &state.history, sc.repetition_penalty);
        self.apply_top_k(&mut scored, sc.top_k);

        if sc.temperature > 0.0 {
            Ok(sample_from_distribution(&scored))
        } else {
            Ok(argmax(&scored))
        }
    }

    /// Divides all logits by the temperature. No-op when temperature is 0.
    fn apply_temperature(
        &self,
        scored: &mut [(llama_cpp_2::token::LlamaToken, f32)],
        temperature: f32,
    ) {
        if temperature > 0.0 {
            for (_, logit) in scored.iter_mut() {
                *logit /= temperature;
            }
        }
    }

    /// Penalizes tokens that appear in the recent history window.
    fn apply_repetition_penalty(
        &self,
        scored: &mut [(llama_cpp_2::token::LlamaToken, f32)],
        history: &[llama_cpp_2::token::LlamaToken],
        penalty: f32,
    ) {
        if (penalty - 1.0).abs() < f32::EPSILON {
            return;
        }
        let recent: std::collections::HashSet<llama_cpp_2::token::LlamaToken> =
            history.iter().rev().take(256).copied().collect();
        for (id, logit) in scored.iter_mut() {
            if recent.contains(id) {
                if *logit > 0.0 {
                    *logit /= penalty;
                } else {
                    *logit *= penalty;
                }
            }
        }
    }

    /// Retains only the top-k candidates by logit value. No-op when k is 0.
    fn apply_top_k(
        &self,
        scored: &mut Vec<(llama_cpp_2::token::LlamaToken, f32)>,
        k: usize,
    ) {
        if k == 0 || k >= scored.len() {
            return;
        }
        scored.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
    }
}

fn argmax(scored: &[(llama_cpp_2::token::LlamaToken, f32)]) -> llama_cpp_2::token::LlamaToken {
    scored
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(id, _)| *id)
        .unwrap_or(llama_cpp_2::token::LlamaToken::new(0))
}

fn sample_from_distribution(
    scored: &[(llama_cpp_2::token::LlamaToken, f32)],
) -> llama_cpp_2::token::LlamaToken {
    let max_logit = scored.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = scored.iter().map(|(_, s)| (s - max_logit).exp()).sum();
    if exp_sum <= 0.0 {
        return argmax(scored);
    }
    let mut rng = rand::rng();
    let target: f32 = rng.random();
    let mut cumulative = 0.0f32;
    for (id, score) in scored {
        cumulative += (score - max_logit).exp() / exp_sum;
        if cumulative >= target {
            return *id;
        }
    }
    scored.last().map(|(id, _)| *id).unwrap_or(llama_cpp_2::token::LlamaToken::new(0))
}
