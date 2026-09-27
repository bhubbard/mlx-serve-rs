use rand::Rng;

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub stop: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(50),
            repetition_penalty: 1.0,
            stop: Vec::new(),
        }
    }
}

pub struct Sampler;

impl Sampler {
    pub fn sample(
        logits: &[f32],
        params: &SamplingParams,
        past_tokens: &[i32],
    ) -> i32 {
        if logits.is_empty() {
            return 0;
        }

        let mut filtered_logits = logits.to_vec();
        let vocab_size = filtered_logits.len();

        // Repetition penalty
        if (params.repetition_penalty - 1.0).abs() > 1e-4 {
            for &token in past_tokens {
                let idx = token as usize;
                if idx < vocab_size {
                    if filtered_logits[idx] > 0.0 {
                        filtered_logits[idx] /= params.repetition_penalty;
                    } else {
                        filtered_logits[idx] *= params.repetition_penalty;
                    }
                }
            }
        }

        // Greedy decoding if temperature is near 0
        if params.temperature <= 1e-5 {
            return filtered_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i32)
                .unwrap_or(0);
        }

        // Apply temperature
        let inv_t = 1.0 / params.temperature;
        for val in &mut filtered_logits {
            *val *= inv_t;
        }

        // Softmax
        let max_logit = filtered_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        let mut probs: Vec<f32> = filtered_logits
            .iter()
            .map(|&l| {
                let p = (l - max_logit).exp();
                exp_sum += p;
                p
            })
            .collect();

        if exp_sum > 0.0 {
            for p in &mut probs {
                *p /= exp_sum;
            }
        }

        // Top-K / Top-P
        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(top_k) = params.top_k {
            if top_k > 0 && top_k < indexed.len() {
                indexed.truncate(top_k);
            }
        }

        if params.top_p < 1.0 {
            let mut cum = 0.0f32;
            let mut cutoff = indexed.len();
            for (i, (_, p)) in indexed.iter().enumerate() {
                cum += p;
                if cum >= params.top_p {
                    cutoff = (i + 1).min(indexed.len());
                    break;
                }
            }
            indexed.truncate(cutoff);
        }

        let total: f32 = indexed.iter().map(|(_, p)| *p).sum();
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen::<f32>() * total;

        let mut acc = 0.0f32;
        for (idx, p) in indexed {
            acc += p;
            if r <= acc {
                return idx as i32;
            }
        }

        0
    }
}
