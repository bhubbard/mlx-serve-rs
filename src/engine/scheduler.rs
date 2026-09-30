use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use crate::engine::model::MlxModel;
use crate::engine::sampler::{Sampler, SamplingParams};
use crate::error::{Result, ServerError};

#[derive(Debug)]
pub struct ServerMetrics {
    pub total_requests: AtomicU64,
    pub total_generated_tokens: AtomicU64,
    pub total_prompt_tokens: AtomicU64,
    pub active_requests: AtomicUsize,
}

impl Default for ServerMetrics {
    fn default() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            total_generated_tokens: AtomicU64::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            active_requests: AtomicUsize::new(0),
        }
    }
}

#[derive(Debug)]
pub struct GenerationRequest {
    pub model_id: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub tx: Option<mpsc::UnboundedSender<String>>,
}

#[derive(Debug)]
pub struct Scheduler {
    pub models: HashMap<String, MlxModel>,
    pub default_model: String,
    pub metrics: Arc<ServerMetrics>,
}

impl Scheduler {
    pub fn new(default_model: String) -> Result<Self> {
        let mut models = HashMap::new();

        // Initialize default MLX model
        let model = MlxModel::new(
            default_model.clone(),
            32000,
            128,
            2,
            4,
        )?;
        models.insert(default_model.clone(), model);

        Ok(Self {
            models,
            default_model,
            metrics: Arc::new(ServerMetrics::default()),
        })
    }

    pub fn list_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    pub fn tokenize(text: &str) -> Vec<i32> {
        let mut tokens = Vec::new();
        for b in text.as_bytes() {
            tokens.push((*b as i32) % 32000);
        }
        if tokens.is_empty() {
            tokens.push(1); // BOS
        }
        tokens
    }

    pub fn detokenize(token: i32) -> String {
        let b = (token.abs() % 256) as u8;
        if b.is_ascii_graphic() || b == b' ' || b == b'\n' {
            (b as char).to_string()
        } else {
            " ".to_string()
        }
    }

    pub async fn generate_stream(
        scheduler: Arc<Mutex<Self>>,
        model_id: &str,
        prompt: &str,
        max_tokens: usize,
        sampling: SamplingParams,
        tx: mpsc::UnboundedSender<String>,
    ) -> Result<()> {
        let prompt_tokens = Self::tokenize(prompt);
        let prompt_len = prompt_tokens.len();

        let mut sched = scheduler.lock().await;
        let metrics = sched.metrics.clone();

        metrics.active_requests.fetch_add(1, Ordering::SeqCst);
        metrics.total_requests.fetch_add(1, Ordering::SeqCst);
        metrics.total_prompt_tokens.fetch_add(prompt_len as u64, Ordering::SeqCst);

        let model = match sched.models.get_mut(model_id) {
            Some(m) => m,
            None => {
                metrics.active_requests.fetch_sub(1, Ordering::SeqCst);
                return Err(ServerError::ModelNotFound(model_id.to_string()));
            }
        };

        // 1. Prefill
        let mut logits = model.prefill(&prompt_tokens)?;
        let mut past_tokens = prompt_tokens.clone();
        let mut generated_tokens = Vec::new();

        for _ in 0..max_tokens {
            let next_token = Sampler::sample(&logits, &sampling, &past_tokens);
            generated_tokens.push(next_token);
            past_tokens.push(next_token);

            let token_str = Self::detokenize(next_token);
            let should_stop = sampling.stop.iter().any(|s| token_str.contains(s)) || next_token == 2;

            if tx.send(token_str).is_err() {
                break; // Client disconnected
            }

            if should_stop {
                break;
            }

            logits = model.decode_step(next_token)?;
        }

        metrics.total_generated_tokens.fetch_add(generated_tokens.len() as u64, Ordering::SeqCst);
        metrics.active_requests.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }

    pub async fn generate_full(
        scheduler: Arc<Mutex<Self>>,
        model_id: &str,
        prompt: &str,
        max_tokens: usize,
        sampling: SamplingParams,
    ) -> Result<(String, usize, usize)> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let sched_clone = scheduler.clone();
        let model_id_owned = model_id.to_string();
        let prompt_owned = prompt.to_string();

        let gen_handle = tokio::spawn(async move {
            Self::generate_stream(
                sched_clone,
                &model_id_owned,
                &prompt_owned,
                max_tokens,
                sampling,
                tx,
            ).await
        });

        let mut output = String::new();
        let mut count = 0;
        while let Some(chunk) = rx.recv().await {
            output.push_str(&chunk);
            count += 1;
        }

        gen_handle.await.map_err(|e| ServerError::Internal(e.to_string()))??;
        let prompt_len = prompt.len().max(1);
        Ok((output, prompt_len, count))
    }

    pub async fn embed(
        scheduler: Arc<Mutex<Self>>,
        model_id: &str,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        let mut tokenized_batches = Vec::new();
        for text in texts {
            tokenized_batches.push(Self::tokenize(text));
        }

        let mut sched = scheduler.lock().await;
        let model = match sched.models.get_mut(model_id) {
            Some(m) => m,
            None => return Err(ServerError::ModelNotFound(model_id.to_string())),
        };

        let mut embeddings = Vec::new();
        for tokens in tokenized_batches {
            let emb = model.embed(&tokens)?;
            embeddings.push(emb);
        }
        Ok(embeddings)
    }
}
