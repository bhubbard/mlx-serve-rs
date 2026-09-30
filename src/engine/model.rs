use mlx_rs::module::Module;
use mlx_rs::nn::{Embedding, Linear};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{concatenate, softmax_axis, transpose_axes};
use mlx_rs::Array;
use crate::error::Result;

#[derive(Debug)]
pub struct KVCache {
    pub keys: Vec<Option<Array>>,
    pub values: Vec<Option<Array>>,
}

impl KVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            keys: vec![None; num_layers],
            values: vec![None; num_layers],
        }
    }

    pub fn update(&mut self, layer: usize, k: &Array, v: &Array) -> Result<(Array, Array)> {
        let full_k = match &self.keys[layer] {
            Some(prev_k) => concatenate(&[prev_k, k], 2)?,
            None => k.clone(),
        };

        let full_v = match &self.values[layer] {
            Some(prev_v) => concatenate(&[prev_v, v], 2)?,
            None => v.clone(),
        };

        self.keys[layer] = Some(full_k.clone());
        self.values[layer] = Some(full_v.clone());
        Ok((full_k, full_v))
    }

    pub fn reset(&mut self) {
        for k in &mut self.keys {
            *k = None;
        }
        for v in &mut self.values {
            *v = None;
        }
    }
}

#[derive(Debug)]
pub struct TransformerBlock {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub mlp1: Linear,
    pub mlp2: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub layer_idx: usize,
}

impl TransformerBlock {
    pub fn new(hidden_dim: usize, num_heads: usize, layer_idx: usize) -> Result<Self> {
        let head_dim = hidden_dim / num_heads;
        let intermediate = hidden_dim * 4;
        Ok(Self {
            q_proj: Linear::new(hidden_dim as i32, hidden_dim as i32)?,
            k_proj: Linear::new(hidden_dim as i32, hidden_dim as i32)?,
            v_proj: Linear::new(hidden_dim as i32, hidden_dim as i32)?,
            out_proj: Linear::new(hidden_dim as i32, hidden_dim as i32)?,
            mlp1: Linear::new(hidden_dim as i32, intermediate as i32)?,
            mlp2: Linear::new(intermediate as i32, hidden_dim as i32)?,
            num_heads,
            head_dim,
            layer_idx,
        })
    }

    pub fn forward(&mut self, x: &Array, cache: Option<&mut KVCache>) -> Result<Array> {
        let shape = x.shape();
        let bsz = shape[0];
        let seq_len = shape[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q_s = q.reshape(&[bsz, seq_len, self.num_heads as i32, self.head_dim as i32])?;
        let q_h = transpose_axes(&q_s, &[0, 2, 1, 3])?;

        let k_s = k.reshape(&[bsz, seq_len, self.num_heads as i32, self.head_dim as i32])?;
        let k_h = transpose_axes(&k_s, &[0, 2, 1, 3])?;

        let v_s = v.reshape(&[bsz, seq_len, self.num_heads as i32, self.head_dim as i32])?;
        let v_h = transpose_axes(&v_s, &[0, 2, 1, 3])?;

        let (full_k, full_v) = if let Some(c) = cache {
            c.update(self.layer_idx, &k_h, &v_h)?
        } else {
            (k_h, v_h)
        };

        let kv_seq_len = full_k.shape()[2] as usize;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let scale_arr = Array::from_f32(scale);
        let k_t = transpose_axes(&full_k, &[0, 1, 3, 2])?;
        let mut scores = q_h.matmul(&k_t)?.multiply(&scale_arr)?;

        if seq_len > 1 {
            let mut mask = vec![0.0f32; (seq_len as usize) * kv_seq_len];
            for q_idx in 0..(seq_len as usize) {
                for k_idx in 0..kv_seq_len {
                    if k_idx > (kv_seq_len - seq_len as usize) + q_idx {
                        mask[q_idx * kv_seq_len + k_idx] = -10000.0f32;
                    }
                }
            }
            let mask_arr = Array::from_slice(&mask, &[1, 1, seq_len, kv_seq_len as i32]);
            scores = scores.add(&mask_arr)?;
        }

        let weights = softmax_axis(&scores, -1, None)?;
        let context = weights.matmul(&full_v)?;
        let context_t = transpose_axes(&context, &[0, 2, 1, 3])?;
        let attn_flat = context_t.reshape(&[bsz, seq_len, (self.num_heads * self.head_dim) as i32])?;
        let attn_out = self.out_proj.forward(&attn_flat)?;

        // Residual + LayerNorm
        let h1 = x.add(&attn_out)?;
        let norm1 = rms_norm(&h1, 1e-5)?;

        // MLP
        let mlp_h = self.mlp1.forward(&norm1)?;
        let act = mlx_rs::nn::silu(&mlp_h)?;
        let mlp_out = self.mlp2.forward(&act)?;

        let h2 = norm1.add(&mlp_out)?;
        rms_norm(&h2, 1e-5)
    }
}

#[derive(Debug)]
pub struct MlxModel {
    pub model_id: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub embedding: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub lm_head: Linear,
    pub kv_cache: KVCache,
}

impl MlxModel {
    pub fn new(model_id: String, vocab_size: usize, hidden_size: usize, num_layers: usize, num_heads: usize) -> Result<Self> {
        let embedding = Embedding::new(vocab_size as i32, hidden_size as i32)?;
        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            blocks.push(TransformerBlock::new(hidden_dim_round(hidden_size, num_heads), num_heads, i)?);
        }
        let lm_head = Linear::new(hidden_size as i32, vocab_size as i32)?;
        let kv_cache = KVCache::new(num_layers);

        Ok(Self {
            model_id,
            vocab_size,
            hidden_size,
            num_layers,
            embedding,
            blocks,
            lm_head,
            kv_cache,
        })
    }

    pub fn reset_cache(&mut self) {
        self.kv_cache.reset();
    }

    /// Prefill prompt sequence tokens and produce logits for the next token
    pub fn prefill(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        self.reset_cache();
        let token_arr = Array::from_slice(tokens, &[1, tokens.len() as i32]);
        let mut h = self.embedding.forward(&token_arr)?;

        for block in &mut self.blocks {
            h = block.forward(&h, Some(&mut self.kv_cache))?;
        }

        let seq_len = h.shape()[1] as usize;
        let last_hidden = h.index((.., (seq_len - 1) as i32..seq_len as i32, ..));
        let logits = self.lm_head.forward(&last_hidden)?;
        Ok(logits.as_slice::<f32>().to_vec())
    }

    /// Single autoregressive decode step using cached keys and values
    pub fn decode_step(&mut self, token: i32) -> Result<Vec<f32>> {
        let token_arr = Array::from_slice(&[token], &[1, 1]);
        let mut h = self.embedding.forward(&token_arr)?;

        for block in &mut self.blocks {
            h = block.forward(&h, Some(&mut self.kv_cache))?;
        }

        let logits = self.lm_head.forward(&h)?;
        Ok(logits.as_slice::<f32>().to_vec())
    }

    /// Compute dense text embedding vector via mean pooling over token embeddings
    pub fn embed(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        let token_arr = Array::from_slice(tokens, &[1, tokens.len() as i32]);
        let mut h = self.embedding.forward(&token_arr)?;

        for block in &mut self.blocks {
            h = block.forward(&h, None)?;
        }

        let pooled = h.mean_axis(1, false)?;
        // L2 normalize
        let sq = pooled.square()?;
        let sum_sq = sq.sum_axis(-1, true)?;
        let eps = Array::from_f32(1e-12);
        let norm = sum_sq.add(&eps)?.sqrt()?;
        let normalized = pooled.divide(&norm)?;

        Ok(normalized.as_slice::<f32>().to_vec())
    }
}

fn hidden_dim_round(dim: usize, heads: usize) -> usize {
    ((dim + heads - 1) / heads) * heads
}

fn rms_norm(x: &Array, eps: f32) -> Result<Array> {
    let sq = x.square()?;
    let mean_sq = sq.mean_axis(-1, true)?;
    let eps_arr = Array::from_f32(eps);
    let rms = mean_sq.add(&eps_arr)?.sqrt()?;
    Ok(x.divide(&rms)?)
}
