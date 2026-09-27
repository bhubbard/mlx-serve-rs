use mlx_serve_rs::engine::model::{KVCache, MlxModel};

#[test]
fn test_kv_cache() -> anyhow::Result<()> {
    let mut cache = KVCache::new(2);
    let k = mlx_rs::Array::zeros::<f32>(&[1, 2, 4, 16])?;
    let v = mlx_rs::Array::zeros::<f32>(&[1, 2, 4, 16])?;

    let (full_k, full_v) = cache.update(0, &k, &v)?;
    assert_eq!(full_k.shape(), &[1, 2, 4, 16]);
    assert_eq!(full_v.shape(), &[1, 2, 4, 16]);

    cache.reset();
    assert!(cache.keys[0].is_none());
    Ok(())
}

#[test]
fn test_mlx_model_inference() -> anyhow::Result<()> {
    let mut model = MlxModel::new(
        "test-model".to_string(),
        1000,
        64,
        2,
        2,
    )?;

    // Prefill
    let prompt_tokens = vec![1, 2, 3, 4];
    let logits = model.prefill(&prompt_tokens)?;
    assert_eq!(logits.len(), 1000);

    // Decode step
    let next_logits = model.decode_step(5)?;
    assert_eq!(next_logits.len(), 1000);

    // Embed
    let emb = model.embed(&prompt_tokens)?;
    assert_eq!(emb.len(), 64);

    // Verify embedding is L2 normalized
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4);

    Ok(())
}
