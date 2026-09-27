use mlx_serve_rs::engine::sampler::{Sampler, SamplingParams};

#[test]
fn test_greedy_sampler() {
    let logits = vec![0.1f32, 2.5, 0.4, 0.2];
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let token = Sampler::sample(&logits, &params, &[]);
    assert_eq!(token, 1);
}

#[test]
fn test_repetition_penalty() {
    let logits = vec![2.0f32, 2.8, 0.1, 0.1];
    let params = SamplingParams {
        temperature: 0.0,
        repetition_penalty: 2.0, // 2.8 / 2.0 = 1.4 < 2.0
        ..Default::default()
    };
    let token = Sampler::sample(&logits, &params, &[1]);
    assert_eq!(token, 0); // Token 0 wins after penalizing token 1
}
