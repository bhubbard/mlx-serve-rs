use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use mlx_serve_rs::api::create_router;
use mlx_serve_rs::engine::Scheduler;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower::ServiceExt;

fn setup_test_app() -> axum::Router {
    let scheduler = Scheduler::new("default-test-model".to_string()).unwrap();
    let state = Arc::new(Mutex::new(scheduler));
    create_router(state)
}

#[tokio::test]
async fn test_health() -> anyhow::Result<()> {
    let app = setup_test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())?,
        )
        .await?;

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await?;
    let body: Value = serde_json::from_slice(&bytes)?;
    assert_eq!(body["status"], "healthy");
    assert_eq!(body["engine"], "mlx-serve-rs");
    Ok(())
}

#[tokio::test]
async fn test_list_models() -> anyhow::Result<()> {
    let app = setup_test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())?,
        )
        .await?;

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await?;
    let body: Value = serde_json::from_slice(&bytes)?;
    assert_eq!(body["object"], "list");
    let models = body["data"].as_array().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["id"], "default-test-model");
    Ok(())
}

#[tokio::test]
async fn test_chat_completions_json() -> anyhow::Result<()> {
    let app = setup_test_app();
    let payload = json!({
        "model": "default-test-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": false
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&payload)?))?,
        )
        .await?;

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await?;
    let body: Value = serde_json::from_slice(&bytes)?;
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert!(body["usage"]["total_tokens"].as_u64().unwrap() > 0);
    Ok(())
}

#[tokio::test]
async fn test_chat_completions_stream_sse() -> anyhow::Result<()> {
    let app = setup_test_app();
    let payload = json!({
        "model": "default-test-model",
        "messages": [
            {"role": "user", "content": "Hi"}
        ],
        "max_tokens": 4,
        "stream": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&payload)?))?,
        )
        .await?;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );

    let bytes = to_bytes(response.into_body(), 1024 * 1024).await?;
    let text = String::from_utf8(bytes.to_vec())?;
    assert!(text.contains("data: "));
    assert!(text.contains("[DONE]"));
    Ok(())
}

#[tokio::test]
async fn test_text_completions() -> anyhow::Result<()> {
    let app = setup_test_app();
    let payload = json!({
        "model": "default-test-model",
        "prompt": "Once upon a time",
        "max_tokens": 4
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&payload)?))?,
        )
        .await?;

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await?;
    let body: Value = serde_json::from_slice(&bytes)?;
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["choices"].as_array().unwrap().len(), 1);
    Ok(())
}

#[tokio::test]
async fn test_embeddings_api() -> anyhow::Result<()> {
    let app = setup_test_app();
    let payload = json!({
        "model": "default-test-model",
        "input": ["First document", "Second document"]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&payload)?))?,
        )
        .await?;

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await?;
    let body: Value = serde_json::from_slice(&bytes)?;
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["embedding"].as_array().unwrap().len(), 128);
    Ok(())
}

#[tokio::test]
async fn test_ollama_compatibility() -> anyhow::Result<()> {
    let app = setup_test_app();

    // 1. GET /api/tags
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/tags")
                .body(Body::empty())?,
        )
        .await?;
    assert_eq!(response.status(), StatusCode::OK);

    // 2. POST /api/generate
    let payload = json!({
        "model": "default-test-model",
        "prompt": "Hello Ollama"
    });

    let gen_resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&payload)?))?,
        )
        .await?;
    assert_eq!(gen_resp.status(), StatusCode::OK);
    let bytes = to_bytes(gen_resp.into_body(), 1024 * 1024).await?;
    let body: Value = serde_json::from_slice(&bytes)?;
    assert_eq!(body["done"], true);
    Ok(())
}

#[tokio::test]
async fn test_metrics() -> anyhow::Result<()> {
    let app = setup_test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())?,
        )
        .await?;

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await?;
    let body: Value = serde_json::from_slice(&bytes)?;
    assert!(body.get("total_requests").is_some());
    assert!(body.get("active_requests").is_some());
    Ok(())
}
