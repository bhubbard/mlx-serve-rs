use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::{mpsc, Mutex};
use uuid::Uuid;

use crate::api::sse::{format_chat_chunk, format_done};
use crate::engine::sampler::SamplingParams;
use crate::engine::scheduler::Scheduler;
use crate::error::{Result, ServerError};
use crate::types::*;

pub type AppState = Arc<Mutex<Scheduler>>;

pub fn create_router(state: AppState) -> Router {
    Router::new()
        // OpenAI Endpoints
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        // Ollama Endpoints
        .route("/api/tags", get(ollama_tags))
        .route("/api/chat", post(ollama_chat))
        .route("/api/generate", post(ollama_generate))
        // Health & Metrics
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/metrics", get(metrics))
        .with_state(state)
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// GET /v1/models
async fn list_models(State(state): State<AppState>) -> Result<Json<ModelListResponse>> {
    let sched = state.lock().await;
    let model_names = sched.list_models();
    let cards = model_names
        .into_iter()
        .map(|id| ModelCard {
            id,
            object: "model".to_string(),
            created: current_timestamp(),
            owned_by: "mlx-serve".to_string(),
        })
        .collect();

    Ok(Json(ModelListResponse {
        object: "list".to_string(),
        data: cards,
    }))
}

// POST /v1/chat/completions
async fn chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Result<Response> {
    let created = current_timestamp();
    let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let max_tokens = payload.max_tokens.unwrap_or(128);

    let sampling = SamplingParams {
        temperature: payload.temperature,
        top_p: payload.top_p,
        repetition_penalty: payload.repetition_penalty,
        stop: payload.stop.unwrap_or_default(),
        ..Default::default()
    };

    // Flatten chat messages into conversational prompt
    let mut prompt = String::new();
    for msg in &payload.messages {
        let role_str = match msg.role {
            Role::System => "System",
            Role::User => "User",
            Role::Assistant => "Assistant",
            Role::Tool => "Tool",
        };
        prompt.push_str(&format!("{}: {}\n", role_str, msg.content));
    }
    prompt.push_str("Assistant: ");

    if payload.stream {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let state_clone = state.clone();
        let model_id = payload.model.clone();
        let chunk_id = id.clone();

        tokio::spawn(async move {
            let _ = Scheduler::generate_stream(
                state_clone,
                &model_id,
                &prompt,
                max_tokens,
                sampling,
                tx,
            ).await;
        });

        let stream = async_stream::stream! {
            // First chunk with role
            yield Ok::<Event, axum::Error>(format_chat_chunk(
                &chunk_id,
                &payload.model,
                created,
                "",
                Some(Role::Assistant),
                None,
            ));

            while let Some(token) = rx.recv().await {
                yield Ok::<Event, axum::Error>(format_chat_chunk(
                    &chunk_id,
                    &payload.model,
                    created,
                    &token,
                    None,
                    None,
                ));
            }

            // Final stop chunk
            yield Ok::<Event, axum::Error>(format_chat_chunk(
                &chunk_id,
                &payload.model,
                created,
                "",
                None,
                Some("stop".to_string()),
            ));

            // [DONE]
            yield Ok::<Event, axum::Error>(format_done());
        };

        Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        let (text, prompt_tokens, comp_tokens) = Scheduler::generate_full(
            state,
            &payload.model,
            &prompt,
            max_tokens,
            sampling,
        ).await?;

        let resp = ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model: payload.model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: Role::Assistant,
                    content: text,
                    name: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens: comp_tokens,
                total_tokens: prompt_tokens + comp_tokens,
            },
        };

        Ok(Json(resp).into_response())
    }
}

// POST /v1/completions
async fn completions(
    State(state): State<AppState>,
    Json(payload): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>> {
    let created = current_timestamp();
    let id = format!("cmpl-{}", Uuid::new_v4().simple());
    let max_tokens = payload.max_tokens.unwrap_or(64);

    let sampling = SamplingParams {
        temperature: payload.temperature,
        top_p: payload.top_p,
        stop: payload.stop.unwrap_or_default(),
        ..Default::default()
    };

    let (text, prompt_tokens, comp_tokens) = Scheduler::generate_full(
        state,
        &payload.model,
        &payload.prompt,
        max_tokens,
        sampling,
    ).await?;

    Ok(Json(CompletionResponse {
        id,
        object: "text_completion".to_string(),
        created,
        model: payload.model,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: Some("stop".to_string()),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: comp_tokens,
            total_tokens: prompt_tokens + comp_tokens,
        },
    }))
}

// POST /v1/embeddings
async fn embeddings(
    State(state): State<AppState>,
    Json(payload): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>> {
    let inputs: Vec<String> = match payload.input {
        serde_json::Value::String(s) => vec![s],
        serde_json::Value::Array(arr) => arr
            .into_iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => return Err(ServerError::BadRequest("Invalid embedding input".to_string())),
    };

    let embs = Scheduler::embed(state, &payload.model, &inputs).await?;
    let data = embs
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingData {
            object: "embedding".to_string(),
            index,
            embedding,
        })
        .collect();

    Ok(Json(EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: payload.model,
        usage: Usage {
            prompt_tokens: inputs.len() * 4,
            completion_tokens: 0,
            total_tokens: inputs.len() * 4,
        },
    }))
}

// GET /api/tags (Ollama compatible)
async fn ollama_tags(State(state): State<AppState>) -> Result<Json<OllamaTagsResponse>> {
    let sched = state.lock().await;
    let models = sched
        .list_models()
        .into_iter()
        .map(|name| OllamaModelTag {
            name,
            modified_at: chrono::Utc::now().to_rfc3339(),
            size: 4096000000,
        })
        .collect();

    Ok(Json(OllamaTagsResponse { models }))
}

// POST /api/chat (Ollama compatible)
async fn ollama_chat(
    State(state): State<AppState>,
    Json(payload): Json<OllamaChatRequest>,
) -> Result<Json<serde_json::Value>> {
    let mut prompt = String::new();
    for msg in &payload.messages {
        prompt.push_str(&format!("{}: {}\n", serde_json::to_string(&msg.role).unwrap_or_default(), msg.content));
    }
    prompt.push_str("assistant: ");

    let (text, _, _) = Scheduler::generate_full(
        state,
        &payload.model,
        &prompt,
        128,
        SamplingParams::default(),
    ).await?;

    Ok(Json(serde_json::json!({
        "model": payload.model,
        "message": {
            "role": "assistant",
            "content": text
        },
        "done": true
    })))
}

// POST /api/generate (Ollama compatible)
async fn ollama_generate(
    State(state): State<AppState>,
    Json(payload): Json<OllamaGenerateRequest>,
) -> Result<Json<serde_json::Value>> {
    let (text, _, _) = Scheduler::generate_full(
        state,
        &payload.model,
        &payload.prompt,
        128,
        SamplingParams::default(),
    ).await?;

    Ok(Json(serde_json::json!({
        "model": payload.model,
        "response": text,
        "done": true
    })))
}

// GET /health
async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "engine": "mlx-serve-rs",
        "device": "apple_silicon",
        "timestamp": current_timestamp(),
    }))
}

// GET /metrics
async fn metrics(State(state): State<AppState>) -> Json<serde_json::Value> {
    let sched = state.lock().await;
    Json(serde_json::json!({
        "active_requests": sched.metrics.active_requests.load(Ordering::Relaxed),
        "total_requests": sched.metrics.total_requests.load(Ordering::Relaxed),
        "total_generated_tokens": sched.metrics.total_generated_tokens.load(Ordering::Relaxed),
        "total_prompt_tokens": sched.metrics.total_prompt_tokens.load(Ordering::Relaxed),
    }))
}
