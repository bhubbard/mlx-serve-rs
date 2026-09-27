use axum::response::sse::Event;
use crate::types::{ChatCompletionChunk, ChatCompletionChunkChoice, DeltaMessage, Role};

pub fn format_chat_chunk(
    id: &str,
    model: &str,
    created: u64,
    delta: &str,
    role: Option<Role>,
    finish_reason: Option<String>,
) -> Event {
    let chunk = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: DeltaMessage {
                role,
                content: if delta.is_empty() { None } else { Some(delta.to_string()) },
            },
            finish_reason,
        }],
    };

    let data = serde_json::to_string(&chunk).unwrap_or_default();
    Event::default().data(data)
}

pub fn format_done() -> Event {
    Event::default().data("[DONE]")
}
