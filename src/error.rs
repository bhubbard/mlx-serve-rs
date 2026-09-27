use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ServerError {
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Internal engine error: {0}")]
    Internal(String),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match &self {
            ServerError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "invalid_request_error", msg.clone()),
            ServerError::ModelNotFound(msg) => (StatusCode::NOT_FOUND, "model_not_found", msg.clone()),
            ServerError::Mlx(e) => (StatusCode::INTERNAL_SERVER_ERROR, "mlx_error", e.to_string()),
            ServerError::Io(e) => (StatusCode::INTERNAL_SERVER_ERROR, "io_error", e.to_string()),
            ServerError::Json(e) => (StatusCode::BAD_REQUEST, "json_error", e.to_string()),
            ServerError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", msg.clone()),
        };

        let body = Json(json!({
            "error": {
                "message": message,
                "type": error_type,
                "param": null,
                "code": status.as_u16(),
            }
        }));

        (status, body).into_response()
    }
}

pub type Result<T> = std::result::Result<T, ServerError>;
