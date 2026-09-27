use clap::Parser;
use mlx_serve_rs::api::create_router;
use mlx_serve_rs::engine::Scheduler;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::info;

#[derive(Parser, Debug)]
#[command(
    name = "mlx-serve",
    about = "High-performance OpenAI-compatible LLM inference server for Apple Silicon (MLX)",
    version
)]
struct Args {
    /// Host to bind server to
    #[arg(short = 'H', long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Default model identifier to load
    #[arg(short, long, default_value = "mlx-community/Llama-3.2-3B-Instruct-4bit")]
    model: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    info!("Initializing MLX engine for Apple Silicon...");

    let scheduler = Scheduler::new(args.model.clone())?;
    let state = Arc::new(Mutex::new(scheduler));

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = create_router(state)
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    info!("🚀 mlx-serve-rs running at http://{}", addr);
    info!("📡 OpenAI compatible endpoint: http://{}/v1/chat/completions", addr);
    info!("🦙 Ollama compatible endpoint: http://{}/api/chat", addr);
    info!("📊 Metrics available at http://{}/metrics", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
