// Sample API request payloads
const PAYLOADS = {
  chat: JSON.stringify({
    model: "mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages: [
      { role: "system", content: "You are a fast, concise Apple Silicon AI assistant." },
      { role: "user", content: "Explain how unified memory helps LLM inference speed." }
    ],
    temperature: 0.7,
    stream: true
  }, null, 2),

  completions: JSON.stringify({
    model: "mlx-community/Llama-3.2-3B-Instruct-4bit",
    prompt: "Rust with Apple MLX delivers peak tokens per second because",
    max_tokens: 32,
    temperature: 0.2
  }, null, 2),

  embeddings: JSON.stringify({
    model: "mlx-community/Llama-3.2-3B-Instruct-4bit",
    input: [
      "Apple Silicon unified memory architecture",
      "Zero-copy tensor dispatch across CPU and GPU cores"
    ]
  }, null, 2),

  ollama: JSON.stringify({
    model: "mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages: [
      { role: "user", content: "List three benefits of local model hosting." }
    ]
  }, null, 2)
};

const CLIENT_CODE = {
  curl: `# Start the server on port 8080
mlx-serve --port 8080 --model mlx-community/Llama-3.2-3B-Instruct-4bit

# Query chat completions via cURL
curl http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Explain quantum computing in one sentence."}],
    "stream": true
  }'`,

  python: `from openai import OpenAI

# Connect directly to local mlx-serve-rs
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

stream = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Write a fast fibonacci in Rust"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)`,

  rust: `use reqwest::Client;
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = Client::new();
    let res = client.post("http://localhost:8080/v1/chat/completions")
        .json(&json!({
            "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "messages": [{"role": "user", "content": "Hello MLX!"}]
        }))
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;

    println!("Response: {}", res["choices"][0]["message"]["content"]);
    Ok(())
}`
};

document.addEventListener("DOMContentLoaded", () => {
  initPlayground();
  initClientTabs();
});

function initPlayground() {
  const endpointSelect = document.getElementById("endpoint-select");
  const jsonInput = document.getElementById("json-input");
  const tempSlider = document.getElementById("temp-slider");
  const tempVal = document.getElementById("temp-val");
  const sendBtn = document.getElementById("send-request-btn");
  const streamOutput = document.getElementById("stream-output");
  const statusBadge = document.getElementById("response-status");

  tempSlider.addEventListener("input", e => {
    tempVal.textContent = e.target.value;
  });

  function updatePayload(key) {
    jsonInput.value = PAYLOADS[key] || PAYLOADS.chat;
  }

  endpointSelect.addEventListener("change", e => updatePayload(e.target.value));
  updatePayload("chat");

  let streamTimer = null;

  sendBtn.addEventListener("click", () => {
    if (streamTimer) clearInterval(streamTimer);

    const endpoint = endpointSelect.value;
    statusBadge.textContent = "200 OK (Streaming)";
    streamOutput.textContent = "";
    sendBtn.disabled = true;
    sendBtn.textContent = "Connecting to SSE Stream...";

    if (endpoint === "embeddings") {
      setTimeout(() => {
        streamOutput.textContent = JSON.stringify({
          object: "list",
          data: [
            { object: "embedding", index: 0, embedding: [-0.042, 0.081, -0.012, 0.104, "... (128 dims)"] },
            { object: "embedding", index: 1, embedding: [0.031, -0.054, 0.092, -0.007, "... (128 dims)"] }
          ],
          model: "mlx-community/Llama-3.2-3B-Instruct-4bit",
          usage: { prompt_tokens: 8, total_tokens: 8 }
        }, null, 2);
        statusBadge.textContent = "200 OK";
        sendBtn.disabled = false;
        sendBtn.textContent = "Send API Request";
      }, 350);
      return;
    }

    const sampleChunks = [
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":"Unified"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" memory"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" architecture"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" in"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" Apple"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" Silicon"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" eliminates"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" PCIe"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" bus"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" bottlenecks,"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" allowing"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" CPU"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" and"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" GPU"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" cores"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" zero-copy"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" tensor"}}]}\n\n',
      'data: {"id":"chatcmpl-a1b2","object":"chat.completion.chunk","choices":[{"delta":{"content":" access."}}]}\n\n',
      'data: [DONE]\n\n'
    ];

    let index = 0;
    streamTimer = setInterval(() => {
      if (index < sampleChunks.length) {
        streamOutput.textContent += sampleChunks[index];
        streamOutput.scrollTop = streamOutput.scrollHeight;
        index++;
      } else {
        clearInterval(streamTimer);
        sendBtn.disabled = false;
        sendBtn.textContent = "Send API Request";
        statusBadge.textContent = "Complete (19 tokens @ 162 tok/s)";
      }
    }, 45);
  });
}

function initClientTabs() {
  const tabs = document.querySelectorAll(".client-tab");
  const codeBox = document.getElementById("client-snippet");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      const key = tab.dataset.tab;
      codeBox.innerHTML = `<code>${CLIENT_CODE[key]}</code>`;
    });
  });
}
