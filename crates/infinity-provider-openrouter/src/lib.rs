//! OpenRouter provider and shared OpenAI-compatible streaming adapter.
//!
//! [`run`] serves one backend using Infinity's native Unix-socket transport.
//! OpenRouter and Groq have separate binaries, credentials, model catalogs,
//! request options, and model metadata; the stream decoder is shared.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use infinity_provider_protocol::{remote::serve_provider, *};
use serde_json::{Value, json};
use std::{collections::BTreeMap, io::Write, sync::Arc, time::Duration};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

struct ChatProvider {
    client: reqwest::Client,
    key: String,
    models: Vec<ModelEntry>,
    idle: Duration,
    completion_url: String,
    backend: Backend,
}

fn fail(class: ErrorClass, message: impl Into<String>) -> CompletionError {
    CompletionError::provider(class, message)
}

fn api_error(status: u16, value: &Value) -> CompletionError {
    let error = value.get("error").unwrap_or(value);
    let code = error.get("code").and_then(Value::as_str).unwrap_or("");
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("Model API request failed");
    let lower = message.to_lowercase();
    let class = if code == "context_length_exceeded"
        || lower.contains("context length")
        || lower.contains("context window")
    {
        ErrorClass::ContextOverflow
    } else if status == 429 {
        ErrorClass::Throttled
    } else if status == 408 || status >= 500 {
        ErrorClass::Transient
    } else {
        ErrorClass::Fatal
    };
    // Avoid logging backend request bodies, which may contain private prompts.
    fail(
        class,
        format!("Model API error (status {status}, code {code})"),
    )
}

fn image(image: &Image) -> Result<Value, CompletionError> {
    let url = match &image.data {
        ImageSource::Url(url) => url.clone(),
        ImageSource::Base64(data) => {
            let mime = image
                .media_type
                .as_ref()
                .ok_or_else(|| fail(ErrorClass::Fatal, "Base64 image requires media_type"))?;
            format!("data:{};base64,{data}", mime.to_mime_type())
        }
    };
    Ok(json!({"type":"image_url", "image_url":{"url":url}}))
}

fn request_body(model: &ModelEntry, request: CompletionRequest) -> Result<Value, CompletionError> {
    let mut messages = Vec::new();
    if let Some(preamble) = request.preamble {
        messages.push(json!({"role":"system","content":preamble}));
    }
    for message in request.chat_history {
        match message {
            Message::User { content } => {
                let mut parts = Vec::new();
                for item in content {
                    match item {
                        UserContent::Text(text) => {
                            parts.push(json!({"type":"text","text":text.text}))
                        }
                        UserContent::Image(img) => parts.push(image(&img)?),
                        UserContent::ToolResult(result) => {
                            if !parts.is_empty() {
                                messages.push(
                                    json!({"role":"user","content":std::mem::take(&mut parts)}),
                                );
                            }
                            let mut tool_parts = Vec::new();
                            for item in result.content {
                                tool_parts.push(match item {
                                    ToolResultContent::Text(text) => {
                                        json!({"type":"text","text":text.text})
                                    }
                                    ToolResultContent::Image(img) => image(&img)?,
                                });
                            }
                            // OpenRouter supports multimodal tool results on compatible models.
                            let content = if tool_parts.iter().all(|p| p["type"] == "text") {
                                Value::String(
                                    tool_parts
                                        .iter()
                                        .filter_map(|p| p["text"].as_str())
                                        .collect::<Vec<_>>()
                                        .join("\n"),
                                )
                            } else {
                                Value::Array(tool_parts)
                            };
                            messages.push(json!({"role":"tool","tool_call_id":result.call_id.unwrap_or(result.id),"content":content}));
                        }
                    }
                }
                if !parts.is_empty() {
                    messages.push(json!({"role":"user","content":parts}));
                }
            }
            Message::Assistant { content } => {
                let mut text = String::new();
                let mut calls = Vec::new();
                let mut details = Vec::new();
                let mut reasoning = String::new();
                for item in content {
                    match item {
                        AssistantContent::Text(t) => text.push_str(&t.text),
                        AssistantContent::ToolCall(t) => calls.push(json!({"id":t.call_id.unwrap_or(t.id),"type":"function","function":{"name":t.function.name,"arguments":t.function.arguments.to_string()}})),
                        AssistantContent::Reasoning(r) => {
                            reasoning.push_str(&r.display_text());
                            // Preserve OpenRouter reasoning_details, including opaque signatures.
                            for block in r.content {
                                if let ReasoningContent::Encrypted(raw) = block {
                                    let value: Value = serde_json::from_str(&raw)?;
                                    if let Value::Array(values) = value { details.extend(values); }
                                }
                            }
                        }
                    }
                }
                let mut msg = json!({"role":"assistant","content": if text.is_empty() { Value::Null } else { Value::String(text) }});
                if !calls.is_empty() {
                    msg["tool_calls"] = json!(calls);
                }
                if !details.is_empty() {
                    msg["reasoning_details"] = json!(details);
                } else if !reasoning.is_empty() {
                    msg["reasoning"] = json!(reasoning);
                }
                messages.push(msg);
            }
        }
    }
    let max = request
        .max_tokens
        .or(model.max_output_tokens)
        .unwrap_or(8192);
    let max = model.max_output_tokens.map_or(max, |limit| max.min(limit));
    let mut body = json!({"model":model.model_id,"messages":messages,"stream":true,"stream_options":{"include_usage":true},"max_tokens":max});
    if !request.tools.is_empty() {
        body["tools"] = Value::Array(request.tools.into_iter().map(|t| json!({"type":"function","function":{"name":t.name,"description":t.description,"parameters":t.parameters}})).collect());
    }
    if let Some(extra) = request.additional_params {
        let extra = extra
            .as_object()
            .ok_or_else(|| fail(ErrorClass::Fatal, "additional_params must be an object"))?;
        for (key, value) in extra {
            if [
                "model",
                "messages",
                "stream",
                "stream_options",
                "tools",
                "max_tokens",
                "max_completion_tokens",
                "n",
            ]
            .contains(&key.as_str())
            {
                return Err(fail(
                    ErrorClass::Fatal,
                    format!("Reserved additional parameter: {key}"),
                ));
            }
            body[key] = value.clone();
        }
    }
    Ok(body)
}

#[derive(Default)]
struct Call {
    id: String,
    name: String,
    arguments: String,
}
#[derive(Default)]
struct Decoder {
    calls: BTreeMap<u64, Call>,
    reasoning: String,
    details: BTreeMap<u64, Value>,
    usage: Option<Usage>,
    finished: bool,
}
impl Decoder {
    fn chunk(&mut self, value: Value) -> Result<Vec<StreamChunk>, CompletionError> {
        if value.get("error").is_some() {
            let code = value["error"]["code"].as_u64().unwrap_or(500) as u16;
            return Err(api_error(code, &value));
        }
        if let Some(u) = value.get("usage").filter(|v| v.is_object()).or_else(|| {
            value
                .get("x_groq")
                .and_then(|x| x.get("usage"))
                .filter(|v| v.is_object())
        }) {
            self.usage = Some(Usage {
                input_tokens: u["prompt_tokens"].as_u64().unwrap_or(0),
                output_tokens: u["completion_tokens"].as_u64().unwrap_or(0),
                total_tokens: u["total_tokens"].as_u64().unwrap_or(0),
                cached_input_tokens: u["prompt_tokens_details"]["cached_tokens"]
                    .as_u64()
                    .unwrap_or(0),
            });
        }
        let mut out = Vec::new();
        let Some(choice) = value["choices"].as_array().and_then(|c| c.first()) else {
            return Ok(out);
        };
        if let Some(reason) = choice["finish_reason"].as_str() {
            if reason == "error" {
                return Err(fail(
                    ErrorClass::Transient,
                    "Model API stream ended with error",
                ));
            }
            self.finished = true;
        }
        let delta = &choice["delta"];
        if let Some(text) = delta["content"].as_str().filter(|s| !s.is_empty()) {
            out.push(StreamChunk::Text(text.into()));
        }
        if let Some(text) = delta["reasoning"].as_str() {
            self.reasoning.push_str(text);
            out.push(StreamChunk::ReasoningDelta {
                id: None,
                text: text.into(),
            });
        }
        if let Some(details) = delta["reasoning_details"].as_array() {
            for detail in details {
                let index = detail["index"].as_u64().unwrap_or(0);
                let current = self.details.entry(index).or_insert_with(|| json!({}));
                if let Some(fields) = detail.as_object() {
                    for (key, value) in fields {
                        if ["text", "data", "summary", "signature"].contains(&key.as_str())
                            && value.is_string()
                        {
                            let mut text = current[key].as_str().unwrap_or("").to_owned();
                            text.push_str(value.as_str().expect("string checked"));
                            current[key] = json!(text);
                        } else {
                            current[key] = value.clone();
                        }
                    }
                }
            }
        }
        if let Some(calls) = delta["tool_calls"].as_array() {
            for delta in calls {
                let index = delta["index"]
                    .as_u64()
                    .ok_or_else(|| fail(ErrorClass::Transient, "Tool delta missing index"))?;
                let call = self.calls.entry(index).or_default();
                if let Some(id) = delta["id"].as_str() {
                    call.id = id.into();
                }
                if let Some(name) = delta["function"]["name"].as_str() {
                    call.name.push_str(name);
                }
                if let Some(args) = delta["function"]["arguments"].as_str() {
                    call.arguments.push_str(args);
                }
            }
        }
        Ok(out)
    }
    fn finish(self) -> Result<Vec<StreamChunk>, CompletionError> {
        if !self.finished {
            return Err(fail(
                ErrorClass::Transient,
                "Stream ended without finish_reason",
            ));
        }
        let mut out = Vec::new();
        if !self.reasoning.is_empty() || !self.details.is_empty() {
            let mut content = Vec::new();
            if !self.reasoning.is_empty() {
                content.push(ReasoningContent::Text {
                    text: self.reasoning,
                    signature: None,
                });
            }
            if !self.details.is_empty() {
                content.push(ReasoningContent::Encrypted(serde_json::to_string(
                    &self.details.into_values().collect::<Vec<_>>(),
                )?));
            }
            out.push(StreamChunk::Reasoning(Reasoning { id: None, content }));
        }
        for call in self.calls.into_values() {
            if call.id.is_empty() || call.name.is_empty() {
                return Err(fail(ErrorClass::Transient, "Incomplete tool call"));
            }
            let args = if call.arguments.is_empty() {
                json!({})
            } else {
                serde_json::from_str(&call.arguments)
                    .map_err(|_| fail(ErrorClass::Transient, "Invalid tool argument JSON"))?
            };
            out.push(StreamChunk::ToolCall(ToolCall::new(
                call.id, call.name, args,
            )));
        }
        out.push(StreamChunk::Final(FinalResponse { usage: self.usage }));
        Ok(out)
    }
}

#[async_trait]
impl ModelProvider for ChatProvider {
    async fn list_models(&self) -> Result<Vec<ModelEntry>, BoxError> {
        Ok(self.models.clone())
    }
    async fn invoke_model(
        &self,
        id: &str,
        request: CompletionRequest,
    ) -> Result<ModelStream, CompletionError> {
        let model = self
            .models
            .iter()
            .find(|m| m.model_id == id)
            .ok_or_else(|| fail(ErrorClass::Fatal, "Unknown model id"))?;
        let body = backend_request(self.backend, model, request)?;
        let response = tokio::time::timeout(
            self.idle,
            self.client
                .post(&self.completion_url)
                .bearer_auth(&self.key)
                .header("X-Title", "Infinity Code")
                .json(&body)
                .send(),
        )
        .await
        .map_err(|_| fail(ErrorClass::Transient, "Model API response timeout"))?
        .map_err(|_| fail(ErrorClass::Transient, "Model API connection failed"))?;
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let value = tokio::time::timeout(self.idle, response.json::<Value>())
                .await
                .map_err(|_| fail(ErrorClass::Transient, "Error response timeout"))?
                .map_err(|_| fail(ErrorClass::Transient, "Invalid error response"))?;
            return Err(api_error(status, &value));
        }
        let idle = self.idle;
        Ok(Box::pin(async_stream::try_stream! {
            let mut events = response.bytes_stream().eventsource();
            let mut decoder = Decoder::default();
            loop {
                let event = tokio::time::timeout(idle,events.next()).await
                    .map_err(|_| fail(ErrorClass::Transient,"Model API stream idle timeout"))?
                    .ok_or_else(|| fail(ErrorClass::Transient,"Model API stream disconnected before [DONE]"))?
                    .map_err(|_| fail(ErrorClass::Transient,"Model API SSE transport error"))?;
                if event.data.trim() == "[DONE]" {
                    for chunk in decoder.finish()? { yield chunk; }
                    break;
                }
                let value = serde_json::from_str(&event.data).map_err(|_| fail(ErrorClass::Transient,"Invalid Model API stream JSON"))?;
                for chunk in decoder.chunk(value)? { yield chunk; }
            }
        }))
    }
}

/// API and environment namespace used by one provider process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    OpenRouter,
    Groq,
}
impl Backend {
    fn prefix(self) -> &'static str {
        match self {
            Self::OpenRouter => "OPENROUTER",
            Self::Groq => "GROQ",
        }
    }
    fn base(self) -> &'static str {
        match self {
            Self::OpenRouter => "https://openrouter.ai/api/v1",
            Self::Groq => "https://api.groq.com/openai/v1",
        }
    }
}

fn backend_request(
    backend: Backend,
    model: &ModelEntry,
    request: CompletionRequest,
) -> Result<Value, CompletionError> {
    let mut body = request_body(model, request)?;
    if backend == Backend::Groq {
        let object = body
            .as_object_mut()
            .expect("bug: request must be an object");
        let max = object
            .remove("max_tokens")
            .expect("bug: request has max_tokens");
        object.insert("max_completion_tokens".into(), max);
        // Groq does not accept OpenRouter's provider/routing options.
        for key in [
            "provider",
            "models",
            "route",
            "transforms",
            "plugins",
            "reasoning",
        ] {
            if object.contains_key(key) {
                return Err(fail(
                    ErrorClass::Fatal,
                    format!("{key} is an OpenRouter option; use Groq-native parameters"),
                ));
            }
        }
        for key in ["logprobs", "logit_bias", "top_logprobs"] {
            if object.contains_key(key) {
                return Err(fail(
                    ErrorClass::Fatal,
                    format!("Groq does not support {key}"),
                ));
            }
        }
        if model.model_id.starts_with("openai/gpt-oss-") {
            if object.contains_key("reasoning_format") {
                return Err(fail(
                    ErrorClass::Fatal,
                    "Groq GPT-OSS uses include_reasoning, not reasoning_format",
                ));
            }
            object.entry("parallel_tool_calls").or_insert(json!(false));
        }
        if let Some(messages) = object.get_mut("messages").and_then(Value::as_array_mut) {
            for message in messages {
                if let Some(message) = message.as_object_mut() {
                    message.remove("reasoning_details");
                    message.remove("reasoning");
                    if message.get("role").and_then(Value::as_str) == Some("user")
                        && let Some(parts) = message.get("content").and_then(Value::as_array)
                    {
                        if parts.iter().any(|p| p["type"] != "text") {
                            return Err(fail(
                                ErrorClass::Fatal,
                                "This Groq adapter currently supports text input only",
                            ));
                        }
                        let text = parts
                            .iter()
                            .filter_map(|p| p["text"].as_str())
                            .collect::<Vec<_>>()
                            .join("\n");
                        message.insert("content".into(), json!(text));
                    }
                    if message.get("role").and_then(Value::as_str) == Some("tool")
                        && !message.get("content").is_some_and(Value::is_string)
                    {
                        return Err(fail(ErrorClass::Fatal, "Groq tool results must be text"));
                    }
                }
            }
        }
    }
    Ok(body)
}

// Reviewed against https://console.groq.com/docs/models and /docs/tool-use/overview.
// Live /models supplies availability and context_window, but not tool capability.
fn groq_output_limit(id: &str) -> Option<u64> {
    match id {
        "openai/gpt-oss-20b" | "openai/gpt-oss-120b" | "openai/gpt-oss-safeguard-20b" => {
            Some(65_536)
        }
        "llama-3.3-70b-versatile" => Some(32_768),
        "llama-3.1-8b-instant" | "meta-models/muse-spark-open" | "minimaxai/minimax-m2.7" => {
            Some(131_072)
        }
        "qwen/qwen3.8-27b" => Some(16_384),
        _ => None,
    }
}

fn model_entry(backend: Backend, model: &Value, id: &str) -> Result<ModelEntry, BoxError> {
    let (context, max, image) = match backend {
        Backend::OpenRouter => {
            if !model["supported_parameters"]
                .as_array()
                .is_some_and(|p| p.iter().any(|p| p == "tools"))
            {
                return Err(format!("Model does not advertise tool support: {id}").into());
            }
            (
                model["context_length"].as_u64(),
                model["top_provider"]["max_completion_tokens"].as_u64(),
                model["architecture"]["input_modalities"]
                    .as_array()
                    .is_some_and(|a| a.iter().any(|m| m == "image")),
            )
        }
        Backend::Groq => {
            if model["active"] == false {
                return Err(format!("Inactive Groq model: {id}").into());
            }
            let limit = groq_output_limit(id).ok_or_else(|| {
                format!("Groq model not in this adapter's reviewed tool-capable list: {id}")
            })?;
            (
                model["context_window"].as_u64(),
                Some(
                    model["max_completion_tokens"]
                        .as_u64()
                        .filter(|n| *n > 0)
                        .unwrap_or(limit)
                        .min(limit),
                ),
                false,
            )
        }
    };
    Ok(ModelEntry {
        model_id: id.into(),
        display_name: model["name"].as_str().unwrap_or(id).into(),
        context_window: usize::try_from(
            context
                .filter(|n| *n > 0)
                .ok_or("Missing positive context window")?,
        )?,
        max_output_tokens: max.filter(|n| *n > 0),
        supports_image_input: image,
    })
}

fn select_models(
    backend: Backend,
    catalog: &Value,
    selected: &[String],
) -> Result<(Vec<ModelEntry>, Vec<String>), BoxError> {
    let data = catalog["data"]
        .as_array()
        .ok_or("Model catalog has no data array")?;
    let mut models = Vec::new();
    let mut warnings = Vec::new();
    for id in selected {
        if models.iter().any(|m: &ModelEntry| &m.model_id == id) {
            continue;
        }
        match data.iter().find(|m| m["id"].as_str() == Some(id)) {
            None => warnings.push(format!(
                "Model not found in {} catalog: {id}",
                backend.prefix()
            )),
            Some(model) => match model_entry(backend, model, id) {
                Ok(entry) => models.push(entry),
                Err(e) => warnings.push(e.to_string()),
            },
        }
    }
    if models.is_empty() {
        return Err(format!(
            "No usable models in {}_MODELS. {}",
            backend.prefix(),
            warnings.join("; ")
        )
        .into());
    }
    Ok((models, warnings))
}

async fn load_models(
    client: &reqwest::Client,
    key: &str,
    backend: Backend,
    selected: &[String],
) -> Result<Vec<ModelEntry>, BoxError> {
    let response = client
        .get(format!("{}/models", backend.base()))
        .bearer_auth(key)
        .timeout(Duration::from_secs(30))
        .send()
        .await?;
    if !response.status().is_success() {
        return Err(format!(
            "{} catalog HTTP {} (check API key, permissions and network)",
            backend.prefix(),
            response.status()
        )
        .into());
    }
    let catalog: Value = response.json().await?;
    let (models, warnings) = select_models(backend, &catalog, selected)?;
    for warning in warnings {
        eprintln!("Warning: {warning}");
    }
    Ok(models)
}

/// Start a native Infinity provider. Credentials never cross between backends.
pub async fn run(backend: Backend) -> Result<(), BoxError> {
    let prefix = backend.prefix();
    let key_name = format!("{prefix}_API_KEY");
    let models_name = format!("{prefix}_MODELS");
    let key = std::env::var(&key_name)
        .map_err(|_| format!("Set {key_name} before starting the Infinity daemon"))?;
    if key.trim().is_empty() {
        return Err(format!("{key_name} is empty").into());
    }
    let selected: Vec<String> = std::env::var(&models_name)
        .map_err(|_| format!("Set {models_name} to comma-separated tool-capable model IDs"))?
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    if selected.is_empty() {
        return Err(format!("{models_name} is empty").into());
    }
    let seconds = std::env::var(format!("{prefix}_TIMEOUT_SECS"))
        .unwrap_or_else(|_| "120".into())
        .parse::<u64>()?;
    if seconds == 0 {
        return Err("Timeout must be greater than zero".into());
    }
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .build()?;
    let models = load_models(&client, &key, backend, &selected).await?;
    eprintln!("{} provider: loaded {} model(s)", prefix, models.len());
    let provider = Arc::new(ChatProvider {
        client,
        key,
        models,
        idle: Duration::from_secs(seconds),
        completion_url: format!("{}/chat/completions", backend.base()),
        backend,
    });
    let (path, server) = serve_provider(provider)?;
    println!("{}", path.display());
    std::io::stdout().flush()?;
    server.await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model() -> ModelEntry {
        ModelEntry {
            model_id: "test/model".into(),
            display_name: "Test".into(),
            context_window: 32000,
            max_output_tokens: Some(4096),
            supports_image_input: true,
        }
    }
    #[test]
    fn fragmented_parallel_tools_and_usage() {
        let mut d = Decoder::default();
        d.chunk(json!({"choices":[{"delta":{"tool_calls":[
            {"index":0,"id":"a","function":{"name":"one","arguments":"{\"x\":"}},
            {"index":1,"id":"b","function":{"name":"two","arguments":"{}"}}
        ]}}]}))
        .expect("first chunk");
        d.chunk(json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]})).expect("second chunk");
        d.chunk(json!({"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25,"prompt_tokens_details":{"cached_tokens":10}}})).expect("usage");
        let chunks = d.finish().expect("complete");
        assert!(
            matches!(&chunks[0],StreamChunk::ToolCall(t) if t.id == "a" && t.function.arguments == json!({"x":1}))
        );
        assert!(matches!(&chunks[1],StreamChunk::ToolCall(t) if t.id == "b"));
        assert!(
            matches!(&chunks[2],StreamChunk::Final(f) if f.usage.expect("usage").cached_input_tokens == 10)
        );
    }
    #[test]
    fn incomplete_stream_and_malformed_tools_fail() {
        assert_eq!(
            Decoder::default()
                .finish()
                .expect_err("must reject truncation")
                .class(),
            ErrorClass::Transient
        );
        let mut d = Decoder::default();
        d.chunk(json!({"choices":[{"finish_reason":"tool_calls","delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"x","arguments":"{"}}]}}]})).expect("chunk");
        assert_eq!(
            d.finish().expect_err("invalid args").class(),
            ErrorClass::Transient
        );
    }
    #[test]
    fn classifies_backend_errors() {
        for (status, error, class) in [
            (429, json!({}), ErrorClass::Throttled),
            (503, json!({}), ErrorClass::Transient),
            (401, json!({}), ErrorClass::Fatal),
            (402, json!({}), ErrorClass::Fatal),
            (
                400,
                json!({"code":"context_length_exceeded"}),
                ErrorClass::ContextOverflow,
            ),
        ] {
            assert_eq!(api_error(status, &error).class(), class);
        }
    }
    #[test]
    fn tool_history_and_token_cap() {
        let request = CompletionRequest {
            preamble: Some("system".into()),
            max_tokens: Some(9999),
            chat_history: vec![
                Message::assistant("hi"),
                Message::Assistant {
                    content: vec![AssistantContent::ToolCall(ToolCall::new(
                        "a",
                        "run",
                        json!({"x":1}),
                    ))],
                },
                Message::User {
                    content: vec![UserContent::ToolResult(ToolResult {
                        id: "a".into(),
                        call_id: None,
                        content: vec![ToolResultContent::text("ok")],
                    })],
                },
            ],
            ..Default::default()
        };
        let b = request_body(&model(), request).expect("request");
        assert_eq!(b["max_tokens"], 4096);
        assert_eq!(
            b["messages"][2]["tool_calls"][0]["function"]["arguments"],
            "{\"x\":1}"
        );
        assert_eq!(
            b["messages"][3],
            json!({"role":"tool","tool_call_id":"a","content":"ok"})
        );
    }
    #[test]
    fn protects_protocol_fields() {
        let request = CompletionRequest {
            additional_params: Some(json!({"stream":false})),
            ..Default::default()
        };
        assert_eq!(
            request_body(&model(), request)
                .expect_err("must reject override")
                .class(),
            ErrorClass::Fatal
        );
    }
    #[test]
    fn opaque_reasoning_round_trip() {
        let mut d = Decoder::default();
        d.chunk(json!({"choices":[{"delta":{"reasoning_details":[{"index":0,"type":"reasoning.encrypted","data":"abc","format":"test"}]}}]})).expect("reasoning");
        d.chunk(json!({"choices":[{"delta":{"reasoning_details":[{"index":0,"data":"def"}]},"finish_reason":"stop"}]})).expect("finish");
        let chunks = d.finish().expect("complete");
        let StreamChunk::Reasoning(r) = &chunks[0] else {
            panic!("reasoning required")
        };
        let b = request_body(
            &model(),
            CompletionRequest {
                chat_history: vec![Message::Assistant {
                    content: vec![AssistantContent::Reasoning(r.clone())],
                }],
                ..Default::default()
            },
        )
        .expect("request");
        assert_eq!(b["messages"][0]["reasoning_details"][0]["data"], "abcdef");
    }
    #[test]
    fn base64_image_needs_mime() {
        let img = Image {
            data: ImageSource::Base64("abc".into()),
            media_type: None,
        };
        assert!(image(&img).is_err());
        assert_eq!(
            image(&Image {
                media_type: Some(ImageMediaType::PNG),
                ..img
            })
            .expect("image")["image_url"]["url"],
            "data:image/png;base64,abc"
        );
    }
    #[tokio::test]
    async fn sse_handles_fragmented_bytes_comments_and_usage() {
        let wire = ": keepalive\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\r\n\r\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"total_tokens\":2}}\n\ndata: [DONE]\n\n";
        let fragments: Vec<Result<Vec<u8>, std::io::Error>> =
            wire.as_bytes().chunks(3).map(|b| Ok(b.to_vec())).collect();
        let mut events = futures_util::stream::iter(fragments).eventsource();
        let mut d = Decoder::default();
        let mut output = Vec::new();
        while let Some(event) = events.next().await {
            let event = event.expect("SSE event");
            if event.data == "[DONE]" {
                output.extend(d.finish().expect("finish"));
                break;
            }
            output.extend(
                d.chunk(serde_json::from_str(&event.data).expect("JSON"))
                    .expect("decode"),
            );
        }
        assert!(matches!(&output[0],StreamChunk::Text(t) if t == "hi"));
        assert!(
            matches!(&output[1],StreamChunk::Final(f) if f.usage.expect("usage").total_tokens == 2)
        );
    }
    fn provider(url: String) -> ChatProvider {
        ChatProvider {
            client: reqwest::Client::builder()
                .no_proxy()
                .build()
                .expect("client"),
            key: "test-key".into(),
            models: vec![model()],
            idle: Duration::from_secs(5),
            completion_url: url,
            backend: Backend::OpenRouter,
        }
    }
    #[tokio::test]
    async fn unix_transport_lists_models_and_rejects_unknown_model() {
        let (path, server) =
            serve_provider(Arc::new(provider("http://127.0.0.1:1".into()))).expect("serve");
        let task = tokio::spawn(server);
        let remote = remote::RemoteModelProvider::new(path.clone());
        let models = remote.list_models().await.expect("list over Unix socket");
        assert_eq!(models[0].model_id, "test/model");
        match remote
            .invoke_model("missing", CompletionRequest::default())
            .await
        {
            Err(e) => assert_eq!(e.class(), ErrorClass::Fatal),
            Ok(_) => panic!("Unknown model must fail"),
        }
        task.abort();
        std::fs::remove_file(path).expect("remove test socket");
    }
    async fn check_http_backend(backend: Backend) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("listen");
        let url = format!(
            "http://{}/chat/completions",
            listener.local_addr().expect("address")
        );
        let server = tokio::task::spawn_blocking(move || {
            let (mut socket, _) = listener.accept().expect("accept");
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("timeout");
            let mut data = Vec::new();
            let mut buf = [0; 4096];
            loop {
                let n = socket.read(&mut buf).expect("request bytes");
                assert!(n > 0, "request incomplete");
                data.extend_from_slice(&buf[..n]);
                if let Some(end) = data.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&data[..end]).to_lowercase();
                    let len: usize = headers
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .expect("length")
                        .trim()
                        .parse()
                        .expect("number");
                    if data.len() < end + 4 + len {
                        continue;
                    }
                    assert!(headers.contains("authorization: bearer test-key"));
                    let body: Value = serde_json::from_slice(&data[end + 4..end + 4 + len])
                        .expect("request JSON");
                    assert_eq!(body["model"], "test/model");
                    if backend == Backend::Groq {
                        assert_eq!(body["messages"][0]["content"], "Hello");
                        assert_eq!(body["max_completion_tokens"], 4096);
                        assert!(body.get("max_tokens").is_none());
                    } else {
                        assert_eq!(body["messages"][0]["content"][0]["text"], "Hello");
                        assert_eq!(body["max_tokens"], 4096);
                    }
                    assert_eq!(body["stream"], true);
                    break;
                }
            }
            let body = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello!\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"total_tokens\":4}}\n\ndata: [DONE]\n\n";
            write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",body.len(),body).expect("response");
        });
        let mut p = provider(url);
        p.backend = backend;
        let mut stream = p
            .invoke_model(
                "test/model",
                CompletionRequest {
                    chat_history: vec![Message::user("Hello")],
                    ..Default::default()
                },
            )
            .await
            .expect("invoke");
        assert!(
            matches!(stream.next().await.expect("text").expect("ok"),StreamChunk::Text(t) if t == "Hello!")
        );
        assert!(
            matches!(stream.next().await.expect("final").expect("ok"),StreamChunk::Final(f) if f.usage.expect("usage").total_tokens == 4)
        );
        assert!(stream.next().await.is_none());
        server.await.expect("mock server");
    }
    #[tokio::test]
    async fn openrouter_http_end_to_end() {
        check_http_backend(Backend::OpenRouter).await;
    }
    #[tokio::test]
    async fn groq_http_end_to_end() {
        check_http_backend(Backend::Groq).await;
    }
    #[test]
    fn backend_credentials_and_endpoints_are_separate() {
        assert_eq!(Backend::Groq.prefix(), "GROQ");
        assert_eq!(Backend::Groq.base(), "https://api.groq.com/openai/v1");
        assert_eq!(Backend::OpenRouter.prefix(), "OPENROUTER");
        assert_eq!(Backend::OpenRouter.base(), "https://openrouter.ai/api/v1");
    }
    #[test]
    fn groq_catalog_preserves_order_skips_invalid_and_deduplicates() {
        let catalog = json!({"data":[
            {"id":"openai/gpt-oss-20b","context_window":131072,"active":true},
            {"id":"llama-3.3-70b-versatile","context_window":131072,"active":true},
            {"id":"whisper-large-v3","context_window":448,"active":true},
            {"id":"llama-3.1-8b-instant","context_window":131072,"active":false}
        ]});
        let selected: Vec<String> = [
            "missing",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
            "whisper-large-v3",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-20b",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        let (models, warnings) =
            select_models(Backend::Groq, &catalog, &selected).expect("select models");
        assert_eq!(warnings.len(), 3);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].model_id, "llama-3.3-70b-versatile");
        assert_eq!(models[1].max_output_tokens, Some(65536));
        assert!(!models[1].supports_image_input);
        assert!(select_models(Backend::Groq, &catalog, &["missing".into()]).is_err());
    }
    #[test]
    fn openrouter_catalog_skips_bad_entry_but_keeps_all_valid_models() {
        let catalog = json!({"data":[
            {"id":"first","context_length":10000,"supported_parameters":["tools"]},
            {"id":"second","context_length":20000,"supported_parameters":["tools"]},
            {"id":"embedding","context_length":2000}
        ]});
        let (models, warnings) = select_models(
            Backend::OpenRouter,
            &catalog,
            &["first".into(), "embedding".into(), "second".into()],
        )
        .expect("select");
        assert_eq!(models.len(), 2);
        assert_eq!(models[1].model_id, "second");
        assert_eq!(warnings.len(), 1);
    }
    #[test]
    fn groq_parameters_and_reasoning_history() {
        let mut entry = model();
        entry.model_id = "openai/gpt-oss-20b".into();
        let request = CompletionRequest {
            additional_params: Some(json!({"reasoning_effort":"high"})),
            chat_history: vec![Message::Assistant {
                content: vec![
                    AssistantContent::Reasoning(Reasoning::new_with_signature("thinking", None)),
                    AssistantContent::Text("answer".into()),
                ],
            }],
            ..Default::default()
        };
        let body = backend_request(Backend::Groq, &entry, request).expect("Groq request");
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["parallel_tool_calls"], false);
        assert!(body["messages"][0].get("reasoning").is_none());
        for extra in [
            json!({"provider":{}}),
            json!({"reasoning":{}}),
            json!({"reasoning_format":"parsed"}),
            json!({"logprobs":true}),
        ] {
            assert!(
                backend_request(
                    Backend::Groq,
                    &entry,
                    CompletionRequest {
                        additional_params: Some(extra),
                        ..Default::default()
                    }
                )
                .is_err()
            );
        }
    }
    #[test]
    fn groq_usage_in_x_groq_and_reasoning_deltas() {
        let mut decoder = Decoder::default();
        let chunks = decoder
            .chunk(json!({"choices":[{"delta":{"reasoning":"thinking"}}]}))
            .expect("chunk");
        assert!(matches!(&chunks[0],StreamChunk::ReasoningDelta {text,..} if text == "thinking"));
        decoder.chunk(json!({"choices":[{"delta":{},"finish_reason":"stop"}],"x_groq":{"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}})).expect("usage");
        let final_chunks = decoder.finish().expect("finish");
        assert!(
            matches!(final_chunks.last().expect("final"),StreamChunk::Final(f) if f.usage.expect("usage").total_tokens == 7)
        );
    }
}
