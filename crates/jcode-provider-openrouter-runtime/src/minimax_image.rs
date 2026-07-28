//! MiniMax image generation producer.
//!
//! MiniMax chat models run through the OpenAI-compatible `/chat/completions`
//! runtime, which has no Responses-style hosted `image_generation` tool. To
//! wire MiniMax text-to-image into the existing provider-native generated-image
//! event, persistence, display, and visual-context pipeline, this module:
//!
//! 1. Advertises an `image_generation` function tool to MiniMax chat models so
//!    the model can request image generation mid-turn.
//! 2. Intercepts completed `image_generation` tool calls in the chat stream and
//!    executes them client-side against MiniMax's `/v1/image_generation`
//!    endpoint, downloading the returned image URL into the same
//!    `.jcode/generated-images/` directory used by the OpenAI producer.
//! 3. Emits the provider-agnostic `StreamEvent::GeneratedImage` (plus a
//!    `StreamEvent::ToolResult` so the chat turn continues without the agent
//!    loop trying to execute the tool locally).
//!
//! The downstream pipeline (`StreamEvent::GeneratedImage` -> persistence,
//! side-panel, inline display, visual-context feedback) is already
//! provider-agnostic and is reused unchanged. See
//! `crates/jcode-provider-openai/src/stream.rs` (`handle_openai_image_generation_item`)
//! for the reference producer this mirrors.
//!
//! Endpoint and request/response shape come from the MiniMax image generation
//! reference: `POST /v1/image_generation` with `{"model","prompt",...}`, response
//! `{"data":{"image_urls":[...]}, "base_resp":{"status_code":0}}`.

use crate::ProviderAuth;
use anyhow::Result;
use jcode_message_types::StreamEvent;
use reqwest::Client;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

/// The synthetic tool name shared with the OpenAI image generation path
/// (`jcode_base::message::GENERATED_IMAGE_TOOL_NAME`). Re-declared locally so
/// the runtime crate does not pull a new base dependency edge for one const.
pub(crate) const MINIMAX_IMAGE_GENERATION_TOOL_NAME: &str = "image_generation";

/// Default MiniMax image model id used when the model does not specify one.
pub(crate) const MINIMAX_DEFAULT_IMAGE_MODEL: &str = "image-01";

/// Whether the active OpenAI-compatible profile should expose the MiniMax image
/// generation tool. Only the `minimax` profile speaks the `/v1/image_generation`
/// endpoint, and only chat models (not the image models themselves) should be
/// offered the tool.
pub(crate) fn profile_supports_minimax_image_generation(
    profile_id: Option<&str>,
    model: &str,
) -> bool {
    if !matches!(profile_id, Some(id) if id.eq_ignore_ascii_case("minimax")) {
        return false;
    }
    let lowered = model.trim().to_ascii_lowercase();
    // The image models themselves are not chat models and must not be offered
    // the tool. MiniMax image model ids start with `image-`.
    !lowered.starts_with("image-")
}

/// The OpenAI chat-completions function-tool descriptor advertised to MiniMax
/// chat models so they can request image generation. Parameter names match the
/// MiniMax `/v1/image_generation` request body.
pub(crate) fn minimax_image_generation_tool_json() -> Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": MINIMAX_IMAGE_GENERATION_TOOL_NAME,
            "description": "Generate an image from a text prompt using MiniMax image generation. Returns the saved image file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate."
                    },
                    "model": {
                        "type": "string",
                        "description": "Image model id, e.g. image-01.",
                        "default": MINIMAX_DEFAULT_IMAGE_MODEL
                    },
                    "subject_reference": {
                        "type": "object",
                        "description": "Optional subject reference for image-to-image generation."
                    },
                    "aspect_ratio": { "type": "string" },
                    "width": { "type": "integer" },
                    "height": { "type": "integer" },
                    "response_format": {
                        "type": "string",
                        "description": "Output format: url or base64.",
                        "default": "url"
                    },
                    "seed": { "type": "integer" },
                    "n": { "type": "integer" },
                    "prompt_optimizer": { "type": "boolean" }
                },
                "required": ["prompt"]
            }
        }
    })
}

/// Extract the text prompt and image model id from the tool-call arguments JSON
/// the model emitted. Falls back to the default image model when absent so the
/// MiniMax endpoint always receives a valid `model` field.
fn parse_minimax_image_arguments(arguments: &str) -> Result<(String, String), String> {
    let value: Value = if arguments.trim().is_empty() {
        Value::Object(serde_json::Map::new())
    } else {
        serde_json::from_str(arguments).map_err(|err| format!("invalid tool arguments: {err}"))?
    };
    let prompt = value
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .ok_or_else(|| "missing required `prompt` field".to_string())?
        .to_string();
    let model = value
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .unwrap_or(MINIMAX_DEFAULT_IMAGE_MODEL)
        .to_string();
    Ok((prompt, model))
}

/// Build the MiniMax `/v1/image_generation` request body from the tool-call
/// arguments, carrying through only the optional fields the model supplied.
pub(crate) fn build_minimax_image_request_body(arguments: &str) -> Result<Value, String> {
    let (prompt, model) = parse_minimax_image_arguments(arguments)?;
    let raw: Value = if arguments.trim().is_empty() {
        Value::Object(serde_json::Map::new())
    } else {
        serde_json::from_str(arguments).map_err(|err| format!("invalid tool arguments: {err}"))?
    };
    let mut body = serde_json::json!({ "model": model, "prompt": prompt });
    if let Some(obj) = body.as_object_mut() {
        for key in [
            "subject_reference",
            "aspect_ratio",
            "width",
            "height",
            "response_format",
            "seed",
            "n",
            "prompt_optimizer",
        ] {
            if let Some(value) = raw.get(key) {
                obj.insert(key.to_string(), value.clone());
            }
        }
    }
    Ok(body)
}

/// Extract the first image URL from a MiniMax image generation response.
/// Returns `Err` when the response shape indicates failure.
pub(crate) fn parse_minimax_image_response(response: &Value) -> Result<String, String> {
    let status_code = response
        .get("base_resp")
        .and_then(|v| v.get("status_code"))
        .and_then(Value::as_i64);
    if let Some(code) = status_code
        && code != 0
    {
        let message = response
            .get("base_resp")
            .and_then(|v| v.get("status_message"))
            .and_then(Value::as_str)
            .unwrap_or("unknown error");
        return Err(format!(
            "MiniMax image generation failed (status {code}): {message}"
        ));
    }
    response
        .get("data")
        .and_then(|d| d.get("image_urls"))
        .and_then(Value::as_array)
        .and_then(|urls| urls.first())
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| "MiniMax image generation response missing data.image_urls".to_string())
}

/// Map a MiniMax/MiniMax `response_format` value to a file extension. Defaults
/// to `png` because the side-panel reader keys the format label off this.
fn extension_for_response_format(response_format: Option<&str>) -> &'static str {
    match response_format
        .unwrap_or("png")
        .to_ascii_lowercase()
        .as_str()
    {
        "jpeg" | "jpg" => "jpg",
        "webp" => "webp",
        "gif" => "gif",
        _ => "png",
    }
}

/// Sanitize an id into a filename-safe slug, mirroring the OpenAI producer.
fn safe_filename_segment(id: &str) -> String {
    let safe: String = id
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
        .take(80)
        .collect();
    if safe.is_empty() {
        "image".to_string()
    } else {
        safe
    }
}

/// Persist already-downloaded image bytes to the shared `.jcode/generated-images`
/// directory and write the sidecar metadata JSON with the exact contract the
/// side-panel reader (`jcode_base::generated_image`) expects. Returns the
/// generated-image event plus the markdown summary text.
///
/// This is the pure, network-free half of the producer and is unit-tested.
pub(crate) fn persist_minimax_generated_image(
    image_bytes: &[u8],
    output_format: &str,
    tool_call_id: &str,
    request_body: &Value,
) -> Result<(StreamEvent, String), String> {
    let extension =
        extension_for_response_format(request_body.get("response_format").and_then(Value::as_str));
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default();
    let dir = std::env::current_dir()
        .unwrap_or_else(|_| std::env::temp_dir())
        .join(".jcode")
        .join("generated-images");
    std::fs::create_dir_all(&dir)
        .map_err(|err| format!("failed to create generated image directory: {err}"))?;

    let safe_id = safe_filename_segment(tool_call_id);
    let filename = format!("{timestamp_ms}-{safe_id}.{extension}");
    let path = dir.join(filename);
    std::fs::write(&path, image_bytes)
        .map_err(|err| format!("failed to save generated image: {err}"))?;

    let metadata_path = path.with_extension("json");
    let byte_count = std::fs::metadata(&path)
        .map(|m| m.len())
        .unwrap_or_default();
    let metadata = serde_json::json!({
        "schema_version": 1,
        "provider": "minimax",
        "native_tool": MINIMAX_IMAGE_GENERATION_TOOL_NAME,
        "id": tool_call_id,
        "status": "completed",
        "created_at_unix_ms": timestamp_ms,
        "image_path": path.display().to_string(),
        "output_format": output_format,
        "byte_count": byte_count,
        "revised_prompt": serde_json::Value::Null,
        "response_item": request_body,
    });
    let metadata_path_string = match serde_json::to_vec_pretty(&metadata).ok().and_then(|bytes| {
        std::fs::write(&metadata_path, bytes)
            .ok()
            .map(|_| metadata_path.clone())
    }) {
        Some(path) => Some(path.display().to_string()),
        None => {
            jcode_base::logging::warn("Failed to save MiniMax generated image metadata");
            None
        }
    };

    let mut markdown = format!(
        "\n![Generated image]({})\n\nGenerated image saved to `{}`.",
        path.display(),
        path.display()
    );
    if let Some(metadata_path) = metadata_path_string.as_deref() {
        markdown.push_str(&format!("\nMetadata saved to `{}`.", metadata_path));
    }
    markdown.push('\n');

    let event = StreamEvent::GeneratedImage {
        id: tool_call_id.to_string(),
        path: path.display().to_string(),
        metadata_path: metadata_path_string,
        output_format: output_format.to_string(),
        revised_prompt: None,
    };
    Ok((event, markdown))
}

/// Execute a MiniMax image generation tool call end-to-end: POST to
/// `/v1/image_generation`, download the returned image URL, persist it, and
/// return the stream events the agent loop should observe (markdown text delta,
/// generated-image event, and a tool result so the turn continues without local
/// execution). On failure, returns a single errored tool result.
pub(crate) async fn execute_minimax_image_generation(
    client: &Client,
    api_base: &str,
    auth: &ProviderAuth,
    tool_call_id: &str,
    arguments: &str,
) -> Vec<StreamEvent> {
    let request_body = match build_minimax_image_request_body(arguments) {
        Ok(body) => body,
        Err(err) => {
            return vec![tool_result_event(tool_call_id, &err, true)];
        }
    };

    let endpoint = format!("{api_base}/image_generation");
    let send_result = auth.apply(client.post(&endpoint).json(&request_body)).await;
    let response = match send_result {
        Ok(req) => match req.send().await {
            Ok(resp) => resp,
            Err(err) => {
                return vec![tool_result_event(
                    tool_call_id,
                    &format!("MiniMax image request failed: {err}"),
                    true,
                )];
            }
        },
        Err(err) => {
            return vec![tool_result_event(
                tool_call_id,
                &format!("MiniMax image auth failed: {err}"),
                true,
            )];
        }
    };

    let response_json: Value = match response.json().await {
        Ok(v) => v,
        Err(err) => {
            return vec![tool_result_event(
                tool_call_id,
                &format!("MiniMax image response parse failed: {err}"),
                true,
            )];
        }
    };

    let image_url = match parse_minimax_image_response(&response_json) {
        Ok(url) => url,
        Err(err) => return vec![tool_result_event(tool_call_id, &err, true)],
    };

    let image_bytes = match client.get(&image_url).send().await {
        Ok(resp) => match resp.bytes().await {
            Ok(bytes) => bytes.to_vec(),
            Err(err) => {
                return vec![tool_result_event(
                    tool_call_id,
                    &format!("failed to download generated image: {err}"),
                    true,
                )];
            }
        },
        Err(err) => {
            return vec![tool_result_event(
                tool_call_id,
                &format!("failed to download generated image: {err}"),
                true,
            )];
        }
    };

    let output_format = request_body
        .get("response_format")
        .and_then(Value::as_str)
        .unwrap_or("png");
    match persist_minimax_generated_image(&image_bytes, output_format, tool_call_id, &request_body)
    {
        Ok((generated_image, markdown)) => {
            let saved_path = match &generated_image {
                StreamEvent::GeneratedImage { path, .. } => path.clone(),
                _ => String::new(),
            };
            let summary = format!(
                "Generated image ({}) saved to `{}`.",
                output_format, saved_path
            );
            vec![
                StreamEvent::TextDelta(markdown),
                generated_image,
                tool_result_event(tool_call_id, &summary, false),
            ]
        }
        Err(err) => vec![tool_result_event(tool_call_id, &err, true)],
    }
}

fn tool_result_event(tool_call_id: &str, content: &str, is_error: bool) -> StreamEvent {
    StreamEvent::ToolResult {
        tool_use_id: tool_call_id.to_string(),
        content: content.to_string(),
        is_error,
    }
}

/// Track an in-flight `image_generation` tool call as its Start/Delta/End
/// events stream through the interceptor.
#[derive(Default)]
struct PendingImageToolCall {
    id: String,
    arguments: String,
}

/// Wrap the chat-completions event stream so completed `image_generation` tool
/// calls are executed against the MiniMax image endpoint and turned into
/// `StreamEvent::GeneratedImage` + `StreamEvent::ToolResult`. All other events
/// (including the `ToolUseStart`/`ToolInputDelta`/`ToolUseEnd` for the image
/// tool itself, so the assistant turn records the call) are forwarded
/// unchanged.
///
/// The interceptor runs between the SSE stream task and the final event
/// channel: `run_stream_with_retries` writes to `mid_rx`'s sender, and this
/// function drains `mid_rx` and writes to `tx`.
pub(crate) async fn run_minimax_image_interceptor(
    mid_rx: mpsc::Receiver<Result<StreamEvent>>,
    tx: mpsc::Sender<Result<StreamEvent>>,
    client: Client,
    api_base: String,
    auth: ProviderAuth,
) {
    let mut mid_rx = mid_rx;
    let mut pending: Option<PendingImageToolCall> = None;

    while let Some(item) = mid_rx.recv().await {
        let event = match item {
            Ok(event) => event,
            Err(err) => {
                if tx.send(Err(err)).await.is_err() {
                    return;
                }
                continue;
            }
        };

        match event {
            StreamEvent::ToolUseStart { id, name }
                if name == MINIMAX_IMAGE_GENERATION_TOOL_NAME =>
            {
                let pending_id = id.clone();
                pending = Some(PendingImageToolCall {
                    id: pending_id,
                    arguments: String::new(),
                });
                if tx
                    .send(Ok(StreamEvent::ToolUseStart { id, name }))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            StreamEvent::ToolInputDelta(delta) => {
                if let Some(call) = pending.as_mut() {
                    call.arguments.push_str(&delta);
                }
                if tx
                    .send(Ok(StreamEvent::ToolInputDelta(delta)))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            StreamEvent::ToolUseEnd => {
                let completed = pending.take();
                if tx.send(Ok(StreamEvent::ToolUseEnd)).await.is_err() {
                    return;
                }
                if let Some(call) = completed {
                    let events = execute_minimax_image_generation(
                        &client,
                        &api_base,
                        &auth,
                        &call.id,
                        &call.arguments,
                    )
                    .await;
                    for produced in events {
                        if tx.send(Ok(produced)).await.is_err() {
                            return;
                        }
                    }
                }
            }
            other => {
                if tx.send(Ok(other)).await.is_err() {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_cwd() -> tempfile::TempDir {
        let dir = tempfile::Builder::new()
            .prefix("jcode-minimax-image-test-")
            .tempdir()
            .expect("tempdir");
        std::env::set_current_dir(dir.path()).expect("set temp cwd");
        dir
    }

    #[test]
    fn profile_supports_minimax_image_generation_only_for_minimax_chat_models() {
        assert!(profile_supports_minimax_image_generation(
            Some("minimax"),
            "MiniMax-M3"
        ));
        assert!(profile_supports_minimax_image_generation(
            Some("MiniMax"),
            "minimax-m2.7"
        ));
        assert!(!profile_supports_minimax_image_generation(
            Some("openai"),
            "gpt-5"
        ));
        assert!(!profile_supports_minimax_image_generation(
            Some("minimax"),
            "image-01"
        ));
        assert!(!profile_supports_minimax_image_generation(
            None,
            "MiniMax-M3"
        ));
    }

    #[test]
    fn tool_json_has_required_prompt_parameter() {
        let tool = minimax_image_generation_tool_json();
        assert_eq!(tool["type"], "function");
        assert_eq!(tool["function"]["name"], "image_generation");
        let required = tool["function"]["parameters"]["required"]
            .as_array()
            .expect("required array");
        assert!(required.iter().any(|v| v == "prompt"));
    }

    #[test]
    fn build_request_body_includes_prompt_and_default_model() {
        let body = build_minimax_image_request_body(r#"{"prompt":"a red cube"}"#).expect("body");
        assert_eq!(body["prompt"], "a red cube");
        assert_eq!(body["model"], MINIMAX_DEFAULT_IMAGE_MODEL);
    }

    #[test]
    fn build_request_body_rejects_missing_prompt() {
        let err = build_minimax_image_request_body(r#"{"model":"image-01"}"#).expect_err("err");
        assert!(err.contains("prompt"));
    }

    #[test]
    fn build_request_body_carries_optional_fields() {
        let body = build_minimax_image_request_body(
            r#"{"prompt":"sky","aspect_ratio":"16:9","seed":7,"n":2}"#,
        )
        .expect("body");
        assert_eq!(body["aspect_ratio"], "16:9");
        assert_eq!(body["seed"], 7);
        assert_eq!(body["n"], 2);
    }

    #[test]
    fn parse_response_extracts_first_image_url() {
        let response = serde_json::json!({
            "data": {"image_urls": ["https://example.com/a.png", "https://example.com/b.png"]},
            "base_resp": {"status_code": 0}
        });
        assert_eq!(
            parse_minimax_image_response(&response).expect("url"),
            "https://example.com/a.png"
        );
    }

    #[test]
    fn parse_response_reports_nonzero_status() {
        let response = serde_json::json!({
            "data": {},
            "base_resp": {"status_code": 1001, "status_message": "rate limited"}
        });
        let err = parse_minimax_image_response(&response).expect_err("err");
        assert!(err.contains("1001"));
        assert!(err.contains("rate limited"));
    }

    #[test]
    fn persist_writes_image_and_metadata_with_contract_keys() {
        let _dir = temp_cwd();
        let original_dir = std::env::current_dir().expect("cwd");
        // Restore cwd even if assertions fail.
        struct Guard(PathBuf);
        impl Drop for Guard {
            fn drop(&mut self) {
                let _ = std::env::set_current_dir(&self.0);
            }
        }
        let _guard = Guard(original_dir.clone());

        let image_bytes = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        let request_body = serde_json::json!({"model":"image-01","prompt":"a cube"});
        let (event, markdown) =
            persist_minimax_generated_image(&image_bytes, "png", "call_123", &request_body)
                .expect("persist");

        let path = match event {
            StreamEvent::GeneratedImage {
                path,
                metadata_path,
                output_format,
                ..
            } => {
                assert_eq!(output_format, "png");
                assert!(path.ends_with(".png"));
                let metadata_path = metadata_path.expect("metadata path");
                assert!(std::path::Path::new(&path).exists(), "image file missing");
                assert!(
                    std::path::Path::new(&metadata_path).exists(),
                    "metadata file missing"
                );
                let metadata: Value =
                    serde_json::from_slice(&std::fs::read(&metadata_path).expect("read metadata"))
                        .expect("metadata json");
                assert_eq!(metadata["schema_version"], 1);
                assert_eq!(metadata["provider"], "minimax");
                assert_eq!(metadata["native_tool"], "image_generation");
                assert_eq!(metadata["id"], "call_123");
                assert_eq!(metadata["status"], "completed");
                assert!(metadata["byte_count"].as_u64().unwrap_or(0) >= image_bytes.len() as u64);
                assert_eq!(metadata["response_item"]["prompt"], "a cube");
                path
            }
            other => panic!("expected GeneratedImage, got {other:?}"),
        };

        assert!(markdown.contains("![Generated image]"));
        assert!(markdown.contains(&path));
    }

    #[test]
    fn extension_for_response_format_defaults_to_png() {
        assert_eq!(extension_for_response_format(None), "png");
        assert_eq!(extension_for_response_format(Some("png")), "png");
        assert_eq!(extension_for_response_format(Some("jpeg")), "jpg");
        assert_eq!(extension_for_response_format(Some("webp")), "webp");
    }
}
