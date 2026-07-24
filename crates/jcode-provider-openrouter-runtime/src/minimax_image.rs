use super::ProviderAuth;
use anyhow::{Context, Result};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use jcode_message_types::{StreamEvent, ToolDefinition};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

pub(super) const MINIMAX_IMAGE_TOOL_NAME: &str = "image_generation";

pub(super) fn should_enable(profile_id: Option<&str>, tools: &[ToolDefinition]) -> bool {
    profile_id == Some("minimax")
        && !tools
            .iter()
            .any(|tool| tool.name == MINIMAX_IMAGE_TOOL_NAME)
}

pub(super) fn tool_definition() -> Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": MINIMAX_IMAGE_TOOL_NAME,
            "description": "Generate one or more images with the active MiniMax image endpoint. Use this when the user asks to create an image. The generated files are persisted and attached as visual context.",
            "parameters": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image, up to 1500 characters."
                    },
                    "model": {
                        "type": "string",
                        "enum": ["image-01", "image-01-live"],
                        "default": "image-01"
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "enum": ["1:1", "16:9", "4:3", "3:2", "2:3", "3:4", "9:16", "21:9"]
                    },
                    "width": {
                        "type": "integer",
                        "minimum": 512,
                        "maximum": 2048,
                        "multipleOf": 8,
                        "description": "Custom width for image-01. Must be provided with height."
                    },
                    "height": {
                        "type": "integer",
                        "minimum": 512,
                        "maximum": 2048,
                        "multipleOf": 8,
                        "description": "Custom height for image-01. Must be provided with width."
                    },
                    "response_format": {
                        "type": "string",
                        "enum": ["url", "base64"],
                        "default": "url"
                    },
                    "seed": { "type": "integer" },
                    "n": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 9,
                        "default": 1
                    },
                    "prompt_optimizer": {
                        "type": "boolean",
                        "default": false
                    }
                },
                "required": ["prompt"]
            }
        }
    })
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MinimaxImageGenerationRequest {
    prompt: String,
    #[serde(default = "default_image_model")]
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    height: Option<u32>,
    #[serde(default = "default_response_format")]
    response_format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(default = "default_image_count")]
    n: u8,
    #[serde(default)]
    prompt_optimizer: bool,
}

fn default_image_model() -> String {
    "image-01".to_string()
}

fn default_response_format() -> String {
    "url".to_string()
}

const fn default_image_count() -> u8 {
    1
}

impl MinimaxImageGenerationRequest {
    fn from_arguments(arguments: &str) -> Result<Self> {
        let mut request: Self = serde_json::from_str(arguments)
            .context("image_generation arguments must be a JSON object")?;
        request.prompt = request.prompt.trim().to_string();
        request.model = request.model.trim().to_string();
        request.response_format = request.response_format.trim().to_ascii_lowercase();

        if request.prompt.is_empty() {
            anyhow::bail!("prompt is required");
        }
        if request.prompt.chars().count() > 1500 {
            anyhow::bail!("prompt must not exceed 1500 characters");
        }
        if !matches!(request.model.as_str(), "image-01" | "image-01-live") {
            anyhow::bail!("model must be image-01 or image-01-live");
        }
        if !matches!(request.response_format.as_str(), "url" | "base64") {
            anyhow::bail!("response_format must be url or base64");
        }
        if !(1..=9).contains(&request.n) {
            anyhow::bail!("n must be between 1 and 9");
        }
        if request.width.is_some() != request.height.is_some() {
            anyhow::bail!("width and height must be provided together");
        }
        for (name, value) in [("width", request.width), ("height", request.height)] {
            if let Some(value) = value
                && (!(512..=2048).contains(&value) || value % 8 != 0)
            {
                anyhow::bail!("{name} must be between 512 and 2048 and divisible by 8");
            }
        }
        if let Some(aspect_ratio) = request.aspect_ratio.as_deref()
            && !matches!(
                aspect_ratio,
                "1:1" | "16:9" | "4:3" | "3:2" | "2:3" | "3:4" | "9:16" | "21:9"
            )
        {
            anyhow::bail!("unsupported aspect_ratio");
        }

        Ok(request)
    }

    fn metadata_item(&self, response: &Value) -> Value {
        let mut item = serde_json::to_value(self).unwrap_or_else(|_| serde_json::json!({}));
        if let Some(object) = item.as_object_mut() {
            object.remove("prompt");
            object.insert(
                "success_count".to_string(),
                response
                    .pointer("/metadata/success_count")
                    .cloned()
                    .unwrap_or(Value::Null),
            );
            object.insert(
                "failed_count".to_string(),
                response
                    .pointer("/metadata/failed_count")
                    .cloned()
                    .unwrap_or(Value::Null),
            );
        }
        item
    }
}

#[derive(Debug, PartialEq, Eq)]
enum MinimaxImageSource {
    Url(String),
    Base64 {
        bytes: Vec<u8>,
        media_type: Option<String>,
    },
}

fn parse_image_sources(response: &Value, response_format: &str) -> Result<Vec<MinimaxImageSource>> {
    let image_base64 = response
        .pointer("/data/image_base64")
        .and_then(Value::as_array)
        .filter(|values| !values.is_empty());
    let image_urls = response
        .pointer("/data/image_urls")
        .and_then(Value::as_array)
        .filter(|values| !values.is_empty());
    let (values, is_url) = if response_format == "base64" {
        image_base64
            .map(|values| (values, false))
            .or_else(|| image_urls.map(|values| (values, true)))
    } else {
        image_urls
            .map(|values| (values, true))
            .or_else(|| image_base64.map(|values| (values, false)))
    }
    .context("image generation response did not include image data")?;

    values
        .iter()
        .map(|value| {
            let value = value
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .context("image generation response contained an empty image")?;
            if is_url {
                return Ok(MinimaxImageSource::Url(value.to_string()));
            }

            let (encoded, media_type) = if let Some((header, encoded)) = value.split_once(',')
                && header.starts_with("data:")
                && header.ends_with(";base64")
            {
                (
                    encoded,
                    Some(
                        header
                            .trim_start_matches("data:")
                            .trim_end_matches(";base64")
                            .to_string(),
                    ),
                )
            } else {
                (value, None)
            };
            let bytes = BASE64_STANDARD
                .decode(encoded)
                .context("image generation response contained invalid base64")?;
            Ok(MinimaxImageSource::Base64 { bytes, media_type })
        })
        .collect()
}

fn response_status_code(response: &Value) -> Option<i64> {
    let value = response.pointer("/base_resp/status_code")?;
    value
        .as_i64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

fn image_output_format(bytes: &[u8], media_type: Option<&str>) -> &'static str {
    if media_type.is_some_and(|value| value.eq_ignore_ascii_case("image/jpeg"))
        || bytes.starts_with(&[0xff, 0xd8, 0xff])
    {
        "jpeg"
    } else if media_type.is_some_and(|value| value.eq_ignore_ascii_case("image/webp"))
        || (bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP"))
    {
        "webp"
    } else if media_type.is_some_and(|value| value.eq_ignore_ascii_case("image/gif"))
        || bytes.starts_with(b"GIF87a")
        || bytes.starts_with(b"GIF89a")
    {
        "gif"
    } else {
        "png"
    }
}

async fn load_image_source(
    client: &Client,
    source: MinimaxImageSource,
) -> Result<(Vec<u8>, String)> {
    match source {
        MinimaxImageSource::Base64 { bytes, media_type } => {
            let output_format = image_output_format(&bytes, media_type.as_deref()).to_string();
            Ok((bytes, output_format))
        }
        MinimaxImageSource::Url(url) => {
            let response = client
                .get(&url)
                .send()
                .await
                .context("failed to download generated image")?;
            if !response.status().is_success() {
                anyhow::bail!("generated image download returned {}", response.status());
            }
            let media_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            let bytes = response
                .bytes()
                .await
                .context("failed to read generated image download")?
                .to_vec();
            let output_format = image_output_format(&bytes, media_type.as_deref()).to_string();
            Ok((bytes, output_format))
        }
    }
}

async fn execute_image_generation(
    client: &Client,
    api_base: &str,
    auth: &ProviderAuth,
    call_id: &str,
    arguments: &str,
) -> Result<Vec<StreamEvent>> {
    let request = MinimaxImageGenerationRequest::from_arguments(arguments)?;
    let endpoint = format!("{}/image_generation", api_base.trim_end_matches('/'));
    let response = auth
        .apply(client.post(&endpoint).json(&request))
        .await?
        .send()
        .await
        .context("failed to send MiniMax image generation request")?;
    if !response.status().is_success() {
        let status = response.status();
        let body = jcode_base::util::http_error_body(response, "HTTP error").await;
        anyhow::bail!("MiniMax image generation returned {status}: {body}");
    }
    let response: Value = response
        .json()
        .await
        .context("failed to parse MiniMax image generation response")?;
    if response_status_code(&response).unwrap_or_default() != 0 {
        let code = response_status_code(&response).unwrap_or(-1);
        let message = response
            .pointer("/base_resp/status_msg")
            .and_then(Value::as_str)
            .unwrap_or("unknown error");
        anyhow::bail!("MiniMax image generation failed with status {code}: {message}");
    }

    let sources = parse_image_sources(&response, &request.response_format)?;
    let source_count = sources.len();
    let response_id = response
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(call_id);
    let metadata_item = request.metadata_item(&response);
    let mut events = Vec::with_capacity(source_count * 2);

    for (index, source) in sources.into_iter().enumerate() {
        let (bytes, output_format) = load_image_source(client, source).await?;
        let image_id = if source_count == 1 {
            response_id.to_string()
        } else {
            format!("{response_id}-{}", index + 1)
        };
        let persisted = jcode_base::generated_image::persist_generated_image(
            &bytes,
            "minimax",
            MINIMAX_IMAGE_TOOL_NAME,
            &image_id,
            Some("completed"),
            &output_format,
            None,
            metadata_item.clone(),
        )?;

        let mut markdown = format!(
            "\n![Generated image]({})\n\nGenerated image saved to `{}`.",
            persisted.path, persisted.path
        );
        if let Some(metadata_path) = persisted.metadata_path.as_deref() {
            markdown.push_str(&format!("\nMetadata saved to `{metadata_path}`."));
        }
        markdown.push('\n');

        events.push(StreamEvent::GeneratedImage {
            id: persisted.id,
            path: persisted.path,
            metadata_path: persisted.metadata_path,
            output_format: persisted.output_format,
            revised_prompt: persisted.revised_prompt,
        });
        events.push(StreamEvent::TextDelta(markdown));
    }

    Ok(events)
}

#[derive(Debug)]
struct PendingImageCall {
    id: String,
    arguments: String,
}

pub(super) async fn forward_events(
    mut provider_rx: mpsc::Receiver<Result<StreamEvent>>,
    tx: mpsc::Sender<Result<StreamEvent>>,
    client: Client,
    api_base: String,
    auth: ProviderAuth,
) {
    let mut pending_call: Option<PendingImageCall> = None;

    while let Some(event) = provider_rx.recv().await {
        match event {
            Ok(StreamEvent::ToolUseStart { id, name })
                if name == MINIMAX_IMAGE_TOOL_NAME && pending_call.is_none() =>
            {
                pending_call = Some(PendingImageCall {
                    id,
                    arguments: String::new(),
                });
            }
            Ok(StreamEvent::ToolInputDelta(delta)) if pending_call.is_some() => {
                if let Some(call) = pending_call.as_mut() {
                    call.arguments.push_str(&delta);
                }
            }
            Ok(StreamEvent::ToolUseEnd) if pending_call.is_some() => {
                let call = pending_call.take().expect("pending image call");
                match execute_image_generation(&client, &api_base, &auth, &call.id, &call.arguments)
                    .await
                {
                    Ok(events) => {
                        for event in events {
                            if tx.send(Ok(event)).await.is_err() {
                                return;
                            }
                        }
                    }
                    Err(error) => {
                        let detail = error.to_string().replace(['\r', '\n'], " ");
                        let detail = jcode_base::util::truncate_str(&detail, 400);
                        jcode_base::logging::warn(&format!(
                            "MiniMax image generation tool failed: {detail}"
                        ));
                        if tx
                            .send(Ok(StreamEvent::TextDelta(format!(
                                "\n[MiniMax image generation failed: {detail}]\n"
                            ))))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            }
            Ok(StreamEvent::RetryRollback { attempt, max }) => {
                pending_call = None;
                if tx
                    .send(Ok(StreamEvent::RetryRollback { attempt, max }))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            other => {
                if tx.send(other).await.is_err() {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definition_exposes_documented_text_to_image_fields() {
        let tool = tool_definition();
        assert_eq!(
            tool["function"]["name"],
            serde_json::json!("image_generation")
        );
        let properties = &tool["function"]["parameters"]["properties"];
        for field in [
            "model",
            "prompt",
            "aspect_ratio",
            "width",
            "height",
            "response_format",
            "seed",
            "n",
            "prompt_optimizer",
        ] {
            assert!(properties.get(field).is_some(), "missing {field}");
        }
        assert!(properties.get("subject_reference").is_none());
    }

    #[test]
    fn tool_is_enabled_only_for_minimax_without_a_registry_conflict() {
        assert!(should_enable(Some("minimax"), &[]));
        assert!(!should_enable(Some("custom"), &[]));
        assert!(!should_enable(
            Some("minimax"),
            &[ToolDefinition {
                name: MINIMAX_IMAGE_TOOL_NAME.to_string(),
                description: "registry-owned image tool".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }]
        ));
    }

    #[test]
    fn request_defaults_match_the_documented_text_to_image_operation() {
        let request = MinimaxImageGenerationRequest::from_arguments(r#"{"prompt":"city at dusk"}"#)
            .expect("parse image request");
        assert_eq!(request.model, "image-01");
        assert_eq!(request.response_format, "url");
        assert_eq!(request.n, 1);
        assert!(!request.prompt_optimizer);
    }

    #[test]
    fn request_rejects_unpaired_custom_dimensions() {
        let error = MinimaxImageGenerationRequest::from_arguments(
            r#"{"prompt":"city at dusk","width":1024}"#,
        )
        .expect_err("unpaired dimensions should fail");
        assert!(error.to_string().contains("provided together"));
    }

    #[test]
    fn response_parser_handles_url_and_base64_outputs() {
        let url_response = serde_json::json!({
            "data": {"image_urls": ["https://example.com/image.png"]}
        });
        assert_eq!(
            parse_image_sources(&url_response, "url").expect("parse URL response"),
            vec![MinimaxImageSource::Url(
                "https://example.com/image.png".to_string()
            )]
        );

        let base64_response = serde_json::json!({
            "data": {"image_base64": ["data:image/png;base64,aGVsbG8="]}
        });
        assert_eq!(
            parse_image_sources(&base64_response, "base64").expect("parse base64 response"),
            vec![MinimaxImageSource::Base64 {
                bytes: b"hello".to_vec(),
                media_type: Some("image/png".to_string()),
            }]
        );
    }

    #[tokio::test]
    async fn reserved_tool_call_is_intercepted_before_the_application_registry() {
        let (provider_tx, provider_rx) = mpsc::channel(8);
        provider_tx
            .send(Ok(StreamEvent::ToolUseStart {
                id: "call_1".to_string(),
                name: MINIMAX_IMAGE_TOOL_NAME.to_string(),
            }))
            .await
            .expect("send tool start");
        provider_tx
            .send(Ok(StreamEvent::ToolInputDelta("{}".to_string())))
            .await
            .expect("send tool input");
        provider_tx
            .send(Ok(StreamEvent::ToolUseEnd))
            .await
            .expect("send tool end");
        provider_tx
            .send(Ok(StreamEvent::MessageEnd {
                stop_reason: Some("tool_calls".to_string()),
            }))
            .await
            .expect("send message end");
        drop(provider_tx);

        let (tx, mut rx) = mpsc::channel(8);
        forward_events(
            provider_rx,
            tx,
            Client::new(),
            "https://api.minimax.io/v1".to_string(),
            ProviderAuth::None {
                label: "test".to_string(),
            },
        )
        .await;

        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event.expect("forwarded event"));
        }
        assert!(events.iter().all(|event| !matches!(
            event,
            StreamEvent::ToolUseStart { .. }
                | StreamEvent::ToolInputDelta(_)
                | StreamEvent::ToolUseEnd
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta(text) if text.contains("MiniMax image generation failed")
        )));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::MessageEnd { .. }))
        );
    }
}
