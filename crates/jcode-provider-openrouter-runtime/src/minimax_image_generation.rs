use anyhow::{Context, Result, bail};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use jcode_message_types::{ConnectionPhase, ContentBlock, Message, Role, StreamEvent};
use reqwest::Client;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

use super::ProviderAuth;

const IMAGE_GENERATION_TOOL: &str = "image_generation";

pub(super) fn is_minimax_image_model(
    profile_id: Option<&str>,
    api_base: &str,
    model: &str,
) -> bool {
    let is_minimax = profile_id.is_some_and(|id| id.eq_ignore_ascii_case("minimax"))
        || api_base.to_ascii_lowercase().contains("minimax");
    is_minimax && model.trim().to_ascii_lowercase().starts_with("image-")
}

pub(super) fn prompt_from_messages(messages: &[Message]) -> Result<String> {
    let prompt = messages
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .map(|message| {
            message
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text, .. } => Some(text.trim()),
                    _ => None,
                })
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();

    if prompt.is_empty() {
        bail!("MiniMax image generation requires a non-empty user prompt");
    }
    Ok(prompt)
}

pub(super) fn build_request(model: &str, prompt: &str) -> Value {
    json!({
        "model": model,
        "prompt": prompt,
        "response_format": "base64",
        "n": 1,
    })
}

pub(super) async fn run(
    client: Client,
    api_base: String,
    auth: ProviderAuth,
    model: String,
    prompt: String,
    tx: mpsc::Sender<Result<StreamEvent>>,
) -> Result<()> {
    send(
        &tx,
        StreamEvent::ConnectionPhase {
            phase: ConnectionPhase::Connecting,
        },
    )
    .await?;

    let request = build_request(&model, &prompt);
    let url = format!("{}/image_generation", api_base.trim_end_matches('/'));
    let response = auth
        .apply(
            client
                .post(&url)
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .json(&request),
        )
        .await?
        .send()
        .await
        .with_context(|| {
            format!(
                "Failed to send MiniMax image generation request\n  endpoint: {}\n  model: {}\n  auth: {}",
                url,
                model,
                auth.label()
            )
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let body = jcode_base::util::http_error_body(response, "HTTP error").await;
        bail!(
            "MiniMax image generation request failed\n  endpoint: {}\n  model: {}\n  auth: {}\n  status: {}\n  response: {}",
            url,
            model,
            auth.label(),
            status,
            body
        );
    }

    send(
        &tx,
        StreamEvent::ConnectionPhase {
            phase: ConnectionPhase::WaitingForResponse,
        },
    )
    .await?;

    let response_json: Value = response
        .json()
        .await
        .context("Failed to parse MiniMax image generation response")?;
    let status_code = response_json
        .get("base_resp")
        .and_then(|value| value.get("status_code"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    if status_code != 0 {
        let status_msg = response_json
            .get("base_resp")
            .and_then(|value| value.get("status_msg"))
            .and_then(Value::as_str)
            .unwrap_or("unknown provider error");
        bail!(
            "MiniMax image generation failed\n  endpoint: {}\n  model: {}\n  status_code: {}\n  status: {}",
            url,
            model,
            status_code,
            status_msg
        );
    }

    let image_bytes = if let Some(encoded) = response_json
        .get("data")
        .and_then(|data| data.get("image_base64"))
        .and_then(Value::as_array)
        .and_then(|images| images.first())
        .and_then(Value::as_str)
    {
        decode_base64_image(encoded)?
    } else if let Some(image_url) = response_json
        .get("data")
        .and_then(|data| data.get("image_urls"))
        .and_then(Value::as_array)
        .and_then(|images| images.first())
        .and_then(Value::as_str)
    {
        client
            .get(image_url)
            .send()
            .await
            .with_context(|| format!("Failed to download generated image from {}", image_url))?
            .error_for_status()
            .with_context(|| format!("Generated image download failed for {}", image_url))?
            .bytes()
            .await
            .context("Failed to read downloaded MiniMax image bytes")?
            .to_vec()
    } else {
        bail!("MiniMax image generation response did not contain image data");
    };

    let id = response_json
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .unwrap_or("minimax-image-generation")
        .to_string();
    let saved = save_image(&response_json, &request, &model, &prompt, &image_bytes, &id)?;
    let mut markdown = format!(
        "\n![Generated image]({})\n\nGenerated image saved to `{}`.",
        saved.path.display(),
        saved.path.display()
    );
    if let Some(metadata_path) = saved.metadata_path.as_ref() {
        markdown.push_str(&format!(
            "\nMetadata saved to `{}`.",
            metadata_path.display()
        ));
    }
    markdown.push('\n');

    send(&tx, StreamEvent::TextDelta(markdown)).await?;
    send(
        &tx,
        StreamEvent::GeneratedImage {
            id,
            path: saved.path.display().to_string(),
            metadata_path: saved.metadata_path.map(|path| path.display().to_string()),
            output_format: saved.output_format.to_string(),
            revised_prompt: None,
        },
    )
    .await?;
    send(&tx, StreamEvent::MessageEnd { stop_reason: None }).await?;
    Ok(())
}

async fn send(tx: &mpsc::Sender<Result<StreamEvent>>, event: StreamEvent) -> Result<()> {
    tx.send(Ok(event))
        .await
        .map_err(|_| anyhow::anyhow!("image generation event receiver closed"))
}

fn decode_base64_image(encoded: &str) -> Result<Vec<u8>> {
    let payload = encoded
        .split_once(',')
        .map(|(_, payload)| payload)
        .unwrap_or(encoded)
        .trim();
    BASE64_STANDARD
        .decode(payload)
        .context("MiniMax image generation returned invalid base64")
}

struct SavedImage {
    path: PathBuf,
    metadata_path: Option<PathBuf>,
    output_format: &'static str,
}

fn save_image(
    response: &Value,
    request: &Value,
    model: &str,
    prompt: &str,
    image_bytes: &[u8],
    id: &str,
) -> Result<SavedImage> {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    let directory = std::env::current_dir()
        .unwrap_or_else(|_| std::env::temp_dir())
        .join(".jcode")
        .join("generated-images");
    std::fs::create_dir_all(&directory).with_context(|| {
        format!(
            "Failed to create generated image directory {}",
            directory.display()
        )
    })?;

    let (output_format, _) = detect_image_format(image_bytes);
    let safe_id = sanitize_id(id);
    let path = directory.join(format!("{}-{}.{}", timestamp_ms, safe_id, output_format));
    std::fs::write(&path, image_bytes)
        .with_context(|| format!("Failed to save generated image to {}", path.display()))?;

    let mut response_item = response.clone();
    if let Some(data) = response_item.get_mut("data").and_then(Value::as_object_mut) {
        data.remove("image_base64");
        data.remove("image_urls");
    }
    let metadata = json!({
        "schema_version": 1,
        "provider": "minimax",
        "native_tool": IMAGE_GENERATION_TOOL,
        "id": id,
        "status": response.get("base_resp").and_then(|value| value.get("status_msg")),
        "created_at_unix_ms": timestamp_ms,
        "image_path": path.display().to_string(),
        "output_format": output_format,
        "byte_count": image_bytes.len(),
        "revised_prompt": null,
        "response_item": {
            "model": model,
            "prompt": prompt,
            "request": request,
            "response": response_item,
        },
    });
    let metadata_path = path.with_extension("json");
    let metadata_path = match serde_json::to_vec_pretty(&metadata) {
        Ok(bytes) => match std::fs::write(&metadata_path, bytes) {
            Ok(()) => Some(metadata_path),
            Err(error) => {
                jcode_base::logging::warn(&format!(
                    "Failed to save MiniMax generated image metadata: {}",
                    error
                ));
                None
            }
        },
        Err(error) => {
            jcode_base::logging::warn(&format!(
                "Failed to serialize MiniMax generated image metadata: {}",
                error
            ));
            None
        }
    };

    Ok(SavedImage {
        path,
        metadata_path,
        output_format,
    })
}

fn sanitize_id(id: &str) -> String {
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

fn detect_image_format(bytes: &[u8]) -> (&'static str, &'static str) {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        ("png", "image/png")
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        ("jpg", "image/jpeg")
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        ("webp", "image/webp")
    } else if bytes.starts_with(b"GIF8") {
        ("gif", "image/gif")
    } else {
        ("png", "image/png")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_models_are_limited_to_minimax_profiles() {
        assert!(is_minimax_image_model(
            Some("minimax"),
            "https://api.minimax.io/v1",
            "image-01"
        ));
        assert!(is_minimax_image_model(
            Some("minimax"),
            "https://api.minimaxi.com/v1",
            "image-01-live"
        ));
        assert!(!is_minimax_image_model(
            Some("openai"),
            "https://api.openai.com/v1",
            "image-01"
        ));
    }

    #[test]
    fn request_contains_required_minimax_fields() {
        let request = build_request("image-01", "a red kite");
        assert_eq!(request["model"], "image-01");
        assert_eq!(request["prompt"], "a red kite");
        assert_eq!(request["response_format"], "base64");
        assert_eq!(request["n"], 1);
    }

    #[test]
    fn prompt_uses_the_latest_user_text() {
        let messages = vec![
            Message::user("old prompt"),
            Message::assistant_text("ignored"),
            Message::user("new prompt"),
        ];
        assert_eq!(prompt_from_messages(&messages).unwrap(), "new prompt");
    }

    #[test]
    fn save_image_persists_metadata_without_binary_payload() {
        let _lock = jcode_base::storage::lock_test_env();
        let original_dir = std::env::current_dir().expect("current dir");
        let temp = tempfile::tempdir().expect("tempdir");
        std::env::set_current_dir(temp.path()).expect("set current dir");

        let response = json!({
            "id": "img_test",
            "data": {"image_base64": ["AQID"]},
            "base_resp": {"status_code": 0, "status_msg": "success"}
        });
        let request = build_request("image-01", "a red kite");
        let saved = save_image(
            &response,
            &request,
            "image-01",
            "a red kite",
            &[137, 80, 78, 71],
            "img_test",
        )
        .expect("save image");

        assert!(saved.path.exists());
        let metadata_path = saved.metadata_path.expect("metadata path");
        let metadata: Value =
            serde_json::from_slice(&std::fs::read(metadata_path).expect("read metadata"))
                .expect("metadata json");
        assert_eq!(metadata["provider"], "minimax");
        assert_eq!(metadata["native_tool"], IMAGE_GENERATION_TOOL);
        assert!(
            metadata["response_item"]["response"]["data"]
                .get("image_base64")
                .is_none()
        );

        std::env::set_current_dir(original_dir).expect("restore current dir");
    }
}
