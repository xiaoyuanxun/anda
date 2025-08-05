//! Gemini Moonshot API client implementation for Anda Engine
//!
//! This module provides integration with Gemini's API, including:
//! - Client configuration and management
//! - Completion model handling
//! - Response parsing and conversion to Anda's internal formats

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CONTENT_TYPE_JSON, CompletionFeatures, CompletionRequest,
    FunctionDefinition, Message, Resource, ToolCall, Usage as ModelUsage,
};
use log::{Level::Debug, log_enabled};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Duration;

use super::CompletionFeaturesDyn;
use crate::APP_USER_AGENT;

// ================================================================
// Main Gemini Client
// ================================================================
const API_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/openai";
pub static GEMINI_2_5_PRO: &str = "gemini-2.5-pro";
pub static GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";

/// Gemini API client configuration and HTTP client
#[derive(Clone)]
pub struct Client {
    endpoint: String,
    http: reqwest::Client,
}

impl Client {
    /// Creates a new Gemini client instance with the provided API key
    ///
    /// # Arguments
    /// * `api_key` - Gemini API key for authentication
    ///
    /// # Returns
    /// Configured Gemini client instance
    pub fn new(api_key: &str, endpoint: Option<String>) -> Self {
        let endpoint = endpoint.unwrap_or_else(|| API_BASE_URL.to_string());
        let endpoint = if endpoint.is_empty() {
            API_BASE_URL.to_string()
        } else {
            endpoint
        };
        Self {
            endpoint,
            http: reqwest::Client::builder()
                .use_rustls_tls()
                .https_only(true)
                .http2_keep_alive_interval(Some(Duration::from_secs(25)))
                .http2_keep_alive_timeout(Duration::from_secs(15))
                .http2_keep_alive_while_idle(true)
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(180))
                .gzip(true)
                .user_agent(APP_USER_AGENT)
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    let ct: http::HeaderValue = CONTENT_TYPE_JSON.parse().unwrap();
                    headers.insert(http::header::CONTENT_TYPE, ct.clone());
                    headers.insert(http::header::ACCEPT, ct);
                    headers.insert(
                        http::header::AUTHORIZATION,
                        format!("Bearer {}", api_key)
                            .parse()
                            .expect("Bearer token should parse"),
                    );
                    headers
                })
                .build()
                .expect("Gemini reqwest client should build"),
        }
    }

    /// Creates a POST request builder for the specified API path
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.endpoint, path);
        self.http.post(url)
    }

    /// Creates a new completion model instance using the default Gemini model
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(
            self.clone(),
            if model.is_empty() {
                GEMINI_2_5_FLASH
            } else {
                model
            },
        )
    }
}

/// Token usage statistics from Gemini API responses
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
    /// Number of tokens used in the prompt
    pub prompt_tokens: usize,
    /// Number of tokens used in the completion
    pub completion_tokens: usize,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} completion tokens: {}",
            self.prompt_tokens, self.completion_tokens
        )
    }
}

/// Completion response from Gemini API
#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    /// Unique identifier for the completion
    pub id: String,
    /// Object type (typically "chat.completion")
    pub object: String,
    /// Creation timestamp
    pub created: u64,
    /// Model used for the completion
    pub model: String,
    /// List of completion choices
    pub choices: Vec<Choice>,
    /// Token usage statistics
    pub usage: Option<Usage>,
}

impl CompletionResponse {
    fn try_into(mut self, mut full_history: Vec<Value>) -> Result<AgentOutput, BoxError> {
        let choice = self.choices.pop().ok_or("No completion choice")?;
        full_history.push(json!(choice.message));
        let mut output = AgentOutput {
            content: choice.message.content.unwrap_or_default(),
            tool_calls: choice.message.tool_calls.map(|tools| {
                tools
                    .into_iter()
                    .map(|tc| ToolCall {
                        id: tc.id,
                        name: tc.function.name,
                        args: tc.function.arguments,
                        result: None,
                    })
                    .collect()
            }),
            full_history: Some(full_history),
            usage: self
                .usage
                .as_ref()
                .map(|u| ModelUsage {
                    input_tokens: u.prompt_tokens as u64,
                    output_tokens: u.completion_tokens as u64,
                    requests: 1,
                })
                .unwrap_or_default(),
            ..Default::default()
        };

        if !matches!(choice.finish_reason.as_str(), "stop" | "tool_calls") {
            output.failed_reason = Some(choice.finish_reason);
        }
        if let Some(refusal) = choice.message.refusal {
            output.failed_reason = Some(refusal);
        }

        Ok(output)
    }
}

/// Individual completion choice from Gemini API
#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: MessageOutput,
    pub finish_reason: String,
}

/// Output message structure from Gemini API
#[derive(Debug, Deserialize, Serialize)]
pub struct MessageOutput {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    pub refusal: Option<String>,
    pub tool_calls: Option<Vec<ToolCallOutput>>,
}

/// Tool call output structure from Gemini API
#[derive(Debug, Deserialize, Serialize)]
pub struct ToolCallOutput {
    pub id: String,
    pub r#type: String,
    pub function: Function,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: FunctionDefinition,
}

impl From<FunctionDefinition> for ToolDefinition {
    fn from(f: FunctionDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: f,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}

/// Completion model wrapper for Gemini API
#[derive(Clone)]
pub struct CompletionModel {
    /// Gemini client instance
    client: Client,
    /// Model identifier
    pub model: String,
}

impl CompletionModel {
    /// Creates a new completion model instance
    ///
    /// # Arguments
    /// * `client` - Gemini client instance
    /// * `model` - Model identifier string
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl CompletionFeatures for CompletionModel {
    async fn completion(
        &self,
        req: CompletionRequest,
        _resources: Option<Vec<Resource>>,
    ) -> Result<AgentOutput, BoxError> {
        CompletionFeaturesDyn::completion(self, req).await
    }
}

impl CompletionFeaturesDyn for CompletionModel {
    fn completion(&self, mut req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();

        Box::pin(async move {
            // Add system to chat history (if available)
            let has_system = req.system.is_some();
            let mut full_history = if let Some(system) = &req.system {
                vec![json!(Message {
                    role: "system".into(),
                    content: system.to_owned().into(),
                    name: req.system_name.clone(),
                    ..Default::default()
                })]
            } else {
                vec![]
            };

            // Extend existing chat history
            full_history.append(&mut req.chat_history);

            if !req.content_parts.is_empty() {
                full_history.push(json!(Message {
                    role: "user".into(),
                    content: json!(req.content_parts),
                    name: req.prompter_name,
                    ..Default::default()
                }));
            } else if let Some(prompt) = req.prompt_with_context() {
                full_history.push(json!(Message {
                    role: "user".into(),
                    content: prompt.into(),
                    name: req.prompter_name,
                    ..Default::default()
                }));
            }

            let mut body = json!({
                "model": model,
                "messages": full_history.clone(),
            });

            let body = body.as_object_mut().unwrap();
            if let Some(temperature) = req.temperature {
                body.insert("temperature".to_string(), Value::from(temperature));
            }

            if let Some(max_tokens) = req.max_tokens {
                body.insert("max_tokens".to_string(), Value::from(max_tokens));
            }

            if req.response_format.is_some() {
                // Gemini only supports `{"type": "json_object"}`
                body.insert(
                    "response_format".to_string(),
                    json!({"type": "json_object"}),
                );
            }

            if let Some(stop) = req.stop {
                body.insert("stop".to_string(), Value::from(stop));
            }

            if !req.tools.is_empty() {
                body.insert(
                    "tools".to_string(),
                    json!(
                        req.tools
                            .into_iter()
                            .map(ToolDefinition::from)
                            .collect::<Vec<_>>()
                    ),
                );
                body.insert(
                    "tool_choice".to_string(),
                    if req.tool_choice_required {
                        Value::from("required")
                    } else {
                        Value::from("auto")
                    },
                );
            };

            if log_enabled!(Debug) {
                if let Ok(val) = serde_json::to_string(&body) {
                    log::debug!(request = val; "Gemini completions request");
                }
            }

            let response = client.post("/chat/completions").json(body).send().await?;
            if response.status().is_success() {
                let text = response.text().await?;
                match serde_json::from_str::<CompletionResponse>(&text) {
                    Ok(res) => {
                        if log_enabled!(Debug) {
                            if let Ok(val) = serde_json::to_string(&res) {
                                log::debug!(response = val; "Gemini completions response");
                            }
                        }
                        if has_system {
                            full_history.remove(0); // Remove system message from history
                        }
                        res.try_into(full_history)
                    }
                    Err(err) => {
                        Err(format!("Gemini completions error: {}, body: {}", err, text).into())
                    }
                }
            } else {
                let msg = response.text().await?;
                Err(format!("Gemini completions error: {}", msg).into())
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::character::Character;
    use std::time::Instant;

    #[tokio::test(flavor = "current_thread")]
    #[ignore]
    async fn test_gemini() {
        dotenv::dotenv().ok();

        let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY is not set");
        let character_path = format!("{}/../characters/AndaICP.toml", env!("CARGO_MANIFEST_DIR"));
        println!("Character path: {}", character_path);
        let character = std::fs::read_to_string(character_path).expect("Character file not found");
        let character = Character::from_toml(&character).expect("Character should parse");
        let client = Client::new(&api_key, None);
        let now = Instant::now();
        let model = client.completion_model(GEMINI_2_5_FLASH);
        let req = character.to_request("I am Yan, glad to see you".into(), Some("Yan".into()));
        let res = CompletionFeatures::completion(&model, req, None)
            .await
            .unwrap();
        println!("{:#?}", res);
        println!("Took: {:?}", now.elapsed());
    }
}
