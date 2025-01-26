//! DeepSeek API client implementation for Anda Engine
//!
//! This module provides integration with DeepSeek's API, including:
//! - Client configuration and management
//! - Completion model handling
//! - Response parsing and conversion to Anda's internal formats

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CompletionFeatures, CompletionRequest, FunctionDefinition,
    Message, ToolCall, CONTENT_TYPE_JSON,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

use super::CompletionFeaturesDyn;
use crate::APP_USER_AGENT;

// ================================================================
// Main DeepSeek Client
// ================================================================
const DEEKSEEK_API_BASE_URL: &str = "https://api.deepseek.com";
pub static DEEKSEEK_V3: &str = "deepseek-chat";
pub static DEEKSEEK_R1: &str = "deepseek-reasoner";

/// DeepSeek API client configuration and HTTP client
#[derive(Clone)]
pub struct Client {
    endpoint: String,
    http: reqwest::Client,
}

impl Client {
    /// Creates a new DeepSeek client instance with the provided API key
    ///
    /// # Arguments
    /// * `api_key` - DeepSeek API key for authentication
    ///
    /// # Returns
    /// Configured DeepSeek client instance
    pub fn new(api_key: &str) -> Self {
        Self {
            endpoint: DEEKSEEK_API_BASE_URL.to_string(),
            http: reqwest::Client::builder()
                .use_rustls_tls()
                .https_only(true)
                .http2_keep_alive_interval(Some(Duration::from_secs(25)))
                .http2_keep_alive_timeout(Duration::from_secs(15))
                .http2_keep_alive_while_idle(true)
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(60))
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
                .expect("DeepSeek reqwest client should build"),
        }
    }

    /// Creates a POST request builder for the specified API path
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.endpoint, path);
        self.http.post(url)
    }

    /// Creates a new completion model instance using the default DeepSeek model
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

/// Token usage statistics from DeepSeek API responses
#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    /// Number of tokens used in the prompt
    pub prompt_tokens: usize,
    /// Total number of tokens used (prompt + completion)
    pub total_tokens: usize,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} Total tokens: {}",
            self.prompt_tokens, self.total_tokens
        )
    }
}

/// Completion response from DeepSeek API
#[derive(Debug, Deserialize)]
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
    fn try_into(mut self) -> Result<AgentOutput, BoxError> {
        let choice = self.choices.pop().ok_or("No completion choice")?;
        let mut output = AgentOutput {
            content: choice.message.content,
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

/// Individual completion choice from DeepSeek API
#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: MessageOutput,
    pub finish_reason: String,
}

/// Output message structure from DeepSeek API
#[derive(Debug, Deserialize)]
pub struct MessageOutput {
    pub role: String,
    #[serde(default)]
    pub content: String,
    pub refusal: Option<String>,
    pub tool_calls: Option<Vec<ToolCallOutput>>,
}

/// Tool call output structure from DeepSeek API
#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}

/// Completion model wrapper for DeepSeek API
#[derive(Clone)]
pub struct CompletionModel {
    /// DeepSeek client instance
    client: Client,
    /// Model identifier
    pub model: String,
}

impl CompletionModel {
    /// Creates a new completion model instance
    ///
    /// # Arguments
    /// * `client` - DeepSeek client instance
    /// * `model` - Model identifier string
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl CompletionFeatures for CompletionModel {
    async fn completion(&self, req: CompletionRequest) -> Result<AgentOutput, BoxError> {
        CompletionFeaturesDyn::completion(self, req).await
    }
}

impl CompletionFeaturesDyn for CompletionModel {
    fn completion(&self, mut req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();

        Box::pin(async move {
            // Add system to chat history (if available)
            let mut full_history = if let Some(system) = &req.system {
                vec![Message {
                    role: "system".into(),
                    content: system.to_owned().into(),
                    name: req.system_name.clone(),
                    ..Default::default()
                }]
            } else {
                vec![]
            };

            // Add context documents to chat history
            if !req.documents.is_empty() {
                full_history.push(Message {
                    role: "user".into(),
                    content: format!("{}", req.documents).into(),
                    name: req.system_name.clone(),
                    ..Default::default()
                });
            }

            // Extend existing chat history
            full_history.append(&mut req.chat_history);

            full_history.push(Message {
                role: "user".into(),
                content: req.prompt.into(),
                name: req.prompter_name,
                ..Default::default()
            });

            let mut body = json!({
                "model": model,
                "messages": full_history,
                "temperature": req.temperature,
            });
            let body = body.as_object_mut().unwrap();

            if let Some(max_tokens) = req.max_tokens {
                body.insert("max_tokens".to_string(), Value::from(max_tokens));
            }

            if req.response_format.is_some() {
                // DeepSeek only supports `{"type": "json_object"}`
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
                    json!(req
                        .tools
                        .into_iter()
                        .map(ToolDefinition::from)
                        .collect::<Vec<_>>()),
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

            let response = client.post("/chat/completions").json(body).send().await?;
            if response.status().is_success() {
                match response.json::<CompletionResponse>().await {
                    Ok(res) => res.try_into(),
                    Err(err) => Err(format!("DeepSeek completions error: {}", err).into()),
                }
            } else {
                let msg = response.text().await?;
                Err(format!("DeepSeek completions error: {}", msg).into())
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::character::Character;

    #[tokio::test(flavor = "current_thread")]
    #[ignore]
    async fn test_deepseek() {
        dotenv::dotenv().ok();

        let api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEKSEEK_API_KEY is not set");
        let character_path = format!("{}/../characters/AndaICP.toml", env!("CARGO_MANIFEST_DIR"));
        println!("Character path: {}", character_path);
        let character = std::fs::read_to_string(character_path).expect("Character file not found");
        let character = Character::from_toml(&character).expect("Character should parse");
        let client = Client::new(&api_key);
        let model = client.completion_model(DEEKSEEK_V3);
        let req = character.to_request("I am Yan, glad to see you".into(), Some("Yan".into()));
        let res = CompletionFeatures::completion(&model, req).await.unwrap();
        println!("{}", res.content);
    }
}
