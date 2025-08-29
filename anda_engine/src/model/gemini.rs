//! Gemini Moonshot API client implementation for Anda Engine
//!
//! This module provides integration with Gemini's API, including:
//! - Client configuration and management
//! - Completion model handling
//! - Response parsing and conversion to Anda's internal formats

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CompletionFeatures, CompletionRequest, Message, Resource,
};
use log::{Level::Debug, log_enabled};
use serde_json::{Value, json};

use super::{CompletionFeaturesDyn, request_client_builder};
use crate::rfc3339_datetime_now;

pub mod types;

// ================================================================
// Main Gemini Client
// ================================================================
const API_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

pub static GEMINI_2_5_PRO: &str = "gemini-2.5-pro";
pub static GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";

/// Gemini API client configuration and HTTP client
#[derive(Clone)]
pub struct Client {
    endpoint: String,
    api_key: String,
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
            api_key: api_key.to_string(),
            http: request_client_builder()
                .build()
                .expect("Gemini reqwest client should build"),
        }
    }

    /// Sets a custom HTTP client for the client
    pub fn with_client(self, http: reqwest::Client) -> Self {
        Self {
            endpoint: self.endpoint,
            api_key: self.api_key,
            http,
        }
    }

    /// Creates a POST request builder for the specified API path
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.endpoint, path);
        self.http.post(url).header("x-goog-api-key", &self.api_key)
    }

    /// Creates a new completion model instance using the default Gemini model
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(
            self.clone(),
            if model.is_empty() {
                GEMINI_2_5_PRO
            } else {
                model
            },
        )
    }
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
        _resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        CompletionFeaturesDyn::completion(self, req).await
    }
}

impl CompletionFeaturesDyn for CompletionModel {
    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();

        Box::pin(async move {
            let mut greq = types::GenerateContentRequest::default();
            if !req.system.is_empty() {
                greq.system_instruction = Some(types::Content {
                    role: Some(types::Role::Model),
                    parts: vec![types::ContentPart {
                        data: types::PartKind::Text(req.system),
                        ..Default::default()
                    }],
                });
            };

            for msg in req.chat_history {
                greq.contents.push(msg.try_into()?);
            }

            let role = if req.role.as_deref() == Some("assistant") {
                types::Role::Model
            } else {
                types::Role::User
            };

            let mut full_history: Vec<Value> = Vec::new();
            if let Some(prompt) = req.documents.to_message(&rfc3339_datetime_now()) {
                full_history.push(json!(&prompt));
                greq.contents.push(prompt.try_into()?);
            }

            if !req.content_parts.is_empty() {
                full_history.push(json!(Message {
                    role: role.to_string(),
                    content: json!(req.content_parts),
                    name: req.prompter_name.clone(),
                    ..Default::default()
                }));

                greq.contents.push(types::Content {
                    role: Some(role),
                    parts: req
                        .content_parts
                        .into_iter()
                        .map(|v| v.try_into())
                        .collect::<Result<_, _>>()?,
                });
            }

            if !req.prompt.is_empty() {
                full_history.push(json!(Message {
                    role: role.to_string(),
                    content: req.prompt.clone().into(),
                    name: req.prompter_name,
                    ..Default::default()
                }));

                greq.contents.push(types::Content {
                    role: Some(role),
                    parts: vec![types::ContentPart {
                        data: types::PartKind::Text(req.prompt),
                        ..Default::default()
                    }],
                });
            }

            if let Some(temperature) = req.temperature {
                greq.generation_config.temperature = Some(temperature);
            }

            if let Some(max_tokens) = req.max_tokens {
                greq.generation_config.max_output_tokens = Some(max_tokens as i32);
            }

            if let Some(response_format) = req.response_format {
                greq.generation_config.response_mime_type = Some("application/json".to_string());
                if let Some(val) = response_format.get("json_schema") {
                    greq.generation_config.response_schema = Some(val.clone());
                } else if let Some(ty) = response_format.get("type")
                    && ty.as_str() != Some("json_object")
                {
                    greq.generation_config.response_schema = Some(response_format);
                }
            }

            if let Some(stop) = req.stop {
                greq.generation_config.stop_sequences = Some(stop);
            }

            if !req.tools.is_empty() {
                greq.tools = vec![req.tools.into()];
                greq.tool_config = Some(types::ToolConfig::default());
            };

            if log_enabled!(Debug)
                && let Ok(val) = serde_json::to_string(&greq)
            {
                log::debug!(request = val; "Gemini completions request");
            }

            let response = client
                .post(&format!("/{}:generateContent", model))
                .json(&greq)
                .send()
                .await?;
            if response.status().is_success() {
                let text = response.text().await?;

                match serde_json::from_str::<types::GenerateContentResponse>(&text) {
                    Ok(res) => {
                        if log_enabled!(Debug) {
                            log::debug!(
                                request:serde = greq,
                                response:serde = res;
                                "Gemini completions response");
                        } else if res.maybe_failed() {
                            log::warn!(
                                request:serde = greq,
                                response:serde = res;
                                "completions maybe failed");
                        }

                        res.try_into(full_history)
                    }
                    Err(err) => {
                        Err(format!("Gemini completions error: {}, body: {}", err, text).into())
                    }
                }
            } else {
                let status = response.status();
                let msg = response.text().await?;
                log::error!(
                    request:serde = greq;
                    "completions request failed: {status}, body: {msg}",
                );
                Err(format!("Gemini completions error: {}", msg).into())
            }
        })
    }
}
