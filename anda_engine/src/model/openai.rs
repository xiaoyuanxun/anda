//! OpenAI API client implementation for Anda Engine
//!
//! This module provides integration with OpenAI's API, including:
//! - Client configuration and management
//! - Completion model handling
//! - Embedding model handling
//! - Response parsing and conversion to Anda's internal formats

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CompletionRequest, Embedding, Json, Message,
    Usage as ModelUsage,
};
use log::{Level::Debug, log_enabled};
use serde::{Deserialize, Serialize};
use serde_json::json;

pub mod types;

use super::{CompletionFeaturesDyn, EmbeddingFeaturesDyn, request_client_builder};
use crate::{rfc3339_datetime, unix_ms};

// ================================================================
// Main OpenAI Client
// ================================================================
const API_BASE_URL: &str = "https://api.openai.com/v1";

// ================================================================
// OpenAI Embedding API
// ================================================================
/// `text-embedding-3-large` embedding model
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-3-small` embedding model
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-ada-002` embedding model
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

// ================================================================
// OpenAI Completion API
// ================================================================
/// `o1` completion model
pub const O1: &str = "o1";
/// `o1-mini completion model
pub const O3_MINI: &str = "o3-mini";

/// OpenAI API client for handling embeddings and completions
#[derive(Clone)]
pub struct Client {
    endpoint: String,
    api_key: String,
    http: reqwest::Client,
}

impl Client {
    /// Creates a new OpenAI client with the given API key
    ///
    /// # Arguments
    /// * `api_key` - OpenAI API key for authentication
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
                .expect("OpenAI reqwest client should build"),
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

    /// Creates a POST request builder for the given API path
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.endpoint, path);
        self.http.post(url).bearer_auth(&self.api_key)
    }

    /// Creates an embedding model with the given name
    ///
    /// # Arguments
    /// * `model` - Name of the embedding model to use
    ///
    /// # Note
    /// Default embedding dimension of 0 will be used if model is not known
    pub fn embedding_model(&self, model: &str) -> EmbeddingModel {
        let ndims = match model {
            TEXT_EMBEDDING_3_LARGE => 3072,
            TEXT_EMBEDDING_3_SMALL | TEXT_EMBEDDING_ADA_002 => 1536,
            _ => 0,
        };
        EmbeddingModel::new(self.clone(), model, ndims)
    }

    /// Creates a completion model with the given name
    ///
    /// # Arguments
    /// * `model` - Name of the completion model to use
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), if model.is_empty() { O3_MINI } else { model })
    }
}

/// Response structure for OpenAI embedding API
#[derive(Debug, Deserialize, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

impl EmbeddingResponse {
    fn try_into(self, texts: Vec<String>) -> Result<(Vec<Embedding>, ModelUsage), BoxError> {
        if self.data.len() != texts.len() {
            return Err(format!(
                "Expected {} embeddings, got {}",
                texts.len(),
                self.data.len()
            )
            .into());
        }

        Ok((
            self.data
                .into_iter()
                .zip(texts)
                .map(|(embedding, text)| Embedding {
                    text,
                    vec: embedding.embedding,
                })
                .collect(),
            ModelUsage {
                input_tokens: self.usage.prompt_tokens as u64,
                output_tokens: self
                    .usage
                    .total_tokens
                    .saturating_sub(self.usage.prompt_tokens) as u64,
                requests: 1,
            },
        ))
    }
}

/// Individual embedding data from OpenAI response
#[derive(Debug, Deserialize, Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

/// Token usage information from OpenAI API
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    #[serde(default)]
    pub completion_tokens: usize, // no completion_tokens in embeddings API
    pub total_tokens: usize,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {}, completion tokens: {}",
            self.prompt_tokens, self.completion_tokens
        )
    }
}

/// Embedding model implementation for OpenAI API
#[derive(Clone)]
pub struct EmbeddingModel {
    pub model: String,
    client: Client,
    ndims: usize,
}

const MAX_DOCUMENTS: usize = 1024;
impl EmbeddingFeaturesDyn for EmbeddingModel {
    /// The number of dimensions in the embedding vector.
    fn ndims(&self) -> usize {
        self.ndims
    }

    /// Generates embeddings for multiple texts in a batch
    /// Returns a vector of Embedding structs in the same order as input texts
    fn embed(
        &self,
        texts: Vec<String>,
    ) -> BoxPinFut<Result<(Vec<Embedding>, ModelUsage), BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();
        Box::pin(async move {
            if texts.len() > MAX_DOCUMENTS {
                return Err(format!("Too many documents, max is {}", MAX_DOCUMENTS).into());
            }

            let response = client
                .post("/embeddings")
                .json(&json!({
                    "model": model,
                    "input": texts,
                }))
                .send()
                .await?;

            if response.status().is_success() {
                match response.json::<EmbeddingResponse>().await {
                    Ok(res) => res.try_into(texts),
                    Err(err) => Err(format!("OpenAI embeddings error: {}", err).into()),
                }
            } else {
                let msg = response.text().await?;
                Err(format!("OpenAI embeddings error: {}", msg).into())
            }
        })
    }

    /// Generates a single embedding for a query text
    /// Optimized for single text embedding generation
    fn embed_query(&self, text: String) -> BoxPinFut<Result<(Embedding, ModelUsage), BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();
        Box::pin(async move {
            let response = client
                .post("/embeddings")
                .json(&json!({
                    "model": model,
                    "input": text,
                }))
                .send()
                .await?;

            if response.status().is_success() {
                match response.json::<EmbeddingResponse>().await {
                    Ok(mut res) => {
                        let data = res.data.pop().ok_or("no embedding data")?;
                        Ok((
                            Embedding {
                                text: text.to_string(),
                                vec: data.embedding,
                            },
                            ModelUsage {
                                input_tokens: res.usage.prompt_tokens as u64,
                                output_tokens: res
                                    .usage
                                    .total_tokens
                                    .saturating_sub(res.usage.prompt_tokens)
                                    as u64,
                                requests: 1,
                            },
                        ))
                    }
                    Err(err) => Err(format!("OpenAI embeddings error: {}", err).into()),
                }
            } else {
                let msg = response.text().await?;
                Err(format!("OpenAI embeddings error: {}", msg).into())
            }
        })
    }
}

impl EmbeddingModel {
    /// Creates a new embedding model instance
    ///
    /// # Arguments
    /// * `client` - OpenAI client instance
    /// * `model` - Name of the embedding model
    /// * `ndims` - Number of dimensions for the embedding
    pub fn new(client: Client, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_string(),
            ndims,
        }
    }
}

/// Completion model implementation for OpenAI API
#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    pub model: String,
}

impl CompletionModel {
    /// Creates a new completion model instance
    ///
    /// # Arguments
    /// * `client` - OpenAI client instance
    /// * `model` - Name of the completion model
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl CompletionFeaturesDyn for CompletionModel {
    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();

        Box::pin(async move {
            let timestamp = unix_ms();
            let mut raw_history: Vec<Json> = Vec::new();
            let mut chat_history: Vec<Message> = Vec::new();
            let mut oreq = types::CompletionRequest {
                model,
                ..Default::default()
            };
            oreq.additional_parameters.store = Some(false);

            if !req.instructions.is_empty() {
                oreq.instructions = Some(req.instructions);
            };

            for msg in req.raw_history {
                oreq.input.push(serde_json::from_value(msg)?);
            }

            for msg in req.chat_history {
                let vals = types::message_into(msg);
                for val in vals {
                    raw_history.push(serde_json::to_value(&val)?);
                    oreq.input.push(val);
                }
            }

            if let Some(mut msg) = req
                .documents
                .to_message(&rfc3339_datetime(timestamp).unwrap())
            {
                msg.timestamp = Some(timestamp);
                chat_history.push(msg.clone());
                let vals = types::message_into(msg);
                for val in vals {
                    raw_history.push(serde_json::to_value(&val)?);
                    oreq.input.push(val);
                }
            }

            let mut content = req.content;
            if !req.prompt.is_empty() {
                content.push(req.prompt.into());
            }
            if !content.is_empty() {
                let msg = Message {
                    role: req.role.unwrap_or_else(|| "user".to_string()),
                    content,
                    timestamp: Some(timestamp),
                    ..Default::default()
                };

                chat_history.push(msg.clone());
                let vals = types::message_into(msg);
                for val in vals {
                    raw_history.push(serde_json::to_value(&val)?);
                    oreq.input.push(val);
                }
            }

            if let Some(temperature) = req.temperature {
                oreq.temperature = Some(temperature);
            }

            if let Some(max_tokens) = req.max_output_tokens {
                oreq.max_output_tokens = Some(max_tokens as u64);
            }

            if let Some(output_schema) = req.output_schema {
                oreq.additional_parameters.text = Some(types::TextConfig::structured_output(
                    "structured_output".to_string(),
                    output_schema,
                ));
            }

            if !req.tools.is_empty() {
                oreq.tools = req
                    .tools
                    .into_iter()
                    .map(|v| types::ToolDefinition {
                        r#type: "function".to_string(),
                        name: v.name,
                        description: v.description,
                        parameters: v.parameters,
                        strict: v.strict.unwrap_or_default(),
                    })
                    .collect::<Vec<_>>();
                oreq.tool_choice = Some(if req.tool_choice_required {
                    "required".to_string()
                } else {
                    "auto".to_string()
                });
            };

            if log_enabled!(Debug)
                && let Ok(val) = serde_json::to_string(&oreq)
            {
                log::debug!(request = val; "OpenAI completions request");
            }

            let response = client.post("/responses").json(&oreq).send().await?;
            if response.status().is_success() {
                let text = response.text().await?;
                match serde_json::from_str::<types::CompletionResponse>(&text) {
                    Ok(res) => {
                        if log_enabled!(Debug) {
                            log::debug!(
                                request:serde = oreq,
                                response:serde = res;
                                "OpenAI completions response");
                        } else if res.maybe_failed() {
                            log::warn!(
                                request:serde = oreq,
                                response:serde = res;
                                "completions maybe failed");
                        }

                        res.try_into(raw_history, chat_history)
                    }
                    Err(err) => {
                        Err(format!("OpenAI completions error: {}, body: {}", err, text).into())
                    }
                }
            } else {
                let status = response.status();
                let msg = response.text().await?;
                log::error!(
                    request:serde = oreq;
                    "completions request failed: {status}, body: {msg}",
                );
                Err(format!("OpenAI completions error: {}", msg).into())
            }
        })
    }
}
