//! OpenAI API client implementation for Anda Engine
//!
//! This module provides integration with OpenAI's API, including:
//! - Client configuration and management
//! - Completion model handling
//! - Embedding model handling
//! - Response parsing and conversion to Anda's internal formats

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CompletionRequest, Embedding, FunctionDefinition, Message,
    ToolCall, CONTENT_TYPE_JSON,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

use super::{CompletionFeaturesDyn, EmbeddingFeaturesDyn};
use crate::APP_USER_AGENT;

// ================================================================
// Main OpenAI Client
// ================================================================
const OPENAI_API_BASE_URL: &str = "https://api.openai.com/v1";

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
/// `o1-preview` completion model
pub const O1_PREVIEW: &str = "o1-preview";
/// `o1-preview-2024-09-12` completion model
pub const O1_PREVIEW_2024_09_12: &str = "o1-preview-2024-09-12";
/// `o1-mini completion model
pub const O1_MINI: &str = "o1-mini";
/// `o1-mini-2024-09-12` completion model
pub const O1_MINI_2024_09_12: &str = "o1-mini-2024-09-12";
/// `gpt-4o` completion model
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-2024-05-13` completion model
pub const GPT_4O_2024_05_13: &str = "gpt-4o-2024-05-13";
/// `gpt-4-turbo` completion model
pub const GPT_4_TURBO: &str = "gpt-4-turbo";
/// `gpt-4-turbo-2024-04-09` completion model
pub const GPT_4_TURBO_2024_04_09: &str = "gpt-4-turbo-2024-04-09";
/// `gpt-4-turbo-preview` completion model
pub const GPT_4_TURBO_PREVIEW: &str = "gpt-4-turbo-preview";
/// `gpt-4-0125-preview` completion model
pub const GPT_4_0125_PREVIEW: &str = "gpt-4-0125-preview";
/// `gpt-4-1106-preview` completion model
pub const GPT_4_1106_PREVIEW: &str = "gpt-4-1106-preview";
/// `gpt-4-vision-preview` completion model
pub const GPT_4_VISION_PREVIEW: &str = "gpt-4-vision-preview";
/// `gpt-4-1106-vision-preview` completion model
pub const GPT_4_1106_VISION_PREVIEW: &str = "gpt-4-1106-vision-preview";
/// `gpt-4` completion model
pub const GPT_4: &str = "gpt-4";
/// `gpt-4-0613` completion model
pub const GPT_4_0613: &str = "gpt-4-0613";
/// `gpt-4-32k` completion model
pub const GPT_4_32K: &str = "gpt-4-32k";
/// `gpt-4-32k-0613` completion model
pub const GPT_4_32K_0613: &str = "gpt-4-32k-0613";
/// `gpt-3.5-turbo` completion model
pub const GPT_35_TURBO: &str = "gpt-3.5-turbo";
/// `gpt-3.5-turbo-0125` completion model
pub const GPT_35_TURBO_0125: &str = "gpt-3.5-turbo-0125";
/// `gpt-3.5-turbo-1106` completion model
pub const GPT_35_TURBO_1106: &str = "gpt-3.5-turbo-1106";
/// `gpt-3.5-turbo-instruct` completion model
pub const GPT_35_TURBO_INSTRUCT: &str = "gpt-3.5-turbo-instruct";

/// OpenAI API client for handling embeddings and completions
#[derive(Clone)]
pub struct Client {
    endpoint: String,
    http: reqwest::Client,
}

impl Client {
    /// Creates a new OpenAI client with the given API key
    ///
    /// # Arguments
    /// * `api_key` - OpenAI API key for authentication
    pub fn new(api_key: &str) -> Self {
        Self {
            endpoint: OPENAI_API_BASE_URL.to_string(),
            http: reqwest::Client::builder()
                .use_rustls_tls()
                .https_only(true)
                .http2_keep_alive_interval(Some(Duration::from_secs(25)))
                .http2_keep_alive_timeout(Duration::from_secs(15))
                .http2_keep_alive_while_idle(true)
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(120))
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
                .expect("OpenAI reqwest client should build"),
        }
    }

    /// Creates a POST request builder for the given API path
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.endpoint, path);
        self.http.post(url)
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
        CompletionModel::new(self.clone(), model)
    }
}

/// Response structure for OpenAI embedding API
#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

impl EmbeddingResponse {
    fn try_into(self, texts: Vec<String>) -> Result<Vec<Embedding>, BoxError> {
        if self.data.len() != texts.len() {
            return Err(format!(
                "Expected {} embeddings, got {}",
                texts.len(),
                self.data.len()
            )
            .into());
        }

        Ok(self
            .data
            .into_iter()
            .zip(texts)
            .map(|(embedding, text)| Embedding {
                text,
                vec: embedding.embedding,
            })
            .collect())
    }
}

/// Individual embedding data from OpenAI response
#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

/// Token usage information from OpenAI API
#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
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

/// Response structure for OpenAI completion API
#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
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

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: MessageOutput,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize)]
pub struct MessageOutput {
    pub role: String,
    #[serde(default)]
    pub content: String,
    pub refusal: Option<String>,
    pub tool_calls: Option<Vec<ToolCallOutput>>,
}

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
    fn embed(&self, texts: Vec<String>) -> BoxPinFut<Result<Vec<Embedding>, BoxError>> {
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
    fn embed_query(&self, text: String) -> BoxPinFut<Result<Embedding, BoxError>> {
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
                        Ok(Embedding {
                            text: text.to_string(),
                            vec: data.embedding,
                        })
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

    /// Checks if the model is one of the newer OpenAI models
    fn is_new_model(&self) -> bool {
        self.model.starts_with("o1-")
    }
}

impl CompletionFeaturesDyn for CompletionModel {
    fn completion(&self, mut req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let is_new = self.is_new_model();
        let model = self.model.clone();
        let client = self.client.clone();

        Box::pin(async move {
            // Add preamble to chat history (if available)
            let mut full_history = if let Some(system) = &req.system {
                vec![Message {
                    role: if is_new {
                        "developer".into()
                    } else {
                        "system".into()
                    },
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
                if is_new {
                    body.insert("max_completion_tokens".to_string(), Value::from(max_tokens));
                } else {
                    body.insert("max_tokens".to_string(), Value::from(max_tokens));
                }
            }

            if let Some(response_format) = req.response_format {
                body.insert("response_format".to_string(), response_format);
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
                    Err(err) => Err(format!("OpenAI completions error: {}", err).into()),
                }
            } else {
                let msg = response.text().await?;
                Err(format!("OpenAI completions error: {}", msg).into())
            }
        })
    }
}
