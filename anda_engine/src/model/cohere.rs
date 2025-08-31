//! Cohere API client and Anda integration
//!
//! This module provides a client for interacting with Cohere's API, specifically
//! focused on text embedding functionality. It includes support for various
//! Cohere embedding models and handles API communication, error handling,
//! and response parsing.

use anda_core::{BoxError, BoxPinFut, Embedding, Usage};
use serde::Deserialize;
use serde_json::json;

use super::{EmbeddingFeaturesDyn, request_client_builder};

// ================================================================
// Main Cohere Client
// ================================================================
const COHERE_API_BASE_URL: &str = "https://api.cohere.ai";

// ================================================================
// Cohere Embedding API
// ================================================================
/// `embed-english-v3.0` embedding model
pub const EMBED_ENGLISH_V3: &str = "embed-english-v3.0";
/// `embed-english-light-v3.0` embedding model
pub const EMBED_ENGLISH_LIGHT_V3: &str = "embed-english-light-v3.0";
/// `embed-multilingual-v3.0` embedding model
pub const EMBED_MULTILINGUAL_V3: &str = "embed-multilingual-v3.0";
/// `embed-multilingual-light-v3.0` embedding model
pub const EMBED_MULTILINGUAL_LIGHT_V3: &str = "embed-multilingual-light-v3.0";

/// Cohere API client configuration and HTTP client
#[derive(Clone)]
pub struct Client {
    endpoint: String,
    api_key: String,
    http: reqwest::Client,
}

impl Client {
    /// Creates a new Cohere API client with the provided API key
    ///
    /// # Arguments
    /// * `api_key` - Cohere API key for authentication
    pub fn new(api_key: &str, endpoint: Option<String>) -> Self {
        let endpoint = endpoint.unwrap_or_else(|| COHERE_API_BASE_URL.to_string());
        let endpoint = if endpoint.is_empty() {
            COHERE_API_BASE_URL.to_string()
        } else {
            endpoint
        };
        Self {
            endpoint,
            api_key: api_key.to_string(),
            http: request_client_builder()
                .build()
                .expect("Cohere reqwest client should build"),
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
    ///
    /// # Arguments
    /// * `path` - API endpoint path (e.g., "/v1/embed")
    pub fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.endpoint, path);
        self.http.post(url).bearer_auth(&self.api_key)
    }

    /// Creates an embedding model instance with default dimensions
    ///
    /// # Arguments
    /// * `model` - Model identifier (e.g., EMBED_MULTILINGUAL_V3)
    ///
    /// # Returns
    /// EmbeddingModel instance with appropriate dimensions
    pub fn embedding_model(&self, model: &str) -> EmbeddingModel {
        let ndims = match model {
            EMBED_ENGLISH_V3 | EMBED_MULTILINGUAL_V3 => 1024,
            EMBED_ENGLISH_LIGHT_V3 | EMBED_MULTILINGUAL_LIGHT_V3 => 384,
            _ => 0,
        };
        EmbeddingModel::new(self.clone(), model, ndims)
    }
}

/// Response structure for Cohere's embedding API
#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    /// Unique identifier for the request
    pub id: String,
    /// Contains the actual embedding vectors
    pub embeddings: Embeddings,
    /// Original texts that were embedded
    pub texts: Vec<String>,
    /// Metadata about the API response
    #[serde(default)]
    pub meta: Option<Meta>,
}

impl EmbeddingResponse {
    fn try_into(self, texts: Vec<String>) -> Result<(Vec<Embedding>, Usage), BoxError> {
        if self.embeddings.float.len() != texts.len() {
            return Err(format!(
                "Expected {} embeddings, got {}",
                texts.len(),
                self.embeddings.float.len()
            )
            .into());
        }

        Ok((
            self.embeddings
                .float
                .into_iter()
                .zip(texts)
                .map(|(vec, text)| Embedding { text, vec })
                .collect(),
            self.meta.as_ref().map_or(Usage::default(), |m| Usage {
                input_tokens: m.billed_units.input_tokens as u64,
                output_tokens: m.billed_units.output_tokens as u64,
                requests: 1,
            }),
        ))
    }
}

/// Container for different types of embedding vectors
#[derive(Debug, Deserialize)]
pub struct Embeddings {
    #[serde(default)]
    pub float: Vec<Vec<f32>>,
    #[serde(default)]
    pub int8: Vec<Vec<i8>>,
    #[serde(default)]
    pub uint8: Vec<Vec<u8>>,
    #[serde(default)]
    pub binary: Vec<Vec<i8>>,
    #[serde(default)]
    pub ubinary: Vec<Vec<u8>>,
}

/// Metadata about the API response
#[derive(Debug, Deserialize)]
pub struct Meta {
    pub api_version: ApiVersion,
    pub billed_units: BilledUnits,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ApiVersion {
    pub version: String,
    #[serde(default)]
    pub is_deprecated: Option<bool>,
    #[serde(default)]
    pub is_experimental: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct BilledUnits {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub search_units: u32,
    #[serde(default)]
    pub classifications: u32,
}

impl std::fmt::Display for BilledUnits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Input tokens: {}\nOutput tokens: {}\nSearch units: {}\nClassifications: {}",
            self.input_tokens, self.output_tokens, self.search_units, self.classifications
        )
    }
}

/// Cohere embedding model wrapper
#[derive(Clone)]
pub struct EmbeddingModel {
    /// Model identifier
    pub model: String,
    /// Client instance for API communication
    client: Client,
    /// Number of dimensions in the embedding vectors
    ndims: usize,
}

impl EmbeddingModel {
    /// Creates a new embedding model instance
    ///
    /// # Arguments
    /// * `client` - Cohere API client
    /// * `model` - Model identifier
    /// * `ndims` - Number of dimensions in the embedding vectors
    pub fn new(client: Client, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_string(),
            ndims,
        }
    }
}

const MAX_DOCUMENTS: usize = 96;
impl EmbeddingFeaturesDyn for EmbeddingModel {
    /// Returns the number of dimensions for this embedding model
    fn ndims(&self) -> usize {
        self.ndims
    }

    /// Generates embeddings for a batch of texts
    ///
    /// # Arguments
    /// * `texts` - Vector of text strings to embed
    ///
    /// # Returns
    /// Future resolving to a vector of Embedding structs
    ///
    /// https://docs.cohere.com/reference/embed
    /// Maximum number of texts per call is 96.
    /// Tecommend reducing the length of each text to be under 512 tokens for optimal quality.
    fn embed(&self, texts: Vec<String>) -> BoxPinFut<Result<(Vec<Embedding>, Usage), BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();
        Box::pin(async move {
            if texts.len() > MAX_DOCUMENTS {
                return Err(format!("Too many documents, max is {}", MAX_DOCUMENTS).into());
            }

            let response = client
                .post("/v1/embed")
                .json(&json!({
                    "model": model,
                    "input_type": "search_document",
                    "embedding_types": ["float"],
                    "texts": texts,
                }))
                .send()
                .await?;

            if response.status().is_success() {
                match response.json::<EmbeddingResponse>().await {
                    Ok(res) => res.try_into(texts),
                    Err(err) => Err(format!("Cohere embeddings error: {}", err).into()),
                }
            } else {
                let msg = response.text().await?;
                Err(format!("Cohere embeddings error: {}", msg).into())
            }
        })
    }

    /// Generates an embedding for a single query text
    ///
    /// # Arguments
    /// * `text` - Query text to embed
    ///
    /// # Returns
    /// Future resolving to a single Embedding struct
    fn embed_query(&self, text: String) -> BoxPinFut<Result<(Embedding, Usage), BoxError>> {
        let model = self.model.clone();
        let client = self.client.clone();
        Box::pin(async move {
            let response = client
                .post("/v1/embed")
                .json(&json!({
                    "model": model,
                    "input_type": "search_query",
                    "embedding_types": ["float"],
                    "texts": vec![text.clone()],
                }))
                .send()
                .await?;

            if response.status().is_success() {
                match response.json::<EmbeddingResponse>().await {
                    Ok(mut res) => {
                        let data = res.embeddings.float.pop().ok_or("no embedding data")?;
                        let usage = res.meta.as_ref().map_or(Usage::default(), |m| Usage {
                            input_tokens: m.billed_units.input_tokens as u64,
                            output_tokens: m.billed_units.output_tokens as u64,
                            requests: 1,
                        });
                        Ok((Embedding { text, vec: data }, usage))
                    }
                    Err(err) => Err(format!("Cohere embeddings error: {}", err).into()),
                }
            } else {
                let msg = response.text().await?;
                Err(format!("Cohere embeddings error: {}", msg).into())
            }
        })
    }
}
