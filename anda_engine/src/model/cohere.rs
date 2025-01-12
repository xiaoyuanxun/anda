//! Cohere API client and Anda integration
//!
use anda_core::{BoxError, BoxPinFut, Embedding, CONTENT_TYPE_JSON};
use serde::Deserialize;
use serde_json::json;
use std::{future::Future, pin::Pin, time::Duration};

use super::EmbeddingFeaturesDyn;
use crate::APP_USER_AGENT;

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

#[derive(Clone)]
pub struct Client {
    endpoint: String,
    http: reqwest::Client,
}

impl Client {
    pub fn new(api_key: &str) -> Self {
        Self {
            endpoint: COHERE_API_BASE_URL.to_string(),
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
                .expect("Cohere reqwest client should build"),
        }
    }

    pub fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.endpoint, path);
        self.http.post(url)
    }

    /// Note: default embedding dimension of 0 will be used if model is not known.
    /// If this is the case, it's better to use function `embedding_model_with_ndims`
    pub fn embedding_model(&self, model: &str) -> EmbeddingModel {
        let ndims = match model {
            EMBED_ENGLISH_V3 | EMBED_MULTILINGUAL_V3 => 1024,
            EMBED_ENGLISH_LIGHT_V3 | EMBED_MULTILINGUAL_LIGHT_V3 => 384,
            _ => 0,
        };
        EmbeddingModel::new(self.clone(), model, ndims)
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub id: String,
    pub embeddings: Embeddings,
    pub texts: Vec<String>,
    #[serde(default)]
    pub meta: Option<Meta>,
}

impl EmbeddingResponse {
    fn try_into(self, texts: Vec<String>) -> Result<Vec<Embedding>, BoxError> {
        if self.embeddings.float.len() != texts.len() {
            return Err(format!(
                "Expected {} embeddings, got {}",
                texts.len(),
                self.embeddings.float.len()
            )
            .into());
        }

        Ok(self
            .embeddings
            .float
            .into_iter()
            .zip(texts)
            .map(|(vec, text)| Embedding { text, vec })
            .collect())
    }
}

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

#[derive(Clone)]
pub struct EmbeddingModel {
    pub model: String,
    client: Client,
    ndims: usize,
}

impl EmbeddingModel {
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
    fn ndims(&self) -> usize {
        self.ndims
    }

    fn embed(&self, texts: Vec<String>) -> BoxPinFut<Result<Vec<Embedding>, BoxError>> {
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

    fn embed_query(&self, text: String) -> BoxPinFut<Result<Embedding, BoxError>> {
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
                        Ok(Embedding { text, vec: data })
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
