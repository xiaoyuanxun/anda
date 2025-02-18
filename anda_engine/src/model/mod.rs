//! Model integration module for Anda Engine
//!
//! This module provides implementations for various AI model providers, including:
//! - OpenAI (completion and embedding models)
//! - DeepSeek (completion models)
//! - Cohere (embedding models)
//!
//! Each provider implementation includes:
//! - Client configuration and management
//! - API request/response handling
//! - Conversion to Anda's internal data structures
//!
//! The module is designed to be extensible, allowing easy addition of new model providers
//! while maintaining a consistent interface through the `CompletionFeaturesDyn` and
//! `EmbeddingFeaturesDyn` traits.

use anda_core::{
    AgentOutput, BoxError, BoxPinFut, CompletionFeatures, CompletionRequest, Embedding,
    EmbeddingFeatures, ToolCall,
};
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, str::FromStr, sync::Arc};

pub mod cohere;
pub mod deepseek;
pub mod openai;
pub mod xai;

/// Trait for dynamic completion features that can be used across threads
pub trait CompletionFeaturesDyn: Send + Sync + 'static {
    /// Performs a completion request and returns a future with the agent's output
    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>>;
}

/// Trait for dynamic embedding features that can be used across threads
pub trait EmbeddingFeaturesDyn: Send + Sync + 'static {
    /// Returns the number of dimensions for the embedding model
    fn ndims(&self) -> usize;

    /// Embeds multiple texts and returns a future with the resulting embeddings
    fn embed(&self, texts: Vec<String>) -> BoxPinFut<Result<Vec<Embedding>, BoxError>>;

    /// Embeds a single query text and returns a future with the resulting embedding
    fn embed_query(&self, text: String) -> BoxPinFut<Result<Embedding, BoxError>>;
}

/// A placeholder implementation for unimplemented features
#[derive(Clone, Debug)]
pub struct NotImplemented;

impl CompletionFeaturesDyn for NotImplemented {
    fn completion(&self, _req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }
}

impl EmbeddingFeaturesDyn for NotImplemented {
    fn ndims(&self) -> usize {
        0
    }

    fn embed(&self, _texts: Vec<String>) -> BoxPinFut<Result<Vec<Embedding>, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn embed_query(&self, _text: String) -> BoxPinFut<Result<Embedding, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }
}

/// A mock implementation for testing purposes
#[derive(Clone, Debug)]
pub struct MockImplemented;

impl CompletionFeaturesDyn for MockImplemented {
    fn completion(&self, req: CompletionRequest) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        Box::pin(futures::future::ready(Ok(AgentOutput {
            content: req.prompt.clone(),
            tool_calls: if req.tools.is_empty() {
                None
            } else {
                Some(
                    req.tools
                        .iter()
                        .map(|tool| ToolCall {
                            id: tool.name.clone(),
                            name: tool.name.clone(),
                            args: req.prompt.clone(),
                            result: None,
                        })
                        .collect(),
                )
            },
            ..Default::default()
        })))
    }
}

impl EmbeddingFeaturesDyn for MockImplemented {
    fn ndims(&self) -> usize {
        384 // EMBED_MULTILINGUAL_LIGHT_V3
    }

    fn embed(&self, texts: Vec<String>) -> BoxPinFut<Result<Vec<Embedding>, BoxError>> {
        Box::pin(futures::future::ready(Ok(texts
            .into_iter()
            .map(|text| Embedding {
                text,
                vec: vec![0.0; 384],
            })
            .collect())))
    }

    fn embed_query(&self, _text: String) -> BoxPinFut<Result<Embedding, BoxError>> {
        Box::pin(futures::future::ready(Ok(Embedding {
            text: "test".to_string(),
            vec: vec![0.0; 384],
        })))
    }
}

/// Main model struct that combines embedding and completion capabilities
#[derive(Clone)]
pub struct Model {
    /// Embedding feature implementation
    pub embedder: Arc<dyn EmbeddingFeaturesDyn>,
    /// Completion feature implementation
    pub completer: Arc<dyn CompletionFeaturesDyn>,
}

impl Model {
    /// Creates a new Model with specified embedder and completer
    pub fn new(
        completer: Arc<dyn CompletionFeaturesDyn>,
        embedder: Arc<dyn EmbeddingFeaturesDyn>,
    ) -> Self {
        Self {
            embedder,
            completer,
        }
    }

    /// Creates a Model with only completion features
    pub fn with_completer(completer: Arc<dyn CompletionFeaturesDyn>) -> Self {
        Self {
            completer,
            embedder: Arc::new(NotImplemented),
        }
    }

    /// Creates a Model with unimplemented features (returns errors for all operations)
    pub fn not_implemented() -> Self {
        Self {
            completer: Arc::new(NotImplemented),
            embedder: Arc::new(NotImplemented),
        }
    }

    /// Creates a Model with mock implementations for testing
    pub fn mock_implemented() -> Self {
        Self {
            completer: Arc::new(MockImplemented),
            embedder: Arc::new(MockImplemented),
        }
    }
}

impl CompletionFeatures for Model {
    async fn completion(&self, req: CompletionRequest) -> Result<AgentOutput, BoxError> {
        self.completer.completion(req).await
    }
}

impl EmbeddingFeatures for Model {
    fn ndims(&self) -> usize {
        self.embedder.ndims()
    }

    async fn embed(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<Embedding>, BoxError> {
        self.embedder.embed(texts.into_iter().collect()).await
    }

    async fn embed_query(&self, text: &str) -> Result<Embedding, BoxError> {
        self.embedder.embed_query(text.to_string()).await
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum HybridContent {
    Text { text: String },
    Image { image_url: ImageDetail },
    Audio { input_audio: AudioDetail },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ImageDetail {
    pub url: String,
    pub detail: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct AudioDetail {
    pub data: String,
    pub format: String,
}

impl From<String> for HybridContent {
    fn from(text: String) -> Self {
        HybridContent::Text { text }
    }
}

impl From<&str> for HybridContent {
    fn from(text: &str) -> Self {
        text.to_owned().into()
    }
}

impl FromStr for HybridContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_content() {
        let content = HybridContent::Text {
            text: "Hello, world!".to_string(),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert_eq!(json, r#"{"type":"text","text":"Hello, world!"}"#);

        let ct: HybridContent = serde_json::from_str(&json).unwrap();
        assert_eq!(ct, content);

        let ct = HybridContent::from("Hello, world!");
        assert_eq!(ct, content);

        let content = HybridContent::Image {
            image_url: ImageDetail {
                url: "https://example.com/image.jpg".to_string(),
                detail: "high".to_string(),
            },
        };

        let json = serde_json::to_string(&content).unwrap();
        assert_eq!(
            json,
            r#"{"type":"image","image_url":{"url":"https://example.com/image.jpg","detail":"high"}}"#
        );

        let ct: HybridContent = serde_json::from_str(&json).unwrap();
        assert_eq!(ct, content);
    }
}
