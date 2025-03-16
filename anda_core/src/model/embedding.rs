use serde::{Deserialize, Serialize};

use super::Usage;
use crate::BoxError;

/// Represents a text embedding with its original text and vector representation.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Embedding {
    /// The original text that was embedded.
    pub text: String,

    /// The embedding vector (typically high-dimensional float array).
    pub vec: Vec<f32>,
}

/// Provides text embedding capabilities for agents.
pub trait EmbeddingFeatures: Sized {
    /// The number of dimensions in the embedding vector.
    fn ndims(&self) -> usize;

    /// Generates embeddings for multiple texts in a batch.
    /// Returns a vector of Embedding structs in the same order as input texts.
    fn embed(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> impl Future<Output = Result<(Vec<Embedding>, Usage), BoxError>> + Send;

    /// Generates a single embedding for a query text.
    /// Optimized for single text embedding generation.
    fn embed_query(
        &self,
        text: &str,
    ) -> impl Future<Output = Result<(Embedding, Usage), BoxError>> + Send;
}
