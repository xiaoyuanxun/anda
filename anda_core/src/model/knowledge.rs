use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::Value;
use crate::BoxError;

/// Represents a knowledge document with user, text, and metadata.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct Knowledge {
    pub id: String,
    pub user: String,
    pub text: String,
    pub meta: BTreeMap<String, Value>,
}

/// Represents a knowledge document input with user, text, metadata, and vector.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeInput {
    pub user: String,
    pub text: String,
    pub meta: BTreeMap<String, Value>,
    pub vec: Vec<f32>,
}

/// Provides knowledge management capabilities for agents.
pub trait KnowledgeFeatures: Sized {
    /// Performs a semantic search to find top n most similar documents
    /// Returns a list of deserialized knowledge document
    fn knowledge_top_n(
        &self,
        query: &str,
        n: usize,
        user: Option<String>,
    ) -> impl Future<Output = Result<Vec<Knowledge>, BoxError>> + Send;

    /// Retrieves the latest n Knowledge documents created in last N seconds
    fn knowledge_latest_n(
        &self,
        last_seconds: u32,
        n: usize,
        user: Option<String>,
    ) -> impl Future<Output = Result<Vec<Knowledge>, BoxError>> + Send;

    /// Adds a list of Knowledge documents to the knowledge store
    fn knowledge_add(
        &self,
        docs: Vec<KnowledgeInput>,
    ) -> impl std::future::Future<Output = Result<(), BoxError>> + Send;
}
