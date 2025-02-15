//! Core data models and traits for the AI agent system
//!
//! This module defines the fundamental data structures and interfaces used throughout the AI agent system.
//! It includes:
//! - Core message and conversation structures ([`AgentOutput`], [`Message`], [`ToolCall`])
//! - Function definition and tooling support ([`FunctionDefinition`])
//! - Knowledge and document handling ([`Document`], [`Documents`])
//! - Completion request and response structures ([`CompletionRequest`], [`Embedding`])
//! - Core AI capabilities traits ([`CompletionFeatures`], [`EmbeddingFeatures`])
//!
//! The module provides serialization support through `serde` and implements various conversion traits
//! for seamless integration between different data representations.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, future::Future};

use crate::{BoxError, Knowledge};

/// Represents the output of an agent execution
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AgentOutput {
    /// The output content from the agent, may be empty
    pub content: String,

    /// Indicates failure reason if present, None means successful execution
    /// Should be None when finish_reason is "stop" or "tool_calls"
    pub failed_reason: Option<String>,

    /// Tool call that this message is responding to. If this message is a response to a tool call, this field should be set to the tool call ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// full_history will be included in `ctx.completion`'s response,
    /// but not be included in the engine's response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_history: Option<Vec<Value>>,
}

/// Represents a tool call response with it's ID, function name, and arguments
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolCall {
    /// tool call id
    pub id: String,
    /// tool function name
    pub name: String,
    /// tool function  arguments
    pub args: String,
    /// The result of the tool call, auto processed by agents engine, if available
    pub result: Option<String>,
}

/// Represents a message in the agent's conversation history
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Message {
    /// Message role: "developer", "system", "user", "assistant", "tool"
    pub role: String,

    /// The content of the message, can be text or structured data
    pub content: Value,

    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tool call that this message is responding to. If this message is a response to a tool call, this field should be set to the tool call ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Defines a callable function with its metadata and schema
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct FunctionDefinition {
    /// Name of the function
    pub name: String,

    /// Description of what the function does
    pub description: String,

    /// JSON schema defining the function's parameters
    pub parameters: Value,

    /// Whether to enable strict schema adherence when generating the function call. If set to true, the model will follow the exact schema defined in the parameters field. Only a subset of JSON Schema is supported when strict is true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Knowledge document with text and metadata
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    #[serde(flatten)]
    pub additional_props: BTreeMap<String, String>,
}

impl From<Knowledge> for Document {
    fn from(doc: Knowledge) -> Self {
        let mut additional_props = BTreeMap::new();
        additional_props.insert("user".to_string(), doc.user);
        if let Value::Object(obj) = doc.meta {
            for (k, v) in obj {
                additional_props.insert(k, v.to_string());
            }
        }

        Document {
            id: doc.id,
            text: doc.text,
            additional_props,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Documents(pub Vec<Document>);

impl From<Vec<String>> for Documents {
    fn from(texts: Vec<String>) -> Self {
        let mut docs = Vec::new();
        for (i, text) in texts.into_iter().enumerate() {
            docs.push(Document {
                id: format!("doc_{}", i),
                text,
                additional_props: BTreeMap::new(),
            });
        }
        Self(docs)
    }
}

impl From<Vec<Document>> for Documents {
    fn from(docs: Vec<Document>) -> Self {
        Self(docs)
    }
}

impl From<Vec<Knowledge>> for Documents {
    fn from(docs: Vec<Knowledge>) -> Self {
        Self(docs.into_iter().map(Document::from).collect())
    }
}

impl std::ops::Deref for Documents {
    type Target = Vec<Document>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Documents {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AsRef<Vec<Document>> for Documents {
    fn as_ref(&self) -> &Vec<Document> {
        &self.0
    }
}

impl std::fmt::Display for Document {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "<doc id={:?}>", self.id)?;
        if !self.additional_props.is_empty() {
            write!(f, "<meta ")?;
            for (k, v) in &self.additional_props {
                write!(f, "{}={:?} ", k, v)?;
            }
            writeln!(f, "/>")?;
        }
        write!(f, "{:?}\n</doc>\n", self.text)
    }
}

impl std::fmt::Display for Documents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            return Ok(());
        }
        writeln!(f, "<attachments>")?;
        for doc in &self.0 {
            write!(f, "{}", doc)?;
        }
        write!(f, "</attachments>")
    }
}

/// Struct representing a general completion request that can be sent to a completion model provider.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CompletionRequest {
    /// The system message to be sent to the completion model provider, as the "system" role
    pub system: Option<String>,

    /// The name of system role
    pub system_name: Option<String>,

    /// The chat history (raw message) to be sent to the completion model provider
    pub chat_history: Vec<Value>,

    /// The documents to embed into the prompt
    pub documents: Documents,

    /// The prompt to be sent to the completion model provider as "user" role
    /// It can be empty.
    pub prompt: String,

    /// The name of the prompter
    pub prompter_name: Option<String>,

    /// The tools to be sent to the completion model provider
    pub tools: Vec<FunctionDefinition>,

    /// Whether the tool choice is required
    pub tool_choice_required: bool,

    /// The temperature to be sent to the completion model provider
    pub temperature: Option<f64>,

    /// The max tokens to be sent to the completion model provider
    pub max_tokens: Option<usize>,

    /// An object specifying the JSON format that the model must output.
    /// https://platform.openai.com/docs/guides/structured-outputs
    /// The format can be one of the following:
    /// `{ "type": "json_object" }`
    /// `{ "type": "json_schema", "json_schema": {...} }`
    pub response_format: Option<Value>,

    /// The stop sequence to be sent to the completion model provider
    pub stop: Option<Vec<String>>,
}

impl CompletionRequest {
    pub fn context(mut self, id: String, text: String) -> Self {
        self.documents.0.push(Document {
            id,
            text,
            additional_props: BTreeMap::new(),
        });
        self
    }

    pub fn append_documents(mut self, docs: Documents) -> Self {
        self.documents.0.extend(docs.0);
        self
    }

    pub fn append_tools(mut self, tools: Vec<FunctionDefinition>) -> Self {
        self.tools.extend(tools);
        self
    }
}

/// Represents a text embedding with its original text and vector representation
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Embedding {
    /// The original text that was embedded
    pub text: String,

    /// The embedding vector (typically high-dimensional float array)
    pub vec: Vec<f32>,
}

/// Provides LLM completion capabilities for agents
pub trait CompletionFeatures: Sized {
    /// Generates a completion based on the given prompt and context
    fn completion(
        &self,
        req: CompletionRequest,
    ) -> impl Future<Output = Result<AgentOutput, BoxError>> + Send;
}

/// Provides text embedding capabilities for agents
pub trait EmbeddingFeatures: Sized {
    /// The number of dimensions in the embedding vector.
    fn ndims(&self) -> usize;

    /// Generates embeddings for multiple texts in a batch
    /// Returns a vector of Embedding structs in the same order as input texts
    fn embed(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> impl Future<Output = Result<Vec<Embedding>, BoxError>> + Send;

    /// Generates a single embedding for a query text
    /// Optimized for single text embedding generation
    fn embed_query(&self, text: &str) -> impl Future<Output = Result<Embedding, BoxError>> + Send;
}

/// Returns the number of tokens in the given content in the simplest way
pub fn evaluate_tokens(content: &str) -> usize {
    content.len() / 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt() {
        let req = CompletionRequest {
            prompt: "This is a test prompt.".to_string(),
            documents: vec![
                Document {
                    id: "1".to_string(),
                    text: "Test document 1.".to_string(),
                    additional_props: BTreeMap::new(),
                },
                Document {
                    id: "2".to_string(),
                    text: "Test document 2.".to_string(),
                    additional_props: BTreeMap::from([
                        ("key".to_string(), "value".to_string()),
                        ("a".to_string(), "b".to_string()),
                    ]),
                },
            ]
            .into(),
            ..Default::default()
        };
        let prompt = format!("{}\n\n{}", req.documents, req.prompt);
        println!("{}", prompt);
        assert_eq!(
            prompt,
            "<attachments>\n<doc id=\"1\">\n\"Test document 1.\"\n</doc>\n<doc id=\"2\">\n<meta a=\"b\" key=\"value\" />\n\"Test document 2.\"\n</doc>\n</attachments>\n\nThis is a test prompt."
        );
    }
}
