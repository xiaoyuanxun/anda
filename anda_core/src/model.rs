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
use serde_bytes::ByteBuf;
use serde_json::Value;
use std::{collections::BTreeMap, convert::Infallible, future::Future, str::FromStr};

use crate::{BoxError, Knowledge};

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolReqeust {
    /// tool name
    pub name: String,
    /// arguments in JSON string
    pub args: String,
    // pub payment: Option<ByteBuf>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AgentReqeust {
    pub content: Vec<ContentPart>,
    pub attachment: Option<ByteBuf>,
    /// agent name
    pub name: Option<String>,
    pub user: Option<String>,
    // pub payment: Option<ByteBuf>,
}

/// Represents the usage statistics for the agent or tool execution
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
}

impl Usage {
    pub fn accumulate(&mut self, other: &Usage) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
    }
}

/// Represents the output of an agent execution
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AgentOutput {
    /// The output content from the agent, may be empty
    pub content: String,

    /// Indicates failure reason if present, None means successful execution
    /// Should be None when finish_reason is "stop" or "tool_calls"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_reason: Option<String>,

    /// Tool call that this message is responding to. If this message is a response to a tool call, this field should be set to the tool call ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// full_history will be included in `ctx.completion`'s response,
    /// but not be included in the engine's response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_history: Option<Vec<Value>>,

    /// The usage statistics for the agent execution
    #[serde(default)]
    pub usage: Usage,
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

impl FunctionDefinition {
    pub fn name_with_prefix(mut self, prefix: &str) -> Self {
        self.name = format!("{}{}", prefix, self.name);
        self
    }
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

    /// The content parts to be sent to the completion model provider.
    /// prompt will be ignored if content_parts is not empty.
    pub content_parts: Vec<ContentPart>,

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
    /// Adds a document to the request
    pub fn context(mut self, id: String, text: String) -> Self {
        self.documents.0.push(Document {
            id,
            text,
            additional_props: BTreeMap::new(),
        });
        self
    }

    /// Adds multiple documents to the request
    pub fn append_documents(mut self, docs: Documents) -> Self {
        self.documents.0.extend(docs.0);
        self
    }

    /// Adds multiple tools to the request
    pub fn append_tools(mut self, tools: Vec<FunctionDefinition>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Returns the prompt with context if available
    pub fn prompt_with_context(&self) -> Option<String> {
        if self.documents.0.is_empty() && self.prompt.is_empty() {
            return None;
        }

        if self.documents.0.is_empty() {
            Some(self.prompt.clone())
        } else if self.prompt.is_empty() {
            Some(format!("{}", self.documents))
        } else {
            Some(format!("{}\n\n{}", self.documents, self.prompt))
        }
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
    ) -> impl Future<Output = Result<(Vec<Embedding>, Usage), BoxError>> + Send;

    /// Generates a single embedding for a query text
    /// Optimized for single text embedding generation
    fn embed_query(
        &self,
        text: &str,
    ) -> impl Future<Output = Result<(Embedding, Usage), BoxError>> + Send;
}

/// Returns the number of tokens in the given content in the simplest way
pub fn evaluate_tokens(content: &str) -> usize {
    content.len() / 3
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Resource {
    /// The URI of this resource.
    pub uri: String,
    /// A human-readable name for this resource.
    pub name: String,
    /// A description of what this resource represents.
    /// This can be used by clients to improve the LLM's understanding of available resources.
    pub description: String,
    /// https://developer.mozilla.org/zh-CN/docs/Web/HTTP/MIME_types/Common_types
    pub mime_type: String,
    /// The binary data of this resource.
    pub blob: Option<ByteBuf>,
}

/// OpenAI style content part for the completion request
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentPart {
    Text { text: String },
    Image { image_url: ImageDetail },
    Audio { input_audio: AudioDetail },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ImageDetail {
    /// Either a URL of the image or the base64 encoded image data.
    /// https://platform.openai.com/docs/guides/vision
    /// PNG (.png), JPEG (.jpeg and .jpg), WEBP (.webp), and non-animated GIF (.gif).
    pub url: String,
    /// low, high, and auto.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct AudioDetail {
    /// Base64 encoded audio data.
    pub data: String,
    /// The format of the encoded audio data. Currently supports "wav" and "mp3".
    pub format: String,
}

impl From<String> for ContentPart {
    fn from(text: String) -> Self {
        ContentPart::Text { text }
    }
}

impl From<&str> for ContentPart {
    fn from(text: &str) -> Self {
        text.to_owned().into()
    }
}

impl FromStr for ContentPart {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, to_string};

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
        let prompt = req.prompt_with_context().unwrap();
        println!("{}", prompt);
        assert_eq!(
            prompt,
            "<attachments>\n<doc id=\"1\">\n\"Test document 1.\"\n</doc>\n<doc id=\"2\">\n<meta a=\"b\" key=\"value\" />\n\"Test document 2.\"\n</doc>\n</attachments>\n\nThis is a test prompt."
        );

        let msg = json!(Message {
            role: "user".into(),
            content: prompt.into(),
            name: req.prompter_name,
            ..Default::default()
        });
        assert_eq!(
            to_string(&msg).unwrap(),
            r#"{"content":"<attachments>\n<doc id=\"1\">\n\"Test document 1.\"\n</doc>\n<doc id=\"2\">\n<meta a=\"b\" key=\"value\" />\n\"Test document 2.\"\n</doc>\n</attachments>\n\nThis is a test prompt.","role":"user"}"#
        );
    }

    #[test]
    fn test_content_part() {
        let content = ContentPart::Text {
            text: "Hello, world!".to_string(),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert_eq!(json, r#"{"type":"text","text":"Hello, world!"}"#);

        let ct: ContentPart = serde_json::from_str(&json).unwrap();
        assert_eq!(ct, content);

        let ct = ContentPart::from("Hello, world!");
        assert_eq!(ct, content);

        let content = ContentPart::Image {
            image_url: ImageDetail {
                url: "https://example.com/image.jpg".to_string(),
                detail: Some("high".to_string()),
            },
        };

        let json = serde_json::to_string(&content).unwrap();
        assert_eq!(
            json,
            r#"{"type":"image","image_url":{"url":"https://example.com/image.jpg","detail":"high"}}"#
        );

        let ct: ContentPart = serde_json::from_str(&json).unwrap();
        assert_eq!(ct, content);
        let json = serde_json::to_string(&json!(vec![
            ContentPart::Text {
                text: "What's in this image?".to_string(),
            },
            ContentPart::Image {
                image_url: ImageDetail {
                    url: "https://example.com/image.jpg".to_string(),
                    detail: None,
                },
            }
        ]))
        .unwrap();
        assert_eq!(
            json,
            r#"[{"text":"What's in this image?","type":"text"},{"image_url":{"url":"https://example.com/image.jpg"},"type":"image"}]"#
        );
    }
}
