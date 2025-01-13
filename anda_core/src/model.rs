use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, fmt::Write, future::Future};

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
}

/// Represents a tool call response with it's ID, function name, and arguments
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: String,

    /// The result of the tool call, auto processed by agents engine, if available
    pub result: Option<String>,
}

/// Represents a message in the agent's conversation history
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct MessageInput {
    /// Message role: "developer", "system", "user", "assistant", "tool"
    pub role: String,

    /// The content of the message
    pub content: String,

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
    pub parameters: serde_json::Value,

    /// Whether to enable strict schema adherence when generating the function call. If set to true, the model will follow the exact schema defined in the parameters field. Only a subset of JSON Schema is supported when strict is true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Knowledge document with text and metadata
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    #[serde(flatten)]
    pub additional_props: BTreeMap<String, String>,
}

impl std::fmt::Display for Document {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<file id={:?}>\n", self.id)?;
        if !self.additional_props.is_empty() {
            write!(f, "<meta ")?;
            for (k, v) in &self.additional_props {
                write!(f, "{}={:?} ", k, v)?;
            }
            write!(f, "/>\n")?;
        }
        write!(f, "{:?}\n</file>\n", self.text)
    }
}

/// Struct representing a general completion request that can be sent to a completion model provider.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CompletionRequest {
    /// The prompt to be sent to the completion model provider as "developer" or "system" role
    pub prompt: String,

    /// The preamble to be sent to the completion model provider
    pub preamble: Option<String>,

    /// The chat history to be sent to the completion model provider
    pub chat_history: Vec<MessageInput>,

    /// The documents to embed into the prompt
    pub documents: Vec<Document>,

    /// The tools to be sent to the completion model provider
    pub tools: Vec<FunctionDefinition>,

    /// The temperature to be sent to the completion model provider
    pub temperature: Option<f64>,

    /// The max tokens to be sent to the completion model provider
    pub max_tokens: Option<u64>,

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
    pub fn prompt_with_context(&self) -> String {
        if !self.documents.is_empty() {
            let mut w = String::new();
            w.push_str("<attachments>\n");
            for doc in &self.documents {
                if w.write_fmt(format_args!("{doc}")).is_err() {
                    return self.prompt.clone();
                }
            }
            w.push_str("</attachments>\n\n");
            w.push_str(&self.prompt);
            w
        } else {
            self.prompt.clone()
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
pub trait CompletionFeatures<Err>: Sized {
    /// Generates a completion based on the given prompt and context
    ///
    /// # Arguments
    /// * `prompt` - The input prompt for the completion
    /// * `json_output` - Whether to force JSON output format
    /// * `chat_history` - Conversation history as context
    /// * `tools` - Available functions the model can call
    fn completion(
        &self,
        req: CompletionRequest,
    ) -> impl Future<Output = Result<AgentOutput, Err>> + Send;
}

/// Provides text embedding capabilities for agents
pub trait EmbeddingFeatures<Err>: Sized {
    /// The number of dimensions in the embedding vector.
    fn ndims(&self) -> usize;

    /// Generates embeddings for multiple texts in a batch
    /// Returns a vector of Embedding structs in the same order as input texts
    fn embed(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> impl Future<Output = Result<Vec<Embedding>, Err>> + Send;

    /// Generates a single embedding for a query text
    /// Optimized for single text embedding generation
    fn embed_query(&self, text: &str) -> impl Future<Output = Result<Embedding, Err>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_with_context() {
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
            ],
            ..Default::default()
        };
        let prompt = req.prompt_with_context();
        println!("{}", prompt);
        assert_eq!(
            prompt,
            "<attachments>\n<file id=\"1\">\n\"Test document 1.\"\n</file>\n<file id=\"2\">\n<meta a=\"b\" key=\"value\" />\n\"Test document 2.\"\n</file>\n</attachments>\n\nThis is a test prompt."
        );
    }
}
