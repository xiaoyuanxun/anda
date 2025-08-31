use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;

use crate::{AgentOutput, BoxError, ContentPart, FunctionDefinition, Json, Message, Resource};

/// Provides LLM completion capabilities for agents.
pub trait CompletionFeatures: Sized {
    /// Generates a completion based on the given request and optional resources.
    fn completion(
        &self,
        req: CompletionRequest,
        resources: Vec<Resource>,
    ) -> impl Future<Output = Result<AgentOutput, BoxError>> + Send;
}

/// Represents a general completion request that can be sent to a completion model provider.
#[derive(Debug, Clone, Default)]
pub struct CompletionRequest {
    /// The system instructions to be sent to the completion model provider, as the "system" role.
    pub instructions: String,

    /// The name of role, defaulting to "user".
    pub role: Option<String>,

    /// The chat history to be sent to the completion model provider.
    pub chat_history: Vec<Message>,

    /// raw_history is the model specialized history used by anda_engine.
    /// It should be empty in most cases.
    pub raw_history: Vec<Json>,

    /// The documents to embed into the prompt.
    pub documents: Documents,

    /// The prompt to be sent to the completion model provider as role
    /// It can be empty.
    pub prompt: String,

    /// The content parts to be sent to the completion model provider.
    /// It can be empty.
    pub content: Vec<ContentPart>,

    /// The tools to be sent to the completion model provider.
    pub tools: Vec<FunctionDefinition>,

    /// Whether the tool choice is required.
    pub tool_choice_required: bool,

    /// The temperature to be sent to the completion model provider. [0.0, 2.0]
    pub temperature: Option<f64>,

    /// An upper bound for the number of tokens that can be generated for a response,
    pub max_output_tokens: Option<usize>,

    /// An object specifying the JSON format that the model must output.
    pub output_schema: Option<Json>,

    /// The stop sequence to be sent to the completion model provider.
    pub stop: Option<Vec<String>>,
}

impl CompletionRequest {
    /// Adds a document to the request.
    pub fn context(mut self, id: String, text: String) -> Self {
        self.documents.docs.push(Document {
            content: text.into(),
            metadata: BTreeMap::from([("id".to_string(), id.into())]),
        });
        self
    }

    /// Adds multiple documents to the request.
    pub fn append_documents(mut self, docs: Documents) -> Self {
        self.documents.docs.extend(docs.docs);
        self
    }

    /// Adds multiple tools to the request.
    pub fn append_tools(mut self, tools: Vec<FunctionDefinition>) -> Self {
        self.tools.extend(tools);
        self
    }
}

/// A document with metadata and content.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct Document {
    /// The metadata of the document.
    pub metadata: BTreeMap<String, Json>,

    /// The content of the document.
    pub content: Json,
}

impl From<&Resource> for Document {
    fn from(res: &Resource) -> Self {
        let mut metadata = BTreeMap::from([("type".to_string(), "Resource".into())]);
        if let Json::Object(mut val) = json!(res) {
            val.remove("blob");
            metadata.extend(val);
        };

        Self {
            metadata,
            content: Json::Null,
        }
    }
}

/// Collection of knowledge documents.
#[derive(Clone, Debug)]
pub struct Documents {
    /// The tag of the document collection. Defaults to "documents".
    tag: String,
    /// The documents in the collection.
    docs: Vec<Document>,
}

impl Default for Documents {
    fn default() -> Self {
        Self {
            tag: "documents".to_string(),
            docs: Vec::new(),
        }
    }
}

impl Documents {
    /// Creates a new document collection.
    pub fn new(tag: String, docs: Vec<Document>) -> Self {
        Self { tag, docs }
    }

    /// Sets the tag of the document collection.
    pub fn with_tag(self, tag: String) -> Self {
        Self { tag, ..self }
    }

    /// Returns the tag of the document collection.
    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// Converts the document collection to a message.
    pub fn to_message(&self, rfc3339_datetime: &str) -> Option<Message> {
        if self.docs.is_empty() {
            return None;
        }

        Some(Message {
            role: "user".into(),
            content: vec![format!("Current Datetime: {}\n---\n{}", rfc3339_datetime, self).into()],
            name: Some("$system".into()),
            ..Default::default()
        })
    }
}

impl From<Vec<String>> for Documents {
    fn from(texts: Vec<String>) -> Self {
        let mut docs = Vec::new();
        for (i, text) in texts.into_iter().enumerate() {
            docs.push(Document {
                content: text.into(),
                metadata: BTreeMap::from([
                    ("_id".to_string(), i.into()),
                    ("type".to_string(), "Text".into()),
                ]),
            });
        }
        Self {
            docs,
            ..Default::default()
        }
    }
}

impl From<Vec<Document>> for Documents {
    fn from(docs: Vec<Document>) -> Self {
        Self {
            docs,
            ..Default::default()
        }
    }
}

impl std::ops::Deref for Documents {
    type Target = Vec<Document>;

    fn deref(&self) -> &Self::Target {
        &self.docs
    }
}

impl std::ops::DerefMut for Documents {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.docs
    }
}

impl AsRef<Vec<Document>> for Documents {
    fn as_ref(&self) -> &Vec<Document> {
        &self.docs
    }
}

impl std::fmt::Display for Document {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        json!(self).fmt(f)
    }
}

impl std::fmt::Display for Documents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.docs.is_empty() {
            return Ok(());
        }
        writeln!(f, "<{}>", self.tag)?;
        for doc in &self.docs {
            writeln!(f, "{}", doc)?;
        }
        write!(f, "</{}>", self.tag)
    }
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
                    metadata: BTreeMap::from([("_id".to_string(), 1.into())]),
                    content: "Test document 1.".into(),
                },
                Document {
                    metadata: BTreeMap::from([
                        ("_id".to_string(), 2.into()),
                        ("key".to_string(), "value".into()),
                        ("a".to_string(), "b".into()),
                    ]),
                    content: "Test document 2.".into(),
                },
            ]
            .into(),
            ..Default::default()
        };
        // println!("{}", req.documents);

        assert_eq!(
            req.documents.to_string(),
            "<documents>\n{\"content\":\"Test document 1.\",\"metadata\":{\"_id\":1}}\n{\"content\":\"Test document 2.\",\"metadata\":{\"_id\":2,\"a\":\"b\",\"key\":\"value\"}}\n</documents>"
        );
        let documents = req.documents.with_tag("my_docs".to_string());
        assert_eq!(
            documents.to_string(),
            "<my_docs>\n{\"content\":\"Test document 1.\",\"metadata\":{\"_id\":1}}\n{\"content\":\"Test document 2.\",\"metadata\":{\"_id\":2,\"a\":\"b\",\"key\":\"value\"}}\n</my_docs>"
        );
    }
}
