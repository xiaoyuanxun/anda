use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, convert::Infallible, str::FromStr};

use super::{AgentOutput, FunctionDefinition, Knowledge, Resource, Value};
use crate::BoxError;

/// Provides LLM completion capabilities for agents.
pub trait CompletionFeatures: Sized {
    /// Generates a completion based on the given request and optional resources.
    fn completion(
        &self,
        req: CompletionRequest,
        resources: Option<Vec<Resource>>,
    ) -> impl Future<Output = Result<AgentOutput, BoxError>> + Send;
}

/// Represents a general completion request that can be sent to a completion model provider.
#[derive(Debug, Clone, Default)]
pub struct CompletionRequest {
    /// The system message to be sent to the completion model provider, as the "system" role.
    pub system: Option<String>,

    /// The name of system role.
    pub system_name: Option<String>,

    /// The chat history (raw message) to be sent to the completion model provider.
    pub chat_history: Vec<Value>,

    /// The documents to embed into the prompt.
    pub documents: Documents,

    /// The prompt to be sent to the completion model provider as "user" role
    /// It can be empty.
    pub prompt: String,

    /// The name of the prompter.
    pub prompter_name: Option<String>,

    /// The content parts to be sent to the completion model provider.
    /// prompt will be ignored if content_parts is not empty.
    pub content_parts: Vec<ContentPart>,

    /// The tools to be sent to the completion model provider.
    pub tools: Vec<FunctionDefinition>,

    /// Whether the tool choice is required.
    pub tool_choice_required: bool,

    /// The temperature to be sent to the completion model provider.
    pub temperature: Option<f64>,

    /// The max tokens to be sent to the completion model provider.
    pub max_tokens: Option<usize>,

    /// An object specifying the JSON format that the model must output.
    /// https://platform.openai.com/docs/guides/structured-outputs
    /// The format can be one of the following:
    /// `{ "type": "json_object" }`
    /// `{ "type": "json_schema", "json_schema": {...} }`
    pub response_format: Option<Value>,

    /// The stop sequence to be sent to the completion model provider.
    pub stop: Option<Vec<String>>,
}

impl CompletionRequest {
    /// Adds a document to the request.
    pub fn context(mut self, id: String, text: String) -> Self {
        self.documents.0.push(Document {
            id,
            text,
            ..Default::default()
        });
        self
    }

    /// Adds multiple documents to the request.
    pub fn append_documents(mut self, docs: Documents) -> Self {
        self.documents.0.extend(docs.0);
        self
    }

    /// Adds multiple tools to the request.
    pub fn append_tools(mut self, tools: Vec<FunctionDefinition>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Returns the prompt with context if available.
    pub fn prompt_with_context(&self) -> Option<String> {
        if self.documents.0.is_empty() && self.prompt.is_empty() {
            return None;
        }

        if self.documents.0.is_empty() {
            Some(self.prompt.clone())
        } else if self.prompt.is_empty() {
            Some(format!("{}", self.documents))
        } else {
            Some(format!("{}\n---\n{}", self.documents, self.prompt))
        }
    }
}

/// Represents a message send to LLM for completion.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Message {
    /// Message role: "system", "user", "assistant", "tool".
    pub role: String,

    /// The content of the message, can be text or JSON array.
    pub content: Value,

    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tool call that this message is responding to. If this message is a response to a tool call, this field should be set to the tool call ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Knowledge document with text and additional props.
#[derive(Clone, Debug, Default)]
pub struct Document {
    /// The unique identifier for the document in local.
    pub id: String,

    /// The text content of the document.
    pub text: String,

    /// Additional properties for the document.
    pub metadata: BTreeMap<String, String>,
}

impl From<Knowledge> for Document {
    fn from(doc: Knowledge) -> Self {
        let mut metadata = BTreeMap::new();
        metadata.insert("user".to_string(), doc.user);

        for (k, v) in doc.meta {
            if let Ok(v) = serde_json::to_string(&v) {
                metadata.insert(k, v);
            }
        }

        Document {
            id: doc.id,
            text: doc.text,
            metadata,
        }
    }
}

/// Collection of knowledge documents.
#[derive(Clone, Debug, Default)]
pub struct Documents(pub Vec<Document>);

impl From<Vec<String>> for Documents {
    fn from(texts: Vec<String>) -> Self {
        let mut docs = Vec::new();
        for (i, text) in texts.into_iter().enumerate() {
            docs.push(Document {
                id: format!("doc_{}", i),
                text,
                metadata: BTreeMap::new(),
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
        if !self.metadata.is_empty() {
            write!(f, "<meta ")?;
            for (k, v) in &self.metadata {
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

/// OpenAI style content part for the completion request.
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
                    metadata: BTreeMap::new(),
                },
                Document {
                    id: "2".to_string(),
                    text: "Test document 2.".to_string(),
                    metadata: BTreeMap::from([
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
            "<attachments>\n<doc id=\"1\">\n\"Test document 1.\"\n</doc>\n<doc id=\"2\">\n<meta a=\"b\" key=\"value\" />\n\"Test document 2.\"\n</doc>\n</attachments>\n---\nThis is a test prompt."
        );

        let msg = json!(Message {
            role: "user".into(),
            content: prompt.into(),
            name: req.prompter_name,
            ..Default::default()
        });
        assert_eq!(
            to_string(&msg).unwrap(),
            r#"{"content":"<attachments>\n<doc id=\"1\">\n\"Test document 1.\"\n</doc>\n<doc id=\"2\">\n<meta a=\"b\" key=\"value\" />\n\"Test document 2.\"\n</doc>\n</attachments>\n---\nThis is a test prompt.","role":"user"}"#
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
