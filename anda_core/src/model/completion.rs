use anda_db_schema::{FieldType, FieldTyped};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::BTreeMap, convert::Infallible, str::FromStr};

use crate::{AgentOutput, BoxError, FunctionDefinition, Json, Resource};

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
    /// The system message to be sent to the completion model provider, as the "system" role.
    pub system: String,

    /// The name of role, defaulting to "user".
    pub role: Option<String>,

    /// The chat history (raw message) to be sent to the completion model provider.
    pub chat_history: Vec<Json>,

    /// The documents to embed into the prompt.
    pub documents: Documents,

    /// The prompt to be sent to the completion model provider as role
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
    pub response_format: Option<Json>,

    /// The stop sequence to be sent to the completion model provider.
    pub stop: Option<Vec<String>>,
}

impl CompletionRequest {
    /// Adds a document to the request.
    pub fn context(mut self, id: String, text: String) -> Self {
        self.documents.0.push(Document {
            content: text.into(),
            metadata: BTreeMap::from([("id".to_string(), id.into())]),
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
}

/// Represents a message send to LLM for completion.
#[derive(Debug, Clone, Default, Deserialize, Serialize, FieldTyped)]
pub struct Message {
    /// Message role: "system", "user", "assistant", "tool".
    pub role: String,

    /// The content of the message, can be text or JSON array.
    pub content: Json,

    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tool call that this message is responding to. If this message is a response to a tool call, this field should be set to the tool call ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// A document with metadata and content.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Document {
    /// The metadata of the document.
    pub metadata: BTreeMap<String, Json>,

    /// The content of the document.
    pub content: Json,
}

impl From<&Resource> for Document {
    fn from(res: &Resource) -> Self {
        let mut metadata = BTreeMap::from([("_type".to_string(), "Resource".into())]);
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
#[derive(Clone, Debug, Default)]
pub struct Documents(pub Vec<Document>);

impl Documents {
    pub fn to_message(&self, rfc3339_datetime: &str) -> Option<Message> {
        if self.0.is_empty() {
            return None;
        }

        Some(Message {
            role: "user".into(),
            content: format!("Current Datetime: {}\n---\n{}", rfc3339_datetime, self).into(),
            name: Some("$system".into()),
            tool_call_id: None,
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
                    ("_type".to_string(), "Text".into()),
                ]),
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
        writeln!(f, "{}", json!(self))
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
        println!("{}", req.documents);
        assert_eq!(
            req.documents.to_string(),
            "<attachments>\n{\"content\":\"Test document 1.\",\"metadata\":{\"_id\":1}}\n{\"content\":\"Test document 2.\",\"metadata\":{\"_id\":2,\"a\":\"b\",\"key\":\"value\"}}\n</attachments>"
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
