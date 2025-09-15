use anda_core::{ContentPart, Resource};
use anda_db_schema::{AndaDBSchema, FieldEntry, FieldType, Schema, SchemaError};
use anda_engine::context::EngineCard;
use candid::Principal;
use ic_auth_types::Xid;
use isolang::Language;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    str::FromStr,
};
use url::Url;

#[derive(Debug, Default, Clone, Deserialize, Serialize, AndaDBSchema)]
pub struct Thread {
    /// The unique identifier for this resource in the Anda DB collection "threads".
    pub _id: u64,

    #[field_type = "Bytes"]
    #[unique]
    pub id: Xid,

    /// The name of the thread.
    pub name: String,

    /// The primary language of the thread.
    pub language: String,

    pub image: String,

    pub tags: Vec<String>,

    /// The description of the thread.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    // private, protected, public
    #[field_type = "Text"]
    pub visibility: ThreadVisibility,

    /// The status of the thread: active, suspended, banned.
    #[field_type = "Text"]
    pub status: ThreadStatus,

    pub max_participants: u64,

    #[field_type = "Array<Bytes>"]
    pub controllers: BTreeSet<Principal>,

    #[field_type = "Array<Bytes>"]
    pub managers: BTreeSet<Principal>,

    #[field_type = "Map<Bytes, U64>"]
    pub participants: BTreeMap<Principal, u64>,

    #[field_type = "Array<Json>"]
    pub agents: Vec<EngineCard>,

    /// The timestamp when the thread was created.
    pub created_at: u64,

    /// The timestamp when the thread was last updated.
    pub updated_at: u64,
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct ThreadInfo {
    pub _id: u64,
    pub id: Xid,
    pub name: String,
    pub language: String,
    pub image: String,
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub visibility: ThreadVisibility,
    pub status: ThreadStatus,
    pub max_participants: u64,
    pub controllers: BTreeSet<Principal>,
    pub created_at: u64,
    pub updated_at: u64,
}

impl Thread {
    pub fn has_permission(&self, user: &Principal, permission: ThreadPermission) -> bool {
        match permission {
            ThreadPermission::Read => {
                self.status == ThreadStatus::Active
                    && (self.visibility == ThreadVisibility::Public
                        || self.participants.contains_key(user))
            }
            ThreadPermission::Write => {
                self.status == ThreadStatus::Active && self.participants.contains_key(user)
            }
            ThreadPermission::Manage => {
                self.controllers.contains(user) || self.managers.contains(user)
            }
            ThreadPermission::Control => self.controllers.contains(user),
        }
    }

    pub fn to_state(&self) -> ThreadState {
        ThreadState {
            visibility: self.visibility,
            status: self.status,
            updated_at: self.updated_at,
            participants: self.participants.len() as u64,
            max_participants: self.max_participants,
            latest_message_by: None,
            latest_message_id: 0,
            latest_message_at: 0,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ThreadPermission {
    Read,
    Write,
    Manage,
    Control,
}

#[derive(Debug, Default, Clone, Deserialize, Serialize, PartialEq, Eq, JsonSchema)]
pub struct UpdateThreadInfo {
    /// The name of the thread.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// The primary language of the thread.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,

    /// The description of the thread.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility: Option<ThreadVisibility>,
}

impl UpdateThreadInfo {
    pub fn validate_and_normalize(&mut self) -> Result<(), String> {
        if let Some(name) = &mut self.name {
            *name = name.trim().to_string();
            if name.is_empty() {
                return Err("Thread name cannot be empty".to_string());
            }
            if name.len() > 128 {
                return Err("Thread name is too long".to_string());
            }
        }
        if let Some(language) = &mut self.language {
            let lang = Language::from_str(language)
                .map_err(|err| format!("Invalid language code: {}", err))?;
            *language = lang.to_name().to_string();
        }
        if let Some(image) = &self.image {
            if image.len() > 256 {
                return Err("Thread image URL is too long".to_string());
            }
            if let Err(e) = Url::parse(image) {
                return Err(format!("Thread image URL is invalid: {}", e));
            }
        }
        // TODO: more validation
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ThreadState {
    pub visibility: ThreadVisibility,
    pub status: ThreadStatus,
    pub updated_at: u64,
    pub participants: u64,
    pub max_participants: u64,
    pub latest_message_by: Option<Principal>,
    pub latest_message_id: u64,
    pub latest_message_at: u64,
}

#[derive(Copy, Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ThreadVisibility {
    #[default]
    Private,
    Protected,
    Public,
}

impl fmt::Display for ThreadVisibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ThreadVisibility::Private => "private",
            ThreadVisibility::Protected => "protected",
            ThreadVisibility::Public => "public",
        };
        write!(f, "{}", s)
    }
}

#[derive(Copy, Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ThreadStatus {
    #[default]
    Active,
    Suspended,
    Unavailable,
}

impl fmt::Display for ThreadStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ThreadStatus::Active => "active",
            ThreadStatus::Suspended => "suspended",
            ThreadStatus::Unavailable => "unavailable",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, AndaDBSchema)]
pub struct Message {
    #[serde(default)]
    pub _id: u64,

    /// Message role: "system", "user", "assistant".
    pub role: String,

    /// The content of the message
    #[field_type = "Array<Json>"]
    pub content: Vec<ContentPart>,

    /// The resources associated with the message.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<Resource>,

    /// The user ID of the message sender.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[field_type = "Option<Bytes>"]
    pub user: Option<Principal>,

    /// The timestamp of the message.
    #[serde(default)]
    pub timestamp: u64,

    #[serde(default)]
    pub reply_to: u64, // 0 means not a reply
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_status() {
        let thread = Thread {
            _id: 0,
            id: Xid::new(),
            name: "Test Thread".to_string(),
            ..Default::default()
        };
        let rt = ThreadStatus::Active.to_string();
        println!("{}", rt);
        assert_eq!(rt, "active");
        let rt = serde_json::to_string(&thread).unwrap();
        println!("{}", rt);
        assert!(rt.contains(r#","status":"active","#));

        let t: Thread = serde_json::from_str(&rt).unwrap();
        assert_eq!(t.status, ThreadStatus::Active);
    }
}
