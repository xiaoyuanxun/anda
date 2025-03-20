use candid::CandidType;
use serde::{Deserialize, Serialize};

use super::{ByteArrayB64, ByteBufB64};

/// Represents a resource that can be sent to agents or tools.
#[derive(Debug, Default, CandidType, Serialize, Deserialize, Clone)]
pub struct Resource {
    /// A tag that identifies the type of this resource.
    pub tag: String,

    /// The URI of this resource.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,

    /// A human-readable name for this resource.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// A description of what this resource represents.
    /// This can be used by clients to improve the LLM's understanding of available resources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// MIME type, https://developer.mozilla.org/zh-CN/docs/Web/HTTP/MIME_types/Common_types
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,

    /// The binary data of this resource.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<ByteBufB64>,

    /// The size of the resource in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<usize>,

    /// The SHA3-256 hash of the resource.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<ByteArrayB64<32>>,
}

/// Extracts resources with the given tags from the list of resources.
pub fn select_resources(resources: &mut Vec<Resource>, tags: &[&str]) -> Option<Vec<Resource>> {
    if tags.is_empty() {
        return None;
    }

    if tags.first() == Some(&"*") {
        return Some(std::mem::take(resources));
    }

    // nightly feature:
    // {
    //     let res: Vec<Resource> = resources
    //         .extract_if(.., |r| tags.contains(&r.tag.as_str()))
    //         .collect();
    //     if res.is_empty() { None } else { Some(res) }
    // }

    {
        let mut res = Vec::new();
        let mut i = 0;

        while i < resources.len() {
            if tags.contains(&resources[i].tag.as_str()) {
                res.push(resources.remove(i));
            } else {
                i += 1;
            }
        }
        if res.is_empty() { None } else { Some(res) }
    }
}
