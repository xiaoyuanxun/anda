use serde::{Deserialize, Serialize};
use serde_bytes::ByteArray;
use std::{ops::Deref, str::FromStr};

/// Represents a unique identifier with 12 bytes.
/// Based on the xid. See: https://github.com/rs/xid
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Xid(pub ByteArray<12>);

pub const EMPTY_THREAD: Xid = Xid(ByteArray::new([0; 12]));

impl From<Xid> for xid::Id {
    fn from(thread: Xid) -> Self {
        xid::Id(*thread.0)
    }
}

impl From<xid::Id> for Xid {
    fn from(id: xid::Id) -> Self {
        Self(id.0.into())
    }
}

impl FromStr for Xid {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let id = xid::Id::from_str(s).map_err(|err| format!("{err:?}"))?;
        Ok(Self(id.0.into()))
    }
}

impl std::fmt::Display for Xid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        xid::Id(*self.0).fmt(f)
    }
}

impl AsRef<[u8; 12]> for Xid {
    fn as_ref(&self) -> &[u8; 12] {
        self.0.as_ref()
    }
}

impl Deref for Xid {
    type Target = [u8; 12];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Default for Xid {
    fn default() -> Self {
        EMPTY_THREAD
    }
}

impl Xid {
    pub fn new() -> Self {
        Self(xid::new().0.into())
    }

    /// Returns the xid of the thread.
    pub fn xid(&self) -> xid::Id {
        xid::Id(*self.0)
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_ref()
    }

    pub fn is_empty(&self) -> bool {
        self == &EMPTY_THREAD
    }
}
