use anda_db_schema::{AndaDBSchema, FieldEntry, FieldType, Schema, SchemaError};
use candid::Principal;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Represents a state for a user to access the engine.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, AndaDBSchema)]
pub struct User {
    /// The unique identifier for this resource in the Anda DB collection "users".
    pub _id: u64,

    /// The principal ID of the user.
    #[field_type = "Bytes"]
    #[unique]
    pub id: Principal,

    /// The set of features that the user has access to.
    pub features: BTreeSet<String>,

    /// The status of the user, -2: banned, -1: suspended, 0: active.
    pub status: i8,

    /// The subscription tier of the user. 0: free, 1: premium, 2: enterprise.
    pub subscription_tier: u8,

    /// The Unix timestamp when the subscription expires, in milliseconds.
    pub subscription_expiry: u64,

    /// The credit balance of the user.
    pub credit_balance: u64,

    /// The Unix timestamp when the credit expires, in milliseconds.
    pub credit_expiry: u64,

    /// The Unix timestamp when the user was last accessed, in milliseconds.
    pub last_access: u64,

    /// The number of agent requests made by the user.
    pub agent_requests: u64,

    /// The number of tool requests made by the user.
    pub tool_requests: u64,

    /// The number of credit consumed by the user.
    pub credit_consumed: u64,
}

#[derive(Debug)]
pub struct UserState {
    pub(crate) user: RwLock<User>,
}

impl UserState {
    pub fn new(user: User) -> Self {
        Self {
            user: RwLock::new(user),
        }
    }

    /// Returns the user ID.
    pub fn with_user<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&User) -> R,
    {
        f(&self.user.read())
    }

    /// Returns the subscription tier and expiry.
    pub fn subscription(&self) -> (u8, u64) {
        self.with_user(|user| (user.subscription_tier, user.subscription_expiry))
    }

    /// Returns the credit balance and expiry.
    pub fn credit(&self) -> (u64, u64) {
        self.with_user(|user| (user.credit_balance, user.credit_expiry))
    }

    /// Checks if the user has permission to access the engine.
    pub fn has_permission(&self, caller: &Principal, now_ms: u64) -> bool {
        self.with_user(|user| {
            &user.id == caller
                && user.status >= 0
                && (user.subscription_expiry > now_ms
                    || (user.credit_expiry > now_ms && user.credit_balance > 0))
        })
    }

    /// Consumes the credit from the user.
    pub fn consume_credit(&self, credit: u64, now_ms: u64) -> bool {
        let mut user = self.user.write();
        if user.credit_expiry > now_ms && user.credit_balance >= credit {
            user.credit_balance -= credit;
            true
        } else {
            false
        }
    }

    pub fn increment_agent_requests(&self, now_ms: u64) {
        let mut user = self.user.write();
        user.agent_requests = user.agent_requests.saturating_add(1);
        user.last_access = now_ms;
    }

    pub fn increment_tool_requests(&self, now_ms: u64) {
        let mut user = self.user.write();
        user.tool_requests = user.tool_requests.saturating_add(1);
        user.last_access = now_ms;
    }

    /// Topup the credit balance for the user.
    pub fn topup_credit(&self, credit: u64, expiry_ms: u64) {
        let mut user = self.user.write();
        user.credit_balance = user.credit_balance.saturating_add(credit);
        user.credit_expiry = expiry_ms;
    }

    /// Updates the subscription tier and expiry for the user.
    pub fn update_subscription(&self, tier: u8, expiry_ms: u64) {
        let mut user = self.user.write();
        user.subscription_tier = tier;
        user.subscription_expiry = expiry_ms;
    }

    /// Updates the features for the user.
    pub fn update_features(&self, features: BTreeSet<String>) {
        let mut user = self.user.write();
        user.features = features;
    }

    /// Updates the status for the user.
    pub fn update_status(&self, status: i8) {
        let mut user = self.user.write();
        user.status = status;
    }
}
