use anda_core::{
    BoxError, FunctionDefinition, Resource, StateFeatures, Tool, ToolOutput, UpdateVersion, Value,
    gen_schema_for,
};
use candid::Principal;
use ic_cose_types::ANONYMOUS;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, sync::Arc};
use structured_logger::unix_ms;

use super::Management;
use crate::context::BaseCtx;

/// Represents a state for a user to access the engine.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct UserState {
    pub(crate) user: Principal,

    /// The status of the user, -2: banned, -1: suspended, 0: active.
    pub(crate) status: i8,

    /// The subscription tier of the user. 0: free, 1: premium, 2: enterprise.
    pub(crate) subscription_tier: u8,

    /// The Unix timestamp when the subscription expires, in milliseconds.
    pub(crate) subscription_expiry: u64,

    /// The credit balance of the user.
    pub(crate) credit_balance: u64,

    /// The Unix timestamp when the credit expires, in milliseconds.
    pub(crate) credit_expiry: u64,

    /// The set of features that the user has access to.
    pub(crate) features: BTreeSet<String>,

    /// The Unix timestamp when the user was last accessed, in milliseconds.
    pub(crate) last_access: u64,

    /// The number of agent requests made by the user.
    pub(crate) agent_requests: u64,

    /// The number of tool requests made by the user.
    pub(crate) tool_requests: u64,

    /// The number of credit consumed by the user.
    pub(crate) credit_consumed: u64,

    pub(crate) version: Option<UpdateVersion>,
}

impl UserState {
    pub fn new(user: Principal) -> Self {
        Self {
            user,
            status: 0,
            subscription_tier: 0,
            subscription_expiry: 0,
            credit_balance: 0,
            credit_expiry: 0,
            features: BTreeSet::new(),
            last_access: 0,
            agent_requests: 0,
            tool_requests: 0,
            credit_consumed: 0,
            version: None,
        }
    }

    /// Returns the user ID.
    pub fn user(&self) -> &Principal {
        &self.user
    }

    /// Returns the subscription tier and expiry.
    pub fn subscription(&self) -> (u8, u64) {
        (self.subscription_tier, self.subscription_expiry)
    }

    /// Returns the credit balance and expiry.
    pub fn credit(&self) -> (u64, u64) {
        (self.credit_balance, self.credit_expiry)
    }

    /// Returns the features of the user.
    pub fn features(&self) -> &BTreeSet<String> {
        &self.features
    }

    /// Checks if the user has permission to access the engine.
    pub fn has_permission(&self, caller: &Principal, now_ms: u64) -> bool {
        &self.user == caller
            && self.status >= 0
            && (self.subscription_expiry > now_ms
                || (self.credit_expiry > now_ms && self.credit_balance > 0))
    }

    /// Consumes the credit from the user.
    pub fn consume_credit(&mut self, credit: u64, now_ms: u64) -> bool {
        if self.credit_expiry > now_ms && self.credit_balance >= credit {
            self.credit_balance -= credit;
            true
        } else {
            false
        }
    }

    /// Topup the credit balance for the user.
    pub(crate) fn topup_credit(&mut self, credit: u64, expiry_ms: u64) {
        self.credit_balance = self.credit_balance.saturating_add(credit);
        self.credit_expiry = expiry_ms;
    }

    /// Updates the subscription tier and expiry for the user.
    pub(crate) fn update_subscription(&mut self, tier: u8, expiry_ms: u64) {
        self.subscription_tier = tier;
        self.subscription_expiry = expiry_ms;
    }

    /// Updates the features for the user.
    pub(crate) fn update_features(&mut self, features: BTreeSet<String>) {
        self.features = features;
    }

    /// Updates the status for the user.
    pub(crate) fn update_status(&mut self, status: i8) {
        self.status = status;
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum UserStateToolArgs {
    GetUserState {
        user: String,
    },
    TopupCredit {
        user: String,
        credit: u64,
        expiry: u64,
    },
    UpdateSubscription {
        user: String,
        tier: u8,
        expiry: u64,
    },
    UpdateFeatures {
        user: String,
        features: BTreeSet<String>,
    },
    UpdateStatus {
        user: String,
        status: i8,
    },
    DeleteUserState {
        user: String,
    },
}

pub struct UserStateTool {
    management: Arc<Management>,
    schema: Value,
}

impl UserStateTool {
    pub const NAME: &'static str = "sys_user_state";

    pub fn new(management: Arc<Management>) -> Self {
        let schema = gen_schema_for::<UserStateToolArgs>();
        Self { management, schema }
    }

    fn min_expiry() -> u64 {
        unix_ms() + 1000 * 60 * 60 * 24
    }
}

impl Tool<BaseCtx> for UserStateTool {
    type Args = UserStateToolArgs;
    type Output = Option<UserState>;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        "Manages user state.".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: self.schema.clone(),
            strict: Some(true),
        }
    }

    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        resources: Option<Vec<Resource>>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        if resources.is_some() {
            return Err("resources are not supported".into());
        }

        let caller = ctx.caller();
        if caller == ANONYMOUS {
            return Err("anonymous user is not allowed".into());
        }

        match args {
            UserStateToolArgs::GetUserState { user } => {
                let user = Principal::from_text(&user)?;

                if self.management.is_manager(&caller) || user == caller {
                    let state = self.management.get_user_state(&user).await?;
                    Ok(ToolOutput::new(Some(state)))
                } else {
                    Err("caller does not have permission".into())
                }
            }

            UserStateToolArgs::TopupCredit {
                user,
                credit,
                expiry,
            } => {
                let user = Principal::from_text(&user)?;
                if expiry < Self::min_expiry() {
                    return Err("expiry is too short".into());
                }

                if self.management.is_manager(&caller) {
                    let mut state = self.management.load_user_state(&user).await?;
                    state.topup_credit(credit, expiry);
                    let res = self.management.save_user_state(state.clone()).await?;
                    state.version = Some(res);
                    Ok(ToolOutput::new(Some(state)))
                } else {
                    Err("caller does not have permission".into())
                }
            }

            UserStateToolArgs::UpdateSubscription { user, tier, expiry } => {
                let user = Principal::from_text(&user)?;
                if expiry < Self::min_expiry() {
                    return Err("expiry is too short".into());
                }

                if tier > 10 {
                    return Err(format!("tier {tier} is too high").into());
                }

                if self.management.is_manager(&caller) {
                    let mut state = self.management.load_user_state(&user).await?;
                    state.update_subscription(tier, expiry);
                    let res = self.management.save_user_state(state.clone()).await?;
                    state.version = Some(res);
                    Ok(ToolOutput::new(Some(state)))
                } else {
                    Err("caller does not have permission".into())
                }
            }

            UserStateToolArgs::UpdateFeatures { user, features } => {
                let user = Principal::from_text(&user)?;

                if self.management.is_manager(&caller) {
                    let mut state = self.management.load_user_state(&user).await?;
                    state.update_features(features);
                    let res = self.management.save_user_state(state.clone()).await?;
                    state.version = Some(res);
                    Ok(ToolOutput::new(Some(state)))
                } else {
                    Err("caller does not have permission".into())
                }
            }

            UserStateToolArgs::UpdateStatus { user, status } => {
                let user = Principal::from_text(&user)?;
                if status < -2 || status > 0 {
                    return Err(format!("invalid status {status}").into());
                }

                if self.management.is_manager(&caller) {
                    let mut state = self.management.load_user_state(&user).await?;
                    state.update_status(status);
                    let res = self.management.save_user_state(state.clone()).await?;
                    state.version = Some(res);
                    Ok(ToolOutput::new(Some(state)))
                } else {
                    Err("caller does not have permission".into())
                }
            }

            UserStateToolArgs::DeleteUserState { user } => {
                let user = Principal::from_text(&user)?;

                if self.management.is_manager(&caller) {
                    self.management.delete_user_state(&user).await?;
                    Ok(ToolOutput::new(None))
                } else {
                    Err("caller does not have permission".into())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineBuilder;

    #[tokio::test]
    async fn test_user_state_tool() {
        let engine = EngineBuilder::new();
        let ctx = engine.mock_ctx();
        let management = Arc::new(Management::new(&ctx.base, ctx.id()));

        let tool = UserStateTool::new(management);
        let definition = tool.definition();
        println!("{}", serde_json::to_string_pretty(&definition).unwrap());
    }
}
