use anda_core::{
    BoxError, FunctionDefinition, Resource, StateFeatures, Tool, ToolOutput, UpdateVersion, Value,
    gen_schema_for,
};
use candid::{CandidType, Principal};
use ic_cose_types::ANONYMOUS;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, sync::Arc};
use structured_logger::unix_ms;

use super::Management;
use crate::context::BaseCtx;

/// Represents a state for a user to access the engine.
#[derive(Debug, Clone, CandidType, Deserialize, Serialize, PartialEq, Eq)]
pub struct UserState {
    pub user: Principal,

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

    /// The set of features that the user has access to.
    pub features: BTreeSet<String>,

    /// The Unix timestamp when the user was last accessed, in milliseconds.
    pub last_access: u64,

    /// The number of agent requests made by the user.
    pub agent_requests: u64,

    /// The number of tool requests made by the user.
    pub tool_requests: u64,

    /// The number of credit consumed by the user.
    pub credit_consumed: u64,

    pub version: Option<UpdateVersion>,
}

#[derive(Debug, Clone)]
pub struct UserStateWrapper {
    pub(crate) state: UserState,
}

impl UserStateWrapper {
    pub fn new(user: Principal) -> Self {
        Self {
            state: UserState {
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
            },
        }
    }

    /// Returns the user ID.
    pub fn user(&self) -> &Principal {
        &self.state.user
    }

    /// Returns the subscription tier and expiry.
    pub fn subscription(&self) -> (u8, u64) {
        (self.state.subscription_tier, self.state.subscription_expiry)
    }

    /// Returns the credit balance and expiry.
    pub fn credit(&self) -> (u64, u64) {
        (self.state.credit_balance, self.state.credit_expiry)
    }

    /// Returns the features of the user.
    pub fn features(&self) -> &BTreeSet<String> {
        &self.state.features
    }

    /// Checks if the user has permission to access the engine.
    pub fn has_permission(&self, caller: &Principal, now_ms: u64) -> bool {
        &self.state.user == caller
            && self.state.status >= 0
            && (self.state.subscription_expiry > now_ms
                || (self.state.credit_expiry > now_ms && self.state.credit_balance > 0))
    }

    /// Consumes the credit from the user.
    pub fn consume_credit(&mut self, credit: u64, now_ms: u64) -> bool {
        if self.state.credit_expiry > now_ms && self.state.credit_balance >= credit {
            self.state.credit_balance -= credit;
            true
        } else {
            false
        }
    }

    pub(crate) fn increment_agent_requests(&mut self, now_ms: u64) {
        self.state.agent_requests = self.state.agent_requests.saturating_add(1);
        self.state.last_access = now_ms;
    }

    pub(crate) fn increment_tool_requests(&mut self, now_ms: u64) {
        self.state.tool_requests = self.state.tool_requests.saturating_add(1);
        self.state.last_access = now_ms;
    }

    /// Topup the credit balance for the user.
    pub(crate) fn topup_credit(&mut self, credit: u64, expiry_ms: u64) {
        self.state.credit_balance = self.state.credit_balance.saturating_add(credit);
        self.state.credit_expiry = expiry_ms;
    }

    /// Updates the subscription tier and expiry for the user.
    pub(crate) fn update_subscription(&mut self, tier: u8, expiry_ms: u64) {
        self.state.subscription_tier = tier;
        self.state.subscription_expiry = expiry_ms;
    }

    /// Updates the features for the user.
    pub(crate) fn update_features(&mut self, features: BTreeSet<String>) {
        self.state.features = features;
    }

    /// Updates the status for the user.
    pub(crate) fn update_status(&mut self, status: i8) {
        self.state.status = status;
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
                    let mut w = self.management.load_user_state(&user).await?;
                    w.topup_credit(credit, expiry);
                    let res = self.management.save_user_state(w.state.clone()).await?;
                    w.state.version = Some(res);
                    Ok(ToolOutput::new(Some(w.state)))
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
                    let mut w = self.management.load_user_state(&user).await?;
                    w.update_subscription(tier, expiry);
                    let res = self.management.save_user_state(w.state.clone()).await?;
                    w.state.version = Some(res);
                    Ok(ToolOutput::new(Some(w.state)))
                } else {
                    Err("caller does not have permission".into())
                }
            }

            UserStateToolArgs::UpdateFeatures { user, features } => {
                let user = Principal::from_text(&user)?;

                if self.management.is_manager(&caller) {
                    let mut w = self.management.load_user_state(&user).await?;
                    w.update_features(features);
                    let res = self.management.save_user_state(w.state.clone()).await?;
                    w.state.version = Some(res);
                    Ok(ToolOutput::new(Some(w.state)))
                } else {
                    Err("caller does not have permission".into())
                }
            }

            UserStateToolArgs::UpdateStatus { user, status } => {
                let user = Principal::from_text(&user)?;
                if !(-2..=0).contains(&status) {
                    return Err(format!("invalid status {status}").into());
                }

                if self.management.is_manager(&caller) {
                    let mut w = self.management.load_user_state(&user).await?;
                    w.update_status(status);
                    let res = self.management.save_user_state(w.state.clone()).await?;
                    w.state.version = Some(res);
                    Ok(ToolOutput::new(Some(w.state)))
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
    use crate::{
        engine::EngineBuilder,
        management::{ManagementBuilder, Visibility},
    };

    #[tokio::test]
    async fn test_user_state_tool() {
        let engine = EngineBuilder::new();
        let ctx = engine.mock_ctx();
        let management =
            Arc::new(ManagementBuilder::new(Visibility::Private, ctx.id()).build(&ctx.base));

        let tool = UserStateTool::new(management);
        let definition = tool.definition();
        println!("{}", serde_json::to_string_pretty(&definition).unwrap());
    }
}
