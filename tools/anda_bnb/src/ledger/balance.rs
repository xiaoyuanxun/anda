//! Enables AI Agent to query the balance of an account for a BNB token
//!
//! This module provides functionality for querying account balances on the BNB Chain network.
//! It implements the [`Tool`] trait to enable AI agents to interact with BNB Chain ledgers.

use anda_core::{BoxError, FunctionDefinition, Resource, Tool, ToolOutput, gen_schema_for};
use anda_engine::context::BaseCtx;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use super::BNBLedgers;

/// Arguments for the balance of an account for a token
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct BalanceOfArgs {
    /// account address
    pub account: String,
    /// Token symbol, e.g. "cake"
    pub symbol: String,
}

/// BNB Chain Ledger BalanceOf tool implementation
#[derive(Debug, Clone)]
pub struct BalanceOfTool {
    ledgers: Arc<BNBLedgers>,
    schema: Value,
}

impl BalanceOfTool {
    pub const NAME: &'static str = "bnb_ledger_balance_of";
    /// Creates a new BalanceOfTool instance
    pub fn new(ledgers: Arc<BNBLedgers>) -> Self {
        let schema = gen_schema_for::<BalanceOfArgs>();

        BalanceOfTool {
            ledgers,
            schema: json!(schema),
        }
    }
}

/// Implementation of the [`Tool`]` trait for BalanceOfTool
/// Enables AI Agent to query the balance of an account for a BNB token
impl Tool<BaseCtx> for BalanceOfTool {
    type Args = BalanceOfArgs;
    type Output = String;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        let tokens = self
            .ledgers
            .ledgers
            .keys()
            .map(|k| k.as_str())
            .collect::<Vec<_>>();
        format!(
            "Query the balance of the specified account on BNB Chain blockchain for the following tokens: {}",
            tokens.join(", ")
        )
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
        data: Self::Args,
        _resources: Option<Vec<Resource>>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        let token_symbol = data.symbol.clone();
        let (address, amount) = self.ledgers.balance_of(ctx, data).await?;
        Ok(ToolOutput::new(format!(
            "Successful {} balance query, user address: {}, balance {}",
            token_symbol,
            address.to_string(),
            amount
        )))

    }
}
