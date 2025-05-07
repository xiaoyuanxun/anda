//! Enables AI Agent to perform BNB token transfers
//!
//! Provides functionality for transferring tokens between accounts on the BNB Chain network.
//! Supports:
//! - Multiple token types
//! - Integration with BNB Chain standards
//! - Atomic transfers with proper error handling

use anda_core::{
    BoxError, FunctionDefinition, Resource, Tool, ToolOutput, gen_schema_for,
};
use anda_engine::context::BaseCtx;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use super::BNBLedgers;

/// Arguments for transferring tokens to an account
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct TransferToArgs {
    /// BNB Chain account address to receive token, e.g. "0xA8c4AAE4ce759072D933bD4a51172257622eF128"
    pub account: String,
    /// Token symbol, e.g. "BNB"
    pub symbol: String,
    /// Token amount, e.g. 1.1 BNB
    pub amount: f64,
}

/// Implementation of the BNB Chain Ledger Transfer tool
#[derive(Debug, Clone)]
pub struct TransferTool {
    ledgers: Arc<BNBLedgers>,
    schema: Value,
}

impl TransferTool {
    pub const NAME: &'static str = "bnb_ledger_transfer";

    pub fn new(ledgers: Arc<BNBLedgers>) -> Self {
        let schema = gen_schema_for::<TransferToArgs>();

        TransferTool { ledgers, schema }
    }
}

/// Implementation of the [`Tool`] trait for TransferTool
/// Enables AI Agent to perform BNB token transfers
impl Tool<BaseCtx> for TransferTool {
    type Args = TransferToArgs;
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
        if tokens.len() > 1 {
            format!(
                "Transfer {} tokens to the specified account on BNB Chain blockchain.",
                tokens.join(", ")
            )
        } else {
            format!(
                "Transfer {} token to the specified account on BNB Chain blockchain.",
                tokens[0]
            )
        }
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
        let (ledger, tx) = self.ledgers.transfer(ctx, data).await?;
        Ok(ToolOutput::new(format!(
            "Successful transfer, receipient address: {}, detail: https://www.bscscan.com/tx/{}",
            ledger,
            tx
        )))
    }
}
