//! Enables AI Agent to perform ICP token transfers
//!
//! Provides functionality for transferring tokens between accounts on the Internet Computer Protocol (ICP) network.
//! Supports:
//! - Multiple token types (e.g., ICP, PANDA)
//! - Memo fields for transaction identification
//! - Integration with ICP ledger standards
//! - Atomic transfers with proper error handling

use anda_core::{fix_json_schema, BoxError, FunctionDefinition, StateFeatures, Tool};
use anda_engine::context::BaseCtx;
use num_traits::cast::ToPrimitive;
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

use super::ICPLedgers;

/// Arguments for transferring tokens to an account
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct TransferToArgs {
    /// ICP account address (principal) to receive token, e.g. "77ibd-jp5kr-moeco-kgoar-rro5v-5tng4-krif5-5h2i6-osf2f-2sjtv-kqe"
    pub account: String,
    /// Token symbol, e.g. "ICP"
    pub symbol: String,
    /// Token amount, e.g. 1.1 ICP
    pub amount: f64,
}

/// Implementation of the ICP Ledger Transfer tool
#[derive(Debug, Clone)]
pub struct TransferTool {
    ledgers: Arc<ICPLedgers>,
    schema: Value,
}

impl TransferTool {
    pub const NAME: &'static str = "icp_ledger_transfer";

    pub fn new(ledgers: Arc<ICPLedgers>) -> Self {
        let mut schema = schema_for!(TransferToArgs);
        fix_json_schema(&mut schema);

        TransferTool {
            ledgers,
            schema: json!(schema),
        }
    }
}

/// Implementation of the [`Tool`] trait for TransferTool
/// Enables AI Agent to perform ICP token transfers
impl Tool<BaseCtx> for TransferTool {
    const CONTINUE: bool = true;
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
                "Transfer {} tokens to the specified account on ICP blockchain.",
                tokens.join(", ")
            )
        } else {
            format!(
                "Transfer {} token to the specified account on ICP blockchain.",
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

    async fn call(&self, ctx: BaseCtx, data: Self::Args) -> Result<Self::Output, BoxError> {
        let (ledger, tx) = self.ledgers.transfer(&ctx, ctx.id(), data).await?;
        Ok(format!(
            "Successful, transaction ID: {}, detail: https://www.icexplorer.io/token/details/{}",
            tx.0.to_u64().unwrap_or(0),
            ledger.to_text()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anda_engine::context::mock;
    use candid::{decode_args, encode_args, Nat, Principal};
    use icrc_ledger_types::icrc1::{
        account::principal_to_subaccount,
        transfer::{TransferArg, TransferError},
    };
    use std::collections::BTreeMap;

    #[tokio::test(flavor = "current_thread")]
    async fn test_icp_ledger_transfer() {
        let panda_ledger = Principal::from_text("druyg-tyaaa-aaaaq-aactq-cai").unwrap();
        let ledgers = ICPLedgers {
            ledgers: BTreeMap::from([
                (
                    String::from("ICP"),
                    (
                        Principal::from_text("ryjl3-tyaaa-aaaaa-aaaba-cai").unwrap(),
                        8,
                    ),
                ),
                (String::from("PANDA"), (panda_ledger, 8)),
            ]),
            from_user_subaccount: true,
        };
        let ledgers = Arc::new(ledgers);
        let tool = TransferTool::new(ledgers.clone());
        let definition = tool.definition();
        assert_eq!(definition.name, "icp_ledger_transfer");
        let s = serde_json::to_string_pretty(&definition).unwrap();
        println!("{}", s);
        // {
        //     "name": "icp_ledger_transfer",
        //     "description": "Transfer ICP, PANDA tokens to the specified account on ICP blockchain.",
        //     "parameters": {
        //       "additionalProperties": false,
        //       "description": "Arguments for transferring tokens to an account",
        //       "properties": {
        //         "account": {
        //           "description": "ICP account address (principal) to receive token, e.g. \"77ibd-jp5kr-moeco-kgoar-rro5v-5tng4-krif5-5h2i6-osf2f-2sjtv-kqe\"",
        //           "type": "string"
        //         },
        //         "amount": {
        //           "description": "Token amount, e.g. 1.1 ICP",
        //           "type": "number"
        //         },
        //         "symbol": {
        //           "description": "Token symbol, e.g. \"ICP\"",
        //           "type": "string"
        //         }
        //       },
        //       "required": [
        //         "account",
        //         "amount",
        //         "symbol"
        //       ],
        //       "title": "TransferToArgs",
        //       "type": "object"
        //     },
        //     "strict": true
        // }

        let args = TransferToArgs {
            account: Principal::anonymous().to_string(),
            symbol: "PANDA".to_string(),
            amount: 9999.000012345678,
        };
        let mocker = mock::MockCanisterCaller::new(|canister, method, args| {
            if method == "icrc1_balance_of" {
                return encode_args((Nat::from(999900001234u64),)).unwrap();
            }
            assert_eq!(canister, &panda_ledger);
            assert_eq!(method, "icrc1_transfer");
            let (args,): (TransferArg,) = decode_args(&args).unwrap();
            println!("{:?}", args);
            assert_eq!(
                args.from_subaccount,
                Some(principal_to_subaccount(Principal::anonymous()))
            );
            assert_eq!(args.to.owner, Principal::anonymous());
            assert_eq!(args.amount, Nat::from(999900001234u64));

            let res: Result<Nat, TransferError> = Ok(Nat::from(321u64));
            encode_args((res,)).unwrap()
        });

        let (_, res) = ledgers
            .transfer(&mocker, Principal::anonymous(), args)
            .await
            .unwrap();
        assert_eq!(res, Nat::from(321u64));
    }
}
