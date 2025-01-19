use anda_core::{BoxError, CanisterCaller, FunctionDefinition, Tool};
use anda_engine::context::BaseCtx;
use candid::{Nat, Principal};
use icrc_ledger_types::icrc1::{
    account::{principal_to_subaccount, Account},
    transfer::{Memo, TransferArg, TransferError},
};
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;
use serde_json::{json, Value};

const MAX_MEMO_LEN: usize = 32;

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct TransferToArgs {
    account: String,
    amount: f64,
    memo: Option<String>, // should less than 32 bytes
}

#[derive(Debug, Clone)]
pub struct ICPLedgerTransfer {
    canister: Principal,
    schema: Value,
    symbol: String,
    decimals: u8,
    from_user_subaccount: bool,
}

impl ICPLedgerTransfer {
    pub fn new(
        canister: Principal,
        symbol: String, // token symbol, e.g. "ICP"
        decimals: u8,   // token decimals, e.g. 8
        from_user_subaccount: bool,
    ) -> ICPLedgerTransfer {
        let mut schema = schema_for!(TransferToArgs);
        schema.meta_schema = None; // Remove the $schema field

        ICPLedgerTransfer {
            canister,
            schema: json!(schema),
            symbol,
            decimals,
            from_user_subaccount,
        }
    }

    async fn transfer(
        &self,
        ctx: &impl CanisterCaller,
        data: TransferToArgs,
    ) -> Result<Nat, BoxError> {
        let owner = Principal::from_text(&data.account)?;
        let from_subaccount = if self.from_user_subaccount {
            Some(principal_to_subaccount(owner))
        } else {
            None
        };

        let amount = (data.amount * 10u64.pow(self.decimals as u32) as f64) as u64;
        let res: Result<Nat, TransferError> = ctx
            .canister_update(
                &self.canister,
                "icrc1_transfer",
                (TransferArg {
                    from_subaccount,
                    to: Account {
                        owner,
                        subaccount: None,
                    },
                    amount: amount.into(),
                    memo: data.memo.map(|m| {
                        let mut buf = String::new();
                        for c in m.chars() {
                            if buf.len() + c.len_utf8() > MAX_MEMO_LEN {
                                break;
                            }
                            buf.push(c);
                        }
                        Memo(ByteBuf::from(buf.into_bytes()))
                    }),
                    fee: None,
                    created_at_time: None,
                },),
            )
            .await?;
        res.map_err(|err| format!("failed to transfer tokens, error: {:?}", err).into())
    }
}

impl Tool<BaseCtx> for ICPLedgerTransfer {
    const CONTINUE: bool = true;
    type Args = TransferToArgs;
    type Output = Nat;

    fn name(&self) -> String {
        format!("icp_ledger_transfer_{}", self.symbol)
    }

    fn description(&self) -> String {
        format!(
            "Transfer {} tokens to the specified account on ICP network.",
            self.symbol
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

    async fn call(&self, ctx: BaseCtx, data: Self::Args) -> Result<Self::Output, BoxError> {
        self.transfer(&ctx, data).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anda_engine::context::mock;
    use candid::{decode_args, encode_args};

    #[tokio::test(flavor = "current_thread")]
    async fn test_icp_ledger_transfer() {
        let ledger = Principal::from_text("druyg-tyaaa-aaaaq-aactq-cai").unwrap();
        let tool = ICPLedgerTransfer::new(ledger, "PANDA".to_string(), 8, true);
        let definition = tool.definition();
        assert_eq!(definition.name, "icp_ledger_transfer_PANDA");
        let s = serde_json::to_string_pretty(&definition).unwrap();
        println!("{}", s);
        // {
        //     "name": "icp_ledger_transfer_PANDA",
        //     "description": "Transfer PANDA tokens to the specified account on ICP network.",
        //     "parameters": {
        //       "properties": {
        //         "account": {
        //           "type": "string"
        //         },
        //         "amount": {
        //           "format": "double",
        //           "type": "number"
        //         },
        //         "memo": {
        //           "type": [
        //             "string",
        //             "null"
        //           ]
        //         }
        //       },
        //       "required": [
        //         "account",
        //         "amount"
        //       ],
        //       "title": "TransferToArgs",
        //       "type": "object"
        //     },
        //     "strict": true
        // }

        let args = TransferToArgs {
            account: Principal::anonymous().to_string(),
            amount: 9999.000012345678,
            memo: Some("test memo".to_string()),
        };
        let mocker = mock::MockCanisterCaller::new(|canister, method, args| {
            assert_eq!(canister, &ledger);
            assert_eq!(method, "icrc1_transfer");
            let (args,): (TransferArg,) = decode_args(&args).unwrap();
            println!("{:?}", args);
            assert_eq!(
                args.from_subaccount,
                Some(principal_to_subaccount(Principal::anonymous()))
            );
            assert_eq!(args.to.owner, Principal::anonymous());
            assert_eq!(args.amount, Nat::from(999900001234u64));
            assert_eq!(args.memo, Some(Memo(ByteBuf::from("test memo".as_bytes()))));

            let res: Result<Nat, TransferError> = Ok(Nat::from(321u64));
            encode_args((res,)).unwrap()
        });

        let res = tool.transfer(&mocker, args).await.unwrap();
        assert_eq!(res, Nat::from(321u64));
    }
}
