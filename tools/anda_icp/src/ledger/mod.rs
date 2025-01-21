use anda_core::{BoxError, CanisterCaller};
use candid::{Nat, Principal};
use icrc_ledger_types::{
    icrc::generic_metadata_value::MetadataValue,
    icrc1::{
        account::{principal_to_subaccount, Account},
        transfer::{Memo, TransferArg, TransferError},
    },
};
use num_traits::cast::ToPrimitive;
use serde_bytes::ByteBuf;
use std::collections::{BTreeMap, BTreeSet};

pub mod balance;
pub mod transfer;

const MAX_MEMO_LEN: usize = 32;

/// ICP Ledger Transfer tool implementation
#[derive(Debug, Clone)]
pub struct ICPLedgers {
    /// Map of token symbols to their corresponding canister ID and decimals places
    pub ledgers: BTreeMap<String, (Principal, u8)>,
    /// Flag indicating whether to use user-specific subaccounts for transfers
    pub from_user_subaccount: bool,
}

impl ICPLedgers {
    /// Creates a new ICPLedgerTransfer instance
    /// 
    /// # Arguments
    /// * `ctx` - Canister caller context
    /// * `ledger_canisters` - Set of 1 to N ICP token ledger canister IDs
    /// * `from_user_subaccount` - When false, the from account is the Agent's main account.
    ///                            When true, the from account is a user-specific subaccount derived from the Agent's main account.
    pub async fn load(
        ctx: &impl CanisterCaller,
        ledger_canisters: BTreeSet<Principal>,
        from_user_subaccount: bool,
    ) -> Result<ICPLedgers, BoxError> {
        if ledger_canisters.is_empty() {
            return Err("No ledger canister specified".into());
        }
        let mut ledgers = BTreeMap::new();
        for canister in ledger_canisters {
            let res: Vec<(String, MetadataValue)> =
                ctx.canister_query(&canister, "icrc1_metadata", ()).await?;
            let mut symbol = "ICP".to_string();
            let mut decimals = -1i8;
            for (k, v) in res {
                match k.as_str() {
                    // icrc1:symbol
                    "icrc1:symbol" => {
                        if let MetadataValue::Text(s) = v {
                            symbol = s;
                        }
                    }
                    // icrc1:decimals
                    "icrc1:decimals" => {
                        if let MetadataValue::Nat(n) = v {
                            decimals = n.0.to_i8().unwrap_or(-1)
                        }
                    }
                    _ => {}
                }
            }

            if decimals > -1 {
                ledgers.insert(symbol, (canister, decimals as u8));
            }
        }

        Ok(ICPLedgers {
            ledgers,
            from_user_subaccount,
        })
    }

    /// Performs the token transfer operation
    ///
    /// # Arguments
    /// * `ctx` - Canister caller context
    /// * `args` - Transfer arguments containing destination account, amount, and memo
    ///
    /// # Returns
    /// Result containing the transaction ID (Nat) or an error
    async fn transfer(
        &self,
        ctx: &impl CanisterCaller,
        args: transfer::TransferToArgs,
    ) -> Result<Nat, BoxError> {
        let owner = Principal::from_text(&args.account)?;
        let from_subaccount = if self.from_user_subaccount {
            Some(principal_to_subaccount(owner))
        } else {
            None
        };
        let (canister, decimals) = self
            .ledgers
            .get(&args.symbol)
            .ok_or_else(|| format!("Token {} is not supported", args.symbol))?;

        let amount = (args.amount * 10u64.pow(*decimals as u32) as f64) as u64;
        let res: Result<Nat, TransferError> = ctx
            .canister_update(
                &canister,
                "icrc1_transfer",
                (TransferArg {
                    from_subaccount,
                    to: Account {
                        owner,
                        subaccount: None,
                    },
                    amount: amount.into(),
                    memo: args.memo.map(|m| {
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

    /// Retrieves the balance of a specific account for a given token
    ///
    /// # Arguments
    /// * `ctx` - Canister caller context
    /// * `args` - Balance query arguments containing account and token symbol
    ///
    /// # Returns
    /// Result containing the balance (Nat) or an error
    async fn balance_of(
        &self,
        ctx: &impl CanisterCaller,
        args: balance::BalanceOfArgs,
    ) -> Result<Nat, BoxError> {
        let owner = Principal::from_text(&args.account)?;

        let (canister, _) = self
            .ledgers
            .get(&args.symbol)
            .ok_or_else(|| format!("Token {} is not supported", args.symbol))?;
        let account = Account {
            owner,
            subaccount: None,
        };

        let res: Nat = ctx
            .canister_query(&canister, "icrc1_balance_of", (account,))
            .await?;
        Ok(res)
    }
}
