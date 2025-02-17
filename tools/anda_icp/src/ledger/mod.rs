//! Module for interacting with ICP Ledgers using ICRC-1 standard
//!
//! This module provides functionality for:
//! - Loading and managing multiple ICP ledger canisters
//! - Transferring tokens between accounts
//! - Querying account balances
//!
//! The implementation supports:
//! - Multiple token symbols (though primarily designed for ICP)
//! - Configurable subaccount usage for transfers
//! - ICRC-1 standard compliant operations
//!
//! # Examples
//! ```rust,ignore
//! use anda_icp::ledger::ICPLedgers;
//! use anda_core::CanisterCaller;
//! use std::collections::BTreeSet;
//!
//! async fn example(ctx: &impl CanisterCaller) {
//!     let canisters = BTreeSet::from([Principal::from_text("ryjl3-tyaaa-aaaaa-aaaba-cai").unwrap()]);
//!     let ledgers = ICPLedgers::load(ctx, canisters, false).await.unwrap();
//!     // Use ledgers for transfers or balance queries
//! }
//! ```

use anda_core::{BoxError, CanisterCaller};
use candid::{Nat, Principal};
use icrc_ledger_types::{
    icrc::generic_metadata_value::MetadataValue,
    icrc1::{
        account::{principal_to_subaccount, Account},
        transfer::{TransferArg, TransferError},
    },
};
use num_traits::cast::ToPrimitive;
use std::collections::{BTreeMap, BTreeSet};

pub mod balance;
pub mod transfer;

pub use balance::*;
pub use transfer::*;

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
    /// Result containing the ledger ID and transaction ID (Nat) or an error
    async fn transfer(
        &self,
        ctx: &impl CanisterCaller,
        me: Principal,
        args: transfer::TransferToArgs,
    ) -> Result<(Principal, Nat), BoxError> {
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
        let balance: Nat = ctx
            .canister_query(
                canister,
                "icrc1_balance_of",
                (Account {
                    owner: me,
                    subaccount: from_subaccount,
                },),
            )
            .await?;

        if balance < amount {
            return Err("insufficient balance".into());
        }

        let res: Result<Nat, TransferError> = ctx
            .canister_update(
                canister,
                "icrc1_transfer",
                (TransferArg {
                    from_subaccount,
                    to: Account {
                        owner,
                        subaccount: None,
                    },
                    amount: amount.into(),
                    memo: None,
                    fee: None,
                    created_at_time: None,
                },),
            )
            .await?;
        log::info!(
            account = args.account,
            symbol = args.symbol,
            amount = args.amount,
            result = res.is_ok();
            "icrc1_transfer",
        );
        res.map(|v| (*canister, v))
            .map_err(|err| format!("failed to transfer tokens, error: {:?}", err).into())
    }

    /// Retrieves the balance of a specific account for a given token
    ///
    /// # Arguments
    /// * `ctx` - Canister caller context
    /// * `args` - Balance query arguments containing account and token symbol
    ///
    /// # Returns
    /// Result containing the ledger ID and token balance (f64) or an error
    async fn balance_of(
        &self,
        ctx: &impl CanisterCaller,
        args: balance::BalanceOfArgs,
    ) -> Result<(Principal, f64), BoxError> {
        let owner = Principal::from_text(&args.account)?;

        let (canister, decimals) = self
            .ledgers
            .get(&args.symbol)
            .ok_or_else(|| format!("Token {} is not supported", args.symbol))?;
        let account = Account {
            owner,
            subaccount: None,
        };

        let res: Nat = ctx
            .canister_query(canister, "icrc1_balance_of", (account,))
            .await?;

        let amount = res.0.to_f64().unwrap_or_default() / 10u64.pow(*decimals as u32) as f64;
        log::info!(
            account = args.account,
            symbol = args.symbol,
            balance = amount;
            "balance_of",
        );
        Ok((*canister, amount))
    }
}
