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

use anda_core::{context::HttpFeatures, BoxError, CanisterCaller, CONTENT_TYPE_JSON};
use anda_engine::context::BaseCtx;
use candid::{Nat, Principal};
use icrc_ledger_types::{
    icrc::generic_metadata_value::MetadataValue,
    icrc1::{
        account::{Account, principal_to_subaccount},
        transfer::{TransferArg, TransferError},
    },
};
use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use serde_json::json;

use alloy::{
    consensus::TxEnvelope, primitives::{utils::{format_units, parse_units}, Address, U256}, sol, sol_types::SolInterface
};
use core::str::FromStr;

pub mod balance;
pub mod transfer;

pub use balance::*;
pub use transfer::*;

use crate::utils_evm::*;

// Codegen from artifact.
sol!(
    #[allow(missing_docs)]
    #[sol(rpc)]
    #[derive(Debug)]
    ERC20STD,
    "artifacts/ERC20Example.json"
);

/// ICP Ledger Transfer tool implementation
#[derive(Debug, Clone)]
pub struct BSCLedgers {
    /// Map of token symbols to their corresponding canister ID and decimals places
    pub ledgers: BTreeMap<String, (Address, u8)>,
}

impl BSCLedgers {
    /// Creates a new ICPLedgerTransfer instance
    ///
    /// # Arguments
    /// * `ctx` - Canister caller context
    /// * `ledger_canisters` - Set of 1 to N ICP token ledger canister IDs
    /// * `from_user_subaccount` - When false, the from account is the Agent's main account.
    ///   When true, the from account is a user-specific subaccount derived from the Agent's main account.
    pub async fn load(
        ctx: &impl CanisterCaller,
        ledger_canisters: BTreeSet<Principal>,
        from_user_subaccount: bool,
    ) -> Result<BSCLedgers, BoxError> {
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

        let mut _ledgers = BTreeMap::new();
        Ok(BSCLedgers {
            ledgers: _ledgers,
        })
    }

    /// Performs the token transfer operation
    ///
    /// # Arguments
    /// * `ctx` - BSC/EVM caller context
    /// * `args` - Transfer arguments containing destination account, amount, and memo
    ///
    /// # Returns
    /// Result containing the ledger ID and transaction ID (Nat) or an error
    async fn transfer(
        &self,
        ctx: &BaseCtx, // Todo: impl HttpFeatures
        me: Address,
        args: transfer::TransferToArgs,
    ) -> Result<(Address, Nat), BoxError> {
        use hex;
        use std::str::FromStr;

        let to_addr = Address::from_str(&args.account)?;  
        let to_amount = &args.amount;

        let (token_addr, decimals) = self
            .ledgers
            .get(&args.symbol)
            .ok_or_else(|| format!("Token {} is not supported", args.symbol))?;

        let balance_of_args = BalanceOfArgs {
            account: me.to_string().clone(),
            symbol: args.symbol.clone(),
        };
        let balance = self.balance_of(&ctx, balance_of_args).await?.1;
        if balance < *to_amount  {
            return Err("Insufficient balance".into());
        }

        // Generate Transfer Call Data
        let call_args = ERC20STD::transferCall{
            to: to_addr,
            amount: parse_units(&to_amount.to_string(), *decimals)?.into(),
        };
        let amount = call_args.amount.to_be_bytes_vec();
        let call_data = ERC20STD::ERC20STDCalls::transfer(call_args).abi_encode();
        log::debug!("call_data: {:?}", call_data);

        // let nonce = get_nonce(sender_address, rpc_url).await?;  // Todo: get real time nonce

        let chain_id: u64 = 97; // Todo: adjust chain id by testnet or mainnet
        let tx_evm = TxEvm {
            nonce: 1,
            gas_price: 5_000_000_000u128, // 5 Gwei
            gas_limit: 21000u64, // Adjust based on contract
            to: &token_addr,
            value: U256::ZERO, // For ERC20 transfers
            data: &call_data,
            v: chain_id,
            r: U256::ZERO,
            s: U256::ZERO,
        };

        // let derivation_path: &[&[u8]] = &[b"44'", b"60'", b"10'", b"20", b"30"];  // Todo: after the way to retrieve derivation_path is confirmed
        // let signature = ctx.secp256k1_sign_message_ecdsa(derivation_path, &tx_hash).await?;
        
        let root_secret_org = dotenv::var("ROOT_SECRET")?;  // Todo: Read root secret from ctx
        let root_secret = const_hex::decode(&root_secret_org)?;
        let sk = generate_secret_key(root_secret.as_slice())?;
        let stream = tx_evm.compose_tx_raw(&sk)?;

        let body = json!({
            "id": 1,
            "jsonrpc":"2.0",
            "method":"eth_sendRawTransaction",
            "params":[
                format!("0x{}", hex::encode::<&[u8]>(stream.as_raw())), 
            ],    
        });

        log::debug!("Transfer req body: {:#?}", body);
        let body = serde_json::to_vec(&body)?;

        let url = "https://bsc-testnet.bnbchain.org";  // Todo: pass it as a parameter in call

        let res = ctx.https_call(
            url, 
            http::Method::POST, 
            Some(get_http_header()), 
            Some(body)
        ).await?;
        let body = res.text().await?;
        log::debug!("BSC transfer res body: {:#?}", body);

        let amount = BigUint::from_bytes_be(amount.as_slice());
        Ok((to_addr, Nat(amount)))
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
        ctx: &BaseCtx,
        args: balance::BalanceOfArgs,
    ) -> Result<(Address, f64), BoxError> {
        let url = "https://bsc-testnet.bnbchain.org";  // Todo: pass it as a parameter in call

        let owner_addr = Address::from_str(&args.account)?;
        let call_args = ERC20STD::balanceOfCall{
                account: Address::from_str(&args.account).unwrap()
            };
        let call_data = ERC20STD::ERC20STDCalls::balanceOf(call_args).abi_encode();
        log::debug!("call_data: {:?}", call_data);

        let (token_addr, _decimals) = self
            .ledgers
            .get(&args.symbol)
            .ok_or_else(|| format!("Token {} is not supported", args.symbol))?;

        let body = json!({
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{
                "to": token_addr.to_string(),
                "data": format!("0x{}", hex::encode(call_data)),
            }, "latest"],
            "id": 1
        });
        let body = serde_json::to_vec(&body)?;

        let res = ctx.https_call(
                url, 
                http::Method::POST, 
                Some(get_http_header()), 
                Some(body)
            ).await?;
        let body = res.json::<JsonRpcResponse>().await?;
        log::debug!("BSC balance query: {:#?}", body);
        let balance = get_balance(body)?;

        log::info!(  // Todo: why not log in test
            account = args.account,
            symbol = args.symbol,
            balance = balance;
            "balance_of_bsc",
        );

        return Ok((owner_addr, balance));
    }
}
