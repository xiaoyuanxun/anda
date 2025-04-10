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
    consensus::TxEnvelope, 
    network::{AnyNetwork, EthereumWallet, NetworkWallet}, 
    primitives::{utils::{format_units, parse_units}, Address, U256}, 
    providers::ProviderBuilder, 
    signers::local::PrivateKeySigner, 
    sol, sol_types::SolInterface
};
use core::str::FromStr;

pub mod balance;
pub mod transfer;

pub use balance::*;
pub use transfer::*;

use crate::{signer::{convert_to_boxed, derive_address_from_pubkey, AndaSigner}, utils_evm::*};

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
        ctx: BaseCtx, // Todo: impl HttpFeatures
        me: Address,
        args: transfer::TransferToArgs,
    ) -> Result<(Address, Nat), BoxError> {
        use hex;
        use std::str::FromStr;
        use anda_core::KeysFeatures;

        let url = "https://bsc-testnet.bnbchain.org";  // Todo: pass it as a parameter in call
        let chain_id: u64 = 97;
        let derivation_path: &[&[u8]] = &[b"44'", b"60'", b"10'", b"20", b"30"];  // Todo: how to retrieve derivation path?

        // Create an anda signer
        let signer = AndaSigner::new(
            ctx.clone(), 
            convert_to_boxed(derivation_path),
            Some(chain_id),
        ).await?;

        // Todo: to remove
        // let root_secret_org = dotenv::var("ROOT_SECRET").unwrap();
        // let root_secret = const_hex::decode(&root_secret_org).unwrap();
        // let sk = ic_secp256k1::PrivateKey::generate_from_seed(&root_secret);
        // let sk = sk.serialize_sec1();
        // let sk = hex::encode(&sk);
        // let signer: PrivateKeySigner = sk.as_str().parse().expect("should parse private key");

        // // Create an Ethereum wallet from the signer
        let wallet = EthereumWallet::from(signer);
        // Get sender EVM address
        let sender_address = NetworkWallet::<AnyNetwork>::default_signer_address(&wallet);
        log::debug!("Sender EVM address: {:?}", sender_address);                
        
        // Create a provider with the wallet.
        let provider = ProviderBuilder::new().
                wallet(wallet).on_http(reqwest::Url::parse(url).unwrap());

        let to_addr = Address::from_str(&args.account)?;  
        let to_amount = &args.amount;

        let (token_addr, decimals) = self
            .ledgers
            .get(&args.symbol)
            .ok_or_else(|| format!("Token {} is not supported", args.symbol))?;

        // Create contract instance, get token symbol and decimals
        let contract = ERC20STD::new(*token_addr, provider);
        let symbol = contract.symbol().call().await?._0;
        let decimals = contract.decimals().call().await?._0;
        let balance = contract.balanceOf(sender_address).call().await?._0;
        log::debug!("symbol: {:?}, decimals: {:?}, balance: {:?}", symbol.clone(), decimals, balance);
        
        // Balance check
        let to_amount = parse_units(&to_amount.to_string(), decimals)?.into();
        if balance < to_amount  {
            return Err("Insufficient balance".into());
        }

        // Transfer token
        log::debug!("BSC transfer. amount: {:?}, transfer to_addr: {:?}", to_amount, to_addr);
        let res = contract.transfer(to_addr, to_amount).send().await?.watch().await;
        log::debug!("BSC transfer result: {:#?}", res);

        Ok((to_addr, Nat::from(0u32)))
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
        ctx: BaseCtx,
        args: balance::BalanceOfArgs,
    ) -> Result<(Address, f64), BoxError> {
        let url = "https://bsc-testnet.bnbchain.org";  // Todo: pass it as a parameter in call
        let chain_id: u64 = 97;
        let derivation_path: &[&[u8]] = &[b"44'", b"60'", b"10'", b"20", b"30"];  // Todo: how to retrieve derivation path?

        // Create an anda signer
        let signer = AndaSigner::new(
            ctx, 
            convert_to_boxed(derivation_path),
            Some(chain_id),
        ).await.unwrap();

        // Create an Ethereum wallet from the signer
        let wallet = EthereumWallet::from(signer);

        // Create a provider with the wallet.
        let provider = ProviderBuilder::new().
                wallet(wallet).on_http(reqwest::Url::parse(url).unwrap());

        let user_addr = Address::from_str(&args.account)?;

        let (token_addr, _decimals) = self
            .ledgers
            .get(&args.symbol)
            .ok_or_else(|| format!("Token {} is not supported", args.symbol))?;

        let contract = ERC20STD::new(*token_addr, provider);

        let symbol = contract.symbol().call().await.unwrap()._0;
        let decimals = contract.decimals().call().await.unwrap()._0;
        let balance = contract.balanceOf( user_addr).call().await.unwrap()._0;
        log::debug!("Query balance. user_addr: {:?}, token_addr: {:?}. \
                    symbol: {:?}, decimals: {:?}, balance query: {:?}", 
                    user_addr, token_addr, symbol.clone(), decimals, balance);

        let balance = get_balance(balance)?;

        log::info!(  // Todo: why not log in test
            account = args.account,
            symbol = args.symbol,
            balance = balance;
            "balance_of_bsc",
        );

        return Ok((user_addr, balance));
    }
}
