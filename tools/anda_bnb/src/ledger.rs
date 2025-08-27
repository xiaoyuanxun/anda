//! Module for interacting with BNB Chain Ledgers
//!
//! This module provides functionality for:
//! - Loading and managing BNB Chain token contracts
//! - Transferring tokens between accounts
//! - Querying account balances
//!
//! The implementation supports:
//! - Multiple token symbols
//! - ERC20 standard compliant operations
//!
//! # Examples
//! ,ignore
//! use anda_bnb::ledger::BNBLedgers;
//!
//! async fn example() {
//!     let ledgers = BNBLedgers::load().await.unwrap();
//!     // Use ledgers for transfers or balance queries
//! }
//!

use alloy::{
    network::{AnyNetwork, EthereumWallet, NetworkWallet},
    primitives::{Address, FixedBytes, utils::parse_units},
    providers::ProviderBuilder,
    sol,
};
use anda_core::BoxError;
use anda_engine::context::BaseCtx;
use core::str::FromStr;
use std::collections::{BTreeMap, BTreeSet};

pub mod balance;
pub mod transfer;

pub use balance::*;
pub use transfer::*;

use crate::{helper::*, signer::AndaEvmSigner};

// Codegen from artifact.
sol!(
    #[allow(missing_docs)]
    #[sol(rpc)]
    #[derive(Debug)]
    ERC20STD,
    "artifacts/ERC20Example.json"
);

// Read the BNB_RPC from environment variable
pub fn bnb_rpc() -> String {
    dotenv::var("BNB_RPC").unwrap_or_else(|_| "https://bsc-testnet.bnbchain.org".to_string())
}

// public static derivation path
pub static DRVT_PATH: &[&[u8]] = &[b"44'", b"60'", b"10'", b"20", b"30"];

/// BNB Ledger Transfer tool implementation
#[derive(Debug, Clone)]
pub struct BNBLedgers {
    provider_url: reqwest::Url,
    chain_id: u64,
    derivation_path: Vec<Vec<u8>>,
    /// Map of token symbols to their corresponding canister ID and decimals places
    pub ledgers: BTreeMap<String, (Address, u8)>,
}

impl BNBLedgers {
    /// Loads a BNBLedgers instance by retrieving token information from the BNB token contract
    pub async fn load(
        provider_url: String,
        chain_id: u64,
        derivation_path: Vec<Vec<u8>>,
        tokens: BTreeSet<String>,
    ) -> Result<BNBLedgers, BoxError> {
        // Create a provider
        let provider_url: reqwest::Url = provider_url.parse()?;
        let provider = ProviderBuilder::new().connect_http(provider_url.clone());
        let mut ledgers: BTreeMap<String, (Address, u8)> = BTreeMap::new();

        for token in tokens {
            // Get token symbol and decimals
            // ERC20 token contract address and instance
            let token_addr = Address::parse_checksummed(token, None)?;
            let contract = ERC20STD::new(token_addr, provider.clone());
            let symbol = contract.symbol().call().await?;
            let decimals = contract.decimals().call().await?;
            log::debug!(
                "symbol: {:?}, token_addr: {:?}, decimals: {:?}",
                symbol,
                token_addr,
                decimals
            );
            // Add token to ledgers map
            ledgers.insert(symbol, (token_addr, decimals));
        }

        // Create ledgers instance
        let ledgers = BNBLedgers {
            provider_url,
            chain_id,
            derivation_path,
            ledgers,
        };

        Ok(ledgers)
    }

    /// Performs the token transfer operation
    ///
    /// # Arguments
    /// * `ctx` - EVM caller context
    /// * `args` - Transfer arguments containing destination account, amount, and memo
    ///
    /// # Returns
    /// Result containing the account address and transaction ID or an error
    async fn transfer(
        &self,
        ctx: BaseCtx,
        args: transfer::TransferToArgs,
    ) -> Result<(Address, FixedBytes<32>), BoxError> {
        use std::str::FromStr;

        // Create an anda signer
        let signer =
            AndaEvmSigner::new(ctx, self.derivation_path.clone(), Some(self.chain_id)).await?;

        // Create an Ethereum wallet from the signer
        let wallet = EthereumWallet::from(signer);
        // Get sender EVM address
        let sender_address = NetworkWallet::<AnyNetwork>::default_signer_address(&wallet);
        log::debug!("Sender EVM address: {:?}", sender_address);
        
        // Create a provider with the wallet.
        let provider = ProviderBuilder::new()
            .with_simple_nonce_management()
            .with_gas_estimation()
            .wallet(wallet)
            .connect_http(self.provider_url.clone());

        // Get receiver address, transfer amount, and token address to transfer
        let to_addr = Address::from_str(&args.account)?;
        let to_amount = &args.amount;
        let (token_addr, decimals) = self
            .ledgers
            .get(&args.symbol)
            .ok_or_else(|| format!("Token {} is not supported", args.symbol))?;

        // Create contract instance, get token symbol and decimals
        let contract = ERC20STD::new(*token_addr, provider);
        // Balance check
        let balance = contract.balanceOf(sender_address).call().await?;
        if log::log_enabled!(log::Level::Debug) {
            let balance = get_balance(balance)?;
            log::debug!(
                "symbol: {:?}, decimals: {:?}, balance: {:?}",
                args.symbol,
                decimals,
                balance
            );
        }

        let to_amount = parse_units(&to_amount.to_string(), *decimals)?.into();
        if balance < to_amount {
            return Err("Insufficient balance".into());
        }

        // Transfer token
        log::debug!(
            "BNB transfer. amount: {:?}, transfer to_addr: {:?}",
            to_amount,
            to_addr
        );

        let pending_tx = contract.transfer(to_addr, to_amount).send().await?;
        log::debug!("BNB transfer pending tx: {:?}", pending_tx);

        let res = pending_tx.watch().await?;

        Ok((to_addr, res))
    }

    /// Retrieves the balance of a specific account for a given token
    ///
    /// # Arguments
    /// * `ctx` - EVM caller context
    /// * `args` - Balance query arguments containing account and token symbol
    ///
    /// # Returns
    /// Result containing the account address and token balance (f64) or an error
    async fn balance_of(
        &self,
        _ctx: BaseCtx,
        args: balance::BalanceOfArgs,
    ) -> Result<(Address, f64), BoxError> {
        // Create a provider
        let provider = ProviderBuilder::new().connect_http(self.provider_url.clone());

        // Read the account address from the arguments
        let user_addr = Address::from_str(&args.account)?;

        // Read the token address and decimals
        let (token_addr, decimals) = self
            .ledgers
            .get(&args.symbol)
            .ok_or_else(|| format!("Token {} is not supported", args.symbol))?;

        // Create contract instance, get token symbol and decimals
        let contract = ERC20STD::new(*token_addr, provider);
        // Query balance
        let balance = contract.balanceOf(user_addr).call().await.unwrap();

        // Convert balance to f64
        let balance = get_balance(balance)?;
        log::info!(
            user_addr = user_addr.to_string(),
            token_addr = token_addr.to_string(),
            symbol = args.symbol,
            decimals = decimals,
            balance = balance;
            "balance_of_bnb"
        );

        Ok((user_addr, balance))
    }
}
