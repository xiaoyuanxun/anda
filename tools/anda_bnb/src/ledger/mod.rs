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

use anda_core::BoxError;
use anda_engine::context::BaseCtx;
use std::collections::BTreeMap;

use alloy::{
    network::{AnyNetwork, EthereumWallet, NetworkWallet},
    primitives::{utils::parse_units, Address, FixedBytes}, 
    providers::ProviderBuilder, sol
};
use core::str::FromStr;

pub mod balance;
pub mod transfer;

pub use balance::*;
pub use transfer::*;

use crate::{signer::AndaEvmSigner, utils_evm::*};

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

// public static url of BNB BEP20 contract address
pub static TOKEN_ADDR: &str = "0xDE3a190D9D26A8271Ae9C27573c03094A8A2c449";  // BNB testnet

// public static chain id of BNB
pub static CHAIN_ID: u64 = 97;  // BNB testnet

// public static derivation path
pub static DRVT_PATH: &[&[u8]] = &[b"44'", b"60'", b"10'", b"20", b"30"];

/// BNB Ledger Transfer tool implementation
#[derive(Debug, Clone)]
pub struct BNBLedgers {
    /// Map of token symbols to their corresponding canister ID and decimals places
    pub ledgers: BTreeMap<String, (Address, u8)>,
}

impl BNBLedgers {
    /// Loads a BNBLedgers instance by retrieving token information from the BNB token contract
    ///
    /// # Returns
    /// A `Result` containing the initialized `BNBLedgers` with token symbol, address, and decimals,
    /// or an error if token information retrieval fails
    ///
    /// # Errors
    /// Returns a `BoxError` if RPC connection fails, token symbol/decimals cannot be retrieved,
    /// or address parsing encounters an issue
    pub async fn load() -> Result<BNBLedgers, BoxError> {
        // Create a provider
        let rpc_url = bnb_rpc().parse()?;
        let provider = ProviderBuilder::new().on_http(rpc_url);

        // ERC20 token contract address and instance
        let token_addr =  Address::parse_checksummed(TOKEN_ADDR, None).unwrap();
        let contract = ERC20STD::new(token_addr, provider.clone());

        // Get token symbol and decimals
        let symbol = contract.symbol().call().await?;
        let decimals = contract.decimals().call().await?;
        log::debug!("symbol: {:?}, decimals: {:?}", symbol.clone(), decimals);
        
        // Create ledgers instance
        let ledgers = BNBLedgers {
            ledgers: BTreeMap::from([
                (
                    symbol.clone(),
                    (
                        token_addr,
                        decimals,
                    ),
                ),
            ])
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
        let signer = AndaEvmSigner::new(
            ctx,
            DRVT_PATH.iter()
                .map(|&s| s.to_vec())
                .collect(),
            Some(CHAIN_ID),
        ).await?;

        // Create an Ethereum wallet from the signer
        let wallet = EthereumWallet::from(signer);
        // Get sender EVM address
        let sender_address = NetworkWallet::<AnyNetwork>::default_signer_address(&wallet);
        log::debug!("Sender EVM address: {:?}", sender_address);                
        
        // Create a provider with the wallet.
        let provider = ProviderBuilder::new()
                .with_simple_nonce_management()
                .with_gas_estimation()
                .wallet(wallet).on_http(reqwest::Url::parse(bnb_rpc().as_ref()).unwrap());  // Todo: read rpc url from web3 client

        // Get receiver address, transfer amount, and token address to transfer
        let to_addr = Address::from_str(&args.account)?;  
        let to_amount = &args.amount;
        let (token_addr, _decimals) = self
            .ledgers
            .get(&args.symbol)
            .ok_or_else(|| format!("Token {} is not supported", args.symbol))?;

        // Create contract instance, get token symbol and decimals
        let contract = ERC20STD::new(*token_addr, provider);
        let symbol = contract.symbol().call().await?;
        let decimals = contract.decimals().call().await?;
        // Balance check
        let balance = contract.balanceOf(sender_address).call().await?;
        log::debug!("symbol: {:?}, decimals: {:?}, balance: {:?}", &symbol, decimals, balance);
        let to_amount = parse_units(&to_amount.to_string(), decimals)?.into();
        if balance < to_amount  {
            return Err("Insufficient balance".into());
        }

        // Transfer token
        log::debug!("BNB transfer. amount: {:?}, transfer to_addr: {:?}", to_amount, to_addr);
        let pending_tx = contract.transfer(to_addr, to_amount).send().await?;
        log::debug!("BNB transfer pending tx: {:?}", pending_tx);
        let res = pending_tx.watch().await?;
        log::debug!("BNB transfer result: {:#?}", res);

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
        let provider = ProviderBuilder::new()
                    .on_http(reqwest::Url::parse(bnb_rpc().as_ref()).unwrap());  // Todo: read rpc url from web3 client

        // Read the account address from the arguments
        let user_addr = Address::from_str(&args.account)?;

        // Read the token address and decimals
        let (token_addr, _decimals) = self
            .ledgers
            .get(&args.symbol)
            .ok_or_else(|| format!("Token {} is not supported", args.symbol))?;

        // Create contract instance, get token symbol and decimals
        let contract = ERC20STD::new(*token_addr, provider);
        let symbol = contract.symbol().call().await.unwrap();
        let decimals = contract.decimals().call().await.unwrap();
        // Query balance
        let balance = contract.balanceOf( user_addr).call().await.unwrap();

        // Convert balance to f64
        let balance = get_balance(balance)?;
        log::info!(
            user_addr = user_addr.to_string(),
            token_addr = token_addr.to_string(),
            symbol = &symbol,
            decimals = decimals,
            balance = balance;
            "balance_of_bnb"
        );

        return Ok((user_addr, balance));
    }
}