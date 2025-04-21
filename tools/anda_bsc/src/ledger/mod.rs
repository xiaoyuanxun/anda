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

use crate::{signer::AndaSigner, utils_evm::*};

// Codegen from artifact.
sol!(
    #[allow(missing_docs)]
    #[sol(rpc)]
    #[derive(Debug)]
    ERC20STD,
    "artifacts/ERC20Example.json"
);

// Read the BSC_RPC from environment variable
pub fn bsc_rpc() -> String {
    dotenv::var("BSC_RPC").unwrap_or_else(|_| "https://bsc-testnet.bnbchain.org".to_string())
}

// public static url of BSC BEP20 contract address
pub static TOKEN_ADDR: &str = "0xDE3a190D9D26A8271Ae9C27573c03094A8A2c449";  // BSC testnet

// public static chain id of BSC
pub static CHAIN_ID: u64 = 97;  // BSC testnet

// public static derivation path
pub static DRVT_PATH: &[&[u8]] = &[b"44'", b"60'", b"10'", b"20", b"30"];

/// ICP Ledger Transfer tool implementation
#[derive(Debug, Clone)]
pub struct BSCLedgers {
    /// Map of token symbols to their corresponding canister ID and decimals places
    pub ledgers: BTreeMap<String, (Address, u8)>,
}

impl BSCLedgers {
    /// Loads a BSCLedgers instance by retrieving token information from the BSC token contract
    ///
    /// # Returns
    /// A `Result` containing the initialized `BSCLedgers` with token symbol, address, and decimals,
    /// or an error if token information retrieval fails
    ///
    /// # Errors
    /// Returns a `BoxError` if RPC connection fails, token symbol/decimals cannot be retrieved,
    /// or address parsing encounters an issue
    pub async fn load() -> Result<BSCLedgers, BoxError> {
        // Create a provider
        let rpc_url = bsc_rpc().parse()?;
        let provider = ProviderBuilder::new().on_http(rpc_url);

        // ERC20 token contract address and instance
        let token_addr =  Address::parse_checksummed(TOKEN_ADDR, None).unwrap();
        let contract = ERC20STD::new(token_addr, provider.clone());

        // Get token symbol and decimals
        let symbol = contract.symbol().call().await?;
        let decimals = contract.decimals().call().await?;
        log::debug!("symbol: {:?}, decimals: {:?}", symbol.clone(), decimals);
        
        // Create ledgers instance
        let ledgers = BSCLedgers {
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
        let signer = AndaSigner::new(
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
                .wallet(wallet).on_http(reqwest::Url::parse(bsc_rpc().as_ref()).unwrap());  // Todo: read rpc url from web3 client

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
        log::debug!("BSC transfer. amount: {:?}, transfer to_addr: {:?}", to_amount, to_addr);
        let pending_tx = contract.transfer(to_addr, to_amount).send().await?;
        log::debug!("BSC transfer pending tx: {:?}", pending_tx);
        let res = pending_tx.watch().await?;
        log::debug!("BSC transfer result: {:#?}", res);

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
                    .on_http(reqwest::Url::parse(bsc_rpc().as_ref()).unwrap());  // Todo: read rpc url from web3 client

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
        log::debug!("Query balance. user_addr: {:?}, token_addr: {:?}. \
                    symbol: {:?}, decimals: {:?}, balance query: {:?}", 
                    user_addr, token_addr, &symbol, decimals, balance);

        // Convert balance to f64
        let balance = get_balance(balance)?;
        log::info!(  // Todo: why not log in test
            account = args.account,
            symbol = args.symbol,
            balance = balance;
            "balance_of_bsc"
        );

        return Ok((user_addr, balance));
    }
}
