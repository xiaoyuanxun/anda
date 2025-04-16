//! Enables AI Agent to query the balance of an account for a ICP token
//!
//! This module provides functionality for querying account balances on the ICP network.
//! It implements the [`Tool`] trait to enable AI agents to interact with ICP ledgers.

use anda_core::{BoxError, FunctionDefinition, Resource, Tool, ToolOutput, gen_schema_for};
use anda_engine::context::BaseCtx;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use super::BSCLedgers;

/// Arguments for the balance of an account for a token
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct BalanceOfArgs {
    /// ICP account address (principal) to query, e.g. "77ibd-jp5kr-moeco-kgoar-rro5v-5tng4-krif5-5h2i6-osf2f-2sjtv-kqe"
    pub account: String,
    /// Token symbol, e.g. "ICP"
    pub symbol: String,
}

/// ICP Ledger BalanceOf tool implementation
#[derive(Debug, Clone)]
pub struct BalanceOfTool {
    ledgers: Arc<BSCLedgers>,
    schema: Value,
}

impl BalanceOfTool {
    pub const NAME: &'static str = "bsc_ledger_balance_of";
    /// Creates a new BalanceOfTool instance
    pub fn new(ledgers: Arc<BSCLedgers>) -> Self {
        let schema = gen_schema_for::<BalanceOfArgs>();

        BalanceOfTool {
            ledgers,
            schema: json!(schema),
        }
    }
}

/// Implementation of the [`Tool`]` trait for BalanceOfTool
/// Enables AI Agent to query the balance of an account for a ICP token
impl Tool<BaseCtx> for BalanceOfTool {
    type Args = BalanceOfArgs;
    type Output = f64;

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
        format!(
            "Query the balance of the specified account on ICP blockchain for the following tokens: {}",
            tokens.join(", ")
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

    async fn call(
        &self,
        ctx: BaseCtx,
        data: Self::Args,
        _resources: Option<Vec<Resource>>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        let (_, amount) = self.ledgers.balance_of(ctx, data).await?;
        Ok(ToolOutput::new(amount))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candid::Principal;
    use std::collections::BTreeMap;
    use alloy::{
        hex, primitives::Address, providers::ProviderBuilder
    };
    use alloy::network::EthereumWallet;
    use anda_web3_client::client::{
        Client as Web3Client, load_identity
    };
    use anda_engine::{
        extension::extractor::Extractor,
        context::{Web3SDK, Web3ClientFeatures},
        engine::EngineBuilder,
    };
    use rand::Rng;
    use crate::ledger::{BSC_RPC, DRVT_PATH, TOKEN_ADDR};
    use crate::signer::{convert_to_boxed, derive_address_from_pubkey, AndaSigner};
    use super::super::ERC20STD;
    use anda_core::KeysFeatures;

    #[tokio::test]
    async fn test_bsc_ledger_balance() {
        let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Debug)
        .try_init();

        // Generate random bytes for identity and root secret
        let mut rng = rand::rng();
        let random_bytes: Vec<u8> = (0..32).map(|_| rng.random()).collect();
        let id_secret = hex::encode(random_bytes);
        let root_secret_org = dotenv::var("ROOT_SECRET").unwrap();  // Todo: use fixed root secret for test

        // Parse and validate cryptographic secrets
        let identity = load_identity(&id_secret).unwrap();
        let root_secret = const_hex::decode(&root_secret_org).unwrap();
        let root_secret: [u8; 48] = root_secret
            .try_into()
            .map_err(|_| format!("invalid root_secret: {:?}", &root_secret_org))
            .unwrap();

        // Initialize Web3 client for ICP network interaction
        let web3 = Web3Client::builder()
            // .with_ic_host(url)  // Todo: Why local url would hang the programe?
            .with_identity(Arc::new(identity))
            .with_root_secret(root_secret)
            .build().await.unwrap();

        let rpc_url = BSC_RPC.parse().unwrap();
        let provider = ProviderBuilder::new().on_http(rpc_url);
        // ERC20 token contract address and instance
        let token_addr =  Address::parse_checksummed(TOKEN_ADDR, None).unwrap();
        let contract = ERC20STD::new(token_addr, provider.clone());

        // Get token symbol and decimals
        let symbol = contract.symbol().call().await.unwrap();
        let decimals = contract.decimals().call().await.unwrap();
        log::debug!("symbol: {:?}, decimals: {:?}", symbol.clone(), decimals);

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

        let ledgers = Arc::new(ledgers);
        let tool = BalanceOfTool::new(ledgers.clone());
        let definition = tool.definition();
        assert_eq!(definition.name, "bsc_ledger_balance_of");
        let s = serde_json::to_string_pretty(&definition).unwrap();

        #[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
        struct TestStruct {
            name: String,
            age: Option<u8>,
        }    
        let agent = Extractor::<TestStruct>::default();

        let engine_ctx = EngineBuilder::new()
                    .with_name("BSC_TEST".to_string()).unwrap()
                    .with_web3_client(Arc::new(Web3SDK::from_web3(Arc::new(web3))))
                    .register_agent(agent).unwrap()
                    .mock_ctx();
        let base_ctx = engine_ctx.base.clone();

        // Derive EVM address from derivation path
        let pubkey_bytes = base_ctx.secp256k1_public_key(DRVT_PATH)
            .await
            .map_err(|e| 
                format!("Failed to get public key from derivation path: {:?}. Error: {:?}", DRVT_PATH, e.to_string()
                ) 
            ).unwrap();
        let user_address = derive_address_from_pubkey(&pubkey_bytes).unwrap();
        log::debug!("User pubkey: {:?}, User EVM address: {:?}",
                    hex::encode(pubkey_bytes), user_address);

        let args = BalanceOfArgs {
            account: user_address.to_string(),
            symbol: symbol.clone(),
        };
        tool.call(base_ctx, args, None).await.unwrap();
    }
}
