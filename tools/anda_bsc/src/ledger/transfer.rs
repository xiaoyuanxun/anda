//! Enables AI Agent to perform ICP token transfers
//!
//! Provides functionality for transferring tokens between accounts on the Internet Computer Protocol (ICP) network.
//! Supports:
//! - Multiple token types (e.g., ICP, PANDA)
//! - Memo fields for transaction identification
//! - Integration with ICP ledger standards
//! - Atomic transfers with proper error handling

use anda_core::{
    BoxError, FunctionDefinition, Resource, Tool, ToolOutput, gen_schema_for,
};
use anda_engine::context::BaseCtx;
use num_traits::cast::ToPrimitive;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::str::FromStr;
use alloy::primitives::Address;
use super::BSCLedgers;

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
    ledgers: Arc<BSCLedgers>,
    schema: Value,
}

impl TransferTool {
    pub const NAME: &'static str = "bsc_ledger_transfer";

    pub fn new(ledgers: Arc<BSCLedgers>) -> Self {
        let schema = gen_schema_for::<TransferToArgs>();

        TransferTool { ledgers, schema }
    }
}

/// Implementation of the [`Tool`] trait for TransferTool
/// Enables AI Agent to perform ICP token transfers
impl Tool<BaseCtx> for TransferTool {
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

    async fn call(
        &self,
        ctx: BaseCtx,
        data: Self::Args,
        _resources: Option<Vec<Resource>>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        let address = Address::from_str(&data.account)?; // Todo: pass sender address as a parameter in call
        let (ledger, tx) = self.ledgers.transfer(&ctx, address, data).await?;
        Ok(ToolOutput::new(format!(
            "Successful, transaction ID: {}, detail: https://www.icexplorer.io/token/details/{}", // Todo: change for BS
            tx.0.to_u64().unwrap_or(0),
            ledger.to_string()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anda_engine::{
        extension::extractor::Extractor,
        context::Web3SDK,
        engine::EngineBuilder,
    };
    use anda_web3_client::client::{
        Client as Web3Client, load_identity
    };
    use alloy::{
        hex, primitives::address, providers::ProviderBuilder
    };
    use std::collections::BTreeMap;
    use rand::Rng;
    use super::super::{BSCLedgers, ERC20STD};
    use super::super::super::utils_evm::{
        generate_secret_key, derive_evm_address
    };

    #[tokio::test(flavor = "current_thread")]
    async fn test_bsc_ledger_transfer() {
        let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Debug)
        .try_init();

        // Generate random bytes for identity and root secret
        let mut rng = rand::thread_rng();
        let random_bytes: Vec<u8> = (0..32).map(|_| rng.r#gen()).collect();
        let id_secret = hex::encode(random_bytes);
        let random_bytes: Vec<u8> = (0..48).map(|_| rng.r#gen()).collect();  // Todo: why gen is a keyword?
        // let root_secret_org = hex::encode(random_bytes);
        let root_secret_org = dotenv::var("ROOT_SECRET").unwrap();  // Todo: Read root secret from rng

        // Parse and validate cryptographic secrets
        let identity = load_identity(&id_secret).unwrap();
        let root_secret = const_hex::decode(&root_secret_org).unwrap();
        let root_secret: [u8; 48] = root_secret
            .try_into()
            .map_err(|_| format!("invalid root_secret: {:?}", &root_secret_org))
            .unwrap();

        // Initialize Web3 client for ICP network interaction
        let web3 = Web3Client::builder()
        .with_ic_host("https://bsc-testnet.bnbchain.org")
        .with_identity(Arc::new(identity))
        .with_root_secret(root_secret)
        .build().await.unwrap();
    
        // Derive EVM address from derivation path
        // let derivation_path: &[&[u8]] = &[b"44'", b"60'", b"10'", b"20", b"30"];  // Todo: how to retrieve derivation path?
        // let pk = web3.ed25519_public_key(&derivation_path).await.unwrap();

        // Generate sepc256k1 secrete key from root secret
        let sk = generate_secret_key(&root_secret.as_slice()).unwrap();
        derive_evm_address(&sk); // For debugging

        let rpc_url = "https://bsc-testnet.bnbchain.org".parse().unwrap();
        let provider = ProviderBuilder::new().on_http(rpc_url);
        let token_addr = address!("0xDE3a190D9D26A8271Ae9C27573c03094A8A2c449");
        let contract = ERC20STD::new(token_addr, provider.clone());

        // Get token symbol.
        let symbol = contract.symbol().call().await.unwrap()._0;
        let decimals = contract.decimals().call().await.unwrap()._0;
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
        let tool = TransferTool::new(ledgers.clone());
        let definition = tool.definition();
        assert_eq!(definition.name, "bsc_ledger_transfer");
        assert_eq!(tool.description().contains(&symbol), true);

        let to_addr = address!("0xDE3a190D9D26A8271Ae9C27573c03094A8A2c449");
        let transfer_amount = 0.0;  // Todo: set a positive amount
        let transfer_to_args = TransferToArgs {
            account: to_addr.to_string(),
            symbol: symbol.clone(),
            amount: transfer_amount,
        };

        #[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
        struct TestStruct {
            name: String,
            age: Option<u8>,
        }    
        let agent = Extractor::<TestStruct>::default();

        let engine_ctx = EngineBuilder::new()
                    // .with_id(principal)  // Todo: how to retrieve sender address?
                    .with_name("BSC_TEST".to_string()).unwrap()
                    .with_web3_client(Arc::new(Web3SDK::from_web3(Arc::new(web3.clone()))))
                    .register_agent(agent).unwrap()
                    .mock_ctx();
        let base_ctx = engine_ctx.base.clone();
        tool.call(base_ctx, transfer_to_args, None).await.unwrap();
    }
}
