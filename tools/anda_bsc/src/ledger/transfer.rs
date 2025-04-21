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
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
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
        let (ledger, tx) = self.ledgers.transfer(ctx, data).await?;
        Ok(ToolOutput::new(format!(
            "Successful transfer, receipient address: {}, detail: https://www.bscscan.com/tx/{}",
            ledger.to_string(),
            tx.to_string()
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
    use alloy::primitives::address;
    use super::super::BSCLedgers;

    #[tokio::test(flavor = "current_thread")]
    async fn test_bsc_ledger_transfer() {
        let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Debug)
        .try_init();

        // Generate random bytes for identity and root secret
        let id_secret = dotenv::var("ID_SECRET").unwrap();
        let root_secret_org = dotenv::var("ROOT_SECRET").unwrap();

        // Parse cryptographic secrets
        let identity = load_identity(&id_secret).unwrap();
        let root_secret = const_hex::decode(&root_secret_org).unwrap();
        let root_secret: [u8; 48] = root_secret
            .try_into()
            .map_err(|_| format!("invalid root_secret: {:?}", &root_secret_org))
            .unwrap();

        // Create a BSC ledger instance
        let ledgers = BSCLedgers::load().await.unwrap();
        let ledgers = Arc::new(ledgers);
        let tool = TransferTool::new(ledgers.clone());
        let definition = tool.definition();
        assert_eq!(definition.name, "bsc_ledger_transfer");
        assert_eq!(tool.description()
                    .contains(ledgers.ledgers.clone().first_key_value().unwrap().0), true);

        // Create an agent for testing
        #[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
        struct TestStruct {
            name: String,
            age: Option<u8>,
        }    
        let agent = Extractor::<TestStruct>::default();

        // Initialize Web3 client for BSC network
        let web3 = Web3Client::builder()
            .with_identity(Arc::new(identity))
            .with_root_secret(root_secret)
            .build().await.unwrap();
    
        // Create a context for testing
        let engine_ctx = EngineBuilder::new()
                    .with_name("BSC_TEST".to_string()).unwrap()
                    .with_web3_client(Arc::new(Web3SDK::from_web3(Arc::new(web3))))
                    .register_agent(agent).unwrap()
                    .mock_ctx();
        let base_ctx = engine_ctx.base;

        // Iterate through the ledgers and perform transfers
        for (symbol, _) in ledgers.ledgers.clone() {
            // Init transfer arguments
            let to_addr = address!("0xA8c4AAE4ce759072D933bD4a51172257622eF128");  // Receiver addr
            let transfer_amount = 0.00012;
            let transfer_to_args = TransferToArgs {
                account: to_addr.to_string(),
                symbol: symbol.clone(),
                amount: transfer_amount,
            };
            
            // Call tool to transfer tokens
            let res = tool.call(base_ctx.clone(), transfer_to_args, None).await;
            assert!(res.is_ok(), "Transfer failed: {:?}", res);
            println!("Transfer result: {:#?}", res.unwrap());            
        }

    }
}
