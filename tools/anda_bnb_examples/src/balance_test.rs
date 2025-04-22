/// Tests the balance retrieval functionality for BNB ledgers.
///
/// This async function demonstrates the process of:
/// - Loading cryptographic identity and secrets
/// - Initializing a Web3 client for BNB Chain
/// - Creating BNB ledgers and a balance query tool
/// - Deriving a user's EVM address
/// - Querying balances for different token symbols
///
/// # Examples
/// 
/// test_bnb_ledger_balance().await;
/// 
use std::sync::Arc;

use alloy::hex;
use anda_web3_client::client::{
    Client as Web3Client, load_identity
};
use anda_engine::{
    extension::extractor::Extractor,
    context::Web3SDK,
    engine::EngineBuilder,
};
use anda_bnb::ledger::{BNBLedgers, BalanceOfArgs, BalanceOfTool, DRVT_PATH};
use anda_bnb::signer::derive_address_from_pubkey;
use anda_core::{KeysFeatures, Tool};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use structured_logger::Builder;

pub async fn test_bnb_ledger_balance() {
    Builder::with_level("debug").init();

    // Read identity and root secret
    let id_secret = dotenv::var("ID_SECRET").unwrap();
    let root_secret_org = dotenv::var("ROOT_SECRET").unwrap();

    // Parse and validate cryptographic secrets
    let identity = load_identity(&id_secret).unwrap();
    let root_secret = hex::decode(&root_secret_org).unwrap();
    let root_secret: [u8; 48] = root_secret
        .try_into()
        .map_err(|_| format!("invalid root_secret: {:?}", &root_secret_org))
        .unwrap();

    // Initialize Web3 client for BNB Chain network interaction
    let web3 = Web3Client::builder()
        .with_identity(Arc::new(identity))
        .with_root_secret(root_secret)
        .build().await.unwrap();

    // Create ledgers instance
    let ledgers = BNBLedgers::load().await.unwrap();
    let ledgers = Arc::new(ledgers);
    let tool = BalanceOfTool::new(ledgers.clone());
    let definition = tool.definition();
    assert_eq!(definition.name, "bnb_ledger_balance_of");

    // Create an agent for testing
    #[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
    struct TestStruct {
        name: String,
        age: Option<u8>,
    }    
    let agent = Extractor::<TestStruct>::default();

    // Create a context for testing
    let engine_ctx = EngineBuilder::new()
                .with_name("BNB_TEST".to_string()).unwrap()
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

    // Iterate through the ledgers and perform balance queries
    for (symbol, _) in ledgers.ledgers.clone() {
        // Create arguments for balance query
        let args = BalanceOfArgs {
            account: user_address.to_string(),
            symbol: symbol.into(),
        };

        // Call the tool to query balance
        let res = tool.call(base_ctx.clone(), args, None).await;
        assert!(res.is_ok(), "Balance query failed: {:?}", res);
        println!("Balance query result: {:#?}", res.unwrap());
    }
}
