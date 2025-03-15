use anda_core::{AgentInput, AgentOutput, BoxError, HttpFeatures, ToolInput, ToolOutput};
use anda_web3_client::client::{Client as Web3Client, load_identity};
use base64::{Engine, prelude::BASE64_URL_SAFE_NO_PAD};
use ciborium::value::Value;
use clap::{Parser, Subcommand};
use rand::{RngCore, thread_rng};
use std::sync::Arc;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[clap(short, long, default_value = "https://icp-api.io")]
    ic_host: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Generate random bytes with the given length and format
    RandBytes {
        /// Length of the random bytes, default is 32
        #[arg(short, long, default_value = "32")]
        len: usize,
        /// Output format: hex or base64, default is hex
        #[arg(short, long, default_value = "hex")]
        format: String,
    },

    /// make an signed RPC call to the endpoint with the given ICP identity, method and args.
    /// The RPC response from the endpoint should be string.
    /// Example: `anda_engine_cli rpc -i ./identity.pem -e 'https://andaicp.anda.bot/proposal'  -m start_x_bot`
    Rpc {
        #[arg(short, long, default_value = "http://127.0.0.1:8042/default")]
        endpoint: String,

        /// Path to ICP identity pem file or 32 bytes identity secret in hex.
        #[arg(short, long, env = "ID_SECRET", default_value = "Anonymous")]
        id_secret: String,

        /// RPC method name
        #[arg(short, long)]
        method: String,

        /// RPC arguments, default is []
        #[arg(short, long)]
        args: Option<Vec<String>>,
    },

    /// Run an AI agent with the given prompt and name on the endpoint.
    AgentRun {
        #[arg(short, long, default_value = "http://127.0.0.1:8042/default")]
        endpoint: String,

        /// Path to ICP identity pem file or 32 bytes identity secret in hex.
        #[arg(short, long, env = "ID_SECRET", default_value = "Anonymous")]
        id_secret: String,

        #[arg(short, long)]
        prompt: String,

        #[arg(short, long)]
        name: Option<String>,
    },

    /// Call a tool with the given name and args on the endpoint.
    ToolCall {
        #[arg(short, long, default_value = "http://127.0.0.1:8042/default")]
        endpoint: String,

        /// Path to ICP identity pem file or 32 bytes identity secret in hex.
        #[arg(short, long, env = "ID_SECRET", default_value = "Anonymous")]
        id_secret: String,

        #[arg(short, long)]
        name: String,

        #[arg(short, long)]
        args: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    dotenv::dotenv().ok();
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::RandBytes { len, format }) => {
            let mut rng = thread_rng();
            let mut bytes = vec![0u8; (*len).min(1024)];
            rng.fill_bytes(&mut bytes);
            match format.as_str() {
                "hex" => {
                    println!("{}", const_hex::encode(&bytes));
                }
                "base64" => {
                    println!("{}", BASE64_URL_SAFE_NO_PAD.encode(&bytes));
                }
                _ => {
                    println!("{:?}", bytes);
                }
            }
        }

        Some(Commands::Rpc {
            endpoint,
            id_secret,
            method,
            args,
        }) => {
            let identity = load_identity(id_secret)?;
            let web3 = Web3Client::builder()
                .with_ic_host(&cli.ic_host)
                .with_identity(Arc::new(identity))
                .with_allow_http(true, None)
                .build()
                .await?;

            println!("principal: {}", web3.get_principal());
            let args = args.clone().unwrap_or_default();

            let res: Value = web3.https_signed_rpc(endpoint, method, &args).await?;
            println!("{:?}", res);
        }

        Some(Commands::AgentRun {
            endpoint,
            id_secret,
            prompt,
            name,
        }) => {
            let identity = load_identity(id_secret)?;
            let web3 = Web3Client::builder()
                .with_ic_host(&cli.ic_host)
                .with_identity(Arc::new(identity))
                .with_allow_http(true, None)
                .build()
                .await?;

            println!("principal: {}", web3.get_principal());

            let res: AgentOutput = web3
                .https_signed_rpc(
                    endpoint,
                    "agent_run",
                    &(&AgentInput {
                        name: name.clone().unwrap_or_else(|| "".to_string()),
                        prompt: prompt.clone(),
                        ..Default::default()
                    },),
                )
                .await?;
            println!("{:?}", res);
        }

        Some(Commands::ToolCall {
            endpoint,
            id_secret,
            name,
            args,
        }) => {
            let identity = load_identity(id_secret)?;
            let web3 = Web3Client::builder()
                .with_ic_host(&cli.ic_host)
                .with_identity(Arc::new(identity))
                .with_allow_http(true, None)
                .build()
                .await?;

            println!("principal: {}", web3.get_principal());
            let args: serde_json::Value = serde_json::from_str(args)?;

            let res: ToolOutput<serde_json::Value> = web3
                .https_signed_rpc(
                    endpoint,
                    "tool_call",
                    &(&ToolInput {
                        name: name.clone(),
                        args,
                        ..Default::default()
                    },),
                )
                .await?;
            println!("{}", serde_json::to_string_pretty(&res)?);
        }

        None => {
            println!("no command");
        }
    }

    Ok(())
}
