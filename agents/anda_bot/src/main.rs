use agent_twitter_client::scraper::Scraper;
use anda_core::{BoxError, EmbeddingFeatures, Path};
use anda_engine::{
    context::{derivation_path_with, TEEClient, Web3SDK},
    engine::{Engine, EngineBuilder, ROOT_PATH},
    extension::{
        attention::Attention,
        character::{Character, CharacterAgent},
        google::GoogleSearchTool,
        segmenter::DocumentSegmenter,
    },
    model::{cohere, deepseek, openai, Model},
    store::Store,
    APP_USER_AGENT,
};
use anda_icp::ledger::{BalanceOfTool, ICPLedgers};
use anda_lancedb::{
    knowledge::KnowledgeStore,
    lancedb::{DynObjectStore, LanceVectorStore, LocalFileSystem},
};
use anda_web3_client::client::{load_identity, Client as Web3Client};
use axum::{routing, Router};
use candid::Principal;
use clap::{Parser, Subcommand};
use ed25519_consensus::SigningKey;
use ic_agent::{
    identity::{BasicIdentity, Identity},
    Agent,
};
use ic_cose::client::CoseSDK;
use ic_cose_types::{
    types::{object_store::CHUNK_SIZE, setting::SettingPath},
    CanisterCaller,
};
use ic_object_store::{
    agent::build_agent,
    client::{Client, ObjectStoreClient},
};
use ic_tee_agent::setting::decrypt_payload;
use std::collections::BTreeSet;
use std::{net::SocketAddr, sync::Arc, time::Duration};
use structured_logger::{async_json::new_writer, get_env_level, unix_ms, Builder};
use tokio::{net::TcpStream, signal, sync::RwLock};
use tokio_util::sync::CancellationToken;

mod config;
mod handler;
mod ic_sig_verifier;
mod twitter;

const APP_NAME: &str = env!("CARGO_PKG_NAME");
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

static LOG_TARGET: &str = "anda_bot";
static IC_OBJECT_STORE: &str = "ic://object_store";
static ENGINE_NAME: &str = "Anda.bot";
static COSE_SECRET_PERMANENT_KEY: &str = "v1";
const LOCAL_SERVER_SHUTDOWN_DURATION: Duration = Duration::from_secs(5);

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Port to listen on
    #[clap(short, long, default_value = "8042")]
    port: u16,

    /// ICP API host
    #[clap(short, long, default_value = "https://icp-api.io")]
    ic_host: String,

    /// Path to the character file
    #[clap(
        short,
        long,
        env = "CHARACTER_FILE_PATH",
        default_value = "./Character.toml"
    )]
    character: String,

    /// where the logtail server is running on host (e.g. 127.0.0.1:9999)
    #[clap(short, long)]
    logtail: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    StartTee {
        #[clap(long, default_value = "http://127.0.0.1:8080")]
        tee_host: String,
        /// Basic auth to request TEE service
        #[clap(long)]
        basic_token: String,

        /// COSE canister
        #[clap(long)]
        cose_canister: String,

        /// COSE namespace
        #[clap(long)]
        cose_namespace: String,

        /// COSE canister
        #[clap(long)]
        object_store_canister: String,
    },
    StartLocal {
        /// Path to ICP identity pem file or 32 bytes identity secret in hex.
        #[arg(short, long, env = "ID_SECRET")]
        id_secret: String,

        /// 48 bytes root secret in hex to derive keys
        #[arg(long, env = "ROOT_SECRET")]
        root_secret: String,

        /// Path to the configuration file
        #[clap(long, env = "CONFIG_FILE_PATH", default_value = "./Config.toml")]
        config: String,

        #[clap(long, env = "OBJECT_STORE_PATH", default_value = "./object_store")]
        store_path: String,

        /// Manager principal
        #[clap(long, default_value = "")]
        manager: String,
    },
}

// cargo run -p anda_bot -- start-local
fn main() -> Result<(), BoxError> {
    dotenv::dotenv().ok();

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let cli = Cli::parse();

            let writer = if let Some(logtail) = &cli.logtail {
                let stream = TcpStream::connect(logtail).await?;
                stream.writable().await?;
                new_writer(stream)
            } else {
                new_writer(tokio::io::stdout())
            };
            Builder::with_level(&get_env_level().to_string())
                .with_target_writer("*", writer)
                .init();

            log::info!(target: LOG_TARGET, "bootstrap {}@{}", APP_NAME, APP_VERSION);
            match bootstrap(cli).await {
                Ok(_) => Ok(()),
                Err(err) => {
                    log::error!(target: LOG_TARGET, "bootstrap error: {:?}", err);
                    tokio::time::sleep(Duration::from_secs(3)).await;
                    Err(err)
                }
            }
        })
}

async fn bootstrap(cli: Cli) -> Result<(), BoxError> {
    let character = std::fs::read_to_string(&cli.character)?;
    let character = Character::from_toml(&character)?;
    log::info!("{:?}", character);

    match cli.command {
        Some(Commands::StartTee {
            tee_host,
            basic_token,
            cose_canister,
            cose_namespace,
            object_store_canister,
        }) => {
            bootstrap_tee(
                cli.port,
                cli.ic_host,
                tee_host,
                basic_token,
                cose_canister,
                cose_namespace,
                object_store_canister,
                character,
            )
            .await
        }
        Some(Commands::StartLocal {
            id_secret,
            root_secret,
            config,
            store_path,
            manager,
        }) => {
            let cfg = config::Conf::from_file(&config)?;
            log::debug!("{:?}", cfg);
            let root_secret = const_hex::decode(root_secret)?;
            let root_secret: [u8; 48] =
                root_secret.try_into().map_err(|_| "invalid root_secret")?;

            bootstrap_local(
                cli.port,
                cli.ic_host,
                &id_secret,
                root_secret,
                cfg,
                character,
                store_path,
                manager,
            )
            .await
        }
        None => {
            println!("{}@{}", APP_NAME, APP_VERSION);
            Err("missing subcommand".into())
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn bootstrap_tee(
    port: u16,
    ic_host: String,
    tee_host: String,
    basic_token: String,
    cose_canister: String,
    cose_namespace: String,
    object_store_canister: String,
    character: Character,
) -> Result<(), BoxError> {
    let global_cancel_token = CancellationToken::new();
    let shutdown_future = shutdown_signal(global_cancel_token.clone());

    let engine_name = ENGINE_NAME.to_string();
    let default_agent = character.username.clone();
    let knowledge_table: Path = default_agent.to_ascii_lowercase().into();
    let cose_setting_key: Vec<u8> = default_agent.to_ascii_lowercase().into();

    let cose_canister = Principal::from_text(&cose_canister)?;
    log::info!(target: LOG_TARGET, "start to connect TEE service");
    let tee = TEEClient::new(&tee_host, &basic_token, APP_USER_AGENT, cose_canister);
    let tee_info = tee.connect_tee(global_cancel_token.clone()).await?;
    log::info!(target: LOG_TARGET, "TEEAppInformation: {:?}", tee_info);

    let root_path = Path::from(ROOT_PATH);
    let id_secret = tee
        .a256gcm_key(&derivation_path_with(
            &root_path,
            &[default_agent.as_bytes()],
        ))
        .await?;
    let my_id = BasicIdentity::from_signing_key(SigningKey::from(id_secret));
    let my_principal = my_id.sender()?;
    log::info!(target: LOG_TARGET,
       "sign_in, principal: {:?}", my_principal.to_text());

    let my_agent = build_agent(&ic_host, Arc::new(my_id)).await.unwrap();

    log::info!(target: LOG_TARGET, "start to get admin_master_secret");
    let admin_master_secret = tee
        .get_cose_encrypted_key(&SettingPath {
            ns: cose_namespace.clone(),
            user_owned: false,
            subject: Some(tee_info.id),
            key: COSE_SECRET_PERMANENT_KEY.as_bytes().to_vec().into(),
            version: 0,
        })
        .await?;

    log::info!(target: LOG_TARGET, "start to get encrypted config");
    let encrypted_cfg_path = SettingPath {
        ns: cose_namespace.clone(),
        user_owned: false,
        subject: Some(tee_info.id),
        key: cose_setting_key.into(),
        version: 0,
    };
    let encrypted_cfg = match tee.setting_get(&encrypted_cfg_path).await {
        Ok(setting) => {
            let encrypted_cfg = decrypt_payload(&setting, &admin_master_secret, &[])?;

            config::Conf::from_toml(&String::from_utf8(encrypted_cfg)?)?
        }
        Err(err) => {
            log::info!(
                "get encrypted_cfg error: {:?}\n{:?}",
                err,
                &encrypted_cfg_path
            );

            return Err(err.into());
        }
    };

    // LL Models
    log::info!(target: LOG_TARGET, "start to connect models");
    let model = connect_model(&encrypted_cfg.llm)?;

    // ObjectStore
    log::info!(target: LOG_TARGET, "start to connect object_store");
    let object_store_canister = Principal::from_text(object_store_canister)?;
    let object_store =
        connect_object_store(&tee, Arc::new(my_agent), &root_path, object_store_canister).await?;
    let object_store = Arc::new(object_store);
    let os_state = object_store.get_state().await?;
    log::info!(target: LOG_TARGET, "object_store state: {:?}", os_state);

    log::info!(target: LOG_TARGET, "start to init knowledge_store");
    let knowledge_store =
        connect_knowledge_store(object_store.clone(), knowledge_table, &model).await?;

    let knowledge_store = Arc::new(knowledge_store);
    log::info!(target: LOG_TARGET, "start to build engine");
    let agent = character.build(
        Arc::new(Attention::default()),
        Arc::new(DocumentSegmenter::default()),
        knowledge_store.clone(),
    );

    let mut engine = EngineBuilder::new()
        .with_id(tee_info.id)
        .with_name(engine_name.clone())
        .with_cancellation_token(global_cancel_token.clone())
        .with_web3_client(Arc::new(Web3SDK::from_tee(tee.clone())))
        .with_model(model)
        .with_store(Store::new(object_store));

    if !encrypted_cfg.google.api_key.is_empty() {
        engine = engine.register_tool(GoogleSearchTool::new(
            encrypted_cfg.google.api_key.clone(),
            encrypted_cfg.google.search_engine_id.clone(),
            None,
        ))?;
    }
    if !encrypted_cfg.icp.token_ledgers.is_empty() {
        let token_ledgers: BTreeSet<Principal> = encrypted_cfg
            .icp
            .token_ledgers
            .iter()
            .flat_map(|t| Principal::from_text(t).map_err(|_| format!("invalid token: {}", t)))
            .collect();

        let ledgers = ICPLedgers::load(&tee, token_ledgers, false).await?;
        let ledgers = Arc::new(ledgers);
        engine = engine.register_tool(BalanceOfTool::new(ledgers.clone()))?;
    }

    engine = engine.register_agent(agent.clone())?;

    let agent = Arc::new(agent);
    let engine = Arc::new(engine.build(default_agent.clone())?);
    let x_status = Arc::new(RwLock::new(handler::ServiceStatus::Running));
    let app_state = handler::AppState {
        web3: Arc::new(handler::Web3SDK::Tee(tee)),
        x_status: x_status.clone(),
        info: Arc::new(handler::AppInformation {
            id: my_principal,
            name: engine_name,
            start_time_ms: unix_ms(),
            default_agent,
            object_store_canister: Some(object_store_canister),
            caller: Principal::anonymous(),
        }),
        knowledge_store,
        cose_namespace,
        manager: "".to_string(),
    };

    match tokio::try_join!(
        start_server(
            format!("127.0.0.1:{}", port),
            app_state,
            global_cancel_token.clone()
        ),
        start_x(
            encrypted_cfg.x,
            engine,
            agent,
            global_cancel_token.clone(),
            x_status
        ),
        shutdown_future
    ) {
        Ok(_) => Ok(()),
        Err(err) => {
            log::error!(target: LOG_TARGET, "server error: {:?}", err);
            Err(err)
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn bootstrap_local(
    port: u16,
    ic_host: String,
    id_secret: &str,
    root_secret: [u8; 48],
    cfg: config::Conf,
    character: Character,
    store_path: String,
    manager: String,
) -> Result<(), BoxError> {
    let global_cancel_token = CancellationToken::new();
    let shutdown_future = shutdown_signal(global_cancel_token.clone());

    let engine_name = ENGINE_NAME.to_string();
    let default_agent = character.username.clone();
    let knowledge_table: Path = default_agent.to_ascii_lowercase().into();

    let identity = load_identity(id_secret)?;
    let web3 = Web3Client::builder()
        .with_ic_host(&ic_host)
        .with_identity(Arc::new(identity))
        .with_root_secret(root_secret)
        .build()
        .await?;
    let my_principal = web3.get_principal();
    log::info!(target: LOG_TARGET, "start local service, principal: {:?}", my_principal.to_text());

    // LL Models
    log::info!(target: LOG_TARGET, "start to connect models");
    let model = connect_model(&cfg.llm)?;

    // ObjectStore
    log::info!(target: LOG_TARGET, "start to connect object_store");
    let object_store = LocalFileSystem::new_with_prefix(store_path)?;
    let object_store = Arc::new(object_store);

    log::info!(target: LOG_TARGET, "start to init knowledge_store");
    let knowledge_store =
        connect_knowledge_store(object_store.clone(), knowledge_table, &model).await?;

    let knowledge_store = Arc::new(knowledge_store);
    log::info!(target: LOG_TARGET, "start to build engine");
    let agent = character.build(
        Arc::new(Attention::default()),
        Arc::new(DocumentSegmenter::default()),
        knowledge_store.clone(),
    );

    let mut engine = EngineBuilder::new()
        .with_id(my_principal)
        .with_name(engine_name.clone())
        .with_cancellation_token(global_cancel_token.clone())
        .with_web3_client(Arc::new(Web3SDK::from_web3(Arc::new(web3.clone()))))
        .with_model(model)
        .with_store(Store::new(object_store));

    if !cfg.google.api_key.is_empty() {
        engine = engine.register_tool(GoogleSearchTool::new(
            cfg.google.api_key.clone(),
            cfg.google.search_engine_id.clone(),
            None,
        ))?;
    }
    if !cfg.icp.token_ledgers.is_empty() {
        let token_ledgers: BTreeSet<Principal> = cfg
            .icp
            .token_ledgers
            .iter()
            .flat_map(|t| Principal::from_text(t).map_err(|_| format!("invalid token: {}", t)))
            .collect();

        let ledgers = ICPLedgers::load(&web3, token_ledgers, false).await?;
        let ledgers = Arc::new(ledgers);
        engine = engine.register_tool(BalanceOfTool::new(ledgers.clone()))?;
    }

    engine = engine.register_agent(agent.clone())?;

    let agent = Arc::new(agent);
    let engine = Arc::new(engine.build(default_agent.clone())?);
    let x_status = Arc::new(RwLock::new(handler::ServiceStatus::Running));
    let app_state = handler::AppState {
        web3: Arc::new(handler::Web3SDK::Web3(web3)),
        x_status: x_status.clone(),
        info: Arc::new(handler::AppInformation {
            id: my_principal,
            name: engine_name,
            start_time_ms: unix_ms(),
            default_agent,
            object_store_canister: None,
            caller: Principal::anonymous(),
        }),
        knowledge_store,
        cose_namespace: "".to_string(),
        manager,
    };

    match tokio::try_join!(
        start_server(
            format!("127.0.0.1:{}", port),
            app_state,
            global_cancel_token.clone()
        ),
        start_x(cfg.x, engine, agent, global_cancel_token.clone(), x_status),
        shutdown_future
    ) {
        Ok(_) => Ok(()),
        Err(err) => {
            log::error!(target: LOG_TARGET, "server error: {:?}", err);
            Err(err)
        }
    }
}

async fn connect_object_store(
    tee: &TEEClient,
    ic_agent: Arc<Agent>,
    root_path: &Path,
    object_store_canister: Principal,
) -> Result<ObjectStoreClient, BoxError> {
    let aes_secret = tee
        .a256gcm_key(&derivation_path_with(
            root_path,
            &[IC_OBJECT_STORE.as_bytes(), b"A256GCM"],
        ))
        .await?;

    // ensure write access to object store
    let my_principal = ic_agent.get_principal()?;
    let res: Result<bool, String> = tee
        .canister_query(
            &object_store_canister,
            "is_member",
            ("manager", &my_principal),
        )
        .await?;
    if !res? {
        let res: Result<(), String> = tee
            .canister_update(
                &object_store_canister,
                "admin_add_managers",
                (vec![&my_principal],),
            )
            .await?;
        res?;
    }
    let client = Client::new(ic_agent, object_store_canister, Some(aes_secret));
    Ok(ObjectStoreClient::new(Arc::new(client)))
}

async fn connect_knowledge_store(
    object_store: Arc<DynObjectStore>,
    namespace: Path,
    model: &Model,
) -> Result<KnowledgeStore, BoxError> {
    let mut store = LanceVectorStore::new_with_object_store(
        IC_OBJECT_STORE.to_string(),
        object_store,
        Some(CHUNK_SIZE),
        Some(model.embedder.clone()),
    )
    .await?;

    log::info!(target: LOG_TARGET, "knowledge_store start init");
    let ks =
        KnowledgeStore::init(&mut store, namespace, model.ndims() as u16, Some(1024 * 10)).await?;
    log::info!(target: LOG_TARGET, "knowledge_store ks: {:?}", ks.name());
    Ok(ks)
}

fn connect_model(cfg: &config::Llm) -> Result<Model, BoxError> {
    if cfg.openai_api_key.is_empty() {
        Ok(Model::new(
            Arc::new(
                deepseek::Client::new(
                    &cfg.deepseek_api_key,
                    if cfg.deepseek_endpoint.is_empty() {
                        None
                    } else {
                        Some(cfg.deepseek_endpoint.clone())
                    },
                )
                .completion_model(if cfg.deepseek_model.is_empty() {
                    deepseek::DEEKSEEK_V3
                } else {
                    &cfg.deepseek_model
                }),
            ),
            Arc::new(
                cohere::Client::new(&cfg.cohere_api_key)
                    .embedding_model(&cfg.cohere_embedding_model),
            ),
        ))
    } else {
        let cli = openai::Client::new(
            &cfg.openai_api_key,
            if cfg.deepseek_endpoint.is_empty() {
                None
            } else {
                Some(cfg.deepseek_endpoint.clone())
            },
        );
        Ok(Model::new(
            Arc::new(cli.completion_model(&cfg.openai_completion_model)),
            Arc::new(cli.embedding_model(&cfg.openai_embedding_model)),
        ))
    }
}

async fn start_server(
    addr: String,
    app_state: handler::AppState,
    cancel_token: CancellationToken,
) -> Result<(), BoxError> {
    let app = Router::new()
        .route("/.well-known/app", routing::get(handler::get_information))
        .route("/proposal", routing::post(handler::add_proposal))
        .with_state(app_state);

    let addr: SocketAddr = addr.parse()?;
    let listener = create_reuse_port_listener(addr).await?;

    log::warn!(target: LOG_TARGET,
                "{}@{} listening on {:?}", APP_NAME, APP_VERSION, addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = cancel_token.cancelled().await;
            tokio::time::sleep(LOCAL_SERVER_SHUTDOWN_DURATION).await;
        })
        .await?;
    Ok(())
}

async fn start_x(
    cfg: config::X,
    engine: Arc<Engine>,
    agent: Arc<CharacterAgent<KnowledgeStore>>,
    cancel_token: CancellationToken,
    status: Arc<RwLock<handler::ServiceStatus>>,
) -> Result<(), BoxError> {
    let mut scraper = Scraper::new().await?;

    let cookie_str = cfg.cookie_string.unwrap_or_default();
    if !cookie_str.is_empty() {
        scraper.set_from_cookie_string(&cookie_str).await?;
    } else {
        scraper
            .login(
                cfg.username.clone(),
                cfg.password.clone(),
                cfg.email,
                cfg.two_factor_auth,
            )
            .await?;
    }

    let x = twitter::TwitterDaemon::new(engine, agent, scraper, status);
    x.run(cancel_token).await
}

async fn shutdown_signal(cancel_token: CancellationToken) -> Result<(), BoxError> {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    log::warn!(target: LOG_TARGET, "received termination signal, starting graceful shutdown");
    cancel_token.cancel();
    tokio::time::sleep(LOCAL_SERVER_SHUTDOWN_DURATION).await;

    Ok(())
}

async fn create_reuse_port_listener(addr: SocketAddr) -> Result<tokio::net::TcpListener, BoxError> {
    let socket = match &addr {
        SocketAddr::V4(_) => tokio::net::TcpSocket::new_v4()?,
        SocketAddr::V6(_) => tokio::net::TcpSocket::new_v6()?,
    };

    socket.set_reuseport(true)?;
    socket.bind(addr)?;
    let listener = socket.listen(1024)?;
    Ok(listener)
}
