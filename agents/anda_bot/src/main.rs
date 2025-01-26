use agent_twitter_client::scraper::Scraper;
use anda_core::{BoxError, EmbeddingFeatures, Path};
use anda_engine::{
    context::TEEClient,
    engine::{Engine, EngineBuilder, ROOT_PATH},
    extension::{
        attention::Attention,
        character::{Character, CharacterAgent},
        google::GoogleSearchTool,
        segmenter::DocumentSegmenter,
    },
    model::{cohere, deepseek, openai, Model},
    store::Store,
};
use anda_icp::ledger::{BalanceOfTool, ICPLedgers};
use anda_lancedb::{knowledge::KnowledgeStore, lancedb::LanceVectorStore};
use axum::{routing, Router};
use candid::Principal;
use ciborium::from_reader;
use clap::Parser;
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
use ic_tee_cdk::TEEAppInformation;
use std::collections::BTreeSet;
use std::{net::SocketAddr, sync::Arc, time::Duration};
use structured_logger::{async_json::new_writer, get_env_level, unix_ms, Builder};
use tokio::{net::TcpStream, signal, sync::RwLock, time::sleep};
use tokio_util::sync::CancellationToken;

mod config;
mod handler;
mod twitter;

const APP_NAME: &str = env!("CARGO_PKG_NAME");
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

static LOG_TARGET: &str = "anda_bot";
static IC_OBJECT_STORE: &str = "ic://object_store";
static ENGINE_NAME: &str = "Anda.bot";
static COSE_SECRET_PERMANENT_KEY: &str = "v1";
const LOCAL_SERVER_SHUTDOWN_DURATION: Duration = Duration::from_secs(5);

#[derive(Debug, Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    /// Port to listen on
    #[clap(long, default_value = "8042")]
    port: u16,

    /// Path to the configuration file
    #[clap(long, env = "CONFIG_FILE_PATH", default_value = "./Config.toml")]
    config: String,

    #[clap(long, env = "CHARACTER_FILE_PATH", default_value = "./Character.toml")]
    character: String,

    /// where the logtail server is running on host (e.g. 127.0.0.1:9999)
    #[clap(long)]
    logtail: Option<String>,
}

// cargo run -p anda_bot -- --port 8042 --config agents/anda_bot/nitro_enclave/Config.toml --character agents/anda_bot/nitro_enclave/Character.toml
fn main() -> Result<(), BoxError> {
    let default_stack_size = 2 * 1024 * 1024;
    let stack_size = std::env::var("RUST_MIN_STACK")
        .map(|s| s.parse().expect("RUST_MIN_STACK must be a valid number"))
        .unwrap_or(default_stack_size);

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(stack_size)
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
    let cfg = config::Conf::from_file(&cli.config).unwrap_or_else(|err| {
        println!("config error: {:?}", err);
        panic!("config error: {:?}", err)
    });
    log::info!("{:?}", cfg);

    let character = std::fs::read_to_string(&cli.character)?;
    let character = Character::from_toml(&character)?;
    log::info!("{:?}", character);

    let global_cancel_token = CancellationToken::new();
    let shutdown_future = shutdown_signal(global_cancel_token.clone());

    let engine_name = ENGINE_NAME.to_string();
    let cose_canister = Principal::from_text(&cfg.icp.cose_canister)?;
    let default_agent = character.username.clone();
    let knowledge_table: Path = default_agent.to_ascii_lowercase().into();
    let cose_setting_key: Vec<u8> = default_agent.to_ascii_lowercase().into();

    log::info!(target: LOG_TARGET, "start to connect TEE service");
    let tee = TEEClient::new(&cfg.tee.tee_host, &cfg.tee.basic_token, cose_canister);
    let tee_info = connect_tee(&cfg, &tee, global_cancel_token.clone()).await?;
    log::info!(target: LOG_TARGET, "TEEAppInformation: {:?}", tee_info);

    let root_path = Path::from(ROOT_PATH);
    let id_secret = tee
        .a256gcm_key(&root_path, &[default_agent.as_bytes()])
        .await?;
    let my_id = BasicIdentity::from_signing_key(SigningKey::from(id_secret));
    let my_principal = my_id.sender()?;
    log::info!(target: LOG_TARGET,
       "sign_in, principal: {:?}", my_principal.to_text());

    let my_agent = build_agent(&cfg.icp.api_host, Arc::new(my_id))
        .await
        .unwrap();

    log::info!(target: LOG_TARGET, "start to get admin_master_secret");
    let admin_master_secret = tee
        .get_cose_encrypted_key(&SettingPath {
            ns: cfg.icp.cose_namespace.clone(),
            user_owned: false,
            subject: Some(tee_info.id),
            key: COSE_SECRET_PERMANENT_KEY.as_bytes().to_vec().into(),
            version: 0,
        })
        .await?;

    log::info!(target: LOG_TARGET, "start to get encrypted config");
    let encrypted_cfg_path = SettingPath {
        ns: cfg.icp.cose_namespace.clone(),
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

            cfg.clone()
        }
    };

    // LL Models
    log::info!(target: LOG_TARGET, "start to connect models");
    let model = connect_model(&encrypted_cfg.llm)?;

    // ObjectStore
    log::info!(target: LOG_TARGET, "start to connect object_store");
    let object_store_canister = Principal::from_text(cfg.icp.object_store_canister)?;
    let object_store =
        connect_object_store(&tee, Arc::new(my_agent), &root_path, object_store_canister).await?;
    let object_store = Arc::new(object_store);
    let os_state = object_store.get_state().await?;
    log::info!(target: LOG_TARGET, "object_store state: {:?}", os_state);

    log::info!(target: LOG_TARGET, "start to init knowledge_store");
    let knowledge_store =
        connect_knowledge_store(object_store.clone(), knowledge_table, &model).await?;

    log::info!(target: LOG_TARGET, "start to build engine");
    let agent = character.build(
        Attention::default(),
        DocumentSegmenter::default(),
        knowledge_store,
    );

    let mut engine = EngineBuilder::new()
        .with_id(tee_info.id)
        .with_name(engine_name.clone())
        .with_cancellation_token(global_cancel_token.clone())
        .with_tee_client(tee.clone())
        .with_model(model)
        .with_store(Store::new(object_store));

    if !encrypted_cfg.google.api_key.is_empty() {
        engine = engine.register_tool(GoogleSearchTool::new(
            encrypted_cfg.google.api_key.clone(),
            encrypted_cfg.google.search_engine_id.clone(),
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

        let ledgers = ICPLedgers::load(&tee, token_ledgers, false).await?;
        let ledgers = Arc::new(ledgers);
        engine = engine.register_tool(BalanceOfTool::new(ledgers.clone()))?;
    }

    engine = engine.register_agent(agent.clone())?;

    let agent = Arc::new(agent);
    let engine = Arc::new(engine.build(default_agent.clone())?);
    let x_status = Arc::new(RwLock::new(handler::ServiceStatus::Running));
    let app_state = handler::AppState {
        tee: Arc::new(tee),
        x_status: x_status.clone(),
        cose_namespace: cfg.icp.cose_namespace.clone(),
        info: Arc::new(handler::AppInformation {
            id: my_principal,
            name: engine_name,
            start_time_ms: unix_ms(),
            default_agent,
            object_store_canister: Some(object_store_canister),
            caller: Principal::anonymous(),
        }),
    };

    match tokio::try_join!(
        start_server(
            format!("127.0.0.1:{}", cli.port),
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

async fn connect_tee(
    cfg: &config::Conf,
    tee: &TEEClient,
    cancel_token: CancellationToken,
) -> Result<TEEAppInformation, BoxError> {
    loop {
        if let Ok(tee_info) = tee
            .http
            .get(format!("{}/information", &cfg.tee.tee_host))
            .send()
            .await
        {
            let tee_info = tee_info.bytes().await?;
            let tee_info: TEEAppInformation = from_reader(&tee_info[..])?;
            return Ok(tee_info);
        }

        tokio::select! {
            _ = cancel_token.cancelled() => {
                return Err("connect_tee cancelled".into());
            },
            _ = sleep(Duration::from_secs(2)) => {},
        }
        log::info!(target: LOG_TARGET, "connecting TEE service again");
    }
}

async fn connect_object_store(
    tee: &TEEClient,
    ic_agent: Arc<Agent>,
    root_path: &Path,
    object_store_canister: Principal,
) -> Result<ObjectStoreClient, BoxError> {
    let aes_secret = tee
        .a256gcm_key(root_path, &[IC_OBJECT_STORE.as_bytes(), b"A256GCM"])
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
    object_store: Arc<ObjectStoreClient>,
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
    ks.create_index().await?;
    Ok(ks)
}

fn connect_model(cfg: &config::Llm) -> Result<Model, BoxError> {
    if cfg.openai_api_key.is_empty() {
        Ok(Model::new(
            Arc::new(
                cohere::Client::new(&cfg.cohere_api_key)
                    .embedding_model(&cfg.cohere_embedding_model),
            ),
            Arc::new(
                deepseek::Client::new(&cfg.deepseek_api_key)
                    .completion_model(deepseek::DEEKSEEK_V3),
            ),
        ))
    } else {
        let cli = openai::Client::new(&cfg.openai_api_key);
        Ok(Model::new(
            Arc::new(cli.embedding_model(&cfg.openai_embedding_model)),
            Arc::new(cli.completion_model(&cfg.openai_completion_model)),
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
