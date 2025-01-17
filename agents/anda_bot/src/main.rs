use agent_twitter_client::scraper::Scraper;
use anda_core::{BoxError, EmbeddingFeatures, Path};
use anda_engine::{
    context::TEEClient,
    engine::{Engine, EngineBuilder, ROOT_PATH},
    extension::{
        attention::Attention,
        character::{Character, CharacterAgent},
        segmenter::DocumentSegmenter,
    },
    model::{cohere, deepseek, openai, Model},
    store::Store,
};
use anda_lancedb::{knowledge::KnowledgeStore, lancedb::LanceVectorStore};
use axum::{routing, Router};
use candid::Principal;
use ciborium::from_reader;
use ed25519_consensus::SigningKey;
use ic_agent::identity::{BasicIdentity, Identity};
use ic_cose_types::types::object_store::CHUNK_SIZE;
use ic_object_store::{
    agent::build_agent,
    client::{Client, ObjectStoreClient},
};
use ic_tee_cdk::TEEAppInformation;
use std::{net::SocketAddr, sync::Arc, time::Duration};
use structured_logger::{async_json::new_writer, unix_ms, Builder};
use tokio::{signal, sync::RwLock};
use tokio_util::sync::CancellationToken;

mod config;
mod handler;
mod twitter;

const APP_NAME: &str = env!("CARGO_PKG_NAME");
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

static LOG_TARGET: &str = "bootstrap";
static IC_OBJECT_STORE: &str = "ic://object_store";
const LOCAL_SERVER_SHUTDOWN_DURATION: Duration = Duration::from_secs(5);

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let cfg = config::Conf::new().unwrap_or_else(|err| panic!("config error: {}", err));
    Builder::with_level(cfg.log.level.as_str())
        .with_target_writer("*", new_writer(tokio::io::stdout()))
        .init();

    log::debug!("{:?}", cfg);
    let global_cancel_token = CancellationToken::new();
    let shutdown_future = shutdown_signal(global_cancel_token.clone());

    let character = Character::from_toml(&cfg.character.content)?;
    let default_agent = character.username.clone();
    let engine_name = "Anda.bot".to_string();

    let tee = TEEClient::new(&cfg.tee.tee_host, &cfg.tee.basic_token);
    let info = tee.http.get(&cfg.tee.tee_host).send().await?;
    let info = info.bytes().await?;
    let info: TEEAppInformation = from_reader(&info[..])?;
    log::debug!("TEEAppInformation: {:?}", cfg);

    let model = if cfg.llm.openai_api_key.is_empty() {
        Model::new(
            Arc::new(
                cohere::Client::new(&cfg.llm.cohere_api_key)
                    .embedding_model(&cfg.llm.cohere_embedding_model),
            ),
            Arc::new(deepseek::Client::new(&cfg.llm.deepseek_api_key).completion_model()),
        )
    } else {
        let cli = openai::Client::new(&cfg.llm.openai_api_key);
        Model::new(
            Arc::new(cli.embedding_model(&cfg.llm.openai_embedding_model)),
            Arc::new(cli.completion_model(&cfg.llm.openai_completion_model)),
        )
    };

    let ndims = model.ndims();

    // ObjectStore
    let object_store_canister = Principal::from_text(cfg.icp.object_store_canister)?;
    let root_path = Path::from(ROOT_PATH);
    let (object_store_client, object_store_client_id) = {
        let id_secret = tee
            .a256gcm_key(&root_path, &[IC_OBJECT_STORE.as_bytes()])
            .await?;
        let aes_secret = tee
            .a256gcm_key(&root_path, &[IC_OBJECT_STORE.as_bytes(), b"A256GCM"])
            .await?;
        let sk = SigningKey::from(id_secret);
        let id = BasicIdentity::from_signing_key(sk);
        let object_store_client_id = id.sender()?;
        let agent = build_agent(&cfg.icp.api_host, Arc::new(id)).await.unwrap();
        let cli = Arc::new(Client::new(
            Arc::new(agent),
            object_store_canister,
            Some(aes_secret),
        ));
        (cli, object_store_client_id)
    };

    let object_store_status = object_store_client.head(&Path::from("information")).await;
    let x_status = Arc::new(RwLock::new(if object_store_status.is_err() {
        // object store is not available
        handler::ServiceStatus::Stopped
    } else {
        handler::ServiceStatus::Running
    }));

    let app_state = handler::AppState {
        x_status: x_status.clone(),
        info: Arc::new(handler::AppInformation {
            id: info.id,
            name: engine_name.clone(),
            start_time_ms: unix_ms(),
            default_agent: default_agent.clone(),
            object_store_client: Some(object_store_client_id),
            object_store_canister: Some(object_store_canister),
            caller: Principal::anonymous(),
        }),
    };

    if object_store_status.is_err() {
        match tokio::try_join!(
            start_server(
                format!("127.0.0.1:{}", cfg.server.port),
                app_state,
                global_cancel_token.clone()
            ),
            shutdown_future
        ) {
            Ok(_) => return Ok(()),
            Err(err) => {
                log::error!(target: LOG_TARGET, "server error: {:?}", err);
                return Err(err);
            }
        }
    }

    let object_store = Arc::new(ObjectStoreClient::new(object_store_client));
    let knowledge_store: KnowledgeStore = {
        let mut store = LanceVectorStore::new_with_object_store(
            IC_OBJECT_STORE.to_string(),
            object_store.clone(),
            Some(CHUNK_SIZE),
            None,
        )
        .await?;

        let namespace: Path = default_agent.clone().into();
        let ks = KnowledgeStore::init(&mut store, namespace, ndims as u16, Some(1024 * 10)).await?;

        ks.create_index().await?;
        ks
    };

    let agent = character.build(
        Attention::default(),
        DocumentSegmenter::default(),
        knowledge_store,
    );
    let engine = EngineBuilder::new()
        .with_name(engine_name)
        .with_cancellation_token(global_cancel_token.clone())
        .with_tee_client(tee)
        .with_model(model)
        .with_store(Store::new(object_store))
        .register_agent(agent.clone())?;

    let agent = Arc::new(agent);
    let engine = Arc::new(engine.build(default_agent)?);

    match tokio::try_join!(
        start_server(
            format!("127.0.0.1:{}", cfg.server.port),
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

    if let Some(cookie_str) = cfg.cookie_string {
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
