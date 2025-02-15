use anda_engine::context::TEEClient;
use anda_lancedb::knowledge::KnowledgeStore;
use axum::{
    extract::{Request, State},
    http::StatusCode,
    response::IntoResponse,
};
use candid::Principal;
use ic_cose::client::CoseSDK;
use ic_cose_types::to_cbor_bytes;
use ic_tee_agent::{
    http::{Content, UserSignature, ANONYMOUS_PRINCIPAL, HEADER_IC_TEE_CALLER},
    RPCRequest, RPCResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use structured_logger::unix_ms;
use tokio::sync::RwLock;

use crate::ic_sig_verifier::verify_sig;

#[derive(Clone)]
pub struct AppState {
    pub web3: Arc<Web3SDK>,
    pub x_status: Arc<RwLock<ServiceStatus>>,
    pub info: Arc<AppInformation>,
    pub knowledge_store: Arc<KnowledgeStore>,
    pub cose_namespace: String, // used for TEE
    pub manager: String,        // used for local
}

pub enum Web3SDK {
    Tee(TEEClient),
    Web3(anda_web3_client::client::Client),
}

impl AppState {
    pub async fn is_manager(&self, headers: &http::HeaderMap) -> bool {
        match self.web3.as_ref() {
            Web3SDK::Tee(cli) => {
                // signature is verified by the TEE gateway
                let caller = get_caller(headers);
                caller != ANONYMOUS_PRINCIPAL
                    && cli
                        .namespace_is_member(&self.cose_namespace, "manager", &caller)
                        .await
                        .unwrap_or(false)
            }
            Web3SDK::Web3(_cli) => {
                // verify signature
                let caller = if let Some(sig) = UserSignature::try_from(headers) {
                    match sig.verify_with(unix_ms(), verify_sig, Some(self.info.id), None) {
                        Ok(_) => sig.user,
                        Err(_) => {
                            return false;
                        }
                    }
                } else {
                    return false;
                };
                caller.to_text() == self.manager
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum ServiceStatus {
    #[default]
    Stopped,
    Running,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AppInformation {
    pub id: Principal,
    pub name: String, // engine name
    pub start_time_ms: u64,
    pub default_agent: String,
    pub object_store_canister: Option<Principal>,
    pub caller: Principal,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AppInformationJSON {
    pub id: String,   // TEE service id
    pub name: String, // engine name
    pub start_time_ms: u64,
    pub default_agent: String,
    pub object_store_canister: Option<String>,
    pub caller: String,
}

/// GET /.well-known/app
pub async fn get_information(State(app): State<AppState>, req: Request) -> impl IntoResponse {
    let mut info = app.info.as_ref().clone();
    let headers = req.headers();
    info.caller = get_caller(headers);
    match Content::from(headers) {
        Content::CBOR(_, _) => Content::CBOR(info, None).into_response(),
        _ => Content::JSON(
            AppInformationJSON {
                id: info.id.to_string(),
                name: info.name,
                start_time_ms: info.start_time_ms,
                default_agent: info.default_agent.clone(),
                object_store_canister: info.object_store_canister.as_ref().map(|p| p.to_string()),
                caller: info.caller.to_string(),
            },
            None,
        )
        .into_response(),
    }
}

pub async fn add_proposal(
    State(app): State<AppState>,
    headers: http::HeaderMap,
    ct: Content<RPCRequest>,
) -> impl IntoResponse {
    match ct {
        Content::CBOR(req, _) | Content::JSON(req, _) => {
            let is_manager = app.is_manager(&headers).await;
            if !is_manager {
                return StatusCode::FORBIDDEN.into_response();
            }

            let res = handle_proposal(&req, &app).await;
            Content::CBOR(res, None).into_response()
        }
        _ => StatusCode::UNSUPPORTED_MEDIA_TYPE.into_response(),
    }
}

async fn handle_proposal(req: &RPCRequest, app: &AppState) -> RPCResponse {
    match req.method.as_str() {
        "start_x_bot" => {
            let mut x_status = app.x_status.write().await;
            *x_status = ServiceStatus::Running;
            Ok(to_cbor_bytes(&"Ok").into())
        }
        "stop_x_bot" => {
            let mut x_status = app.x_status.write().await;
            *x_status = ServiceStatus::Stopped;
            Ok(to_cbor_bytes(&"Ok").into())
        }
        "knowledge_store_create_index" => {
            let knowledge_store = app.knowledge_store.clone();
            tokio::spawn(async move {
                if let Err(err) = knowledge_store.create_index().await {
                    log::error!("knowledge_store: failed to create index: {}", err);
                }
            });
            Ok(to_cbor_bytes(&"Ok").into())
        }
        "knowledge_store_optimize" => {
            let knowledge_store = app.knowledge_store.clone();
            tokio::spawn(async move {
                if let Err(err) = knowledge_store.optimize().await {
                    log::error!("knowledge_store: failed to optimize: {}", err);
                }
            });
            Ok(to_cbor_bytes(&"Ok").into())
        }
        _ => Err(format!("unsupported method {}", req.method)),
    }
}

fn get_caller(headers: &http::HeaderMap) -> Principal {
    if let Some(caller) = headers.get(&HEADER_IC_TEE_CALLER) {
        if let Ok(caller) = Principal::from_text(caller.to_str().unwrap_or_default()) {
            caller
        } else {
            ANONYMOUS_PRINCIPAL
        }
    } else {
        ANONYMOUS_PRINCIPAL
    }
}
