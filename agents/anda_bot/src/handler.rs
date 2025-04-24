use anda_engine::context::TEEClient;
use anda_kdb::KnowledgeStore;
use axum::{
    extract::{Request, State},
    http::StatusCode,
    response::IntoResponse,
};
use candid::Principal;
use ic_auth_verifier::envelope::{ANONYMOUS_PRINCIPAL, SignedEnvelope, extract_user, unix_ms};
use ic_cose::client::CoseSDK;
use ic_cose_types::to_cbor_bytes;
use ic_tee_agent::{RPCRequest, RPCResponse, http::Content};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AppState {
    pub web3: Arc<Web3SDK>,
    pub x_status: Arc<RwLock<ServiceStatus>>,
    pub info: Arc<AppInformation>,
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
                let caller = extract_user(headers);
                caller != ANONYMOUS_PRINCIPAL
                    && cli
                        .namespace_is_member(&self.cose_namespace, "manager", &caller)
                        .await
                        .unwrap_or(false)
            }
            Web3SDK::Web3(_cli) => {
                // verify signature
                let caller = if let Some(se) = SignedEnvelope::try_from(headers) {
                    match se.verify(unix_ms(), Some(self.info.id), None) {
                        Ok(_) => se.sender(),
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
    info.caller = extract_user(headers);
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
        _ => Err(format!("unsupported method {}", req.method)),
    }
}
