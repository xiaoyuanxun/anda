use axum::{
    extract::{Request, State},
    response::IntoResponse,
};
use candid::Principal;
use ic_auth_verifier::envelope::extract_user;
use ic_tee_agent::http::Content;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub info: Arc<AppInformation>,
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
