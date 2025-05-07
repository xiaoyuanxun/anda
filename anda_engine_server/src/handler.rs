use anda_core::{AgentInput, ToolInput, Value};
use anda_engine::engine::{Engine, Information};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use candid::Principal;
use ciborium::from_reader;
use ic_auth_verifier::envelope::{ANONYMOUS_PRINCIPAL, SignedEnvelope, unix_ms};
use ic_cose_types::to_cbor_bytes;
use ic_tee_agent::{
    RPCRequest, RPCResponse,
    http::{Content, ContentWithSHA3},
};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::types::*;

#[derive(Clone)]
pub struct AppState {
    pub(crate) engines: Arc<BTreeMap<Principal, Engine>>,
    pub(crate) default_engine: Principal,
    pub(crate) start_time_ms: u64,
}

/// GET /.well-known/information
pub async fn get_information(
    State(app): State<AppState>,
    headers: http::HeaderMap,
) -> impl IntoResponse {
    let caller = if let Some(se) = SignedEnvelope::from_authorization(&headers)
        .or_else(|| SignedEnvelope::from_headers(&headers))
    {
        match se.verify(unix_ms(), None, None) {
            Ok(_) => se.sender(),
            Err(_) => ANONYMOUS_PRINCIPAL,
        }
    } else {
        ANONYMOUS_PRINCIPAL
    };

    let info = AppInformation {
        engines: app
            .engines
            .iter()
            .map(|(_, e)| Information {
                id: e.id(),
                name: e.name(),
                description: e.description(),
                agents: vec![],
                tools: vec![],
                endpoint: "".to_string(),
            })
            .collect(),
        default_engine: app.default_engine,
        start_time_ms: app.start_time_ms,
        caller,
    };

    match Content::from(&headers) {
        Content::CBOR(_, _) => Content::CBOR(info, None).into_response(),
        _ => Content::JSON(info, None).into_response(),
    }
}

/// GET /.well-known/information/{id}
pub async fn get_engine_information(
    State(app): State<AppState>,
    headers: http::HeaderMap,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let id = if &id == "default" {
        app.default_engine
    } else if let Ok(id) = Principal::from_text(&id) {
        id
    } else {
        return (
            StatusCode::BAD_REQUEST,
            format!("invalid engine id: {id:?}"),
        )
            .into_response();
    };

    match app.engines.get(&id) {
        Some(engine) => {
            let info = engine.information();
            match Content::from(&headers) {
                Content::CBOR(_, _) => Content::CBOR(info, None).into_response(),
                _ => Content::JSON(info, None).into_response(),
            }
        }
        None => (
            StatusCode::NOT_FOUND,
            format!("engine {} not found", id.to_text()),
        )
            .into_response(),
    }
}

/// POST /{*id}
pub async fn anda_engine(
    State(app): State<AppState>,
    headers: http::HeaderMap,
    Path(id): Path<String>,
    ct: ContentWithSHA3<RPCRequest>,
) -> impl IntoResponse {
    let id = if &id == "default" {
        app.default_engine
    } else if let Ok(id) = Principal::from_text(&id) {
        id
    } else {
        return (
            StatusCode::BAD_REQUEST,
            format!("invalid engine id: {id:?}"),
        )
            .into_response();
    };

    let (req, hash) = match &ct {
        ContentWithSHA3::CBOR(req, hash) => (req, hash),
        ContentWithSHA3::JSON(req, hash) => (req, hash),
    };

    let caller = if let Some(se) = SignedEnvelope::from_authorization(&headers)
        .or_else(|| SignedEnvelope::from_headers(&headers))
    {
        match se.verify(unix_ms(), Some(id), Some(hash.as_slice())) {
            Ok(_) => se.sender(),
            Err(_) => ANONYMOUS_PRINCIPAL,
        }
    } else {
        ANONYMOUS_PRINCIPAL
    };

    log::info!(
        method = req.method.as_str(),
        agent = id.to_text(),
        caller = caller.to_text();
        "anda_engine",
    );
    let res = engine_run(req, &app, caller, id).await;
    match &ct {
        ContentWithSHA3::CBOR(_, _) => Content::CBOR(res, None).into_response(),
        ContentWithSHA3::JSON(_, _) => Content::JSON(res, None).into_response(),
    }
}

async fn engine_run(
    req: &RPCRequest,
    app: &AppState,
    caller: Principal,
    id: Principal,
) -> RPCResponse {
    let engine = app
        .engines
        .get(&id)
        .ok_or_else(|| format!("engine {} not found", id.to_text()))?;

    match req.method.as_str() {
        "agent_run" => {
            let args: (AgentInput,) = from_reader(req.params.as_slice())
                .map_err(|err| format!("failed to decode params: {err:?}"))?;
            let res = engine
                .agent_run(caller, args.0)
                .await
                .map_err(|err| format!("failed to run agent: {err:?}"))?;
            Ok(to_cbor_bytes(&res).into())
        }
        "tool_call" => {
            let args: (ToolInput<Value>,) = from_reader(req.params.as_slice())
                .map_err(|err| format!("failed to decode params: {err:?}"))?;
            let res = engine
                .tool_call(caller, args.0)
                .await
                .map_err(|err| format!("failed to call tool: {err:?}"))?;
            Ok(to_cbor_bytes(&res).into())
        }
        "information" => {
            let res = engine.information();
            Ok(to_cbor_bytes(&res).into())
        }
        method => Err(format!(
            "{method} on engine {} not implemented",
            id.to_text()
        )),
    }
}
