use anda_core::Resource;
use anda_engine::engine::{Engine, InformationJSON};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use candid::Principal;
use ciborium::from_reader;
use ic_cose_types::to_cbor_bytes;
use ic_tee_agent::{
    RPCRequest, RPCResponse,
    http::{ANONYMOUS_PRINCIPAL, Content, ContentWithSHA3, UserSignature},
};
use std::collections::BTreeMap;
use std::sync::Arc;
use structured_logger::unix_ms;

use crate::{ic_sig_verifier::verify_sig, types::*};

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
    let caller = if let Some(sig) = UserSignature::try_from(&headers) {
        match sig.verify_with(unix_ms(), verify_sig, None, None) {
            Ok(_) => sig.user,
            Err(_) => ANONYMOUS_PRINCIPAL,
        }
    } else {
        ANONYMOUS_PRINCIPAL
    };

    let info = AppInformation {
        engines: app
            .engines
            .iter()
            .map(|(_, e)| e.information(false))
            .collect(),
        default_engine: app.default_engine,
        start_time_ms: app.start_time_ms,
        caller,
    };

    match Content::from(&headers) {
        Content::CBOR(_, _) => Content::CBOR(info, None).into_response(),
        _ => Content::JSON(AppInformationJSON::from(info), None).into_response(),
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
        return (StatusCode::BAD_REQUEST, format!("invalid engine id: {id}")).into_response();
    };

    match app.engines.get(&id) {
        Some(engine) => {
            let info = engine.information(true);
            match Content::from(&headers) {
                Content::CBOR(_, _) => Content::CBOR(info, None).into_response(),
                _ => Content::JSON(InformationJSON::from(info), None).into_response(),
            }
        }
        None => (StatusCode::NOT_FOUND, format!("engine {id} not found")).into_response(),
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
        return (StatusCode::BAD_REQUEST, format!("invalid engine id: {id}")).into_response();
    };

    let (req, hash) = match ct {
        ContentWithSHA3::CBOR(req, hash) => (req, hash),
        ContentWithSHA3::JSON(_, _) => {
            return StatusCode::UNSUPPORTED_MEDIA_TYPE.into_response();
        }
    };

    let caller = if let Some(sig) = UserSignature::try_from(&headers) {
        match sig.verify_with(unix_ms(), verify_sig, Some(id), Some(hash.as_slice())) {
            Ok(_) => sig.user,
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
    let res = engine_run(&req, &app, caller, id).await;
    Content::CBOR(res, None).into_response()
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
        .ok_or_else(|| format!("engine {id} not found"))?;

    match req.method.as_str() {
        "agent_run" => {
            let args: (Option<String>, String, Option<Vec<Resource>>) =
                from_reader(req.params.as_slice())
                    .map_err(|err| format!("failed to decode params: {err:?}"))?;
            let res = engine
                .agent_run(args.0, args.1, args.2, caller, None)
                .await
                .map_err(|err| format!("failed to run agent: {err:?}"))?;
            Ok(to_cbor_bytes(&res).into())
        }
        "tool_call" => {
            let args: (String, String) = from_reader(req.params.as_slice())
                .map_err(|err| format!("failed to decode params: {err:?}"))?;
            let res = engine
                .tool_call(args.0, args.1, caller, None)
                .await
                .map_err(|err| format!("failed to call tool: {err:?}"))?;
            Ok(to_cbor_bytes(&res).into())
        }
        "information" => {
            let args: (bool,) = from_reader(req.params.as_slice())
                .map_err(|err| format!("failed to decode params: {err:?}"))?;
            let res = engine.information(args.0);
            Ok(to_cbor_bytes(&res).into())
        }
        method => Err(format!("{method} on engine {id} not implemented")),
    }
}
