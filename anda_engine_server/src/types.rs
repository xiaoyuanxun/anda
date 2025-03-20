use anda_engine::context::Information;
use candid::Principal;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AppInformation {
    pub engines: Vec<Information>,
    pub default_engine: Principal,
    pub caller: Principal,
    pub start_time_ms: u64,
}
